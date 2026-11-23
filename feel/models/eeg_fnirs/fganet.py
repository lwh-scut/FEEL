import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import griddata


# ----------------------------
# 工具函数：获取真实坐标（模拟）
# ----------------------------
def get_eeg_coords(num_ch=32):
    """模拟EEG电极位置"""
    np.random.seed(42)
    coords = np.random.randn(num_ch, 3)
    coords = coords / np.linalg.norm(coords, axis=1, keepdims=True)
    return coords.astype(np.float32)


def get_fnirs_coords(num_sources=20, num_detectors=20):
    """模拟fNIRS源和探测器位置"""
    np.random.seed(43)
    sources = np.random.randn(num_sources, 3)
    sources = sources / np.linalg.norm(sources, axis=1, keepdims=True)
    detectors = np.random.randn(num_detectors, 3)
    detectors = detectors / np.linalg.norm(detectors, axis=1, keepdims=True)
    return sources.astype(np.float32), detectors.astype(np.float32)


# ----------------------------
# 投影到2D网格（azimuthal equidistant projection + cubic interpolation）
# ----------------------------
def azimuthal_equidistant_projection(coords_3d):
    x, y, z = coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2]
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x)
    r = theta
    proj_x = r * np.cos(phi)
    proj_y = r * np.sin(phi)
    return np.stack([proj_x, proj_y], axis=1)


def project_to_2d_grid(signal, coords_3d, grid_size=16):
    """
    signal: [B, C, T]
    coords_3d: [C, 3]
    返回: [B, grid_size, grid_size, T]
    """
    B, C, T = signal.shape
    device = signal.device
    proj_2d = azimuthal_equidistant_projection(coords_3d)

    linspace = np.linspace(-np.pi, np.pi, grid_size)
    X, Y = np.meshgrid(linspace, linspace)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)

    output = torch.zeros(B, grid_size, grid_size, T, device=device)
    for b in range(B):
        for t in range(T):
            values = signal[b, :, t].cpu().numpy()
            interp_vals = griddata(proj_2d, values, grid_points, method='cubic', fill_value=0.0)
            interp_vals = interp_vals.reshape(grid_size, grid_size)
            output[b, :, :, t] = torch.from_numpy(interp_vals).to(device)
    return output


# ----------------------------
# Spatial Attentive Pooling (SAP)
# ----------------------------
class SpatialAttentivePooling(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(H, W))  # 可学习空间权重

    def forward(self, F):  # F: [B, H, W, T, C]
        B, H, W, T, C = F.shape
        alpha = self.alpha.view(1, H, W, 1, 1)
        weighted = F * alpha
        out = weighted.sum(dim=[1, 2])  # [B, T, C]
        return out


# ----------------------------
# Temporal Attentive Pooling (TAP)
# ----------------------------
class TemporalAttentivePooling(nn.Module):
    def __init__(self, T, C):
        super().__init__()
        self.fc = nn.Linear(C, 1)
        self.T = T

    def forward(self, F_sap):  # F_sap: [B, T, C]
        psi = self.fc(F_sap)  # [B, T, 1]
        psi = F.softmax(psi, dim=1)  # [B, T, 1]
        out = (F_sap * psi).sum(dim=1)  # [B, C]
        return out, psi.squeeze(-1)


# ----------------------------
# Conv3D Block
# ----------------------------
class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# ----------------------------
# FGA Layer (fNIRS-Guided Attention)
# ----------------------------
class FGALayer(nn.Module):
    def __init__(self, H, W, T_fnirs, C_in, C_out):
        super().__init__()
        self.conv_fga = nn.Conv3d(C_in, 1, kernel_size=(3, 3, 3), padding=1)
        self.sap = SpatialAttentivePooling(H, W)  # ✅ 新增：每层独立 SAP
        self.tap = TemporalAttentivePooling(T_fnirs, 1)  # 注意：输入通道是1
        self.H, self.W = H, W
        self.gamma_param = nn.Parameter(torch.tensor(0.0))

    def forward(self, F_eeg, F_fnirs):
        B, C, H, W, T_e = F_eeg.shape
        _, _, _, _, T_f = F_fnirs.shape

        # 1. 从 fNIRS 特征生成注意力图
        A = self.conv_fga(F_fnirs)  # [B, 1, H, W, T_f]
        A_perm = A.permute(0, 2, 3, 4, 1)  # [B, H, W, T_f, 1]

        # 2. 应用 SAP（可学习！）
        A_sap = self.sap(A_perm)  # [B, T_f, 1]  ← 关键修正！

        # 3. 应用 TAP
        A_tap, _ = self.tap(A_sap)  # [B, 1]
        Delta = A_tap.view(B, 1, 1, 1, 1).expand(B, 1, H, W, 1)
        Delta = torch.sigmoid(Delta)

        # 4. 应用注意力 + 残差连接
        gamma = torch.sigmoid(self.gamma_param)
        F_attended = F_eeg * Delta
        F_out = gamma * F_eeg + (1 - gamma) * F_attended
        return F_out


# ----------------------------
# FGANet 主模型（完全对齐 Fig. 5）
# ----------------------------
class FGANet(nn.Module):
    def __init__(self, eeg_ch=32, fnirs_ch=40, grid_size=16,
                 T_eeg=1500, T_fnirs=150, num_classes=2):
        super().__init__()
        self.grid_size = grid_size
        self.eeg_coords = get_eeg_coords(eeg_ch)
        self.fnirs_sources, self.fnirs_detectors = get_fnirs_coords(
            num_sources=fnirs_ch // 2, num_detectors=fnirs_ch // 2
        )
        self.fnirs_coords = np.concatenate([self.fnirs_sources, self.fnirs_detectors], axis=0)

        # === 共享参数的CNN块（EEG分支）===
        self.eeg_conv1 = Conv3DBlock(1, 32, kernel_size=(2, 2, 9), stride=(2, 2, 4))
        self.eeg_conv2 = Conv3DBlock(32, 64, kernel_size=(2, 2, 3), stride=(2, 2, 2))
        self.eeg_conv3 = Conv3DBlock(64, 64, kernel_size=(2, 2, 3), stride=(2, 2, 2))

        # === fNIRS分支（独立）===
        self.fnirs_conv1 = Conv3DBlock(1, 32, kernel_size=(2, 2, 3), stride=(2, 2, 2))
        self.fnirs_conv2 = Conv3DBlock(32, 64, kernel_size=(2, 2, 3), stride=(2, 2, 2))
        self.fnirs_conv3 = Conv3DBlock(64, 64, kernel_size=(2, 2, 3), stride=(2, 2, 2))

        # === FGA Layers（三层）===
        T1_eeg = (T_eeg - 9) // 4 + 1
        T2_eeg = (T1_eeg - 3) // 2 + 1
        T3_eeg = (T2_eeg - 3) // 2 + 1

        T1_fnirs = (T_fnirs - 3) // 2 + 1
        T2_fnirs = (T1_fnirs - 3) // 2 + 1
        T3_fnirs = (T2_fnirs - 3) // 2 + 1

        H1 = W1 = (grid_size - 2) // 2 + 1
        H2 = W2 = (H1 - 2) // 2 + 1
        H3 = W3 = (H2 - 2) // 2 + 1

        self.fga1 = FGALayer(H1, W1, T1_fnirs, 32, 32)
        self.fga2 = FGALayer(H2, W2, T2_fnirs, 64, 64)
        self.fga3 = FGALayer(H3, W3, T3_fnirs, 64, 64)

        # === Pooling & Classifiers ===
        self.sap3 = SpatialAttentivePooling(H3, W3)
        self.tap3 = TemporalAttentivePooling(T3_eeg, 64)

        self.classifier_eeg = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        self.classifier_fusion = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        self.classifier_fnirs = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        # 自适应融合权重
        self.weight_fc = nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, eeg, fnirs):
        B = eeg.size(0)

        # Step 1: 构建3D张量
        eeg_3d = project_to_2d_grid(eeg, self.eeg_coords, self.grid_size)  # [B, H, W, T_e]
        fnirs_3d = project_to_2d_grid(fnirs, self.fnirs_coords, self.grid_size)  # [B, H, W, T_f]

        # 转为 [B, 1, H, W, T]
        eeg_5d = eeg_3d.permute(0, 3, 1, 2).unsqueeze(1).permute(0, 1, 3, 4, 2)  # [B,1,H,W,T]
        fnirs_5d = fnirs_3d.permute(0, 3, 1, 2).unsqueeze(1).permute(0, 1, 3, 4, 2)

        # Step 2: EEG 分支（不带 FGA）
        e1 = self.eeg_conv1(eeg_5d)
        e2 = self.eeg_conv2(e1)
        e3 = self.eeg_conv3(e2)

        # Step 3: fNIRS 分支
        f1 = self.fnirs_conv1(fnirs_5d)
        f2 = self.fnirs_conv2(f1)
        f3 = self.fnirs_conv3(f2)

        # Step 4: Fusion 分支（EEG + FGA）
        fusion1 = self.fga1(e1, f1)  # FGA 作用于 EEG 分支的输出
        fusion2 = self.fga2(e2, f2)
        fusion3 = self.fga3(e3, f3)

        # Step 5: Pooling
        def pool_feature(feat):
            feat_perm = feat.permute(0, 2, 3, 4, 1)  # [B, H, W, T, C]
            sap_out = self.sap3(feat_perm)  # [B, T, C]
            tap_out, _ = self.tap3(sap_out)  # [B, C]
            return tap_out

        eeg_feat = pool_feature(e3)
        fusion_feat = pool_feature(fusion3)
        fnirs_feat = pool_feature(f3)

        y_eeg = self.classifier_eeg(eeg_feat)
        y_fusion = self.classifier_fusion(fusion_feat)
        y_fnirs = self.classifier_fnirs(fnirs_feat)

        # Step 6: 自适应融合
        weight_input = torch.cat([eeg_feat, fusion_feat], dim=1)
        weights = F.softmax(self.weight_fc(weight_input), dim=1)
        xi_eeg, xi_fusion = weights[:, 0], weights[:, 1]
        y_pred = xi_eeg.unsqueeze(1) * y_eeg + xi_fusion.unsqueeze(1) * y_fusion

        # return y_pred, y_eeg, y_fusion, y_fnirs
        return y_pred


# ----------------------------
# 测试脚本
# ----------------------------
if __name__ == "__main__":
    B = 4
    eeg = torch.randn(B, 32, 1500)
    fnirs = torch.randn(B, 40, 150)

    model = FGANet(eeg_ch=32, fnirs_ch=40, grid_size=16, T_eeg=1500, T_fnirs=150)
    y_pred, y_eeg, y_fusion, y_fnirs = model(eeg, fnirs)

    print("✅ Model forward pass successful!")
    print("Final prediction shape:", y_pred.shape)
    print("EEG branch:", y_eeg.shape)
    print("Fusion branch:", y_fusion.shape)
    print("fNIRS branch:", y_fnirs.shape)
