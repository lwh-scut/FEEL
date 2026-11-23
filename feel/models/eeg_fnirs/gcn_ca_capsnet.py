import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from scipy.signal import resample, welch
import numpy as np

# ----------------------------
# 工具函数：微分熵（DE）
# ----------------------------
def differential_entropy(x):
    var = torch.var(x, dim=-1, unbiased=False) + 1e-8
    de = 0.5 * torch.log(2 * np.pi * np.e * var)
    return de


# ----------------------------
# 下采样 EEG 到 fNIRS 采样率
# ----------------------------
def downsample_eeg_to_fnirs(eeg, target_len):
    B, C, T = eeg.shape
    eeg_np = eeg.cpu().numpy()
    downsampled = np.zeros((B, C, target_len))
    for b in range(B):
        for c in range(C):
            downsampled[b, c] = resample(eeg_np[b, c], target_len)
    return torch.from_numpy(downsampled).to(eeg.device)


# ----------------------------
# 构建邻接矩阵（Pearson + top-k%）
# ----------------------------
def build_adjacency_matrix(eeg_ts, fnirs_ts, top_k_percent=40):
    B, _, T = eeg_ts.shape
    combined = torch.cat([eeg_ts, fnirs_ts], dim=1)  # (B, 72, T)
    combined = (combined - combined.mean(dim=-1, keepdim=True)) / (combined.std(dim=-1, keepdim=True) + 1e-8)
    adj = torch.bmm(combined, combined.transpose(-1, -2)) / (T - 1)
    adj = torch.abs(adj)

    num_nodes = combined.shape[1]
    k = int(num_nodes * num_nodes * top_k_percent / 100)
    for b in range(B):
        flat = adj[b].view(-1)
        if k >= flat.numel():
            continue
        threshold = torch.topk(flat, k, largest=True).values[-1]
        mask = adj[b] >= threshold
        adj[b] = mask.float() * adj[b]
    return adj


# ----------------------------
# fNIRS 特征提取（含 0.01–0.1 Hz PSD）
# ----------------------------
def compute_fnirs_features(fnirs_raw, fs=10.0):
    fnirs_np = fnirs_raw.cpu().numpy()
    B, C, T = fnirs_np.shape

    mean_feat = np.mean(fnirs_np, axis=-1)
    var_feat = np.var(fnirs_np, axis=-1)
    std_feat = np.sqrt(var_feat + 1e-8)
    skew_feat = np.mean(((fnirs_np - mean_feat[..., None]) / std_feat[..., None]) ** 3, axis=-1)

    psd_vals = np.zeros((B, C))
    for b in range(B):
        for c in range(C):
            f, Pxx = welch(
                fnirs_np[b, c],
                fs=fs,
                nperseg=min(16, T),
                window='hann'
            )
            valid = (f >= 0.01) & (f <= 0.1)
            if np.any(valid):
                psd_vals[b, c] = np.mean(Pxx[valid])
            else:
                psd_vals[b, c] = np.mean(Pxx)

    eps = 1e-8
    de_feat = 0.5 * np.log(2 * np.pi * np.e * (var_feat + eps))
    features = np.stack([mean_feat, var_feat, skew_feat, psd_vals, de_feat], axis=-1)
    return torch.from_numpy(features).to(fnirs_raw.device).float()


# ----------------------------
# Squash 函数
# ----------------------------
def squash(capsules, dim=-1):
    norm_sq = torch.sum(capsules ** 2, dim=dim, keepdim=True)
    norm = torch.sqrt(norm_sq)
    scale = norm_sq / (1 + norm_sq)
    return scale * capsules / (norm + 1e-8)


import numpy as np
from scipy.signal import butter, filtfilt


def compute_band_de(eeg_raw, fs=1000.0):
    """
    eeg_raw: (B, C, T) - PyTorch tensor, possibly on GPU
    Returns: (B, C, num_bands) - PyTorch tensor on same device as input
    """
    device = eeg_raw.device  # 记住原始设备（比如 cuda:0）

    # 转为 CPU NumPy 数组以供 scipy 处理
    eeg_np = eeg_raw.cpu().numpy()  # (B, C, T)

    bands = {
        'delta': [1, 4],
        'theta': [4, 8],
        'alpha': [8, 13],
        'beta': [13, 30],
        'gamma': [30, 50]
    }

    de_features = []
    for band, (low, high) in bands.items():
        b, a = butter(N=4, Wn=[low, high], btype='band', fs=fs)
        # filtfilt 需要沿时间轴（-1）滤波
        filtered = filtfilt(b, a, eeg_np, axis=-1)  # (B, C, T)
        # 计算微分熵（假设使用 log(var) 作为近似）
        var = np.var(filtered, axis=-1, keepdims=False)  # (B, C)
        # 避免除零或 log(0)
        var = np.clip(var, 1e-6, None)
        de = np.log(var)  # (B, C)
        de_features.append(de)

    # 拼接所有频段：(B, C, 5)
    de_all = np.stack(de_features, axis=-1)  # (B, C, 5)

    # 转回 PyTorch 张量，并移回原始设备
    return torch.from_numpy(de_all).to(device).float()

# ----------------------------
# 严格对齐论文的 GCN-CA-CapsNet
# ----------------------------
class GCN_CA_CapsNet(nn.Module):
    def __init__(self, num_classes=4, eeg_chans=32, fnirs_chans=40):
        super().__init__()
        self.num_emotions = num_classes
        self.num_nodes = eeg_chans + fnirs_chans  # 72
        self.gcn1 = SAGEConv(in_channels=5, out_channels=16)
        self.gcn2 = SAGEConv(in_channels=16, out_channels=16)

        # === Capsule Attention Modules (Fig. 4) ===
        # Node Attention: per-dimension within capsule
        self.node_att_mlp = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.Sigmoid()
        )
        # Hierarchy Attention: scalar per hierarchy (layer1 vs layer2)
        self.hier_att_fc = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        # === Learnable transformation for dynamic routing ===
        # W_ij: (num_primary_caps, num_emotions, out_dim, in_dim)
        self.W_digicap = nn.Parameter(
            torch.randn(self.num_nodes * 2, num_classes, 16, 16)
        )

    def forward(self, eeg_raw, fnirs_raw):
        device = eeg_raw.device
        B = eeg_raw.shape[0]

        eeg_features = compute_band_de(eeg_raw, fs=1000.0)      # (B, 32, 5)
        fnirs_features = compute_fnirs_features(fnirs_raw, fs=10.0)  # (B, 40, 5)

        # ---------- Step 2: Build Graph ----------
        T_fnirs = fnirs_raw.shape[-1]
        eeg_downsampled = downsample_eeg_to_fnirs(eeg_raw, T_fnirs)  # (B, 32, 30)
        adj = build_adjacency_matrix(eeg_downsampled, fnirs_raw, top_k_percent=40)  # (B, 72, 72)

        # ---------- Step 3: Two-layer GCN ----------
        x = torch.cat([eeg_features, fnirs_features], dim=1)  # (B, 72, 5)
        gcn_out1_list = []
        gcn_out2_list = []
        for b in range(B):
            edge_index = adj[b].nonzero().t().contiguous()
            if edge_index.numel() == 0:
                # fallback: fully connected
                idx = torch.arange(self.num_nodes, device=device)
                edge_index = torch.combinations(idx, r=2).t()
                edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            x_b = x[b]  # (72, 5)
            out1 = self.gcn1(x_b, edge_index)  # (72, 16)
            out2 = self.gcn2(out1, edge_index)  # (72, 16)
            gcn_out1_list.append(out1)
            gcn_out2_list.append(out2)
        gcn_out1 = torch.stack(gcn_out1_list, dim=0)  # (B, 72, 16)
        gcn_out2 = torch.stack(gcn_out2_list, dim=0)  # (B, 72, 16)

        # ---------- Step 4: Capsule Attention (Fig. 4) ----------
        # Split hierarchies
        caps_l1 = gcn_out1  # (B, 72, 16)
        caps_l2 = gcn_out2  # (B, 72, 16)

        # Node Attention: per-dimension
        node_att_l1 = self.node_att_mlp(caps_l1)  # (B, 72, 16)
        node_att_l2 = self.node_att_mlp(caps_l2)  # (B, 72, 16)

        # Hierarchy Attention: scalar per hierarchy
        hier_att_l1 = self.hier_att_fc(torch.mean(caps_l1, dim=1))  # (B, 1)
        hier_att_l2 = self.hier_att_fc(torch.mean(caps_l2, dim=1))  # (B, 1)

        # Apply attention
        weighted_l1 = caps_l1 * node_att_l1 * hier_att_l1.unsqueeze(-1)
        weighted_l2 = caps_l2 * node_att_l2 * hier_att_l2.unsqueeze(-1)
        primary_caps = torch.cat([weighted_l1, weighted_l2], dim=1)  # (B, 144, 16)

        # ---------- Step 5: Dynamic Routing with W_ij ----------
        # Reshape for matrix multiplication
        u_tiled = primary_caps.unsqueeze(2).unsqueeze(-1)  # (B, 144, 1, 16, 1)
        W_tiled = self.W_digicap.unsqueeze(0)              # (1, 144, 4, 16, 16)
        u_hat = torch.squeeze(torch.matmul(W_tiled, u_tiled), -1)  # (B, 144, 4, 16)

        # Initialize logits
        b_ij = torch.zeros(B, self.num_nodes * 2, self.num_emotions, 1, device=device)

        # Routing iterations
        for _ in range(3):
            c_ij = F.softmax(b_ij, dim=1)  # (B, 144, 4, 1)
            s_j = torch.sum(c_ij * u_hat, dim=1)  # (B, 4, 16)
            v_j = squash(s_j)  # (B, 4, 16)
            # Agreement
            a_ij = torch.sum(u_hat * v_j.unsqueeze(1), dim=-1, keepdim=True)  # (B, 144, 4, 1)
            b_ij = b_ij + a_ij

        return v_j  # (B, 4, 16)

    def margin_loss(self, caps_output, targets, m_plus=0.9, m_minus=0.1, lambda_=0.5):
        v_mag = torch.norm(caps_output, dim=-1)  # (B, 4)
        T_c = F.one_hot(targets, num_classes=self.num_emotions).float()
        L_c = T_c * F.relu(m_plus - v_mag) ** 2 + \
              lambda_ * (1 - T_c) * F.relu(v_mag - m_minus) ** 2
        return L_c.sum(dim=1).mean()


# ----------------------------
# 使用示例
# ----------------------------
if __name__ == "__main__":
    model = GCN_CA_CapsNet(num_emotions=4).cuda()
    B = 4
    eeg = torch.randn(B, 32, 3000).cuda()
    fnirs = torch.randn(B, 40, 30).cuda()
    labels = torch.randint(0, 4, (B,)).cuda()

    output_caps = model(eeg, fnirs)
    loss = model.margin_loss(output_caps, labels)
    print("Loss:", loss.item())
