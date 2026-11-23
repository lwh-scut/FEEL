# ====== 更新后的 EFNet（适配 CrossEntropyLoss + 2分类） ======
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableSpatialEmbedding(nn.Module):
    def __init__(self, num_channels, grid_h=17, grid_w=17):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.embedding = nn.Parameter(torch.randn(num_channels, grid_h, grid_w))

    def forward(self, x):
        B, T, C = x.shape
        emb = self.embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, C, H, W]
        x = x.unsqueeze(-1).unsqueeze(-1)  # [B, T, C, 1, 1]
        out = (x * emb).sum(dim=2)  # [B, T, H, W]
        return out


class Conv2Plus1D(nn.Module):
    def __init__(self, in_ch, out_ch, kt, ks, st=1, ss=1, pt=0, ps=0):
        super().__init__()
        # Spatial: apply same 2D conv to every time step
        self.spatial_conv = nn.Conv3d(
            in_ch, out_ch,
            kernel_size=(1, ks, ks),
            stride=(1, ss, ss),
            padding=(0, ps, ps),
            bias=False
        )
        self.bn_s = nn.BatchNorm3d(out_ch)

        # Temporal: 1D conv along time, shared across space
        self.temporal_conv = nn.Conv3d(
            out_ch, out_ch,
            kernel_size=(kt, 1, 1),
            stride=(st, 1, 1),
            padding=(pt, 0, 0),
            bias=False
        )
        self.bn_t = nn.BatchNorm3d(out_ch)

    def forward(self, x):
        # x: [B, C, T, H, W]
        x = F.relu(self.bn_s(self.spatial_conv(x)))  # [B, C', T, H', W']
        x = F.relu(self.bn_t(self.temporal_conv(x)))  # [B, C', T', H', W']
        return x


class DownsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, is_eeg=True):
        super().__init__()
        if is_eeg:
            self.conv = Conv2Plus1D(in_ch, out_ch, kt=9, ks=7, st=4, ss=2, pt=4, ps=3)
            self.pool = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(2, 1, 1))
        else:
            self.conv = Conv2Plus1D(in_ch, out_ch, kt=3, ks=7, st=1, ss=2, pt=1, ps=3)
            self.pool = nn.Identity()

    def forward(self, x):
        return self.pool(self.conv(x))


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = Conv2Plus1D(ch, ch, kt=3, ks=3, pt=1, ps=1)
        self.conv2 = Conv2Plus1D(ch, ch, kt=3, ks=3, pt=1, ps=1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class EFNet(nn.Module):
    def __init__(self, eeg_chans=32, fnirs_chans=40, num_classes=2):
        super().__init__()
        assert fnirs_chans == 40, "Expected 40 fNIRS channels (20 HbO + 20 HbR)"
        grid_size = 17

        self.eeg_embed = LearnableSpatialEmbedding(eeg_chans, grid_size, grid_size)

        # 分别为 HbO 和 HbR 创建嵌入层（可共享也可独立）
        # 这里选择独立嵌入（更合理，因 HbO/HbR 空间模式不同）
        self.hbo_embed = LearnableSpatialEmbedding(20, grid_size, grid_size)
        self.hbr_embed = LearnableSpatialEmbedding(20, grid_size, grid_size)

        # EEG 流
        self.eeg_down = DownsampleBlock(1, 32, is_eeg=True)
        self.eeg_res = nn.Sequential(ResidualBlock(32), ResidualBlock(32))

        # fNIRS 流：现在输入是 2 通道（HbO + HbR）
        self.fnirs_down = DownsampleBlock(2, 32, is_eeg=False)
        self.fnirs_res1 = nn.Sequential(ResidualBlock(32), ResidualBlock(32))
        self.fnirs_res2 = nn.Sequential(ResidualBlock(32), ResidualBlock(32))

        # 对齐与融合
        self.align_eeg = nn.Conv3d(32, 32, kernel_size=(5, 1, 1), padding=(2, 0, 0))
        self.fuse_conv = nn.Conv3d(64, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, eeg, fnirs):
        """
        eeg: [B, 1500, 32]
        fnirs: [B, 150, 40]  # 前20: HbO, 后20: HbR
        return: [B, num_classes]
        """
        B = eeg.size(0)

        # EEG path
        eeg_grid = self.eeg_embed(eeg)  # [B, 1500, 17, 17]
        eeg_4d = eeg_grid.unsqueeze(1)  # [B, 1, 1500, 17, 17]
        eeg_feat = self.eeg_down(eeg_4d)
        eeg_feat = self.eeg_res(eeg_feat)

        # fNIRS path —— 正确拆分 HbO / HbR
        hbO = fnirs[:, :, :20]  # [B, 150, 20]
        hbR = fnirs[:, :, 20:]  # [B, 150, 20]

        hbO_grid = self.hbo_embed(hbO)  # [B, 150, 17, 17]
        hbR_grid = self.hbr_embed(hbR)  # [B, 150, 17, 17]

        # 拼接成双通道：[B, 2, 150, 17, 17]
        fnirs_4d = torch.stack([hbO_grid, hbR_grid], dim=1)

        fnirs_feat = self.fnirs_down(fnirs_4d)
        fnirs_feat = self.fnirs_res1(fnirs_feat)

        # Align & Fuse
        eeg_aligned = self.align_eeg(eeg_feat)
        min_t = min(eeg_aligned.size(2), fnirs_feat.size(2))
        eeg_aligned = eeg_aligned[:, :, :min_t]
        fnirs_feat = fnirs_feat[:, :, :min_t]

        fused = torch.cat([eeg_aligned, fnirs_feat], dim=1)
        fused = self.fuse_conv(fused)
        final_feat = self.fnirs_res2(fused)

        pooled = self.global_pool(final_feat).view(B, -1)
        logits = self.classifier(pooled)
        return logits
