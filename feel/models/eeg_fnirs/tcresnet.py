import torch
import torch.nn as nn


class TCResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, downsample=None):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class FlexibleTCResNetBranch(nn.Module):
    """
    可配置下采样策略的 TC-ResNet 分支，适配不同时间长度的输入。
    """

    def __init__(self, input_channels, base_channels=64, n_downsamples=3):
        super().__init__()
        self.n_downsamples = n_downsamples

        # Initial stem
        self.conv1 = nn.Conv1d(input_channels, base_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)

        layers = []
        in_ch = base_channels
        for i in range(n_downsamples):
            out_ch = base_channels * (2 ** (i + 1))  # 64 -> 128 -> 256 -> 512...
            kernel = 3 if i == 0 else 9
            stride = 2

            # 构建下采样路径
            downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
            layers.append(TCResBlock(in_ch, out_ch, kernel_size=kernel, stride=stride, downsample=downsample))
            in_ch = out_ch

        self.layers = nn.Sequential(*layers)
        self.final_channels = in_ch

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layers(x)  # 输出时间长度 ≈ T / (2^n_downsamples)
        x = torch.mean(x, dim=2)  # Global Average Pooling
        return x


class DualModalEmotionNet_CustomSR(nn.Module):
    """
    适配自定义采样率的双模态情绪识别模型：
    - fNIRS: 10 Hz (e.g., 10s window → 100 points)
    - EEG:  1000 Hz (e.g., 10s window → 10,000 points)
    """

    def __init__(self, num_classes=4,
                 fnirs_chans=24, fNIRS_downsamples=2,  # 时间点少，少下采样
                 eeg_chans=16, EEG_downsamples=4,  # 时间点多，多下采样
                 base_ch=64):
        super().__init__()

        self.fNIRS_net = FlexibleTCResNetBranch(fnirs_chans, base_ch, fNIRS_downsamples)
        self.EEG_net = FlexibleTCResNetBranch(eeg_chans, base_ch, EEG_downsamples)

        fused_dim = self.fNIRS_net.final_channels + self.EEG_net.final_channels
        self.classifier = nn.Linear(fused_dim, num_classes)

    def forward(self, EEG_x, fNIRS_x):
        """
        Args:
            fNIRS_x: (B, fNIRS_ch, T_f) e.g., (B, 24, 100)  for 10s @ 10Hz
            EEG_x:   (B, EEG_ch, T_e)   e.g., (B, 16, 10000) for 10s @ 1000Hz
        Returns:
            logits: (B, num_classes)
        """
        f_feat = self.fNIRS_net(fNIRS_x)  # (B, C1)
        e_feat = self.EEG_net(EEG_x)  # (B, C2)
        fused = torch.cat([f_feat, e_feat], dim=1)
        return self.classifier(fused)


# === 使用示例 ===
if __name__ == "__main__":
    # 假设使用 10 秒窗口
    fNIRS_data = torch.randn(4, 24, 100)  # 10 Hz × 10 s = 100
    EEG_data = torch.randn(4, 16, 10000)  # 1000 Hz × 10 s = 10,000

    model = DualModalEmotionNet_CustomSR(
        fNIRS_ch=24, fNIRS_downsamples=2,  # 100 → 25 → 12 (still >1, safe)
        EEG_ch=16, EEG_downsamples=4  # 10000 → 5000 → 2500 → 1250 → 625
    )

    output = model(fNIRS_data, EEG_data)
    print("Output shape:", output.shape)  # [4, 4]

    # 查看参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params / 1e6:.2f} M")
