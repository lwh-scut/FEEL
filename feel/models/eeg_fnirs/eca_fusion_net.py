import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# ECA Module
# ----------------------------
class ECAModule(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECAModule, self).__init__()
        t = int(abs((torch.log2(torch.tensor(channel, dtype=torch.float32)) + b) / gamma))
        k = t if t % 2 == 1 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

# ----------------------------
# Conv Block: (B, F_in, T, C) -> (B, F_out, T, C)
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_t=3, dropout_rate=0.5):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_t, 1),
            padding=(kernel_t // 2, 0),
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

# ----------------------------
# Feature Extractor
# ----------------------------
class FeatureExtractor(nn.Module):
    def __init__(self, config, dropout_rate=0.5):
        super(FeatureExtractor, self).__init__()
        c1, c2, c3, c4 = config
        self.block1 = ConvBlock(1, c1, dropout_rate=dropout_rate)
        self.block2 = ConvBlock(c1, c2, dropout_rate=dropout_rate)
        self.eca1 = ECAModule(c2)
        self.block3 = ConvBlock(c2, c3, dropout_rate=dropout_rate)
        self.block4 = ConvBlock(c3, c4, dropout_rate=dropout_rate)
        self.eca2 = ECAModule(c4)

    def forward(self, x):
        # x: (B, C, T) → (B, 1, T, C)
        x = x.unsqueeze(1)  # (B, 1, T, C)
        x = self.block1(x)   # (B, c1, T, C)
        x = self.block2(x)   # (B, c2, T, C)
        x = self.eca1(x)
        x = self.block3(x)   # (B, c3, T, C)
        x = self.block4(x)   # (B, c4, T, C)
        x = self.eca2(x)
        return x

# ----------------------------
# ECA-FusionNet
# ----------------------------
# ----------------------------
# ECA-FusionNet (with fNIRS upsampling)
# ----------------------------
class ECAFusionNet(nn.Module):
    def __init__(self, eeg_ch=32, fnirs_ch=40, t_length=1500, num_classes=2, dropout_rate=0.5):
        super(ECAFusionNet, self).__init__()
        self.eeg_ch = eeg_ch
        self.fnirs_ch = fnirs_ch
        self.t_target = t_length  # target time length (e.g., 1500)

        self.eeg_extractor = FeatureExtractor([8, 16, 16, 32], dropout_rate)
        self.fnirs_extractor = FeatureExtractor([8, 16, 16, 32], dropout_rate)

        self.classifier_eeg = nn.Linear(32, num_classes)
        self.classifier_fnirs = nn.Linear(32, num_classes)
        self.classifier_fused = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        self.fusion_weights = nn.Parameter(torch.ones(3))

    def forward(self, eeg, fnirs):
        """
        eeg: (B, eeg_ch, T_eeg)   → e.g., (B, 32, 1500)
        fnirs: (B, fnirs_ch, T_fnirs) → e.g., (B, 40, 150)
        """
        # Step 1: Upsample fNIRS to match EEG time length
        if fnirs.shape[-1] != self.t_target:
            # fnirs is (B, C, T_src) → interpolate along last dim to T_target
            fnirs = F.interpolate(fnirs, size=self.t_target, mode='linear', align_corners=False)
            # Now fnirs is (B, C, T_target)

        # Now both eeg and fnirs have same T = self.t_target
        feat_eeg = self.eeg_extractor(eeg)  # (B, 32, T, eeg_ch)
        feat_fnirs = self.fnirs_extractor(fnirs)  # (B, 32, T, fnirs_ch)

        pooled_eeg = F.adaptive_avg_pool2d(feat_eeg, (1, 1)).flatten(1)
        pooled_fnirs = F.adaptive_avg_pool2d(feat_fnirs, (1, 1)).flatten(1)

        logits_eeg = self.classifier_eeg(pooled_eeg)
        logits_fnirs = self.classifier_fnirs(pooled_fnirs)

        fused_feat = torch.cat([pooled_eeg, pooled_fnirs], dim=1)
        logits_fused = self.classifier_fused(fused_feat)

        prob_eeg = F.softmax(logits_eeg, dim=1)
        prob_fnirs = F.softmax(logits_fnirs, dim=1)
        prob_fused = F.softmax(logits_fused, dim=1)

        weights = F.softmax(self.fusion_weights, dim=0)
        final_prob = (
                weights[0] * prob_eeg +
                weights[1] * prob_fnirs +
                weights[2] * prob_fused
        )

        return final_prob

# ----------------------------
# Test
# ----------------------------
if __name__ == "__main__":
    B, T = 4, 1500
    eeg = torch.randn(B, 32, T)
    fnirs = torch.randn(B, 40, 150)

    model = ECAFusionNet(eeg_ch=32, fnirs_ch=40, t_length=T, num_classes=2)
    final_prob = model(eeg, fnirs)

    print("✅ Success! Final prob shape:", final_prob.shape)
