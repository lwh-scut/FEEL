import torch
import torch.nn as nn

from feel.models.eeg.eegnet import EEGNet


class EEGfNIRSNet(nn.Module):
    def __init__(self, eeg_chans, eeg_timepoints, fnirs_chans, fnirs_timepoints, num_classes):
        super().__init__()
        # EEG 分支
        self.eeg_net = EEGNet(
            num_electrodes=eeg_chans, datapoints=eeg_timepoints, num_classes=num_classes

        )
        # 修改 EEGNet 最后一层为特征提取（移除分类头）
        self.eeg_net.classifier_block = nn.Identity()

        # fNIRS 分支：简单 CNN
        self.fnirs_net = nn.Sequential(
            nn.Conv1d(fnirs_chans, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(10),
            nn.Flatten(),
            nn.Linear(32 * 10, 64)
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, eeg, fnirs):
        # eeg: (B, C_eeg, T_eeg)
        # fnirs: (B, C_fnirs, T_fnirs)
        eeg_feat = self.eeg_net(eeg)  # (B, 64)
        fnirs_feat = self.fnirs_net(fnirs)  # (B, 64)
        fused = torch.cat([eeg_feat, fnirs_feat], dim=1)
        out = self.fusion(fused)
        return out
