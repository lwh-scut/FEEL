import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGBranch(nn.Module):
    def __init__(self, timepoints=1500, channels=32):
        super(EEGBranch, self).__init__()
        self.channels = channels
        self.timepoints = timepoints

        # Block 1: Temporal conv (kernel along time)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(7, 1), padding=(3, 0)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(7, 1), padding=(3, 0)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(7, 1), padding=(3, 0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(7, 1)),
            nn.Dropout(0.5),
            nn.BatchNorm2d(32)
        )

        # Block 2: Spatio-temporal conv
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(4, 4), padding=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(4, 4), padding=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(4, 4), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4)),
            nn.Dropout(0.5),
            nn.BatchNorm2d(64)
        )

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, timepoints, channels)
            dummy = self.conv1(dummy)
            dummy = self.conv2(dummy)
            self.flat_features = dummy.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.flat_features, 256)
        self.fc2 = nn.Linear(256, 128)

    def forward(self, x):
        # x shape: (B, T, C) -> add channel dim: (B, 1, T, C)
        x = x.transpose(1, 2)  # (B, T, C)
        x = x.unsqueeze(1)  # (B, 1, T, C)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  # (B, 128)
        return x


class FNIRSBranch(nn.Module):
    def __init__(self, timepoints=150, channels=40):
        super(FNIRSBranch, self).__init__()

        # Block 1: Temporal conv (small kernel)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(4, 1), padding=(2, 0)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(2, 0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # Reduce time: 150 -> 75
            nn.Dropout(0.5),
            nn.BatchNorm2d(32)
        )

        # Block 2: Spatio-temporal conv
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # (75,40) -> (37,20)
            nn.Dropout(0.5),
            nn.BatchNorm2d(64)
        )

        # Compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, timepoints, channels)
            dummy = self.conv1(dummy)
            dummy = self.conv2(dummy)
            self.flat_features = dummy.view(1, -1).size(1)

        self.fc = nn.Linear(self.flat_features, 128)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, T, C)
        x = x.unsqueeze(1)  # (B, 1, T, C)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))  # (B, 128)
        return x


class EFNet(nn.Module):
    def __init__(
            self,
            eeg_timepoints=1500,
            eeg_channels=32,
            fnirs_timepoints=150,
            fnirs_channels=40,
            num_classes=1  # binary classification
    ):
        super(EFNet, self).__init__()
        self.eeg_branch = EEGBranch(timepoints=eeg_timepoints, channels=eeg_channels)
        self.fnirs_branch = FNIRSBranch(timepoints=fnirs_timepoints, channels=fnirs_channels)

        # Fusion layers
        self.fusion_fc1 = nn.Linear(128 + 128, 256)
        self.fusion_fc2 = nn.Linear(256, 64)
        self.classifier = nn.Linear(64, num_classes)

        # L2 regularization will be applied via weight decay in optimizer

    def forward(self, eeg, fnirs):

        eeg_feat = self.eeg_branch(eeg)  # (B, 128)
        fnirs_feat = self.fnirs_branch(fnirs)  # (B, 128)

        fused = torch.cat([eeg_feat, fnirs_feat], dim=1)  # (B, 256)
        fused = F.relu(self.fusion_fc1(fused))
        fused = F.dropout(fused, p=0.5, training=self.training)
        fused = F.relu(self.fusion_fc2(fused))
        out = self.classifier(fused)  # (B, 1)

        if self.classifier.out_features == 1:
            out = torch.sigmoid(out)
        return out
