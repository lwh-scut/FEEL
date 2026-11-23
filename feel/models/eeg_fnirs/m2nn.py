import torch
import torch.nn as nn
import torch.nn.functional as F


class M2NN(nn.Module):
    def __init__(self, num_classes=2, alpha=2.5, beta=5.0):
        super(M2NN, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

        # ========================
        # EEG Branch (32 ch, 1500 t)
        # ========================
        self.eeg_blocks = nn.Sequential(
            # Block 1: Temporal filtering (small time kernels)
            nn.Conv2d(1, 32, kernel_size=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            # Block 2: Temporal filtering
            nn.Conv2d(32, 64, kernel_size=(1, 4), padding=(0, 2)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=(1, 4), padding=(0, 2)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            # Block 3: Spatial projection + temporal feature
            nn.Conv2d(64, 64, kernel_size=(32, 1)),  # across all 32 channels
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=(1, 4)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            # Block 4: Temporal refinement
            nn.Conv2d(64, 128, kernel_size=(1, 4)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=(1, 4)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )

        # ========================
        # fNIRS Branch (40 ch, 150 t)
        # ========================
        self.fnirs_blocks = nn.Sequential(
            # Block 1: Temporal filtering
            nn.Conv2d(1, 32, kernel_size=(1, 4), padding=(0, 2)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=(1, 4), padding=(0, 2)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            # Block 2: Temporal filtering
            nn.Conv2d(32, 64, kernel_size=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            # Block 3: Spatial projection
            nn.Conv2d(64, 64, kernel_size=(40, 1)),  # across all 40 channels
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            # Block 4: Temporal refinement
            nn.Conv2d(64, 128, kernel_size=(1, 2)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=(1, 2)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )

        # Auto-compute flattened feature size
        with torch.no_grad():
            eeg_dummy = torch.randn(1, 1, 32, 1500)
            fnirs_dummy = torch.randn(1, 1, 40, 150)
            eeg_feat = self.eeg_blocks(eeg_dummy)
            fnirs_feat = self.fnirs_blocks(fnirs_dummy)
            eeg_flat_dim = eeg_feat.view(1, -1).size(1)
            fnirs_flat_dim = fnirs_feat.view(1, -1).size(1)

        fusion_dim = eeg_flat_dim + fnirs_flat_dim

        # ========================
        # Fusion & Classification
        # ========================
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ELU()
        )
        self.classifier = nn.Linear(64, num_classes)

        # Center for center loss (will be updated during training)
        self.register_parameter('centers', nn.Parameter(torch.randn(num_classes, 64)))

    def forward(self, eeg, fnirs):
        """
        Args:
            eeg:   (B, 32, 1500)
            fnirs: (B, 40, 150)
        Returns:
            logits: (B, num_classes)
            features: (B, 64) — for center loss
        """
        B = eeg.shape[0]
        eeg = eeg.unsqueeze(1)  # (B, 1, 32, 1500)
        fnirs = fnirs.unsqueeze(1)  # (B, 1, 40, 150)

        eeg_feat = self.eeg_blocks(eeg)
        fnirs_feat = self.fnirs_blocks(fnirs)

        eeg_flat = eeg_feat.view(B, -1)
        fnirs_flat = fnirs_feat.view(B, -1)

        fused = torch.cat([eeg_flat, fnirs_flat], dim=1)
        features = self.fusion_fc(fused)  # (B, 64)
        logits = self.classifier(features)

        return logits, features


# ========================
# Center Loss Module
# ========================
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device='cpu'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, x, labels):
        """
        x: (B, feat_dim)
        labels: (B,)
        """
        batch_centers = self.centers[labels]  # (B, feat_dim)
        loss = F.mse_loss(x, batch_centers, reduction='mean')
        return loss


# ========================
# Test Script
# ========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Random data matching your specs
    B = 8
    eeg = torch.randn(B, 32, 1500).to(device)
    fnirs = torch.randn(B, 40, 150).to(device)
    labels = torch.randint(0, 2, (B,)).to(device)

    model = M2NN(num_classes=2).to(device)
    center_loss_fn = CenterLoss(num_classes=2, feat_dim=64, device=device)

    logits, features = model(eeg, fnirs)

    ce_loss = F.cross_entropy(logits, labels)
    c_loss = center_loss_fn(features, labels)
    total_loss = 2.5 * ce_loss + 5.0 * c_loss

    print("✅ Forward pass successful!")
    print(f"Logits shape: {logits.shape}")
    print(f"Features shape: {features.shape}")
    print(f"CE Loss: {ce_loss.item():.4f}")
    print(f"Center Loss: {c_loss.item():.4f}")
    print(f"Total Loss: {total_loss.item():.4f}")

    total_loss.backward()
    print("✅ Backward pass successful!")
