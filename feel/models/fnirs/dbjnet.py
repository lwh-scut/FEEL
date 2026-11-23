import torch.nn as nn
import torch
from torch.nn.modules.rnn import LSTM
import numpy as np
from torchinfo import summary

class DBJNet(nn.Module):
    def __init__(self, channels, samples, n_class):
        super().__init__()

        self.channels = channels
        self.block2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(8, 1), stride=(4, 1)),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(4, 1), stride=(2, 1)),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(channels, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 256)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64*8*8, 256),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, n_class),
        )

    def forward(self, x):
        x = x[:, :, :]
        mean = torch.mean(x, dim=-1)
        input = x.unsqueeze(1)

        feature0 = input
        feature = self.block2(feature0)
        feature1 = torch.nn.functional.normalize(self.fc1(mean), p=2, dim=1)
        feature2 = torch.nn.functional.normalize(self.fc2(feature), p=2, dim=1)
        feature = torch.cat((feature1, feature2), dim=1)
        logits = self.classifier(feature)
        return logits



if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((16, 40, 160), device=cuda0)
        model = DBJNet(n_class=3, channels=40, samples=160)
        summary(model, input_size=(16, 40, 160))
        model.cuda()
        output = model(x)
        print('output:', output.shape)
