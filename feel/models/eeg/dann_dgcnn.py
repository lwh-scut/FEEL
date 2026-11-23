import torch
import torch.nn.functional as F
from feel.models.eeg.dgcnn import DGCNN
from feel.models.eeg.dgcnn import laplacian
from feel.data.preprocess import feature_extraction

from torch.autograd import Function


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class Discriminator(torch.nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, num_domains=15):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        layers = [
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LeakyReLU(), #######################################
            torch.nn.Dropout(0.5),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.Linear(hidden_dim, num_domains),
            torch.nn.Dropout(0.5),
        ]
        # layers = [
        #     torch.nn.Linear(input_dim, num_domains),  # easy disc
        #     torch.nn.Dropout(0.5),
        # ]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    def initialize(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # He initialization for ReLU
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

class DannDgcnn(DGCNN):
    def __init__(self, num_electrodes=62, in_channels=5, num_classes=3, k=2, relu_is=1, layers=None, dropout_rate=0.5, num_sources=14):
        super(DannDgcnn,self).__init__()
        self.alpha = 0.1
        self.dropout_rate = dropout_rate
        self.layers = layers
        self.k = k
        self.in_channels = in_channels
        self.num_electrodes = num_electrodes
        self.num_classes = num_classes
        self.relu_is = relu_is
        if num_electrodes == 62:
            self.layers = [64]
        elif num_electrodes == 32:
            self.layers = [128]
        self.num_sources = num_sources
        self.discriminator = Discriminator(self.num_electrodes * self.layers[-1], 256, self.num_sources)
        self.leaky_relus = []
        for _ in range(len(self.layers)):
            self.leaky_relus.append(torch.nn.LeakyReLU())

    def featureExtract(self, x):
        adj = self.relu(self.adj + self.adj_bias)
        lap = laplacian(adj)
        for i in range(len(self.layers)):
            x = self.graphConvs[i](x, lap)
            x = self.dropout(x)
            x = self.leaky_relus[i](x)
        return x

    def classify(self, feature):
        x = feature.reshape(feature.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def forward(self, minibatches):
        features = self.featureExtract(minibatches)
        disc_input = features
        disc_input = ReverseLayerF.apply(disc_input, self.alpha)
        disc_input = disc_input.reshape(disc_input.shape[0], -1)
        disc_output = self.discriminator(disc_input)
        outputs = self.classify(features)
        return outputs
