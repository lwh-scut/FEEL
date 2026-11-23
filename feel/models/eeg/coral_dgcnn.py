import torch
import torch.nn.functional as F
from feel.models.eeg.dgcnn import DGCNN
from feel.models.eeg.dgcnn import laplacian
from feel.data.preprocess import feature_extraction


class CoralDgcnn(DGCNN):
    def __init__(self, num_electrodes=62, in_channels=5, num_classes=3, k=2, relu_is=1, layers=None, dropout_rate=0.5):
        super(CoralDgcnn,self).__init__()
        self.alpha = 1
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

    # 计算coral loss，源域之间的协方差
    @staticmethod
    def coralLoss(x, y):
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff

    def getCoralLoss(self, featureList):
        # penalty = 0
        # for i in range(len(featureList)):
        #     for j in range(i+1, len(featureList)):
        #         penalty += self.coralLoss(featureList[i], featureList[j])
        # return penalty
        # featureList = torch.stack(featureList, dim=1)
        featureList = featureList.reshape(featureList.shape[0], -1)
        mean_feature = featureList.mean(dim=0, keepdim=True)
        centered_features = featureList - mean_feature
        covariance_matrix = (centered_features.T @ centered_features) / (featureList.size(0) - 1)

        # 可以根据协方差矩阵的对角线和非对角线的差异来计算 CORAL 损失
        coral_loss = torch.norm(covariance_matrix - torch.eye(covariance_matrix.size(0)).to(featureList.device), p='fro')

        return coral_loss

    def featureExtract(self, x):
        adj = self.relu(self.adj + self.adj_bias)
        lap = laplacian(adj)
        for i in range(len(self.layers)):
            x = self.graphConvs[i](x, lap)
            x = self.dropout(x)
            x = self.b_relus[i](x)
        return x

    def classify(self, feature):
        x = feature.reshape(feature.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def forward(self, minibatches):
        bz = len(minibatches)
        features = self.featureExtract(minibatches)
        outputs = self.classify(features)
        return outputs
