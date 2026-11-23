##
## paper @see https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10976537
## code @see https://github.com/YYingDL/NSAL-DGAT
##
##

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        ca = self.ca(x)
        out = x * ca
        sa = self.sa(out)
        out = out * sa
        return out + residual, ca, sa


#in this code, we will use gcn and global
# 生成邻接矩阵
class GATENet(nn.Module):
    def __init__(self, inc, reduction_ratio=128):
        super(GATENet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(inc, inc // reduction_ratio, bias=False),
                                nn.ELU(inplace=False),
                                nn.Linear(inc // reduction_ratio, inc, bias=False),
                                nn.Tanh(),
                                nn.ReLU(inplace=False))

    def forward(self, x):
        y = self.fc(x)
        return y


class resGCN(nn.Module):
    def __init__(self, inc, outc, band_num):
        super(resGCN, self).__init__()
        self.GConv1 = nn.Conv2d(in_channels=inc,
                                out_channels=outc,
                                kernel_size=(1, 3),
                                stride=(1, 1),
                                padding=(0, 0),
                                groups=band_num,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(outc)
        self.GConv2 = nn.Conv2d(in_channels=outc,
                                out_channels=outc,
                                kernel_size=(1, 1),
                                stride=(1, 1),
                                padding=(0, 1),
                                groups=band_num,
                                bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.ELU = nn.ELU(inplace=False)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, x_p, L):
        x = self.bn2(self.GConv2(self.ELU(self.bn1(self.GConv1(x)))))
        y = torch.einsum('bijk,kp->bijp', (x, L))
        y = self.ELU(torch.add(y, x_p))
        return y


class HGCN(nn.Module):
    def __init__(self, dim, chan_num, band_num):
        super(HGCN, self).__init__()
        self.chan_num = chan_num
        self.dim = dim
        self.resGCN = resGCN(inc=dim * band_num,
                             outc=dim * band_num, band_num=band_num)
        self.ELU = nn.ELU(inplace=False)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Sequential):
                for j in m:
                    if isinstance(j, nn.Linear):
                        nn.init.xavier_uniform_(j.weight, gain=1)

    def forward(self, x, A_ds):
        L = torch.einsum('ik,kp->ip', (A_ds, torch.diag(torch.reciprocal(sum(A_ds)))))
        G = self.resGCN(x, x, L).contiguous()
        return G

class MHGCN(nn.Module):
    def __init__(self, layers, dim, chan_num, band_num, hidden_1, hidden_2):
        super(MHGCN, self).__init__()
        self.chan_num = chan_num
        self.band_num = band_num
        self.A = torch.rand((1, self.chan_num * self.chan_num), dtype=torch.float32, requires_grad=False)
        self.GATENet = GATENet(self.chan_num * self.chan_num, reduction_ratio=128)
        self.HGCN_layers = nn.ModuleList()
        for i in range(layers):
            self.HGCN_layers.append(HGCN(dim=1, chan_num=self.chan_num, band_num=self.band_num))

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Sequential):
                for j in m:
                    if isinstance(j, nn.Linear):
                        nn.init.xavier_uniform_(j.weight, gain=1)

    def forward(self, x):
        self.A = self.A.to(x.device)
        A_ds = self.GATENet(self.A)
        A_ds = A_ds.reshape(self.chan_num, self.chan_num)
        output = []
        output.append(x)
        for i in range(len(self.HGCN_layers)):
            input = x
            output.append(self.HGCN_layers[i](input, A_ds))
            x = output[-1]
        out = torch.cat(output, dim=1)
        return out, A_ds


class Encoder(nn.Module):
    def __init__(self, in_planes=[1500, 32], layers=2, hidden_1=256, hidden_2=64, class_nums=3):
        super(Encoder, self).__init__()
        self.chan_num = in_planes[1]
        self.band_num = in_planes[0]
        self.GGCN = MHGCN(layers=layers, dim=1, chan_num=self.chan_num, band_num=self.band_num, hidden_1=hidden_1,
                          hidden_2=hidden_2)

        self.CBAM =CBAMBlock(channel=(layers + 1) * self.band_num, reduction=4, kernel_size=3)
        self.fc1 = nn.Linear(self.chan_num * (layers + 1) * self.band_num, hidden_2)
        self.fc2 = nn.Linear(hidden_2, hidden_2)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.25)

    def forward(self, x):
        x = x.reshape(x.size(0), 1500, 32)
        x = x.unsqueeze(2)
        g_feat, g_adj = self.GGCN(x)
        g_feat, ca, sa = self.CBAM(g_feat)
        out = self.fc1(g_feat.reshape(g_feat.size(0), -1))
        out = F.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.dropout2(out)
        return out, [g_adj, ca, sa]



class ClassClassifier(nn.Module):
    def __init__(self, hidden_2, num_cls):
        super(ClassClassifier, self).__init__()
        self.classifier = nn.Linear(hidden_2, num_cls)

    def forward(self, x):
        x = self.classifier(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, hidden_1):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(hidden_1, hidden_1)
        self.fc2 = nn.Linear(hidden_1, 1)
        self.dropout1 = nn.Dropout(p=0.25)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class Domain_adaption_model(nn.Module):
    def __init__(self, channels=32, feature_dim=1500, num_of_class=3, layers=2, hidden_1=256, hidden_2=64, device='cuda:1', source_num=3944):
        super(Domain_adaption_model, self).__init__()
        in_planes = [feature_dim, channels]
        self.encoder = Encoder(in_planes=in_planes, layers=layers, hidden_1=hidden_1, hidden_2=hidden_2, class_nums=num_of_class )
        self.cls_classifier = ClassClassifier(hidden_2=hidden_2, num_cls=num_of_class)
        self.source_f_bank = torch.randn(source_num, hidden_2)
        self.source_score_bank = torch.randn(source_num, num_of_class).to(device)
        self.num_of_class = num_of_class
        self.ema_factor = 0.8

    def forward(self, source, target, source_label, source_index):
        source_f, [self.src_adj, self.src_sa, self.src_ca] = self.encoder(source)
        target_f, [self.tar_adj, self.tar_sa, self.tar_ca] = self.encoder(target)

        source_predict = self.cls_classifier(source_f)
        target_predict = self.cls_classifier(target_f)

        source_label_feature = torch.nn.functional.softmax(source_predict, dim=1)
        target_label_feature = torch.nn.functional.softmax(target_predict, dim=1)

        target_label = self.get_target_labels(source_f, source_label_feature, source_index, target_f)
        return source_predict, source_f, target_predict, target_f, [self.src_adj, self.src_sa, self.src_ca], [self.tar_adj, self.tar_sa, self.tar_ca], target_label



    def get_target_labels(self, feature_source_f, source_label_feature, source_index, feature_target_f):
        self.eval()
        output_f = torch.nn.functional.normalize(feature_source_f)
        self.source_f_bank[source_index] = output_f.detach().clone().cpu()
        self.source_score_bank[source_index] = source_label_feature.detach().clone()

        output_f_ = torch.nn.functional.normalize(feature_target_f).cpu().detach().clone()
        distance = output_f_ @ self.source_f_bank.T
        _, idx_near = torch.topk(distance, dim=-1, largest=True, k=7)
        score_near = self.source_score_bank[idx_near]  # batch x K x num_class
        score_near_weight = self.get_weight(score_near)
        score_near_sum_weight = torch.einsum('ijk,ij->ik', score_near, score_near_weight)
        # score_near_sum_weight = torch.mean(score_near, dim=1)  # batch x num_class
        target_predict = torch.nn.functional.softmax(score_near_sum_weight, dim=1)
        return target_predict

    def get_init_banks(self, source, source_index):
        self.eval()
        source_f, source_att = self.encoder(source)

        source_predict = self.cls_classifier(source_f)
        source_label_feature = torch.nn.functional.softmax(source_predict, dim=1)

        self.source_f_bank[source_index] = torch.nn.functional.normalize(source_f).detach().clone().cpu()
        self.source_score_bank[source_index] = source_label_feature.detach().clone()

    def target_predict(self, feature_target):
        self.eval()
        target_f, _ = self.encoder(feature_target)
        target_predict = self.cls_classifier(target_f)
        target_label_feature = torch.nn.functional.softmax(target_predict, dim=1)
        return target_label_feature

    def domain_discrepancy(self, out1, out2, loss_type):
        def huber_loss(e, d=1):
            t = torch.abs(e)
            ret = torch.where(t < d, 0.5 * t ** 2, d * (t - 0.5 * d))
            return torch.mean(ret)

        diff = out1 - out2
        if loss_type == 'L1':
            loss = torch.mean(torch.abs(diff))
        elif loss_type == 'Huber':
            loss = huber_loss(diff)
        else:
            loss = torch.mean(diff * diff)
        return loss

    def get_weight(self, score_near):
        epsilon = 1e-5
        entropy = -(1/score_near.size(1))*torch.sum(score_near*torch.log(score_near+ epsilon), dim=2)
        g = 1 - entropy
        score_near_weight = g / torch.tile(torch.sum(g, dim=1).view(-1, 1), (1, score_near.size(1)))
        return score_near_weight


    def Entropy(self, input_):
        bs = input_.size(0)
        epsilon = 1e-5
        entropy = -input_ * torch.log(input_ + epsilon)
        entropy = torch.sum(entropy, dim=1)
        return entropy




# **************** adversial ***************#

from typing import Optional, Any, Tuple
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
## implementation of domain adversarial traning. For more details, please visit: https://dalib.readthedocs.io/en/latest/index.html
def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct

class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

class WarmStartGradientReverseLayer(nn.Module):

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = np.float64(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1

class DomainAdversarialLoss(nn.Module):
    def __init__(self, domain_discriminator: nn.Module,reduction: Optional[str] = 'mean',max_iter=1000):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1.0, lo=0., hi=1., max_iters=max_iter, auto_step=True)
        self.domain_discriminator = domain_discriminator
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discriminator(f)
        d_s, d_t = d.chunk(2, dim=0)
        d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
        d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
        self.domain_discriminator_accuracy = 0.5 * (binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))
        return 0.5 * (self.bce(d_s, d_label_s) + self.bce(d_t, d_label_t))


class DAANLoss(nn.Module):
    def __init__(self, domain_discriminator: nn.Module, num_class=3, reduction: Optional[str] = 'mean',max_iter=1000):
        super(DAANLoss, self).__init__()
        self.num_class = num_class
        self.grl = WarmStartGradientReverseLayer(alpha=1.0, lo=0., hi=1., max_iters=max_iter, auto_step=True)
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)
        self.local_classifiers = torch.nn.ModuleList()
        self.global_classifiers = domain_discriminator
        for _ in range(num_class):
            self.local_classifiers.append(domain_discriminator)

        self.d_g, self.d_l = 0, 0
        self.dynamic_factor = 0.5

    def forward(self, source, target, source_logits, target_logits):

        global_loss = self.get_global_adversarial_result(source, target)

        #
        # self.d_g = self.d_g + 2 * (1 - 2 * global_loss.cpu().item())
        # self.d_l = self.d_l + 2 * (1 - 2 * (local_loss / self.num_class).cpu().item())

        # adv_loss = (1 - self.dynamic_factor) * global_loss + self.dynamic_factor * local_loss
        return global_loss

    def get_global_adversarial_result(self, f_s, f_t):
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.global_classifiers(f)
        d_s, d_t = d.chunk(2, dim=0)
        d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
        d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
        return 0.5 * (self.bce(d_s, d_label_s) + self.bce(d_t, d_label_t))


    def get_local_adversarial_result(self, feat, logits, source=True):
        loss_adv = 0.0
        for c in range(self.num_class):
            x = feat[c + 1]
            x = self.grl(x)
            softmax_logits = torch.nn.functional.softmax(logits, dim=1)
            logits_c = logits[:, c].reshape((softmax_logits.shape[0], 1)) # (B, 1)
            features_c = logits_c * x
            domain_pred = self.local_classifiers[c](features_c)
            device = domain_pred.device
            if source:
                domain_label = torch.ones(x.size(0), 1).to(device)
            else:
                domain_label = torch.zeros(x.size(0), 1).to(device)
            loss_adv = loss_adv + self.bce(domain_pred, domain_label)
        return 0.5 * loss_adv

    def update_dynamic_factor(self, epoch_length):
        if self.d_g == 0 and self.d_l == 0:
            self.dynamic_factor = 0.5
        else:
            self.d_g = self.d_g / epoch_length
            self.d_l = self.d_l / epoch_length
            self.dynamic_factor = 1 - self.d_g / (self.d_g + self.d_l)
        self.d_g, self.d_l = 0, 0
