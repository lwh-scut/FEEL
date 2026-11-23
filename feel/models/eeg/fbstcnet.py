# This is the PyTorch implementation of the FBSTCNet-M architecture for EEG-based emotion classification.

# Reference:
# "W. Huang, W. Wang, Y. Li, W. Wu. FBSTCNet: A Spatio-Temporal Convolutional Network Integrating Power and Connectivity Features for EEG-Based Emotion Decoding. 2023. (under review)"

from warnings import warn
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.signal import cheb2ord, coherence
import torchaudio
import torch
import numpy as np
from torch import nn
from torch.nn import init

from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler
from torch.optim import AdamW
from braindecode.training import CroppedLoss
from braindecode.util import set_random_seeds
from braindecode.models import get_output_shape
from sklearn.metrics import confusion_matrix

from braindecode.util import np_to_th
from braindecode.models.modules import Expression, Ensure4d
from braindecode.models.functions import (
    safe_log, square, transpose_time_to_spat
)

def get_padding(kernel_size, stride=1, dilation=1, **_):
    if isinstance(kernel_size, tuple):
        kernel_size = max(kernel_size)
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class PowerAndConneMixedNet(nn.Module):

    def __init__(
            self,
            in_chans,
            n_classes,
            fs=100,
            f_trans=2,
            filterRange=None,
            filterStop=None,
            input_window_samples=None,
            n_filters_time=72,
            filter_time_length=25,
            n_filters_spat=72,
            n_filters_power=36,
            pool_time_length=80,
            pool_time_stride=5,
            final_conv_length=35,
            final_conv_stride=25,
            conn_nonlin=coherence,
            pool_mode="mean",
            pool_nonlin=safe_log,
            split_first_layer=True,
            batch_norm=True,
            same_filters_for_features=True,
            batch_norm_alpha=0.1,
            drop_prob=0.5,
    ):
        super().__init__()
        if final_conv_length == "auto":
            assert input_window_samples is not None
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_window_samples = input_window_samples
        self.n_filters_time = n_filters_time
        self.filter_time_length = filter_time_length
        self.n_filters_spat = n_filters_spat
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.final_conv_length = final_conv_length
        self.final_conv_stride = final_conv_stride
        self.same_filters_for_features = same_filters_for_features
        if (self.same_filters_for_features):
            self.n_filters_power = self.n_filters_spat
            self.n_filters_coherence = self.n_filters_spat
        else:
            self.n_filters_power = n_filters_power
            self.n_filters_coherence = self.n_filters_spat - self.n_filters_power

        self.conn_nonlin = conn_nonlin
        self.pool_mode = pool_mode
        self.pool_nonlin = pool_nonlin
        self.split_first_layer = split_first_layer
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.drop_prob = drop_prob
        self.filterRange = filterRange
        self.n_filterbank = len(self.filterRange)
        self.fs = fs
        self.f_trans = f_trans
        self.filterStop = filterStop

        # filter bank
        if self.filterRange is not None:
            self.add_module("filterbank",
                            filterbank(fs=self.fs, frequency_bands=self.filterRange, filterStop=self.filterStop,
                                       f_trans=self.f_trans))
        else:
            self.add_module("filterbank", Ensure4d())  # [numSample × numChannel × numPoint × 1]
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        padding_size = get_padding((self.filter_time_length, 1))

        self.add_module("dimshuffle", Expression(transpose_time_to_spat))  # [numSample × 1 × numPoint × numChannel]

        # temporal convolution
        self.add_module(
            "conv_time",
            nn.Conv2d(
                self.n_filterbank,
                self.n_filters_time,
                (self.filter_time_length, 1),
                stride=1,
                padding=(padding_size, 0),
            ),
        )
        self.add_module("conv_nonlin_exp", Expression(square))
        self.add_module(
            "poolfunc",
            pool_class(
                kernel_size=(pool_time_length, 1),
                stride=(pool_time_stride, 1),
            ),
        )

        # spatial convolutions for power and connectivity-based network, respectively
        self.add_module(
            "conv_spat_power",
            nn.Conv2d(
                self.n_filters_power,
                self.n_filters_power,
                (1, self.in_chans),
                stride=1,
                groups=self.n_filters_power,
                bias=not self.batch_norm,
            ),
        )
        self.add_module(
            "conv_spat_conne",
            nn.Conv2d(
                self.n_filters_coherence,
                self.n_filters_coherence,
                (1, self.in_chans),
                stride=1,
                groups=self.n_filters_coherence,
                bias=not self.batch_norm,
            ),
        )
        if self.batch_norm:
            self.add_module(
                "bnorm_power",
                nn.BatchNorm2d(
                    self.n_filters_power, momentum=self.batch_norm_alpha, affine=True
                ),
            )
            self.add_module(
                "bnorm_conne",
                nn.BatchNorm2d(
                    self.n_filters_coherence, momentum=self.batch_norm_alpha, affine=True
                ),
            )
        # power-based feature extraction
        self.add_module("pool_nonlin_exp", Expression(safe_log))

        # connectivity-based feature extraction
        self.connectivity_exp = coherence_cropped(time_length=fs, time_stride=fs)
        self.add_module("power_drop", nn.Dropout(p=self.drop_prob))
        self.add_module("conne_drop", nn.Dropout(p=self.drop_prob))

        # final convolution
        self.add_module(
            "conv_power_classifier",
            nn.Conv2d(
                self.n_filters_power,
                self.n_classes,
                (self.final_conv_length, 1),
                stride=(self.final_conv_stride, 1),
                bias=True,
            ),
        )
        self.add_module(
            "conv_conn_classifier",
            nn.Conv2d(
                self.n_filters_coherence,
                self.n_classes,
                (self.n_filters_coherence, 1),
                bias=True,
            ),
        )
        self.add_module("conne_shift", Expression(shift_3rd_dim_output))

        self.add_module("softmax", nn.LogSoftmax(dim=1))
        self.add_module("squeeze", Expression(squeeze_final_output))

        init.xavier_uniform_(self.conv_time.weight, gain=1)
        init.constant_(self.conv_time.bias, 0)
        init.xavier_uniform_(self.conv_conn_classifier.weight, gain=1)
        init.constant_(self.conv_conn_classifier.bias, 0)
        init.xavier_uniform_(self.conv_power_classifier.weight, gain=1)
        init.constant_(self.conv_power_classifier.bias, 0)
        if self.batch_norm:
            init.constant_(self.bnorm_power.weight, 1)
            init.constant_(self.bnorm_power.bias, 0)
            init.constant_(self.bnorm_conne.weight, 1)
            init.constant_(self.bnorm_conne.bias, 0)
        init.xavier_uniform_(self.conv_spat_conne.weight, gain=1)
        if not self.batch_norm:
            init.constant_(self.conv_spat_conne.bias, 0)
        init.xavier_uniform_(self.conv_spat_power.weight, gain=1)
        if not self.batch_norm:
            init.constant_(self.conv_spat_power.bias, 0)

    def forward(self, x):

        x = self.filterbank(x)
        x = self.dimshuffle(x)
        x = self.conv_time(x)
        if self.n_filters_power > 0:
            x1 = x[:, 0:self.n_filters_power, :, :]
            x1 = self.conv_spat_power(x1)
            if self.batch_norm:
                x1 = self.bnorm_power(x1)
            x1 = self.conv_nonlin_exp(x1)
            x1 = self.poolfunc(x1)
            x1 = self.pool_nonlin_exp(x1)
            x1 = self.power_drop(x1)
            x1 = self.conv_power_classifier(x1)

        if self.n_filters_coherence > 1:
            if self.same_filters_for_features:
                x2 = x[:, 0:self.n_filters_coherence, :, :]
            else:
                x2 = x[:, self.n_filters_power:self.n_filters_power + self.n_filters_coherence, :, :]
            x2 = self.conv_spat_conne(x2)
            if self.batch_norm:
                x2 = self.bnorm_conne(x2)
            x2 = self.connectivity_exp(x2)
            x2 = self.conv_conn_classifier(x2)
            x2 = self.conne_shift(x2)

        if self.n_filters_power > 0:
            if self.n_filters_coherence > 1:
                xout = torch.cat((x1, x2), dim=2)
            else:
                xout = x1
        else:
            xout = x2
        xout = self.softmax(xout)
        xout = self.squeeze(xout)

        return xout


class filterbank(nn.Module):
    def __init__(self, fs, frequency_bands, filterStop=None, f_trans=1, gpass = 3, gstop = 30):
        super(filterbank, self).__init__()
        self.fs = fs
        self.f_trans = f_trans
        self.frequency_bands = frequency_bands
        self.filterStop = filterStop
        self.gpass = gpass
        self.gstop = gstop
        self.Nyquist_freq = self.fs / 2
        self.nFilter = len(self.frequency_bands)


    def forward(self, x):
        while (len(x.shape) < 4):
            x = x.unsqueeze(-1)
        (n_trials, n_channels, n_samples, temp) = x.size()
        all_filtered = torch.Tensor(np.zeros((n_trials, n_channels, n_samples, self.nFilter))).to(x.device)

        for i in range(self.nFilter):
            (l_freq, h_freq) = self.frequency_bands[i]
            f_pass = np.asarray([l_freq, h_freq])
            if self.filterStop is not None:
                f_stop = np.asarray(self.filterStop[i])
            else:
                f_stop = np.asarray([l_freq - self.f_trans, h_freq + self.f_trans])
            wp = f_pass / self.Nyquist_freq
            ws = f_stop / self.Nyquist_freq
            order, wn = cheb2ord(wp, ws, self.gpass, self.gstop)
            b, a = signal.cheby2(order, self.gstop, ws, btype='bandpass')
            data = x[:,:,:,0]

            torch_a = torch.as_tensor(a,dtype = data.dtype).to(x.device)
            torch_b = torch.as_tensor(b,dtype = data.dtype).to(x.device)
            for j in range(n_trials):
                all_filtered[j,:,:,i] = torchaudio.functional.lfilter(data[j, :, :],torch_a, torch_b)

        return all_filtered

class coherence_cropped(nn.Module):
    def __init__(self, time_length = "auto", time_stride = 1):
        super(coherence_cropped, self).__init__()
        self.time_length = time_length
        self.time_stride = time_stride

    def forward(self, x):
        while (len(x.shape) < 4):
            x = x.unsqueeze(-1)
        (n_trials, n_channels, n_samples, n_slice) = x.size()
        if self.time_length == "auto":
            self.time_length = n_samples
        n_windows_per_slice = int((n_samples - self.time_length) / self.time_stride) + 1
        y = torch.zeros(n_trials, n_channels, n_channels, n_slice * n_windows_per_slice)
        for i_trial in range(n_trials):
            for i_slice in range(n_slice):
                for i in range(n_windows_per_slice):
                    temp = x[i_trial, :, self.time_stride * i:self.time_stride * i + self.time_length, i_slice].squeeze(-1).squeeze(0)
                    y[i_trial, :, :, i_slice * n_windows_per_slice + i] = torch.corrcoef(temp)
        y = y.to(x.device)
        return y

def get_padding(kernel_size, stride=1, dilation=1, **_):
    if isinstance(kernel_size, tuple):
        kernel_size = max(kernel_size)
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

class group_temporal_filter(nn.Module):
    def __init__(self,
                 n_filterbank,
                 n_filters_time,
                 kernel_size_group,
                 stride_size = 1
                 ):
        super(group_temporal_filter, self).__init__()
        self.n_filterbank = n_filterbank
        self.n_filters_time = n_filters_time
        self.kernel_size_group = kernel_size_group if isinstance(kernel_size_group, list) else [kernel_size_group]
        self.n_group = len(self.kernel_size_group)
        self.stride_size = stride_size
        self.filter_list = nn.ModuleList([nn.Conv2d(
                    self.n_filterbank,
                    self.n_filters_time,
                    self.kernel_size_group[i],
                    stride=self.stride_size,
                    padding=(get_padding(self.kernel_size_group[i],self.stride_size),0)
                ) for i in range(self.n_group)])

        for layer in self.filter_list:
            torch.nn.init.xavier_uniform_(layer.weight, gain=1)
            torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        for layer in self.filter_list:
            if 'y' in dir():
                y = torch.cat((y,layer(x)),dim = 1)
            else:
                y = layer(x)
        return y



def coherence(x):
    if x.dim() == 4:
        y = torch.zeros(x.size()[0],x.size()[1],x.size()[1],x.size()[3])
        for i in range(0,x.size()[0]):
            for j in range(0,x.size()[3]):
                temp = x[i,:,:,j].squeeze(-1).squeeze(0)
                y[i,:,:,j] = torch.corrcoef(temp)

    return y




def squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    return x

def squeeze_final_output_2d(x):
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x


def squeeze_3rd_dim_output(x):
    assert x.size()[2] == 1
    x = torch.squeeze(x,2)
    return x

def shift_3rd_dim_output(x):
    assert x.size()[2] == 1
    x = torch.transpose(x,2,3)
    return x
