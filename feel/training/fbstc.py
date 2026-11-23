
# Reference:
# "W. Huang, W. Wang, Y. Li, W. Wu. FBSTCNet: A Spatio-Temporal Convolutional Network Integrating Power and Connectivity Features for EEG-Based Emotion Decoding. 2023. (under review)"


import sys
import os
import time
from datetime import datetime
import scipy.io as scio
import numpy as np
from torch.utils.data import Dataset
import torch
import logging
import csv

# from utils.metric import Metric, SubMetric
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler
from braindecode import EEGClassifier
from torch.optim import AdamW
from braindecode.training import CroppedLoss
from braindecode.util import set_random_seeds
from braindecode.models import get_output_shape
from sklearn.metrics import confusion_matrix


def train(model, dataset_train, dataset_val, dataset_test, device, output_dir="result/", metrics=None,
          metric_choose=None, lr=0.0625 * 0.01, weight_decay=0, scheduler=None, batch_size=16, epochs=40, n_classes=3,
          test_sub_label=None, loss_func=None, loss_param=None):

    ConfM = np.zeros([epochs, n_classes, n_classes])
    confusion_mat_group = np.zeros([epochs, n_classes, n_classes])
    model = model.to(device)
    clf = EEGClassifier(
        model,
        cropped=True,  # cropped decoding
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.nll_loss,
        optimizer=torch.optim.AdamW,
        train_split=None,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        iterator_train__shuffle=True,
        batch_size=batch_size,
        device=device,
    )
    for iep in range(epochs):
        clf.partial_fit(dataset_train, y=None, epochs=1)
        y_pred = clf.predict(dataset_val)
        confusion_mat_group[iep, :, :] = confusion_matrix(dataset_val.tensors[1], y_pred)
        ConfM[iep, :, :] = ConfM[iep, :, :] + confusion_matrix(dataset_val.tensors[1], y_pred)
    correct_all = np.zeros(epochs)
    for iepc in range(epochs):
        correct_all[iepc] = ConfM[iepc][0][0] + ConfM[iepc][1][1] + ConfM[iepc][2][2]
    best_epoch = np.argmax(correct_all) + 1
    best_correct = np.max(correct_all)
    clf = EEGClassifier(
        model,
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.nll_loss,
        optimizer=torch.optim.AdamW,
        train_split=None,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        iterator_train__shuffle=True,
        batch_size=batch_size,
        device=device,
    )
    clf.fit(dataset_train, y=None, epochs=best_epoch)
    y_pred = torch.tensor(clf.predict(dataset_test))

    result = None
    if test_sub_label is None:
        result = Metric(metrics)
    else:
        result = SubMetric(metrics)

    if test_sub_label is None:
        result.update(y_pred, dataset_test.tensors[1])
    else:
        result.update(y_pred, dataset_test.tensors[1], torch.tensor(test_sub_label).to(device))
    result.value()
    result = result.values
    for m in metrics:
        print(f"best_val_{m}: {result[m]:.2f}")
        print(f"best_test_{m}: {result[m]:.2f}")
    return result
