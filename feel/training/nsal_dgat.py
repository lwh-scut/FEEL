import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.optim.optimizer import Optimizer
from typing import Optional
import math
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader,RandomSampler, SequentialSampler
from tqdm import tqdm

from feel.utils.metrics import Metric, SubMetric
from feel.utils.results import save_state

class StepwiseLR_GRL:
    def __init__(self, optimizer: Optimizer, init_lr: Optional[float] = 0.01,
                 gamma: Optional[float] = 0.001, decay_rate: Optional[float] = 0.75, max_iter: Optional[float] = 1000):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0
        self.max_iter = max_iter

    def get_lr(self) -> float:
        lr = self.init_lr / (1.0 + self.gamma * (self.iter_num / self.max_iter)) ** (self.decay_rate)
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if 'lr_mult' not in param_group:
                param_group['lr_mult'] = 1.
            param_group['lr'] = lr * param_group['lr_mult']

        self.iter_num += 1



class CE_Label_Smooth_Loss(nn.Module):
    def __init__(self, classes=3, epsilon=0.05, ):
        super(CE_Label_Smooth_Loss, self).__init__()
        self.classes = classes
        self.epsilon = epsilon

    def forward(self, input, target):
        log_prob = torch.nn.functional.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
                 self.epsilon / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.epsilon))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


def train(model, dann_loss, hidden_2, dataset_train, dataset_val, dataset_test, device, output_dir ="result", metrics=None, metric_choose=None, optimizer=None,
             scheduler=None, batch_size=16, epochs=40, criterion=None, test_sub_label = None,loss_func=None, loss_param=None):
    if metrics is None:
        metrics = ['acc']
    if metric_choose is None:
        metric_choose = metrics[0]
    # data sampler for train and test data
    sampler_train = RandomSampler(dataset_train)
    sampler_val = SequentialSampler(dataset_val)
    sampler_test = SequentialSampler(dataset_test)
    # load dataset
    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train, batch_size=batch_size, num_workers=4, drop_last=True
    )
    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val, batch_size=batch_size, num_workers=4, drop_last=True
    )
    data_loader_test = DataLoader(
        dataset_test, sampler=sampler_test, batch_size=batch_size, num_workers=4,drop_last=True
    )
    test_sub_label_loader = DataLoader(
        test_sub_label, sampler=sampler_test, batch_size=batch_size, num_workers=4
    )
    model = model.to(device)
    interval =1

    getInit(data_loader_train, model, device)
    iteration = math.ceil(len(dataset_train)/batch_size)
    best_metric = {s: 0. for s in metrics}
    for epoch in range(epochs):
        model.train()
        dann_loss.train()
        correct = 0
        count = 0
        _tqdm = tqdm(range(iteration), desc= f"Train Epoch {epoch  + 1}/{epochs}",leave=False)
        if epoch % interval == 0 and epoch > 0:
            loss, metric_value = evaluate(
                model, data_loader_val, device, metrics, criterion
            )
            for m in metrics:
                # if metric is the best, save the model state
                if metric_value[m] > best_metric[m]:
                    best_metric[m] = metric_value[m]
                    save_state(output_dir, model, optimizer, epoch + 1, metric=m)
        source_loader_iter = enumerate(data_loader_train)
        target_loader_iter = enumerate(data_loader_val)
        T = len(dataset_test)//batch_size
        for i in _tqdm:
            try:
                _, (src_data, src_index, src_label_cls) = next(source_loader_iter)
            except Exception as err:
                source_loader_iter = enumerate(data_loader_train)
                _, (src_data, src_index, src_label_cls) = next(source_loader_iter)
            try:
                _, (tar_data, _) = next(target_loader_iter)
            except Exception as err:
                target_loader_iter = enumerate(data_loader_val)
                _, (tar_data, _) = next(target_loader_iter)

            src_data = Variable(src_data.to(device))
            src_index = Variable(src_index.to(device))
            src_label_cls = Variable(src_label_cls.to(device))
            tar_data = Variable(tar_data.to(device))
            src_output_cls, src_feature, tar_output_cls, tar_feature, source_att, target_att, tar_label = model(
                src_data, tar_data, src_label_cls, src_index)
            cls_loss = criterion(src_output_cls, src_label_cls)
            tar_label = torch.argmax(tar_label, dim=1)
            target_loss = criterion(tar_output_cls, tar_label)
            global_transfer_loss = dann_loss(
                src_feature + 0.005 * torch.randn((src_feature.shape[0], (hidden_2))).to(device),
                tar_feature + 0.005 * torch.randn((tar_feature.shape[0], (hidden_2))).to(device),
                src_output_cls, tar_output_cls)
            boost_factor = 2.0 * (2.0 / (1.0 + math.exp(-1 * epoch / 1000)) - 1)
            # update joint loss function
            optimizer.zero_grad()
            loss = cls_loss + global_transfer_loss + boost_factor * (target_loss)
            loss.backward()
            optimizer.step()

            _tqdm.set_postfix_str(f"lr : {optimizer.param_groups[0]['lr']} loss: {loss.item():.2f}")
            # calculate the correct
            _, pred = torch.max(src_output_cls, dim=1)
            if src_label_cls.dim() == 1:
                correct += pred.eq(src_label_cls.data.view_as(pred)).sum()
            elif src_label_cls.dim() == 2:
                correct += pred.eq(torch.argmax(src_label_cls, dim=1).data.view_as(pred)).sum()
            else:
                print(f"警告：未知的标签维度: {src_label_cls.dim()}")
            count += pred.size(0)
        scheduler.step()
        accuracy = float(correct) / count
        print(f"Train epoch {epoch+1}, acc: {accuracy}")
        # _tqdm.set_postfix_str(f"train/epoch {epoch}, accuracy {accuracy} , loss {loss}, cls_loss {cls_loss}")
    model.load_state_dict(torch.load(f"{output_dir}/checkpoint-best{metric_choose}")['model'])
    if test_sub_label is not None:
        _, metric_value = sub_evaluate(model, data_loader_test, test_sub_label_loader, device, metrics, criterion)
    else:
        _, metric_value = evaluate(model, data_loader_test, device, metrics, criterion)
    # print best metrics
    for m in metrics:
        print(f"best_val_{m}: {best_metric[m]:.2f}")
        print(f"best_test_{m}: {metric_value[m]:.2f}")
    return metric_value


def getInit(train_loader, model, device):
    model.eval()
    for _, (tran_input, tran_indx, _ ) in enumerate(train_loader):
        tran_input, tran_indx = tran_input.to(device), tran_indx.to(device)
        tran_input, tran_indx = Variable(tran_input), Variable(tran_indx)
        model.get_init_banks(tran_input, tran_indx)

@torch.no_grad()
@torch.no_grad()
def evaluate(model, data_loader, device, metrics, criterion):
    model.eval()
    metric = Metric(metrics)
    total_loss = 0.0
    num_batches = 0

    if len(data_loader) == 0:
        print("eval state : No data to evaluate.")
        # 返回一个默认的 loss（比如 NaN 或 0）和空指标
        dummy_values = {m: 0.0 for m in metrics}
        return float('nan'), dummy_values

    for idx, (samples, targets) in tqdm(enumerate(data_loader), total=len(data_loader),
                                        desc=f"Evaluating : "):
        samples = Variable(samples.to(device))
        targets = Variable(targets.to(device))

        output = model.target_predict(samples)
        loss = criterion(output, targets)
        total_loss += loss.item()
        num_batches += 1

        _, pred = torch.max(output, dim=1)
        metric.update(pred, targets)

    avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
    print("eval state : " + metric.value())
    return avg_loss, metric.values

@torch.no_grad()
def sub_evaluate(model, data_loader, test_sub_label, device, metrics, criterion):
    model.eval()
    confusion_matrixs = 0
    metric = SubMetric(metrics)
    for idx, ((samples,targets), sub_label) in tqdm(enumerate(zip(data_loader, test_sub_label)), total=len(data_loader),
                                        desc=f"Evaluating : "):
        # load the samples into the device
        samples = Variable(samples.to(device))
        targets = Variable(targets.to(device))
        output = model.target_predict(samples)
        loss = criterion(output, targets.view(-1))
        _, pred = torch.max(output, dim=1)
        metric.update(pred, targets, sub_label)
    print("eval state : " + metric.value())
    return loss, metric.values
