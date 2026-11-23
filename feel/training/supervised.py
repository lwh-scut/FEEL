import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime


def train(model, dataset_train, dataset_val, dataset_test, device, output_dir="result/",
          metrics=None, metric_choose=None, optimizer=None, scheduler=None,
          batch_size=16, epochs=40, criterion=None, loss_func=None, loss_param=None):
    if metrics is None:
        metrics = ['acc', 'f1', 'auc']  # 增加f1和auc指标

    # 训练历史记录
    train_history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'val_auc': []
    }

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
        dataset_val, sampler=sampler_val, batch_size=batch_size, num_workers=4
    )
    data_loader_test = DataLoader(
        dataset_test, sampler=sampler_test, batch_size=batch_size, num_workers=4
    )

    model = model.to(device)
    best_metric = {s: 0. for s in metrics}
    best_model_state = None
    best_val_metric = 0.0  # 记录最佳验证指标
    best_epoch = 0  # 记录最佳epoch

    for epoch in range(epochs):
        model.train()

        # 训练阶段
        train_preds, train_targets, train_losses = [], [], []
        train_bar = tqdm(enumerate(data_loader_train), total=len(data_loader_train),
                         desc=f"Train Epoch {epoch + 1}/{epochs}: lr:{optimizer.param_groups[0]['lr']}")

        for idx, (samples, targets) in train_bar:
            samples = samples.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            outputs = model(samples)
            loss = criterion(outputs, targets) + (0 if loss_func is None else loss_func(loss_param))

            # 收集预测结果
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()
            train_bar.set_postfix_str(f"loss: {loss.item():.2f}")

        if scheduler is not None:
            scheduler.step()

        # 计算训练指标
        train_acc = np.mean(np.array(train_preds) == np.array(train_targets))
        train_f1 = f1_score(train_targets, train_preds, average='macro')
        avg_train_loss = np.mean(train_losses)

        # 验证阶段
        val_results = evaluate(model, data_loader_val, device, metrics, criterion, loss_func, loss_param)

        # 记录训练历史
        train_history['epoch'].append(epoch + 1)
        train_history['train_loss'].append(avg_train_loss)
        train_history['train_acc'].append(train_acc)
        train_history['train_f1'].append(train_f1)
        train_history['val_loss'].append(val_results.get('loss', 0))
        train_history['val_acc'].append(val_results.get('acc', 0))
        train_history['val_f1'].append(val_results.get('f1', 0))
        train_history['val_auc'].append(val_results.get('auc', 0))

        print(f"\033[32m Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"\033[34m Val - Loss: {val_results.get('loss', 0):.4f}, Acc: {val_results.get('acc', 0):.4f}, "
              f"F1: {val_results.get('f1', 0):.4f}, AUC: {val_results.get('auc', 0):.4f}")

        # 保存最佳模型
        for m in metrics:
            if val_results.get(m, 0) > best_metric[m]:
                best_metric[m] = val_results.get(m, 0)
                if m == metric_choose:
                    best_model_state = model.state_dict().copy()
                    best_val_metric = val_results.get(m, 0)
                    best_epoch = epoch + 1
                    # 保存模型状态
                    # from utils.store import save_state
                    # save_state(output_dir, model, optimizer, epoch + 1, metric=m)

    # 加载最佳模型进行测试
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 最终测试
    test_results = evaluate(model, data_loader_test, device, metrics, criterion, loss_func, loss_param)

    # 收集训练和测试数据用于绘制混淆矩阵
    test_data_tensor = dataset_test.tensors[0].cpu() if hasattr(dataset_test, 'tensors') else None
    test_labels_tensor = dataset_test.tensors[1].cpu() if hasattr(dataset_test, 'tensors') else None
    train_data_tensor = dataset_train.tensors[0].cpu() if hasattr(dataset_train, 'tensors') else None
    train_labels_tensor = dataset_train.tensors[1].cpu() if hasattr(dataset_train, 'tensors') else None

    # 返回更多信息用于结果记录
    return {
        **test_results,
        'train_history': train_history,
        'best_model': model,
        'best_model_state': best_model_state,
        'best_val_metric': best_val_metric,
        'best_epoch': best_epoch,
        'test_data': test_data_tensor,
        'test_labels': test_labels_tensor,
        'train_data': train_data_tensor,
        'train_labels': train_labels_tensor
    }


@torch.no_grad()
def evaluate(model, data_loader, device, metrics, criterion, loss_func, loss_param):
    model.eval()
    all_preds, all_targets, all_probs, losses = [], [], [], []

    for idx, (samples, targets) in tqdm(enumerate(data_loader), total=len(data_loader),
                                        desc="Evaluating"):
        samples = samples.to(device)
        targets = targets.to(device)

        outputs = model(samples)
        loss = criterion(outputs, targets) + (0 if loss_func is None else loss_func(loss_param))

        preds = torch.argmax(outputs, dim=1)
        probs = torch.softmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        losses.append(loss.item())

    # 计算各种指标
    results = {}
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # 调试信息：打印标签分布
    unique_targets, target_counts = np.unique(all_targets, return_counts=True)
    unique_preds, pred_counts = np.unique(all_preds, return_counts=True)
    print(f"验证集标签分布: 真实标签{dict(zip(unique_targets, target_counts))}, "
          f"预测标签{dict(zip(unique_preds, pred_counts))}")

    if 'acc' in metrics:
        results['acc'] = np.mean(all_preds == all_targets)

    if 'f1' in metrics:
        results['f1'] = f1_score(all_targets, all_preds, average='macro')

    if 'auc' in metrics:
        try:
            if len(np.unique(all_targets)) > 1:  # 确保有多个类别
                results['auc'] = roc_auc_score(all_targets, all_probs[:, 1])
            else:
                results['auc'] = 0.0
        except:
            results['auc'] = 0.0

    results['loss'] = np.mean(losses)

    return results
