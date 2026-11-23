import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import os

@torch.no_grad()
def evaluate_multimodal(model, data_loader, device, metrics, criterion):
    model.eval()
    all_preds, all_targets, all_probs, losses = [], [], [], []

    for samples in tqdm(data_loader, desc="Evaluating"):
        eeg = samples[0].to(device)
        fnirs = samples[1].to(device)
        targets = samples[2].to(device)

        outputs = model(eeg, fnirs)
        loss = criterion(outputs, targets)

        preds = torch.argmax(outputs, dim=1)
        probs = torch.softmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        losses.append(loss.item())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    results = {'loss': np.mean(losses)}
    if 'acc' in metrics:
        results['acc'] = np.mean(all_preds == all_targets)
    if 'f1' in metrics:
        results['f1'] = f1_score(all_targets, all_preds, average='macro')
    if 'auc' in metrics:
        try:
            if len(np.unique(all_targets)) > 1:
                results['auc'] = roc_auc_score(all_targets, all_probs[:, 1])
            else:
                results['auc'] = 0.0
        except:
            results['auc'] = 0.0

    return results


def train_multimodal(model, dataset_train, dataset_val, dataset_test, device, output_dir="result/",
                     metrics=None, metric_choose=None, optimizer=None, scheduler=None,
                     batch_size=16, epochs=40, criterion=None):

    if metrics is None:
        metrics = ['acc', 'f1', 'auc']
    if metric_choose is None:
        metric_choose = metrics[0]

    sampler_train = RandomSampler(dataset_train)
    sampler_val = SequentialSampler(dataset_val)
    sampler_test = SequentialSampler(dataset_test)

    data_loader_train = DataLoader(dataset_train, sampler=sampler_train, batch_size=batch_size, num_workers=4)
    data_loader_val = DataLoader(dataset_val, sampler=sampler_val, batch_size=batch_size, num_workers=4)
    data_loader_test = DataLoader(dataset_test, sampler=sampler_test, batch_size=batch_size, num_workers=4)

    model = model.to(device)
    best_metric = {m: 0. for m in metrics}
    best_model_state = None
    best_val_metric = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        train_preds, train_targets, train_losses = [], [], []

        train_bar = tqdm(enumerate(data_loader_train), total=len(data_loader_train),
                         desc=f"Train Epoch {epoch + 1}/{epochs}")

        for idx, samples in train_bar:
            eeg = samples[0].to(device)
            fnirs = samples[1].to(device)
            targets = samples[2].to(device)

            optimizer.zero_grad()
            outputs = model(eeg, fnirs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
            train_losses.append(loss.item())
            train_bar.set_postfix_str(f"loss: {loss.item():.2f}")

        if scheduler:
            scheduler.step()

        train_acc = np.mean(np.array(train_preds) == np.array(train_targets))
        train_f1 = f1_score(train_targets, train_preds, average='macro')
        avg_train_loss = np.mean(train_losses)

        val_results = evaluate_multimodal(model, data_loader_val, device, metrics, criterion)

        print(f"\033[32m Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"\033[34m Val - Loss: {val_results['loss']:.4f}, Acc: {val_results.get('acc', 0):.4f}, "
              f"F1: {val_results.get('f1', 0):.4f}, AUC: {val_results.get('auc', 0):.4f}")

        if val_results.get(metric_choose, 0) > best_val_metric:
            best_val_metric = val_results[metric_choose]
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()

    # 测试
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    test_results = evaluate_multimodal(model, data_loader_test, device, metrics, criterion)

    return {
        **test_results,
        'best_val_metric': best_val_metric,
        'best_epoch': best_epoch,
        'best_model_state': best_model_state
    }
