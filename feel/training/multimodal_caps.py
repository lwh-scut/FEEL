import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

@torch.no_grad()
def evaluate_multimodal_caps(model, data_loader, device, metrics):
    """
    专用于 CapsNet 的评估函数：不使用 criterion，直接计算预测和指标
    """
    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    for samples in tqdm(data_loader, desc="Evaluating"):
        eeg = samples[0].to(device)
        fnirs = samples[1].to(device)
        targets = samples[2].to(device)

        caps_output = model(eeg, fnirs)  # (B, num_classes, D)
        v_mag = torch.norm(caps_output, dim=-1)  # (B, num_classes)
        preds = torch.argmax(v_mag, dim=1)
        probs = torch.softmax(v_mag, dim=1)  # 用于 AUC

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    results = {}
    if 'acc' in metrics:
        results['acc'] = np.mean(all_preds == all_targets)
    if 'f1' in metrics:
        results['f1'] = f1_score(all_targets, all_preds, average='macro')
    if 'auc' in metrics:
        try:
            if len(np.unique(all_targets)) > 1:
                if all_probs.shape[1] == 2:
                    results['auc'] = roc_auc_score(all_targets, all_probs[:, 1])
                else:
                    results['auc'] = roc_auc_score(all_targets, all_probs, multi_class='ovr')
            else:
                results['auc'] = 0.0
        except Exception:
            results['auc'] = 0.0

    return results


def train_multimodal_caps(
    model,
    dataset_train,
    dataset_val,
    dataset_test,
    device,
    output_dir="result/",
    metrics=None,
    metric_choose=None,
    optimizer=None,
    scheduler=None,
    batch_size=16,
    epochs=40,
    num_classes=2  # 用于 AUC 判断，可选
):
    """
    专用于 GCN_CA_CapsNet 的训练函数。
    注意：不接受 criterion，因为使用内置 margin_loss。
    """
    if metrics is None:
        metrics = ['acc', 'f1', 'auc']
    if metric_choose is None:
        metric_choose = 'f1'

    sampler_train = RandomSampler(dataset_train)
    sampler_val = SequentialSampler(dataset_val)
    sampler_test = SequentialSampler(dataset_test)

    data_loader_train = DataLoader(dataset_train, sampler=sampler_train, batch_size=batch_size, num_workers=4, pin_memory=True)
    data_loader_val = DataLoader(dataset_val, sampler=sampler_val, batch_size=batch_size, num_workers=4, pin_memory=True)
    data_loader_test = DataLoader(dataset_test, sampler=sampler_test, batch_size=batch_size, num_workers=4, pin_memory=True)

    model = model.to(device)
    best_val_metric = -1.0
    best_model_state = None
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        train_bar = tqdm(enumerate(data_loader_train), total=len(data_loader_train),
                         desc=f"Train Epoch {epoch + 1}/{epochs}")

        for idx, samples in train_bar:
            eeg = samples[0].to(device, non_blocking=True)
            fnirs = samples[1].to(device, non_blocking=True)
            targets = samples[2].to(device, non_blocking=True)

            optimizer.zero_grad()
            caps_output = model(eeg, fnirs)  # (B, num_classes, D)
            loss = model.margin_loss(caps_output, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_bar.set_postfix_str(f"loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(data_loader_train)

        # 验证
        val_results = evaluate_multimodal_caps(model, data_loader_val, device, metrics)
        current_metric = val_results.get(metric_choose, 0.0)

        print(f"\033[32m Train - Loss: {avg_train_loss:.4f}")
        print(f"\033[34m Val - Acc: {val_results.get('acc', 0):.4f}, "
              f"F1: {val_results.get('f1', 0):.4f}, AUC: {val_results.get('auc', 0):.4f}")

        if current_metric > best_val_metric:
            best_val_metric = current_metric
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()

        if scheduler:
            scheduler.step()

    # 测试阶段
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    test_results = evaluate_multimodal_caps(model, data_loader_test, device, metrics)

    return {
        **test_results,
        'best_val_metric': best_val_metric,
        'best_epoch': best_epoch,
        'best_model_state': best_model_state
    }
