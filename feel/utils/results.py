import argparse
import os.path
import time
from pathlib import Path

import torch

def make_output_dir(args, model):
    output_dir = Path(args.output_dir)
    output_dir = output_dir / model
    if args.setting is not None:
        output_dir = output_dir / args.setting
    else:
        output_dir = output_dir / args.experiment_mode
        output_dir = output_dir / args.split_type
    if args.label_used is not None:
        if len(args.label_used) == 1:
            output_dir = output_dir/args.label_used[0]
        else:
            output_dir = output_dir/ "both".join(label for label in args.label_used)
    return output_dir

def save_state(output_dir, model, optimizer, epoch, r_idx='last', rr_idx='last', metric=None, state='best'):
    # compatibility
    if type(output_dir) is argparse.Namespace:
        output_dir = make_output_dir(output_dir, output_dir.model)
    else:
        output_dir = Path(output_dir)
    if not ( r_idx == 'last' and rr_idx == 'last'):
        output_dir = output_dir / str(r_idx)
        output_dir = output_dir / str(rr_idx)

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"An error occurred: {e.strerror}")
    checkpoint_path = output_dir / f'checkpoint-{str(epoch)}' if metric is None \
        else output_dir / f'checkpoint-{state}{metric}'
    save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(save, checkpoint_path)
    print(f"save model to {checkpoint_path}")


def save_data(args, data, label):
    save_dir = Path(args.data_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    mode = {
        'subject-dependent': 'sub-dep',
        'subject-independent': 'sub-In',
        'cross-session': 'cro-sess'
    }
    save_path = save_dir / f'{args.dataset}'
    save_path = save_path / f'{args.feature_type}-tw-{args.time_window}ol-{args.overlap}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(f"Saving Processed Data To {save_path}")


def save_res(args, metric):
    log_dir = Path(args.log_dir)
    # 确保日志目录存在
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    # 使用安全的文件名格式（替换冒号为下划线）
    safe_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / safe_time

    # 如果文件不存在，创建并写入参数
    if not log_file.exists():
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(str(args))

    # 追加写入结果
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write('\n' + str(metric))


def add_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)



import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import torch


def result_log(output_dir, strategy, cross_trail, group_idx, group_metrics,
               lr, batch_size, epochs, seed, k_fold, train_history=None,
               best_model=None, test_data=None, test_labels=None,
               train_data=None, train_labels=None):
    """
    增强版结果记录函数，包含详细指标记录和多种可视化
    """
    # 根据策略确定保存路径
    if strategy == "single-subject":
        save_dir = os.path.join(output_dir, f"subject_{group_idx}")
    elif strategy == "within-subject":
        save_dir = os.path.join(output_dir, "within_subject_results")
    elif strategy == "cross-subject":
        save_dir = os.path.join(output_dir, f"subject_{group_idx}")
    elif strategy == "cross-trial":
        save_dir = os.path.join(output_dir, f"trial_{group_idx}")
    elif strategy == "cross-session":
        save_dir = os.path.join(output_dir, f"session_{group_idx}")
    else:
        save_dir = os.path.join(output_dir, f"group_{group_idx}")

    os.makedirs(save_dir, exist_ok=True)

    # 1. 保存详细的指标结果
    save_detailed_metrics(save_dir, strategy, group_idx, group_metrics,
                          lr, batch_size, epochs, seed, k_fold)

    # 2. 绘制训练曲线（如果有训练历史）
    if train_history and len(train_history) > 0:
        plot_training_curves(save_dir, train_history[0], group_idx)

    # 3. 绘制混淆矩阵（如果有最佳模型和测试数据）
    if (best_model is not None and test_data is not None and
            train_data is not None and len(group_metrics) > 0):
        plot_confusion_matrices(save_dir, best_model, train_data, train_labels,
                                test_data, test_labels, group_idx)

    # 4. 保存K折交叉验证结果
    if len(group_metrics) > 1:
        save_kfold_results(save_dir, group_metrics, group_idx)

    # 5. 生成汇总报告
    generate_summary_report(save_dir, strategy, cross_trail, group_idx, group_metrics,
                            lr, batch_size, epochs, seed, k_fold)

    # 6. 保存组的最佳模型信息
    if len(group_metrics) > 0:
        # 找出最佳折
        best_fold_idx = np.argmax([m.get('val_acc', 0) for m in group_metrics])
        best_fold_metrics = group_metrics[best_fold_idx]

        # 保存最佳模型信息
        best_model_info = {
            'group_index': group_idx,
            'best_fold_index': int(best_fold_idx),
            'best_val_accuracy': float(best_fold_metrics.get('val_acc', 0)),
            'best_test_accuracy': float(best_fold_metrics.get('acc', 0)),
            'best_test_f1': float(best_fold_metrics.get('f1', 0)),
            'best_test_auc': float(best_fold_metrics.get('auc', 0)),
            'best_epoch': best_fold_metrics.get('epoch', 0),
            'timestamp': datetime.now().isoformat()
        }

        print(best_model_info)

        best_model_file = os.path.join(save_dir, f"best_model_info_group_{group_idx}.json")
        with open(best_model_file, 'w', encoding='utf-8') as f:
            json.dump(best_model_info, f, indent=2, ensure_ascii=False)


def save_detailed_metrics(save_dir, strategy, group_idx, group_metrics,
                          lr, batch_size, epochs, seed, k_fold):
    """保存详细的指标结果到JSON文件"""
    metrics_dict = {
        'strategy': strategy,
        'group_index': group_idx,
        'hyperparameters': {
            'learning_rate': lr,
            'batch_size': batch_size,
            'epochs': epochs,
            'random_seed': seed,
            'k_fold': k_fold
        },
        'timestamp': datetime.now().isoformat(),
        'fold_results': []
    }

    for fold_idx, metrics in enumerate(group_metrics):
        fold_result = {
            'fold_index': fold_idx,
            'validation_accuracy': float(metrics.get('val_acc', 0)),  # 添加验证集准确率
            'test_accuracy': float(metrics.get('acc', 0)),
            'test_f1_score': float(metrics.get('f1', 0)),
            'test_auc': float(metrics.get('auc', 0)),
            'test_loss': float(metrics.get('loss', 0)),
            'best_epoch': int(metrics.get('epoch', 0))  # 添加最佳epoch
        }
        metrics_dict['fold_results'].append(fold_result)

    # 计算统计信息
    val_accuracies = [m.get('val_acc', 0) for m in group_metrics]
    accuracies = [m.get('acc', 0) for m in group_metrics]
    f1_scores = [m.get('f1', 0) for m in group_metrics]
    auc_scores = [m.get('auc', 0) for m in group_metrics]

    metrics_dict['statistics'] = {
        'mean_accuracy': float(np.mean(accuracies)),
        'std_accuracy': float(np.std(accuracies)),
        'mean_f1': float(np.mean(f1_scores)),
        'std_f1': float(np.std(f1_scores)),
        'mean_auc': float(np.mean(auc_scores)),
        'std_auc': float(np.std(auc_scores)),
        'best_fold_index': int(np.argmax(val_accuracies)),  # 添加最佳折索引
        'best_validation_accuracy': float(np.max(val_accuracies))  # 添加最佳验证准确率
    }

    # 保存到JSON文件
    metrics_file = os.path.join(save_dir, f"detailed_metrics_group_{group_idx}.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)


def plot_training_curves(save_dir, train_history, group_idx):
    """绘制训练和验证的损失及准确率曲线"""
    if not train_history or 'epoch' not in train_history:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 绘制损失曲线
    if 'train_loss' in train_history and 'val_loss' in train_history:
        ax1.plot(train_history['epoch'], train_history['train_loss'],
                 label='Training Loss', color='blue', linewidth=2, marker='o')
        ax1.plot(train_history['epoch'], train_history['val_loss'],
                 label='Validation Loss', color='red', linewidth=2, marker='s')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 绘制准确率曲线
    if 'train_acc' in train_history and 'val_acc' in train_history:
        ax2.plot(train_history['epoch'], train_history['train_acc'],
                 label='Training Accuracy', color='green', linewidth=2, marker='o')
        ax2.plot(train_history['epoch'], train_history['val_acc'],
                 label='Validation Accuracy', color='orange', linewidth=2, marker='s')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = os.path.join(save_dir, f"training_curves_group_{group_idx}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(save_dir, model, train_data, train_labels,
                            test_data, test_labels, group_idx):
    """绘制训练集和测试集的混淆矩阵"""
    try:
        # 获取模型所在的设备
        device = next(model.parameters()).device

        # 确保数据是tensor格式并移动到正确设备
        if not isinstance(train_data, torch.Tensor):
            train_data = torch.FloatTensor(train_data)
        if not isinstance(test_data, torch.Tensor):
            test_data = torch.FloatTensor(test_data)
        if not isinstance(train_labels, torch.Tensor):
            train_labels = torch.LongTensor(train_labels)
        if not isinstance(test_labels, torch.Tensor):
            test_labels = torch.LongTensor(test_labels)

        # 将数据移动到模型所在的设备
        train_data = train_data.to(device)
        test_data = test_data.to(device)

        # 预测
        model.eval()
        with torch.no_grad():
            train_outputs = model(train_data)
            test_outputs = model(test_data)
            train_preds = torch.argmax(train_outputs, dim=1).cpu().numpy()
            test_preds = torch.argmax(test_outputs, dim=1).cpu().numpy()

        train_labels_np = train_labels.cpu().numpy() if hasattr(train_labels, 'cpu') else train_labels
        test_labels_np = test_labels.cpu().numpy() if hasattr(test_labels, 'cpu') else test_labels

        # 计算混淆矩阵
        from sklearn.metrics import confusion_matrix
        train_cm = confusion_matrix(train_labels_np, train_preds)
        test_cm = confusion_matrix(test_labels_np, test_preds)

        # 绘制混淆矩阵
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 训练集混淆矩阵
        sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                    cbar_kws={'label': 'Count'})
        ax1.set_title(f'Training Set Confusion Matrix\n(Group {group_idx})')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')

        # 测试集混淆矩阵
        sns.heatmap(test_cm, annot=True, fmt='d', cmap='Reds', ax=ax2,
                    cbar_kws={'label': 'Count'})
        ax2.set_title(f'Test Set Confusion Matrix\n(Group {group_idx})')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')

        plt.tight_layout()
        cm_file = os.path.join(save_dir, f"confusion_matrices_group_{group_idx}.png")
        plt.savefig(cm_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 混淆矩阵已保存: {cm_file}")
    except Exception as e:
        print(f"❌ 绘制混淆矩阵时出错: {e}")
        import traceback
        traceback.print_exc()


def save_kfold_results(save_dir, group_metrics, group_idx):
    """保存K折交叉验证结果"""
    if len(group_metrics) <= 1:
        return

    # 创建K折结果表格
    kfold_data = []
    for fold_idx, metrics in enumerate(group_metrics):
        kfold_data.append({
            'Fold': fold_idx + 1,
            'Accuracy': f"{metrics.get('acc', 0):.4f}",
            'F1-Score': f"{metrics.get('f1', 0):.4f}",
            'AUC': f"{metrics.get('auc', 0):.4f}",
            'Loss': f"{metrics.get('loss', 0):.4f}"
        })

    # 添加平均和标准差行
    accuracies = [m.get('acc', 0) for m in group_metrics]
    f1_scores = [m.get('f1', 0) for m in group_metrics]
    auc_scores = [m.get('auc', 0) for m in group_metrics]

    kfold_data.append({
        'Fold': 'Mean ± Std',
        'Accuracy': f"{np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}",
        'F1-Score': f"{np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}",
        'AUC': f"{np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}",
        'Loss': '-'
    })

    # 保存为CSV
    df = pd.DataFrame(kfold_data)
    csv_file = os.path.join(save_dir, f"kfold_results_group_{group_idx}.csv")
    df.to_csv(csv_file, index=False)

    # 保存为文本格式
    txt_file = os.path.join(save_dir, f"kfold_summary_group_{group_idx}.txt")
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(f"K-Fold Cross Validation Results - Group {group_idx}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Number of folds: {len(group_metrics)}\n")
        f.write(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}\n")
        f.write(f"Mean F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}\n")
        f.write(f"Mean AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}\n")


def generate_summary_report(save_dir, strategy, cross_trail, group_idx, group_metrics,
                            lr, batch_size, epochs, seed, k_fold):
    """生成汇总报告"""
    # 计算统计信息
    accuracies = [m.get('acc', 0) for m in group_metrics]
    f1_scores = [m.get('f1', 0) for m in group_metrics]
    auc_scores = [m.get('auc', 0) for m in group_metrics]

    report_file = os.path.join(save_dir, f"summary_report_group_{group_idx}.txt")

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("EXPERIMENT SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("EXPERIMENT SETTINGS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Strategy: {strategy}\n")
        f.write(f"Cross Trail: {cross_trail}\n")
        f.write(f"Group Index: {group_idx}\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Random Seed: {seed}\n")
        f.write(f"K-Fold: {k_fold}\n\n")

        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}\n")
        f.write(f"Mean F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}\n")
        f.write(f"Mean AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}\n\n")

        f.write("DETAILED FOLD RESULTS:\n")
        f.write("-" * 40 + "\n")
        for fold_idx, metrics in enumerate(group_metrics):
            f.write(f"Fold {fold_idx + 1}: ")
            f.write(f"Acc={metrics.get('acc', 0):.4f}, ")
            f.write(f"F1={metrics.get('f1', 0):.4f}, ")
            f.write(f"AUC={metrics.get('auc', 0):.4f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")
