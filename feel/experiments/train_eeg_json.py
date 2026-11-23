from feel.train.cli import resolve_device, training_parser

import json
import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import optim, nn
from tqdm import tqdm


def load_one_json_and_label(file_path, label_dimension):
    """
    子进程：加载单个EEG文件和对应的label
    """
    try:
        # 读取EEG数据
        with open(file_path, 'r', encoding='utf-8') as f:
            eeg_data = json.load(f)['data']

        # 获取标签文件路径
        dir_path = os.path.dirname(file_path)
        label_path = os.path.join(dir_path, "label.txt")
        if not os.path.exists(label_path):
            return None

        # 读取标签
        with open(label_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)

        label_value = label_data.get(f"{label_dimension}_label")
        if label_value is None:
            return None

        return file_path, np.array(eeg_data), label_value

    except Exception:
        return None


def load_dataset(eeg_epoch_dir, label_dimension='valence', num_workers=20):
    """
    并行加载整个EEG数据集及对应的标签（不缓存label）
    """
    # 找到所有EEG JSON文件
    epoch_files = []
    for root, _, files in os.walk(eeg_epoch_dir):
        for file in files:
            if file.endswith('.json'):
                epoch_files.append(os.path.join(root, file))

    print(f"找到 {len(epoch_files)} 个EEG片段文件")

    # 使用多进程并行加载
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(load_one_json_and_label, fp, label_dimension)
                   for fp in epoch_files]

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="并行加载EEG文件", ncols=100):
            res = future.result()
            if res is not None:
                results.append(res)

    # 组织数据结构
    data_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    label_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    all_subject_ids = set()
    all_trail_ids = set()

    for file_path, eeg_data, label_value in results:
        path_parts = file_path.split(os.sep)
        music_id, sub_id = None, None

        for part in path_parts:
            if part.startswith('song'):
                try:
                    music_id = int(part[4:])
                except ValueError:
                    continue
            if part.startswith('sub-') and '_EEG_epochs' in part:
                try:
                    sub_id = part.split('_')[0][4:]
                except Exception:
                    continue

        if music_id is None or sub_id is None:
            continue

        filename = os.path.basename(file_path)
        try:
            epoch_idx = int(os.path.splitext(filename)[0].split('_')[-1])
        except ValueError:
            continue

        session_id = 0
        data_dict[session_id][sub_id][music_id][epoch_idx] = eeg_data
        label_dict[session_id][sub_id][music_id][epoch_idx] = label_value
        all_subject_ids.add(sub_id)
        all_trail_ids.add(music_id)

    # 获取数据形状
    sample_shape = None
    for session_id in data_dict:
        for sub_id in data_dict[session_id]:
            for trail_id in data_dict[session_id][sub_id]:
                for epoch_idx in data_dict[session_id][sub_id][trail_id]:
                    sample_shape = data_dict[session_id][sub_id][trail_id][epoch_idx].shape
                    break
                if sample_shape:
                    break
            if sample_shape:
                break
        if sample_shape:
            break

    if sample_shape is None:
        raise ValueError("没有找到有效的数据样本")

    num_channels, num_points = sample_shape

    print("✅ EEG数据加载完成！")

    return {
        'data_dict': data_dict,
        'label_dict': label_dict,
        'num_classes': 2,
        'subject_ids': sorted(all_subject_ids),
        'trail_ids': sorted(all_trail_ids),
        'num_channels': num_channels,
        'num_points': num_points
    }


def merge_to_part(data_dict, label_dict, strategy, cross_trail=False, sessions=None, train_ratio=0.8):
    """
    根据训练策略，将数据划分为不同的训练和测试组

    参数:
        data_dict: 字典格式的数据，结构为 {session: {subject: {trail: {epoch: data}}}}
        label_dict: 字典格式的标签，结构为 {session: {subject: {trail: {epoch: label}}}}
        strategy: 训练策略 ["single-subject", "within-subject", "cross-subject", "cross-trail"]
        cross_trail: 是否避免trail重叠 (True/False)，默认为False
        sessions: 使用的会话列表（从1开始），默认为所有会话
        train_ratio: 训练集比例（当cross_trail=False时使用），默认为0.8

    返回:
        (result_data, result_label) 元组，其中:
        - result_data: 列表格式的训练测试组，每个元素是包含'train'和'test'的字典
        - result_label: 列表格式的标签组，每个元素是包含'train'和'test'的字典
    """
    # 验证会话设置
    if sessions:
        session_ids = [s - 1 for s in sessions]  # 转换为0-based索引
        for s in session_ids:
            if s not in data_dict:
                raise ValueError(f"会话 {s + 1} 不存在于数据集中")
    else:
        session_ids = list(data_dict.keys())

    # 获取所有被试ID和音乐ID
    all_subject_ids = set()
    all_trail_ids = set()

    for session_id in session_ids:
        for subject_id in data_dict.get(session_id, {}):
            all_subject_ids.add(subject_id)
            for trail_id in data_dict[session_id][subject_id]:
                all_trail_ids.add(trail_id)

    subject_ids = sorted(all_subject_ids)
    trail_ids = sorted(all_trail_ids)

    # 处理不同训练策略
    if strategy == "single-subject":
        # Single-subject: 每个被试一个组
        result_data = []
        result_label = []

        for subject_id in subject_ids:
            subject_data = []
            subject_labels = []
            trail_mapping = defaultdict(list)  # trail_id到epoch索引的映射

            # 收集该被试的所有数据
            for session_id in session_ids:
                if subject_id in data_dict.get(session_id, {}):
                    for trail_id in data_dict[session_id][subject_id]:
                        for epoch_id in sorted(data_dict[session_id][subject_id][trail_id].keys()):
                            data = data_dict[session_id][subject_id][trail_id][epoch_id]
                            label_val = label_dict[session_id][subject_id][trail_id][epoch_id]

                            subject_data.append(data)
                            subject_labels.append(label_val)
                            trail_mapping[trail_id].append(len(subject_data) - 1)

            if cross_trail:
                # 按trail划分：避免同一个trail出现在训练和测试
                all_trails = list(trail_mapping.keys())
                random.shuffle(all_trails)
                split_idx = int(len(all_trails) * train_ratio)
                train_trails = set(all_trails[:split_idx])
                test_trails = set(all_trails[split_idx:])

                train_data = [subject_data[i] for i in range(len(subject_data))
                              if next(t for t, idxs in trail_mapping.items() if i in idxs) in train_trails]
                train_label = [subject_labels[i] for i in range(len(subject_data))
                               if next(t for t, idxs in trail_mapping.items() if i in idxs) in train_trails]
                test_data = [subject_data[i] for i in range(len(subject_data))
                             if next(t for t, idxs in trail_mapping.items() if i in idxs) in test_trails]
                test_label = [subject_labels[i] for i in range(len(subject_data))
                              if next(t for t, idxs in trail_mapping.items() if i in idxs) in test_trails]
            else:
                # 随机划分
                indices = list(range(len(subject_data)))
                random.shuffle(indices)
                split_idx = int(len(indices) * train_ratio)

                train_data = [subject_data[i] for i in indices[:split_idx]]
                train_label = [subject_labels[i] for i in indices[:split_idx]]
                test_data = [subject_data[i] for i in indices[split_idx:]]
                test_label = [subject_labels[i] for i in indices[split_idx:]]

            result_data.append({'train': train_data, 'test': test_data})
            result_label.append({'train': train_label, 'test': test_label})

        return result_data, result_label

    elif strategy == "within-subject":
        # Within-subject: 所有被试合并为一个组
        all_data = []
        all_labels = []
        trail_mapping = defaultdict(list)  # trail_id到epoch索引的映射

        # 收集所有数据
        for session_id in session_ids:
            for subject_id in subject_ids:
                if subject_id in data_dict.get(session_id, {}):
                    for trail_id in data_dict[session_id][subject_id]:
                        for epoch_id in sorted(data_dict[session_id][subject_id][trail_id].keys()):
                            data = data_dict[session_id][subject_id][trail_id][epoch_id]
                            label_val = label_dict[session_id][subject_id][trail_id][epoch_id]

                            all_data.append(data)
                            all_labels.append(label_val)
                            trail_mapping[trail_id].append(len(all_data) - 1)

        if cross_trail:
            # 按trail划分：避免同一个trail出现在训练和测试
            all_trails = list(trail_mapping.keys())
            random.shuffle(all_trails)
            split_idx = int(len(all_trails) * train_ratio)
            train_trails = set(all_trails[:split_idx])
            test_trails = set(all_trails[split_idx:])

            train_data = [all_data[i] for i in range(len(all_data))
                          if next(t for t, idxs in trail_mapping.items() if i in idxs) in train_trails]
            train_label = [all_labels[i] for i in range(len(all_data))
                           if next(t for t, idxs in trail_mapping.items() if i in idxs) in train_trails]
            test_data = [all_data[i] for i in range(len(all_data))
                         if next(t for t, idxs in trail_mapping.items() if i in idxs) in test_trails]
            test_label = [all_labels[i] for i in range(len(all_data))
                          if next(t for t, idxs in trail_mapping.items() if i in idxs) in test_trails]
        else:
            # 随机划分
            indices = list(range(len(all_data)))
            random.shuffle(indices)
            split_idx = int(len(indices) * train_ratio)

            train_data = [all_data[i] for i in indices[:split_idx]]
            train_label = [all_labels[i] for i in indices[:split_idx]]
            test_data = [all_data[i] for i in indices[split_idx:]]
            test_label = [all_labels[i] for i in indices[split_idx:]]

        return [{'train': train_data, 'test': test_data}], [{'train': train_label, 'test': test_label}]

    elif strategy == "cross-subject":
        # Cross-subject: 留一被试法
        result_data = []
        result_label = []

        for test_subject_id in subject_ids:
            train_data = []
            train_label = []
            test_data = []
            test_label = []

            for session_id in session_ids:
                for subject_id in subject_ids:
                    if subject_id in data_dict.get(session_id, {}):
                        for trail_id in data_dict[session_id][subject_id]:
                            for epoch_id in sorted(data_dict[session_id][subject_id][trail_id].keys()):
                                data = data_dict[session_id][subject_id][trail_id][epoch_id]
                                label_val = label_dict[session_id][subject_id][trail_id][epoch_id]

                                if subject_id == test_subject_id:
                                    test_data.append(data)
                                    test_label.append(label_val)
                                else:
                                    train_data.append(data)
                                    train_label.append(label_val)

            result_data.append({'train': train_data, 'test': test_data})
            result_label.append({'train': train_label, 'test': test_label})

        return result_data, result_label

    elif strategy == "cross-trail":
        # Cross-trail: 留一音乐法
        result_data = []
        result_label = []

        for test_trail_id in trail_ids:
            train_data = []
            train_label = []
            test_data = []
            test_label = []

            for session_id in session_ids:
                for subject_id in data_dict.get(session_id, {}):
                    for trail_id in data_dict[session_id][subject_id]:
                        for epoch_id in sorted(data_dict[session_id][subject_id][trail_id].keys()):
                            data = data_dict[session_id][subject_id][trail_id][epoch_id]
                            label_val = label_dict[session_id][subject_id][trail_id][epoch_id]

                            if trail_id == test_trail_id:
                                test_data.append(data)
                                test_label.append(label_val)
                            else:
                                train_data.append(data)
                                train_label.append(label_val)

            result_data.append({'train': train_data, 'test': test_data})
            result_label.append({'train': train_label, 'test': test_label})

        return result_data, result_label


def get_split_index(data, label, strategy, k_fold=1, val_size=0.2, random_state=None):
    """
    简化的划分函数，只处理单次划分（用于非K-fold情况）
    """
    tts = {'train': [], 'val': [], 'test': []}

    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    # 获取测试集索引（固定不变）
    test_indexes = list(range(len(data['test'])))

    # 训练集索引
    train_indexes = list(range(len(data['train'])))
    random.shuffle(train_indexes)

    # 划分验证集
    val_num = int(val_size * len(train_indexes))
    val_indexes = train_indexes[:val_num]
    train_indexes = train_indexes[val_num:]

    tts['train'].append(train_indexes)
    tts['val'].append(val_indexes)
    tts['test'].append(test_indexes)

    return tts


def create_kfold_splits(data, label, k_fold=5, val_size=0.2, random_state=None):
    """
    K-fold 划分：只在训练集上划分 train/val，每折使用同一个 test
    """
    tts = {'train': [], 'val': [], 'test': []}

    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    train_data = data['train']
    train_label = label['train']
    test_data = data['test']   # 固定测试集
    test_label = label['test']

    indexes = list(range(len(train_data)))
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=random_state)

    for train_index, val_index in kf.split(indexes):
        tts['train'].append(train_index.tolist())
        tts['val'].append(val_index.tolist())
        tts['test'].append(list(range(len(test_data))))  # 固定不变

    return tts



def index_to_data(data, label, train_indexes, test_indexes, val_indexes, keep_dim=False, is_kfold=False):
    """
    修正版：在KFold模式下，只使用训练集划分train/val，测试集固定
    """
    if is_kfold:
        # ✅ K-fold 模式：train/val 从 train 里取，test 从 test 里取
        train_data_source = data['train']
        train_label_source = label['train']
        test_data_source = data['test']
        test_label_source = label['test']

        train_data = [train_data_source[i] for i in train_indexes]
        train_label = [train_label_source[i] for i in train_indexes]
        val_data = [train_data_source[i] for i in val_indexes]
        val_label = [train_label_source[i] for i in val_indexes]
        test_data = [test_data_source[i] for i in test_indexes]
        test_label = [test_label_source[i] for i in test_indexes]

    else:
        # 非K-fold模式
        train_data_source = data['train']
        train_label_source = label['train']
        test_data_source = data['test']
        test_label_source = label['test']

        train_data = [train_data_source[i] for i in train_indexes]
        train_label = [train_label_source[i] for i in train_indexes]
        val_data = [train_data_source[i] for i in val_indexes]
        val_label = [train_label_source[i] for i in val_indexes]
        test_data = [test_data_source[i] for i in test_indexes]
        test_label = [test_label_source[i] for i in test_indexes]

    if not keep_dim:
        train_data = np.array(train_data)
        train_label = np.array(train_label)
        val_data = np.array(val_data)
        val_label = np.array(val_label)
        test_data = np.array(test_data)
        test_label = np.array(test_label)

    return train_data, train_label, val_data, val_label, test_data, test_label



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


from feel.models.registry import Model
from feel.utils.results import result_log
from feel.training.supervised import train


def main():
    parser = training_parser("Train EEGNet on JSON music emotion epochs.", "runs/eeg_json", data_dirs=("data_dir",))
    args = parser.parse_args()
    device = resolve_device(args.device)

    output_dir = args.output_dir

    channels = 32
    feature_dim = 512
    num_classes = 2

    strategy = "within-subject"
    cross_trail = False
    seed = 666
    lr = 0.001
    batch_size = 32
    epochs = 2
    k_fold = 5  # 新增K-fold参数

    all_metrics = []  # 存储所有结果

    dataset = load_dataset(args.data_dir, 'arousal', num_workers=20)
    data, label = merge_to_part(data_dict=dataset['data_dict'], label_dict=dataset['label_dict'], strategy=strategy,
                                cross_trail=cross_trail)

    for group_idx in range(len(data)):
        group_data = data[group_idx]
        group_label = label[group_idx]

        print(f"\n=== 处理第 {group_idx + 1}/{len(data)} 组 ===")

        # 根据策略决定是否使用K-fold
        if strategy in ["single-subject", "within-subject"] and k_fold > 1:
            print(f"使用 {k_fold}-折交叉验证")
            tts = create_kfold_splits(group_data, group_label,
                                      k_fold=k_fold,
                                      val_size=0.2,
                                      random_state=seed)
            use_kfold = True
        else:
            print("使用单次划分")
            tts = get_split_index(group_data, group_label,
                                  strategy=strategy,
                                  k_fold=1,
                                  val_size=0.2,
                                  random_state=seed)
            use_kfold = False

        num_folds = len(tts['train'])
        print(f"本组共有 {num_folds} 个训练轮次")

        group_metrics = []  # 当前组的所有折结果
        group_best_model = None  # 存储整个组的最佳模型
        group_best_val_metric = 0.0  # 存储整个组的最佳验证指标
        group_best_fold_idx = -1  # 存储最佳模型所在的折索引

        for fold_idx in range(num_folds):
            # 为每折设置不同的随机种子
            setup_seed(seed + fold_idx * 100 + group_idx * 1000)

            train_indexes = tts['train'][fold_idx]
            test_indexes = tts['test'][fold_idx]
            val_indexes = tts['val'][fold_idx]

            # 分割数据
            train_data, train_label, val_data, val_label, test_data, test_label = \
                index_to_data(group_data, group_label, train_indexes, test_indexes, val_indexes,
                              keep_dim=True, is_kfold=use_kfold)

            if len(val_data) == 0:
                val_data = test_data
                val_label = test_label
                print("验证集为空，使用测试集作为验证集")

            model = Model['EEGNet'](channels, feature_dim, num_classes)
            dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_data), torch.LongTensor(train_label))
            dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.LongTensor(val_label))
            dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.LongTensor(test_label))
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-8)
            criterion = nn.CrossEntropyLoss()

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            print(f"\n--- 第 {fold_idx + 1}/{num_folds} 折训练 ---")
            print(f"训练样本数: {len(train_data)}, 验证样本数: {len(val_data)}, 测试样本数: {len(test_data)}")

            round_result = train(
                model=model,
                dataset_train=dataset_train,
                dataset_val=dataset_val,
                dataset_test=dataset_test,
                device=device,
                output_dir=output_dir,
                metrics=['acc', 'f1', 'auc'],  # 增加指标
                metric_choose='acc',
                optimizer=optimizer,
                batch_size=batch_size,
                epochs=epochs,
                criterion=criterion
            )

            # 记录本折结果
            round_metric = {
                'acc': round_result.get('acc', 0),
                'f1': round_result.get('f1', 0),
                'auc': round_result.get('auc', 0),
                'loss': round_result.get('loss', 0),
                'val_acc': round_result.get('best_val_metric', 0),  # 添加验证集最佳准确率
                'epoch': round_result.get('best_epoch', 0),  # 添加最佳epoch
                'group': group_idx,
                'fold': fold_idx
            }

            # 记录本折结果
            group_metrics.append(round_metric)
            all_metrics.append(round_metric)

            print(f"第 {fold_idx + 1} 折完成，验证集最佳准确率: {round_metric['val_acc']:.4f}, "
                  f"测试集准确率: {round_metric['acc']:.4f}, "
                  f"F1: {round_metric['f1']:.4f}, AUC: {round_metric['auc']:.4f}")

            # 更新整个组的最佳模型
            if round_result.get('best_val_metric', 0) > group_best_val_metric:
                group_best_val_metric = round_result.get('best_val_metric', 0)
                group_best_model = round_result.get('best_model')
                group_best_fold_idx = fold_idx
                print(f"✅ 更新整个组的最佳模型: 折 {fold_idx + 1}, 验证集准确率: {group_best_val_metric:.4f}")

        # 保存整个组的最佳模型
        if group_best_model is not None:
            # 创建组特定的保存目录
            group_save_dir = os.path.join(output_dir, f"group_{group_idx}")
            os.makedirs(group_save_dir, exist_ok=True)

            # 保存模型状态
            model_save_path = os.path.join(group_save_dir, f"best_model_group_{group_idx}.pth")
            torch.save({
                'model_state_dict': group_best_model.state_dict(),
                'val_accuracy': group_best_val_metric,
                'fold_index': group_best_fold_idx,
                'group_index': group_idx
            }, model_save_path)
            print(f"✅ 保存整个组 {group_idx} 的最佳模型: {model_save_path}")

        # 计算当前组的平均性能
        if len(group_metrics) > 0:
            avg_acc = np.mean([m['acc'] for m in group_metrics])
            std_acc = np.std([m['acc'] for m in group_metrics]) if len(group_metrics) > 1 else 0
            avg_f1 = np.mean([m['f1'] for m in group_metrics])
            std_f1 = np.std([m['f1'] for m in group_metrics]) if len(group_metrics) > 1 else 0
            avg_auc = np.mean([m['auc'] for m in group_metrics])
            std_auc = np.std([m['auc'] for m in group_metrics]) if len(group_metrics) > 1 else 0

            print(f"\n第 {group_idx + 1} 组平均性能:")
            print(f"准确率: {avg_acc:.4f} ± {std_acc:.4f}")
            print(f"F1分数: {avg_f1:.4f} ± {std_f1:.4f}")
            print(f"AUC: {avg_auc:.4f} ± {std_auc:.4f}")
            print(f"最佳模型来自折 {group_best_fold_idx + 1}, 验证集准确率: {group_best_val_metric:.4f}")

            # 保存当前组的结果
            result_log(output_dir, strategy, cross_trail, group_idx, group_metrics,
                       lr, batch_size, epochs, seed, k_fold,
                       train_history=[round_result.get('train_history', {})],
                       best_model=group_best_model,  # 使用整个组的最佳模型
                       test_data=round_result.get('test_data'),
                       test_labels=round_result.get('test_labels'),
                       train_data=round_result.get('train_data'),
                       train_labels=round_result.get('train_labels'))

    # 最终统计
    if len(all_metrics) > 0:
        final_avg_acc = np.mean([m['acc'] for m in all_metrics])
        final_std_acc = np.std([m['acc'] for m in all_metrics]) if len(all_metrics) > 1 else 0
        print(f"\n🎯 最终平均准确率: {final_avg_acc:.4f} ± {final_std_acc:.4f}")
        print(f"总训练轮次: {len(all_metrics)}")


if __name__ == "__main__":
    main()
