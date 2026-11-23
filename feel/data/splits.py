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
    正确的K-fold划分函数：每次折都有不同的测试集
    """
    tts = {'train': [], 'val': [], 'test': []}

    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    # 合并所有数据（因为K-fold需要在完整数据集上进行）
    all_data = data['train'] + data['test']
    all_labels = label['train'] + label['test']

    # 创建索引
    indexes = list(range(len(all_data)))
    random.shuffle(indexes)

    # 使用KFold进行真正的K折交叉验证
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=random_state)

    for train_val_index, test_index in kf.split(indexes):
        # train_val_index: 训练+验证集的索引
        # test_index: 测试集的索引（每次折都不同）

        # 在训练验证集中再划分验证集
        train_val_index_list = train_val_index.tolist()
        random.shuffle(train_val_index_list)

        val_num = int(val_size * len(train_val_index_list))
        val_index = train_val_index_list[:val_num]
        train_index = train_val_index_list[val_num:]

        tts['train'].append(train_index)
        tts['val'].append(val_index)
        tts['test'].append(test_index.tolist())  # 每次折都有不同的测试集

    return tts


def create_kfold_splits(data, label, k_fold=5, val_size=0.2, random_state=None):
    """
    正确的K-fold划分函数：每次折都有不同的测试集
    """
    tts = {'train': [], 'val': [], 'test': []}

    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    # 合并所有数据（因为K-fold需要在完整数据集上进行）
    all_data = data['train'] + data['test']
    all_labels = label['train'] + label['test']

    # 创建索引
    indexes = list(range(len(all_data)))
    random.shuffle(indexes)

    # 使用KFold进行真正的K折交叉验证
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=random_state)

    for train_val_index, test_index in kf.split(indexes):
        # train_val_index: 训练+验证集的索引
        # test_index: 测试集的索引（每次折都不同）

        # 在训练验证集中再划分验证集
        train_val_index_list = train_val_index.tolist()
        random.shuffle(train_val_index_list)

        val_num = int(val_size * len(train_val_index_list))
        val_index = train_val_index_list[:val_num]
        train_index = train_val_index_list[val_num:]

        tts['train'].append(train_index)
        tts['val'].append(val_index)
        tts['test'].append(test_index.tolist())  # 每次折都有不同的测试集

    return tts


def index_to_data(data, label, train_indexes, test_indexes, val_indexes, keep_dim=False, is_kfold=False):
    """
    修正的索引到数据转换函数，支持真正的K-fold
    """
    if is_kfold:
        # K-fold模式：需要合并所有数据，因为索引对应完整数据集
        all_data = data['train'] + data['test']
        all_labels = label['train'] + label['test']

        # 从完整数据集中提取
        train_data = [all_data[i] for i in train_indexes]
        train_label = [all_labels[i] for i in train_indexes]
        val_data = [all_data[i] for i in val_indexes]
        val_label = [all_labels[i] for i in val_indexes]
        test_data = [all_data[i] for i in test_indexes]
        test_label = [all_labels[i] for i in test_indexes]

    else:
        # 非K-fold模式：保持原有逻辑
        is_dict_mode = isinstance(data, dict) and 'train' in data
        if is_dict_mode:
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
            train_data = [data[i] for i in train_indexes]
            train_label = [label[i] for i in train_indexes]
            val_data = [data[i] for i in val_indexes]
            val_label = [label[i] for i in val_indexes]
            test_data = [data[i] for i in test_indexes]
            test_label = [label[i] for i in test_indexes]

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


import random
from collections import defaultdict
import numpy as np


def merge_paired_data_to_part(
        paired_samples,
        strategy,
        cross_trail=False,
        train_ratio=0.8,
        random_state=None
):
    """
    对 EEG+fNIRS 配对数据进行划分，支持多种训练策略。

    Args:
        paired_samples: List[Dict], 每个元素含 'eeg', 'fnirs', 'label', 'sub_id', 'music_id'
        strategy: str, ["single-subject", "within-subject", "cross-subject", "cross-trail"]
        cross_trail: bool, 是否按 trail 分割（避免同一 trail 出现在 train/test）
        train_ratio: float, 训练集比例（当 cross_trail=False 时使用）
        random_state: int, 随机种子

    Returns:
        result_groups: List[Dict], 每个元素为 {'train': [...], 'test': [...]},
                       其中每个 item 是原始 paired_samples 中的一个 dict（含 eeg/fnirs/label/sub_id/music_id）
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    # 提取唯一 ID
    all_sub_ids = sorted(set(sample['sub_id'] for sample in paired_samples))
    all_music_ids = sorted(set(sample['music_id'] for sample in paired_samples))

    # 构建索引映射（用于 cross_trail）
    sub_to_indices = defaultdict(list)
    music_to_indices = defaultdict(list)
    for idx, sample in enumerate(paired_samples):
        sub_to_indices[sample['sub_id']].append(idx)
        music_to_indices[sample['music_id']].append(idx)

    result_groups = []

    if strategy == "single-subject":
        for sub_id in all_sub_ids:
            indices = sub_to_indices[sub_id]
            if cross_trail:
                # 按 music_id 分组再划分
                music_groups = defaultdict(list)
                for idx in indices:
                    music_groups[paired_samples[idx]['music_id']].append(idx)
                music_list = list(music_groups.values())
                random.shuffle(music_list)
                split_idx = int(len(music_list) * train_ratio)
                train_musics = [idx for group in music_list[:split_idx] for idx in group]
                test_musics = [idx for group in music_list[split_idx:] for idx in group]
                train_indices = train_musics
                test_indices = test_musics
            else:
                random.shuffle(indices)
                split_idx = int(len(indices) * train_ratio)
                train_indices = indices[:split_idx]
                test_indices = indices[split_idx:]

            train_samples = [paired_samples[i] for i in train_indices]
            test_samples = [paired_samples[i] for i in test_indices]
            result_groups.append({'train': train_samples, 'test': test_samples})

    elif strategy == "within-subject":
        all_indices = list(range(len(paired_samples)))
        if cross_trail:
            # 按 music_id 分组
            music_groups = list(music_to_indices.values())
            random.shuffle(music_groups)
            split_idx = int(len(music_groups) * train_ratio)
            train_indices = [idx for group in music_groups[:split_idx] for idx in group]
            test_indices = [idx for group in music_groups[split_idx:] for idx in group]
        else:
            random.shuffle(all_indices)
            split_idx = int(len(all_indices) * train_ratio)
            train_indices = all_indices[:split_idx]
            test_indices = all_indices[split_idx:]

        train_samples = [paired_samples[i] for i in train_indices]
        test_samples = [paired_samples[i] for i in test_indices]
        result_groups.append({'train': train_samples, 'test': test_samples})

    elif strategy == "cross-subject":
        for test_sub_id in all_sub_ids:
            train_indices = []
            test_indices = []
            for idx, sample in enumerate(paired_samples):
                if sample['sub_id'] == test_sub_id:
                    test_indices.append(idx)
                else:
                    train_indices.append(idx)
            train_samples = [paired_samples[i] for i in train_indices]
            test_samples = [paired_samples[i] for i in test_indices]
            result_groups.append({'train': train_samples, 'test': test_samples})

    elif strategy == "cross-trail":
        for test_music_id in all_music_ids:
            train_indices = []
            test_indices = []
            for idx, sample in enumerate(paired_samples):
                if sample['music_id'] == test_music_id:
                    test_indices.append(idx)
                else:
                    train_indices.append(idx)
            train_samples = [paired_samples[i] for i in train_indices]
            test_samples = [paired_samples[i] for i in test_indices]
            result_groups.append({'train': train_samples, 'test': test_samples})

    else:
        raise ValueError(f"未知策略: {strategy}")

    print(f"✅ 划分完成！共 {len(result_groups)} 个分组")
    return result_groups


def paired_index_to_data(group, keep_dim=False):
    """
    将一个划分组（{'train': [...], 'val': [...], 'test': [...]}）转换为 NumPy 数组。

    注意：此函数假设 val 已经在外部划分好（或设为空列表）
    """

    def _extract(samples):
        if not samples:
            return np.array([]), np.array([]), np.array([])
        eegs = np.stack([s['eeg'] for s in samples], axis=0)
        fnirs = np.stack([s['fnirs'] for s in samples], axis=0)
        labels = np.array([s['label'] for s in samples])
        return eegs, fnirs, labels

    train_eeg, train_fnirs, train_label = _extract(group.get('train', []))
    val_eeg, val_fnirs, val_label = _extract(group.get('val', []))
    test_eeg, test_fnirs, test_label = _extract(group.get('test', []))

    if not keep_dim:
        # 已经是 (N, C, T)，无需 reshape
        pass

    return {
        'eeg': {
            'train': train_eeg, 'val': val_eeg, 'test': test_eeg
        },
        'fnirs': {
            'train': train_fnirs, 'val': val_fnirs, 'test': test_fnirs
        },
        'label': {
            'train': train_label, 'val': val_label, 'test': test_label
        }
    }
