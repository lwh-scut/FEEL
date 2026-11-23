import os
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from tqdm import tqdm
from scipy.io import loadmat
import pandas as pd


# ========== 复用你已有的函数（建议放在 utils.py 中）==========
# （此处省略 load_one_fnirs_and_label, load_one_eeg_and_label, binary_classify_label）
# 但为了完整性，我们假设它们已定义
from feel.data.music_emotion_eeg import load_eeg_dataset
from feel.data.music_emotion_fnirs import load_fnirs_dataset


def load_paired_eeg_fnirs_dataset(
        eeg_dir: str,
        fnirs_dir: str,
        label_csv_path: str,
        label_dimension: str = 'valence',
        num_workers: int = 10,
):
    """
    同时加载配对的 EEG + fNIRS 数据，仅保留两者均存在且标签有效的样本。

    Returns:
        dict with keys:
            - 'paired_samples': List[Dict] 每个元素含 'eeg', 'fnirs', 'label', 'sub_id', 'music_id'
            - 'eeg_shape': (C_eeg, T_eeg)
            - 'fnirs_shape': (C_fnirs, T_fnirs)
            - 'num_samples': int
    """
    print("🔄 正在加载 EEG 数据...")
    eeg_dataset = load_eeg_dataset(
        eeg_dir, label_csv_path, label_dimension, num_workers
    )

    print("🔄 正在加载 fNIRS 数据...")
    fnirs_dataset = load_fnirs_dataset(
        fnirs_dir, label_csv_path, label_dimension, num_workers
    )

    # 提取数据字典
    eeg_data_dict = eeg_dataset['data_dict']
    fnirs_data_dict = fnirs_dataset['data_dict']
    eeg_label_dict = eeg_dataset['label_dict']
    fnirs_label_dict = fnirs_dataset['label_dict']

    paired_samples = []
    session_id = 0  # 你的数据中 session_id 固定为 0

    # 遍历所有可能的 (sub_id, music_id)
    all_sub_ids = set(eeg_data_dict[session_id].keys()) & set(fnirs_data_dict[session_id].keys())
    for sub_id in tqdm(all_sub_ids, desc="🔍 匹配配对样本"):
        eeg_musics = set(eeg_data_dict[session_id][sub_id].keys())
        fnirs_musics = set(fnirs_data_dict[session_id][sub_id].keys())
        common_musics = eeg_musics & fnirs_musics

        for music_id in common_musics:
            # 获取 epoch_idx=0 的数据（你当前只有一个 epoch）
            epoch_idx = 0

            # 检查 EEG 是否存在
            if epoch_idx not in eeg_data_dict[session_id][sub_id][music_id]:
                continue
            if epoch_idx not in eeg_label_dict[session_id][sub_id][music_id]:
                continue

            # 检查 fNIRS 是否存在
            if epoch_idx not in fnirs_data_dict[session_id][sub_id][music_id]:
                continue
            if epoch_idx not in fnirs_label_dict[session_id][sub_id][music_id]:
                continue

            eeg_data = eeg_data_dict[session_id][sub_id][music_id][epoch_idx]
            fnirs_data = fnirs_data_dict[session_id][sub_id][music_id][epoch_idx]
            eeg_label = eeg_label_dict[session_id][sub_id][music_id][epoch_idx]
            fnirs_label = fnirs_label_dict[session_id][sub_id][music_id][epoch_idx]

            # 标签必须一致且非 NaN
            if np.isnan(eeg_label) or np.isnan(fnirs_label):
                continue
            if eeg_label != fnirs_label:
                print(f"⚠️ 标签不一致! sub={sub_id}, music={music_id}: EEG={eeg_label}, fNIRS={fnirs_label}")
                continue

            paired_samples.append({
                'eeg': eeg_data.astype(np.float32),
                'fnirs': fnirs_data.astype(np.float32),
                'label': int(eeg_label),
                'sub_id': sub_id,
                'music_id': music_id
            })

    if not paired_samples:
        raise ValueError("❌ 未找到任何有效的配对样本！请检查路径或标签匹配逻辑。")

    # 获取形状信息
    sample = paired_samples[0]
    eeg_shape = sample['eeg'].shape  # (C_eeg, T_eeg)
    fnirs_shape = sample['fnirs'].shape  # (C_fnirs, T_fnirs)

    print(f"✅ 配对数据加载完成！共 {len(paired_samples)} 个有效样本")
    print(f"   EEG 形状: {eeg_shape[0]} 通道 × {eeg_shape[1]} 时间点")
    print(f"   fNIRS 形状: {fnirs_shape[0]} 通道 × {fnirs_shape[1]} 时间点")

    return {
        'paired_samples': paired_samples,
        'eeg_shape': eeg_shape,
        'fnirs_shape': fnirs_shape,
        'num_samples': len(paired_samples)
    }


# ========== 如果你需要 PyTorch Dataset 类 ==========
from torch.utils.data import Dataset


class PairedEEGfNIRSDataset(Dataset):
    def __init__(self, paired_samples):
        self.samples = paired_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.from_numpy(sample['eeg']),  # (C_eeg, T_eeg)
            torch.from_numpy(sample['fnirs']),  # (C_fnirs, T_fnirs)
            torch.tensor(sample['label'], dtype=torch.long)
        )
