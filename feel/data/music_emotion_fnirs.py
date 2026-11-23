import os
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from scipy.io import loadmat
import pandas as pd


def load_one_fnirs_and_label(mat_path, label_df, label_dimension='valence'):
    """
    加载单个 fNIRS mat 文件，并提取对应标签
    """
    try:
        mat = loadmat(mat_path)
        data = mat['data']  # shape: (40, n_time)
        time = mat['time'][0]  # shape: (n_time,)

        # 截取最后150个时间点
        if data.shape[1] >= 150:
            data = data[:, -150:]
        else:
            # 如果不足150点则补零
            pad_len = 150 - data.shape[1]
            data = np.pad(data, ((0, 0), (pad_len, 0)), mode='constant')

        # 从路径中提取 sub_id 与 music_id
        path_parts = mat_path.split(os.sep)
        music_id = None
        sub_id = None
        for part in path_parts:
            if part.startswith('song'):
                try:
                    music_id = int(part[4:])
                except ValueError:
                    pass
            if part.startswith('sub-'):
                sub_id = part.replace('sub-', '').split('_')[0]

        if music_id is None or sub_id is None:
            print(f"⚠️ 跳过无法识别ID的文件: {mat_path}")
            return None

        sub_id = int(sub_id)

        # 查找对应标签
        label_row = label_df[(label_df['sub_id'] == sub_id) & (label_df['music_id'] == music_id)]
        if label_row.empty:
            label_value = np.nan
        else:
            label_value = label_row.iloc[0][label_dimension]

        return mat_path, data, label_value

    except Exception as e:
        print(f"❌ 加载失败 {mat_path}: {e}")
        return None


def binary_classify_label(label_value):
    """
    将连续标签转换为二分类标签
    >5 变为1，<=5 变为0
    """
    if np.isnan(label_value):
        return np.nan

    # 将标签值转换为二分类
    if label_value > 5:
        return 1
    else:
        return 0


def load_fnirs_dataset(fnirs_dir, label_csv_path, label_dimension='valence', num_workers=10):
    """
    并行加载整个 fNIRS 数据集及对应的标签（不缓存label）
    """
    # 载入标签
    label_df = pd.read_excel(label_csv_path)
    print(f"载入标签 {len(label_df)} 条")

    # 找到所有 .mat 文件
    mat_files = []
    for root, _, files in os.walk(fnirs_dir):
        for file in files:
            if file.endswith('.mat') and 'NIRS' in file:
                mat_files.append(os.path.join(root, file))

    print(f"找到 {len(mat_files)} 个 fNIRS 文件")

    # 并行加载
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(load_one_fnirs_and_label, fp, label_df, label_dimension)
                   for fp in mat_files]

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="并行加载 fNIRS 文件", ncols=100):
            res = future.result()
            if res is not None:
                results.append(res)

    # 组织数据结构
    data_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    label_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    all_subject_ids = set()
    all_trail_ids = set()

    for file_path, fnirs_data, label_value in results:
        # ✅ 在这里清除 NaN 值
        if isinstance(fnirs_data, np.ndarray):
            fnirs_data = np.nan_to_num(fnirs_data, nan=0.0)
        else:
            fnirs_data = np.array(fnirs_data, dtype=np.float32)
            fnirs_data = np.nan_to_num(fnirs_data, nan=0.0)

        path_parts = file_path.split(os.sep)
        music_id = None
        sub_id = None
        for part in path_parts:
            if part.startswith('song'):
                try:
                    music_id = int(part[4:])
                except ValueError:
                    continue
            if part.startswith('sub-'):
                sub_id = part.replace('sub-', '').split('_')[0]

        if music_id is None or sub_id is None:
            continue

        label_value = binary_classify_label(label_value)

        session_id = 0
        epoch_idx = 0  # 这里只有一整段
        data_dict[session_id][sub_id][music_id][epoch_idx] = fnirs_data
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
    print(f"✅ fNIRS 数据加载完成！共 {len(all_subject_ids)} 被试，{len(all_trail_ids)} 首歌曲")
    print(f"数据形状: {num_channels} 通道 × {num_points} 时间点")

    return {
        'data_dict': data_dict,
        'label_dict': label_dict,
        'num_classes': label_df[label_dimension].nunique(),
        'subject_ids': sorted(all_subject_ids),
        'trail_ids': sorted(all_trail_ids),
        'num_channels': num_channels,
        'num_points': num_points
    }
