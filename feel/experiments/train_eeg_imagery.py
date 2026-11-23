from feel.train.cli import resolve_device, training_parser

import os
import numpy as np
import torch
import random
from sklearn.model_selection import KFold
from torch import optim, nn
from tqdm import tqdm
# 假设您有以下导入
from feel.models.registry import Model
from feel.utils.results import result_log
from feel.training.supervised import train

def load_imagery_data(data_dir, subject_id):
    """
    加载四个想象状态的数据

    参数:
        data_dir: 包含.npy文件的目录
        subject_id: 被试ID

    返回:
        dict: 包含四个想象状态数据的字典
    """
    imagery_data = {}

    # 加载四个想象状态的数据
    states = ['forward', 'backward', 'left', 'right']
    for state in states:
        file_path = os.path.join(data_dir, f'{subject_id}_imagery_{state}_data.npy')
        if os.path.exists(file_path):
            data = np.load(file_path)
            imagery_data[state] = data
            print(f"加载 {state} 状态数据: {data.shape}")
        else:
            print(f"警告: 文件 {file_path} 不存在")
            imagery_data[state] = np.array([])

    return imagery_data


def create_dataset(imagery_data):
    """
    从四个想象状态数据创建数据集

    参数:
        imagery_data: 包含四个想象状态数据的字典

    返回:
        data: 所有样本的数据数组 (样本数, 通道数, 时间点数)
        labels: 对应的标签数组 (样本数,)
        num_classes: 类别数量
        num_channels: 通道数量
        num_points: 时间点数
    """
    all_data = []
    all_labels = []

    # 定义状态到标签的映射
    state_to_label = {
        'forward': 0,
        'backward': 1,
        'left': 2,
        'right': 3
    }

    # 收集所有状态的数据
    for state, data in imagery_data.items():
        if data.size > 0:  # 确保有数据
            num_samples = data.shape[0]
            labels = np.full(num_samples, state_to_label[state])
            all_data.append(data)
            all_labels.append(labels)

    # 合并所有数据
    if all_data:
        data_array = np.concatenate(all_data, axis=0)
        label_array = np.concatenate(all_labels, axis=0)

        # 获取数据形状信息
        num_samples, num_channels, num_points = data_array.shape
        num_classes = len(state_to_label)

        return data_array, label_array, num_classes, num_channels, num_points
    else:
        raise ValueError("没有找到有效的想象状态数据")


def merge_to_part(data, labels, strategy="single-subject", cross_trail=False):
    """
    根据训练策略，将数据划分为不同的训练和测试组

    参数:
        data: 所有样本的数据数组 (样本数, 通道数, 时间点数)
        labels: 对应的标签数组 (样本数,)
        strategy: 训练策略 ["single-subject", "within-subject", "cross-subject", "cross-trail"]
        cross_trail: 是否避免trail重叠 (True/False)，默认为False

    返回:
        (result_data, result_label) 元组，其中:
        - result_data: 列表格式的训练测试组，每个元素是包含'train'和'test'的字典
        - result_label: 列表格式的标签组，每个元素是包含'train'和'test'的字典
    """
    # 在这个简化版本中，我们只处理单个被试的数据
    # 对于多被试情况，需要扩展此函数

    # 处理不同训练策略
    if strategy == "single-subject":
        # Single-subject: 每个被试一个组
        # 在这个简化版本中，我们只有一个被试
        result_data = []
        result_label = []

        # 随机划分训练集和测试集
        indices = list(range(len(data)))
        random.shuffle(indices)
        split_idx = int(len(indices) * 0.8)  # 80%训练，20%测试

        train_data = [data[i] for i in indices[:split_idx]]
        train_label = [labels[i] for i in indices[:split_idx]]
        test_data = [data[i] for i in indices[split_idx:]]
        test_label = [labels[i] for i in indices[split_idx:]]

        result_data.append({'train': train_data, 'test': test_data})
        result_label.append({'train': train_label, 'test': test_label})

        return result_data, result_label

    elif strategy == "within-subject":
        # Within-subject: 所有被试合并为一个组
        # 在这个简化版本中，我们只有一个被试
        return [{'train': data, 'test': data}], [{'train': labels, 'test': labels}]

    elif strategy == "cross-subject":
        # Cross-subject: 留一被试法
        # 在这个简化版本中，我们只有一个被试
        result_data = [{'train': data, 'test': data}]
        result_label = [{'train': labels, 'test': labels}]
        return result_data, result_label

    elif strategy == "cross-trail":
        # Cross-trail: 留一音乐法
        # 在这个简化版本中，我们不使用音乐ID
        result_data = [{'train': data, 'test': data}]
        result_label = [{'train': labels, 'test': labels}]
        return result_data, result_label


def get_split_index(data, label, strategy="single-subject", k_fold=5, val_size=0.2, random_state=None):
    """
    根据训练策略获取划分索引

    参数:
        data: 数据字典，格式为 {'train': train_data, 'test': test_data}
        label: 标签字典，格式为 {'train': train_labels, 'test': test_labels}
        strategy: 训练策略 ["single-subject", "within-subject", "cross-subject", "cross-trail"]
        k_fold: K折数量，默认为5
        val_size: 验证集比例，默认为0.2
        random_state: 随机种子，默认为None

    返回:
        tts: 包含 'train', 'val', 'test' 键的字典，每个键对应的值是一个索引列表
    """
    tts = {'train': [], 'val': [], 'test': []}

    # 设置随机种子
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    # 对于single-subject和within-subject策略，可以使用K-fold
    if strategy in ["single-subject", "within-subject"] and k_fold > 1:
        # 合并训练集和测试集（因为K-fold需要整个数据集）
        all_data = data['train'] + data['test']
        all_labels = label['train'] + label['test']
        n_train = len(data['train'])

        # 创建索引
        indexes = list(range(len(all_data)))
        random.shuffle(indexes)

        # 使用KFold进行K折交叉验证
        kf = KFold(n_splits=k_fold, shuffle=True, random_state=random_state)

        for train_val_index, test_index in kf.split(indexes):
            # 在训练验证集中再划分验证集
            train_val_index = train_val_index.tolist()
            val_num = int(val_size * len(train_val_index))
            val_index = train_val_index[:val_num]
            train_index = train_val_index[val_num:]

            tts['train'].append(train_index)
            tts['val'].append(val_index)
            tts['test'].append(test_index)

    # 对于其他策略或当k_fold=1时
    else:
        train_data = data['train']
        train_label = label['train']
        test_data = data['test']
        test_label = label['test']

        # 创建训练集索引
        train_indexes = list(range(len(train_data)))
        random.shuffle(train_indexes)

        # 划分验证集
        val_num = int(val_size * len(train_indexes))
        val_indexes = train_indexes[:val_num]
        train_indexes = train_indexes[val_num:]

        # 创建测试集索引
        test_indexes = list(range(len(test_data)))

        tts['train'] = [train_indexes]
        tts['val'] = [val_indexes]
        tts['test'] = [test_indexes]

    return tts


def index_to_data(data, label, train_indexes, test_indexes, val_indexes, keep_dim=False):
    """
    根据索引提取数据

    返回:
        train_data, train_label, val_data, val_label, test_data, test_label
    """
    # 检查是否是字典结构（cross-subject 或 cross-music 模式）
    is_dict_mode = isinstance(data, dict) and 'train' in data

    if is_dict_mode:
        # 提取训练集和测试集
        train_data_source = data['train']
        train_label_source = label['train']
        test_data_source = data['test']
        test_label_source = label['test']

        # 提取训练集
        train_data = [train_data_source[i] for i in train_indexes]
        train_label = [train_label_source[i] for i in train_indexes]

        # 提取验证集（从训练集中划分）
        val_data = [train_data_source[i] for i in val_indexes]
        val_label = [train_label_source[i] for i in val_indexes]

        # 提取测试集
        test_data = [test_data_source[i] for i in test_indexes]
        test_label = [test_label_source[i] for i in test_indexes]
    else:
        # with-subject 模式
        # 提取训练集
        train_data = [data[i] for i in train_indexes]
        train_label = [label[i] for i in train_indexes]

        # 提取验证集
        val_data = [data[i] for i in val_indexes]
        val_label = [label[i] for i in val_indexes]

        # 提取测试集
        test_data = [data[i] for i in test_indexes]
        test_label = [label[i] for i in test_indexes]

    # 处理 keep_dim 参数
    if not keep_dim:
        # 转换为 NumPy 数组
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


def main():
    parser = training_parser("Train EEGNet on directional motor imagery.", "runs/eeg_imagery", data_dirs=("data_dir",))
    parser.add_argument("--subject-id", required=True, help="Subject ID used in the NPY filenames.")
    args = parser.parse_args()
    device = resolve_device(args.device)

    # 设置参数
    DATA_DIR = args.data_dir
    SUBJECT_ID = args.subject_id
    OUTPUT_DIR = args.output_dir

    # 训练参数
    strategy = "single-subject"
    cross_trail = False
    seed = 666
    lr = 0.001
    batch_size = 32
    epochs = 30
    k_fold = 1  # 使用5折交叉验证

    channels = 32
    feature_dim = 2000
    num_classes = 4

    best_metrics = []

    # 加载想象状态数据
    print(f"加载被试 {SUBJECT_ID} 的想象状态数据...")
    imagery_data = load_imagery_data(DATA_DIR, SUBJECT_ID)

    # 创建数据集
    print("创建数据集...")
    data, labels, num_classes, num_channels, num_points = create_dataset(imagery_data)

    print(f"数据集信息:")
    print(f"  样本数: {len(data)}")
    print(f"  通道数: {num_channels}")
    print(f"  时间点数: {num_points}")
    print(f"  类别数: {num_classes}")

    # 合并数据为部分
    data_part, label_part = merge_to_part(data, labels, strategy, cross_trail)

    # 设置随机种子
    setup_seed(seed)

    # 统一处理所有实验模式
    for group_idx in range(len(data_part)):
        group_data = data_part[group_idx]
        group_label = label_part[group_idx]

        # 获取划分索引
        tts = get_split_index(group_data, group_label, strategy, k_fold=k_fold, val_size=0.2, random_state=seed)

        # 对于with-subject模式，会有k_fold折划分
        num_folds = len(tts['train'])
        print(f"将进行 {num_folds} 折训练")

        for ridx in range(num_folds):
            setup_seed(seed)

            train_indexes = tts['train'][ridx]
            test_indexes = tts['test'][ridx]
            val_indexes = tts['val'][ridx]

            # 根据索引提取数据
            train_data, train_label, val_data, val_label, test_data, test_label = \
                index_to_data(group_data, group_label, train_indexes, test_indexes, val_indexes, keep_dim=True)

            if len(val_data) == 0:
                val_data = test_data
                val_label = test_label

            # 打印数据形状
            print(f"训练数据形状: {np.array(train_data).shape}, 训练标签形状: {np.array(train_label).shape}")
            print(f"验证数据形状: {np.array(val_data).shape}, 验证标签形状: {np.array(val_label).shape}")
            print(f"测试数据形状: {np.array(test_data).shape}, 测试标签形状: {np.array(test_label).shape}")


            model = Model['EEGNet'](num_channels, num_points, num_classes)

            # 创建数据集
            dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_data), torch.LongTensor(train_label))
            dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.LongTensor(val_label))
            dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.LongTensor(test_label))

            # 创建优化器和损失函数
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-8)
            criterion = nn.CrossEntropyLoss()

            # 确保输出目录存在
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)

            print(f"开始第 {ridx + 1}/{num_folds} 折训练")
            print(f"训练样本数: {len(train_data)}, 验证样本数: {len(val_data)}, 测试样本数: {len(test_data)}")

            # 训练模型
            round_metric = train(
                model=model,
                dataset_train=dataset_train,
                dataset_val=dataset_val,
                dataset_test=dataset_test,
                device=device,
                output_dir=OUTPUT_DIR,
                metrics=['acc', 'macro-f1'],
                metric_choose='acc',
                optimizer=optimizer,
                batch_size=batch_size,
                epochs=epochs,
                criterion=criterion
            )

            best_metrics.append(round_metric)

            # 记录结果
            # result_log(OUTPUT_DIR, strategy, cross_trail, group_idx, best_metrics,
            #            lr, batch_size, epochs, seed, num_folds)


if __name__ == "__main__":
    main()
