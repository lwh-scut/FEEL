from feel.train.cli import music_emotion_args, resolve_device


def main():
    args = music_emotion_args(
        "Train E_FNet on the music emotion dataset.",
        "runs/eeg_fnirs/e_fnet",
        data_dirs=('eeg_dir', 'fnirs_dir'),
        epochs=20,
        strategies=('cross-subject', 'single-subject', 'cross-trail'),
    )

    import os
    import numpy as np
    import torch
    from torch import optim, nn

    # ====== 新增导入 ======
    from feel.data.splits import merge_paired_data_to_part, paired_index_to_data
    from feel.data.splits import setup_seed
    from feel.data.music_emotion_eeg_fnirs import load_paired_eeg_fnirs_dataset
    from feel.training.multimodal import train_multimodal  # 我们将创建这个新 trainer
    from feel.models.registry import Model
    from feel.utils.results import result_log

    device = resolve_device(args.device)
    base_output_dir = args.output_dir

    strategies = args.strategy
    label_dimensions = args.label_dimension

    # 注意：现在有两个模态的形状
    eeg_channels = 32
    eeg_timepoints = 1500
    fnirs_channels = 40   # 根据你的数据调整
    fnirs_timepoints = 150  # 根据你的数据调整
    num_classes = 2

    seed = 666
    lr = 0.001
    batch_size = 32
    epochs = args.epochs
    k_fold = 1

    eeg_dir = args.eeg_dir
    fnirs_dir = args.fnirs_dir
    label_csv = args.labels

    for strategy in strategies:
        for label_dimension in label_dimensions:
            print(f"\n{'=' * 60}")
            print(f"开始训练: 策略={strategy}, 标签维度={label_dimension}")
            print(f"{'=' * 60}")

            output_dir = os.path.join(base_output_dir, f"{strategy}_{label_dimension}")
            os.makedirs(output_dir, exist_ok=True)

            cross_trail = (strategy == "cross-trail")
            all_metrics = []

            # === 加载配对数据 ===
            paired_dataset = load_paired_eeg_fnirs_dataset(
                eeg_dir=eeg_dir,
                fnirs_dir=fnirs_dir,
                label_csv_path=label_csv,
                label_dimension=label_dimension,
                num_workers=6
            )
            paired_samples = paired_dataset['paired_samples']

            # === 划分组（按策略）===
            groups = merge_paired_data_to_part(
                paired_samples=paired_samples,
                strategy=strategy,
                cross_trail=cross_trail,
                random_state=seed
            )

            for group_idx, group in enumerate(groups):
                print(f"\n=== 处理第 {group_idx + 1}/{len(groups)} 组 ===")

                # 进一步划分 val（从 train 中切出）
                train_samples = group['train']
                test_samples = group['test']

                if len(train_samples) == 0 or len(test_samples) == 0:
                    print("⚠️ 跳过空组")
                    continue

                # 划分验证集
                random_state_for_val = seed + group_idx * 1000
                np.random.seed(random_state_for_val)
                np.random.shuffle(train_samples)
                val_split = max(1, int(0.2 * len(train_samples)))
                val_samples = train_samples[:val_split]
                train_samples = train_samples[val_split:]

                fold_group = {'train': train_samples, 'val': val_samples, 'test': test_samples}
                data_dict = paired_index_to_data(fold_group, keep_dim=True)

                # 提取数据
                train_eeg = data_dict['eeg']['train']
                train_fnirs = data_dict['fnirs']['train']
                train_label = data_dict['label']['train']

                val_eeg = data_dict['eeg']['val']
                val_fnirs = data_dict['fnirs']['val']
                val_label = data_dict['label']['val']

                test_eeg = data_dict['eeg']['test']
                test_fnirs = data_dict['fnirs']['test']
                test_label = data_dict['label']['test']

                # 清洗标签
                def clean_labels(labels):
                    labels = np.array(labels, dtype=np.float64)
                    labels = np.nan_to_num(labels, nan=0.0)
                    return labels.astype(np.int64)

                train_label = clean_labels(train_label)
                val_label = clean_labels(val_label)
                test_label = clean_labels(test_label)

                # 转换为 tensor
                train_eeg = torch.from_numpy(train_eeg.astype(np.float32)).permute(0, 2, 1)
                train_fnirs = torch.from_numpy(train_fnirs.astype(np.float32)).permute(0, 2, 1)
                train_label = torch.from_numpy(train_label)

                val_eeg = torch.from_numpy(val_eeg.astype(np.float32)).permute(0, 2, 1)
                val_fnirs = torch.from_numpy(val_fnirs.astype(np.float32)).permute(0, 2, 1)
                val_label = torch.from_numpy(val_label)

                test_eeg = torch.from_numpy(test_eeg.astype(np.float32)).permute(0, 2, 1)
                test_fnirs = torch.from_numpy(test_fnirs.astype(np.float32)).permute(0, 2, 1)
                test_label = torch.from_numpy(test_label)

                # 创建自定义 Dataset（或直接传入 tensors）
                from torch.utils.data import TensorDataset
                # 注意：这里我们传入三个张量：eeg, fnirs, label
                dataset_train = TensorDataset(train_eeg, train_fnirs, train_label)
                dataset_val = TensorDataset(val_eeg, val_fnirs, val_label)
                dataset_test = TensorDataset(test_eeg, test_fnirs, test_label)

                # 初始化双模态模型
                model = Model['E_FNet'](
                    eeg_chans=eeg_channels,
                    fnirs_chans=fnirs_channels,
                    num_classes=num_classes
                )

                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-8)
                criterion = nn.CrossEntropyLoss()

                print(f"训练样本: {len(train_eeg)}, 验证: {len(val_eeg)}, 测试: {len(test_eeg)}")

                # 使用支持双输入的训练函数
                round_metric = train_multimodal(
                    model=model,
                    dataset_train=dataset_train,
                    dataset_val=dataset_val,
                    dataset_test=dataset_test,
                    device=device,
                    output_dir=output_dir,
                    metrics=['acc', 'f1', 'auc'],
                    metric_choose='f1',
                    optimizer=optimizer,
                    batch_size=batch_size,
                    epochs=epochs,
                    criterion=criterion
                )

                round_metric.update({
                    'group': group_idx,
                    'fold': 0,  # 单折
                    'strategy': strategy,
                    'label_dimension': label_dimension
                })
                all_metrics.append(round_metric)
                group_metrics = [round_metric]

                print(f"组 {group_idx + 1} 完成，准确率: {round_metric['acc']:.4f}")

                result_log(output_dir, strategy, cross_trail, group_idx, group_metrics,
                           lr, batch_size, epochs, seed, k_fold)

            if all_metrics:
                final_avg_acc = np.mean([m['acc'] for m in all_metrics])
                final_std_acc = np.std([m['acc'] for m in all_metrics]) if len(all_metrics) > 1 else 0
                print(f"\n🎯 {strategy}-{label_dimension} 最终平均准确率: {final_avg_acc:.4f} ± {final_std_acc:.4f}")

    print("\n🎉 所有训练完成！")


if __name__ == "__main__":
    main()
