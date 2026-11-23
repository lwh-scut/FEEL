from feel.train.cli import music_emotion_args, resolve_device


def main():
    args = music_emotion_args(
        "Train DBN on the music emotion dataset.",
        "runs/eeg/dbn",
        data_dirs=('eeg_dir',),
        epochs=20,
        strategies=('cross-subject', 'single-subject', 'cross-trail'),
    )

    import os

    import numpy as np
    import torch
    from torch import optim, nn

    from feel.data.music_emotion_eeg import load_eeg_dataset
    from feel.data.splits import merge_to_part, create_kfold_splits, get_split_index, setup_seed, index_to_data
    from feel.data.preprocess import normalize
    from feel.training.supervised import train
    from feel.models.registry import Model
    from feel.utils.results import result_log

    device = resolve_device(args.device)
    base_output_dir = args.output_dir

    # 定义循环参数
    strategies = args.strategy
    label_dimensions = args.label_dimension

    channels = 32
    feature_dim = 1500
    num_classes = 2

    seed = 666
    lr = 0.001
    batch_size = 32
    epochs = args.epochs
    k_fold = 1  # 新增K-fold参数

    eeg_dir = args.eeg_dir
    label_csv = args.labels

    # 外层循环：训练策略
    for strategy in strategies:
        # 内层循环：标签维度
        for label_dimension in label_dimensions:
            print(f"\n{'=' * 60}")
            print(f"开始训练: 策略={strategy}, 标签维度={label_dimension}")
            print(f"{'=' * 60}")

            # 为当前组合创建输出目录
            output_dir = os.path.join(base_output_dir, f"{strategy}_{label_dimension}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 设置cross_trail标志
            cross_trail = (strategy == "cross-trail")

            all_metrics = []  # 存储当前策略和标签维度的所有结果

            # 加载数据集
            dataset = load_eeg_dataset(eeg_dir, label_csv, label_dimension=label_dimension)
            data, label = merge_to_part(data_dict=dataset['data_dict'], label_dict=dataset['label_dict'],
                                        strategy=strategy, cross_trail=cross_trail)

            for group_idx in range(len(data)):
                group_data = data[group_idx]
                group_label = label[group_idx]

                print(f"\n=== 处理第 {group_idx + 1}/{len(data)} 组 (策略: {strategy}, 标签: {label_dimension}) ===")

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

                    # 清洗 train_label
                    train_label = np.array(train_label, dtype=np.float64)  # 统一成 numpy 数组
                    train_label = np.nan_to_num(train_label, nan=0.0)  # 替换 NaN
                    train_label = train_label.astype(np.int64)  # 转成整数

                    # 清洗 train_data（解决"Creating tensor from list of numpy arrays is slow"警告）
                    train_data = np.array(train_data, dtype=np.float32)
                    val_data = np.array(val_data, dtype=np.float32)
                    test_data = np.array(test_data, dtype=np.float32)

                    val_label = np.nan_to_num(np.array(val_label, dtype=np.float64), nan=0.0).astype(np.int64)
                    test_label = np.nan_to_num(np.array(test_label, dtype=np.float64), nan=0.0).astype(np.int64)

                    train_data, val_data, test_data = normalize(train_data, val_data, test_data, dim='sample', method="minmax")

                    # 创建 TensorDataset
                    dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(train_data),
                                                                   torch.from_numpy(train_label))
                    dataset_val = torch.utils.data.TensorDataset(torch.from_numpy(val_data),
                                                                 torch.from_numpy(val_label))
                    dataset_test = torch.utils.data.TensorDataset(torch.from_numpy(test_data),
                                                                  torch.from_numpy(test_label))

                    model = Model['DBN'](channels, feature_dim, num_classes)
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-8)
                    criterion = nn.CrossEntropyLoss()

                    print(f"\n--- 第 {fold_idx + 1}/{num_folds} 折训练 (策略: {strategy}, 标签: {label_dimension}) ---")
                    print(f"训练样本数: {len(train_data)}, 验证样本数: {len(val_data)}, 测试样本数: {len(test_data)}")

                    round_metric = train(
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

                    # 记录本折结果
                    round_metric['group'] = group_idx
                    round_metric['fold'] = fold_idx
                    round_metric['strategy'] = strategy
                    round_metric['label_dimension'] = label_dimension
                    group_metrics.append(round_metric)
                    all_metrics.append(round_metric)

                    print(f"第 {fold_idx + 1} 折完成，准确率: {round_metric['acc']:.4f}")

                # 计算当前组的平均性能
                if len(group_metrics) > 0:
                    avg_acc = np.mean([m['acc'] for m in group_metrics])
                    std_acc = np.std([m['acc'] for m in group_metrics]) if len(group_metrics) > 1 else 0
                    print(f"\n第 {group_idx + 1} 组平均准确率: {avg_acc:.4f} ± {std_acc:.4f}")

                    # 保存当前组的结果
                    result_log(output_dir, strategy, cross_trail, group_idx, group_metrics,
                               lr, batch_size, epochs, seed, k_fold)

            # 当前策略和标签维度的最终统计
            if len(all_metrics) > 0:
                final_avg_acc = np.mean([m['acc'] for m in all_metrics])
                final_std_acc = np.std([m['acc'] for m in all_metrics]) if len(all_metrics) > 1 else 0
                print(f"\n🎯 {strategy}-{label_dimension} 最终平均准确率: {final_avg_acc:.4f} ± {final_std_acc:.4f}")
                print(f"总训练轮次: {len(all_metrics)}")

            print(f"\n完成训练: 策略={strategy}, 标签维度={label_dimension}")
            print(f"{'=' * 60}")

    print("\n🎉 所有策略和标签维度的训练已完成！")


if __name__ == "__main__":
    main()
