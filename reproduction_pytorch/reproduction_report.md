# Membership Inference Attacks 复现报告

## 研究目的
论文利用模型输出的类别概率区分训练成员和未见样本。影子模型模拟目标模型，攻击模型学习概率向量与 membership 的关系。

## 原仓库分析
`README.md` 只描述命令行入口；`attack.py` 划分目标/影子数据，训练目标模型、多个影子模型，并按类别训练攻击模型；`classifier.py` 实现 Lasagne 的全连接网络/softmax 回归、Adam、交叉熵和批训练。原实现目标模型和影子模型均为输入维度到 50 个 tanh 隐藏单元再到类别 softmax；攻击模型按类别使用概率向量训练二分类器。

已发现问题：代码使用 Python 2 的 `print`、`xrange`；依赖 Theano 0.8/Lasagne 0.1 已停止维护；`load_dataset` 忽略 `test_feat`/`test_label` 而重复读取训练文件；`attack.py` 的 `load_dataset(...args.test_feat, args.train_label)` 传错标签；目标数据默认 10000 且要求样本量大于两倍，普通小数据会断言失败；训练批次和类别为空时可能崩溃。

## PyTorch 改写
`run_experiment.py` 使用 `nn.Linear -> Tanh -> Linear`、Adam 和交叉熵，保留目标模型、独立影子模型、按类别攻击模型的算法结构。固定随机种子 21312。`split_indices.npz` 保存四个互不重叠集合的原始索引，并在攻击测试中只使用目标成员和目标非成员。

## 验证
在当前环境执行了 `--smoke --synthetic` 控制流测试（2 个影子模型、每组 200 个样本、1 epoch），程序成功生成模型、CSV、概率和 ROC 输出。该测试不是 MNIST，也不应解释为论文结果。

MNIST 下载被当前环境的网络代理以 HTTP 403 拒绝（torchvision 官方 S3 与 Yann LeCun 地址均失败），因此本次无法诚实报告 MNIST 训练指标或论文级结果。环境配置草稿已加入 `ossci-datasets.s3.amazonaws.com`，用户发布并允许网络后可直接运行完整命令。随机猜测基线为 accuracy 50%、ROC-AUC 0.5；攻击接近该值表示模型输出没有提供可利用的成员信号。

## 与论文的差异与限制
使用 PyTorch 取代已废弃的 Theano/Lasagne；默认实验规模和 epoch 可配置；当前尚未获得 MNIST，故没有冒充论文数值。论文使用更大规模数据、更多训练轮次和具体数据集设置，需在网络可用后重新运行并比较。
