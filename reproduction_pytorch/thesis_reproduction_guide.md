# Membership Inference Attacks 论文复现：导师汇报与操作讲稿

这份材料说明本项目如何从原始 Python 2 代码迁移到 Python 3/PyTorch，如何准备 MNIST，如何训练目标模型、影子模型和攻击模型，以及如何解释结果。文件中的数字均来自当前仓库已经执行的实验；合成数据 smoke test 只用于检查程序流程，不作为论文实验结果。

## 1. 一句话介绍项目

本项目研究：只观察一个分类模型输出的类别概率，能否推断某个样本是否出现在这个模型的训练集里。原论文的核心方法是使用多个影子模型模拟目标模型，再用影子模型的成员/非成员输出训练攻击模型。

汇报时可以先说：

> 我复现了 Shokri 等人在 2017 年提出的成员推理攻击核心流程。为了使旧代码可运行，我保留原始 Python 2/Theano/Lasagne 代码作为对照，并使用 Python 3/PyTorch 重新实现相同的目标模型、影子模型和按类别攻击模型。随后使用经过格式和哈希校验的真实 MNIST，完成了 2/10 个影子模型、多个 epoch 和多个随机种子的实验。

## 2. 实验环境

实验在 Codex 云环境的 `/workspace/membership-inference` 中执行。主要运行环境为 Linux、CPU、Python 3.12（代码兼容 Python 3.11+）、PyTorch CPU、NumPy、scikit-learn、pandas 和 matplotlib。原始兼容环境也曾安装 Python 2.7、Theano 0.8.2 和 Lasagne 0.1，但现代实验使用 PyTorch。

代码分支是 `reproduce/mnist-pytorch`。主要提交包括：

- `e70bb19`：第一次 PyTorch 复现项目
- `658854b`：增加训练曲线和混淆矩阵
- `efea0dc`：真实 MNIST 3 epoch 结果
- `b36f88b`：20/50 epoch 受控多种子实验
- `da476f8`：官方 test 非成员和真实标签攻击修正

## 3. 原始代码阅读结果

### README.md

README 只给出命令行入口和输入文件格式。输入特征文件每行是逗号分隔的浮点数，标签文件每行是整数标签。

### classifier.py

`classifier.py` 负责构建和训练分类器：

1. 读取特征和标签。
2. 建立输入层、隐藏层和输出层。
3. 使用 tanh 激活函数。
4. 使用 softmax 输出类别概率。
5. 使用交叉熵损失和 Adam 更新参数。
6. 输出训练准确率和测试准确率。

原始神经网络为：

```text
输入特征 → Dense(50) → tanh → Dense(类别数) → softmax
```

### attack.py

`attack.py` 负责完整攻击流程：

1. 从数据中划分目标模型和影子模型数据。
2. 训练目标模型。
3. 训练多个影子模型。
4. 收集成员和非成员样本的概率输出。
5. 按真实类别训练攻击模型。
6. 在目标模型输出上测试成员推理效果。

原代码的问题包括 Python 2 语法、Theano/Lasagne 过时、测试文件参数传错、测试特征被错误地重复读取训练特征，以及小数据集容易触发硬编码断言。

## 4. PyTorch 实现

新实现位于 `reproduction_pytorch/run_experiment.py` 和 `rigorous_experiment_v2.py`。

目标模型和影子模型结构为：

```text
784 个输入像素
      ↓
全连接层：50 个神经元
      ↓
tanh 激活
      ↓
全连接层：10 个输出
      ↓
softmax 概率
```

784 来自 28×28 的 MNIST 图片。输出 10 个概率分别对应数字 0 到 9。

训练时使用：

- Adam 优化器
- 交叉熵损失
- batch size 128
- 固定随机种子
- CPU 训练

`run_experiment.py` 是基础实验入口，`rigorous_experiment.py` 是保持目标模型一致的 2/10 shadow 对比入口，`rigorous_experiment_v2.py` 是严格版本入口。

## 5. MNIST 数据准备

官方源受到当前代理限制，返回 HTTP 403，因此从公开 GitHub 镜像导入了四个 IDX gzip 文件：

```text
https://raw.githubusercontent.com/fgnt/mnist
```

文件包括：

```text
train-images-idx3-ubyte.gz
train-labels-idx1-ubyte.gz
t10k-images-idx3-ubyte.gz
t10k-labels-idx1-ubyte.gz
```

检查内容包括：

- 训练图片 60000 张
- 测试图片 10000 张
- 每张图片 28×28
- 标签范围 0 到 9
- IDX magic number 正确
- gzip 可以解压
- SHA-256 已记录

检查结果在 `mnist_validation.json`。

## 6. 严格的数据划分

v2 实验使用以下划分：

```text
MNIST train
├── 目标模型成员
└── 影子模型训练和非成员数据

MNIST test
└── 目标模型非成员
```

当前配置为：

```text
目标成员：1000
目标非成员：1000
影子训练池：1000
影子非成员池：1000
```

目标成员和影子数据来自训练集，但影子数据只使用目标成员以外的部分。目标非成员来自官方测试集，因此不属于训练数据来源。

每个种子的索引保存在：

```text
rigorous_results_v2/split_seed21312.npz
rigorous_results_v2/split_seed21313.npz
rigorous_results_v2/split_seed21314.npz
```

## 7. 训练目标模型

目标模型先使用目标成员数据训练。训练结束后分别计算：

- 目标训练准确率：对训练时见过的数据预测
- 目标测试准确率：对官方 MNIST test 中的非成员预测
- 泛化差距：训练准确率减去测试准确率

例如：

```text
训练准确率 = 1.000
测试准确率 = 0.900
泛化差距 = 0.100
```

训练准确率高而测试准确率低，代表模型对训练数据更熟悉，成员推理可能有可利用信号。

## 8. 训练影子模型

每个影子模型使用独立的影子训练数据。对每个影子模型：

1. 用影子成员数据训练。
2. 用影子非成员数据测试。
3. 记录成员和非成员样本的 10 类概率。
4. 保存概率和 membership 标签。

影子训练样本标签为 1，影子非成员样本标签为 0。

实验比较 2 个和 10 个影子模型。严格对比时，目标模型只训练一次，2-shadow 和 10-shadow 共享同一个目标权重和目标数据。

## 9. 训练攻击模型

攻击模型输入目标模型的 10 个 softmax 概率，输出二分类结果：

```text
0：非成员
1：成员
```

v2 版本按真实数字标签分组。例如目标样本真实标签为 7，就交给数字 7 的攻击模型。

另外还实现了三个简单基线：

1. 最大预测概率越高，越可能是成员。
2. 交叉熵损失越低，越可能是成员。
3. 预测熵越低，概率越集中，越可能是成员。

阈值只使用影子数据选择，然后用于目标数据，避免目标测试标签泄漏到阈值选择中。

## 10. 已执行的实验

### 旧版 3 epoch 实验

这些结果保留在 `results_mnist_2/` 和 `results_mnist_10/`，用于说明最初的完整流程已经运行成功。它们不能和严格 v2 结果直接比较，因为旧实验的数据规模和非成员来源不同。

### 受控 20/50 epoch 实验

`rigorous_results/` 中保存了 3 个种子、20/50 epoch 的受控对比。2-shadow 和 10-shadow 使用同一个目标模型。

### 严格 v2 实验

`rigorous_results_v2/` 中保存了官方 test 非成员和真实标签攻击的 3 个种子、20 epoch 结果。运行命令：

```bash
cd /workspace/membership-inference/reproduction_pytorch
python rigorous_experiment_v2.py \
  --config config.json \
  --data-root results_mnist_10/data \
  --output rigorous_results_v2 \
  --epochs 20 \
  --seeds 21312,21313,21314
```

## 11. 严格 v2 的实际结果

3 个种子、20 epoch 的均值：

| 影子数 | 方法 | Attack Accuracy | 成员 F1 | Macro F1 | ROC-AUC |
|---:|---|---:|---:|---:|---:|
| 2 | 真实标签神经攻击 | 0.5832 | 0.6652 | 0.5558 | 0.6100 |
| 10 | 真实标签神经攻击 | 0.5947 | 0.6508 | 0.5835 | 0.6169 |
| 2 | 最大置信度 | 0.6175 | 0.7134 | 0.5693 | 0.5966 |
| 10 | 最大置信度 | 0.6175 | 0.7145 | 0.5675 | 0.5966 |
| 2 | 交叉熵损失 | 0.6222 | 0.7196 | 0.5701 | 0.6009 |
| 10 | 交叉熵损失 | 0.6248 | 0.7203 | 0.5754 | 0.6009 |
| 2 | 预测熵 | 0.6132 | 0.7106 | 0.5636 | 0.5952 |
| 10 | 预测熵 | 0.6140 | 0.7114 | 0.5643 | 0.5952 |

随机猜测基线是 Accuracy 0.50、ROC-AUC 0.50。结果略高于随机基线，但不能解释为论文原始规模的数值复现。

## 12. 需要向导师展示的文件

### 代码

- `attack.py`：原始代码
- `classifier.py`：原始分类器
- `run_experiment.py`：基础 PyTorch 实现
- `rigorous_experiment.py`：目标模型一致性实验
- `rigorous_experiment_v2.py`：官方 test 和真实标签严格实验
- `make_presentation_figures.py`：生成汇报图

### 数据与审计

- `mnist_validation.json`：MNIST 文件校验
- `split_seed*.npz`：数据索引
- `data_audit.json`：数据来源记录

### 结果

- `results.csv`：逐种子、逐方法结果
- `summary.csv`：均值和标准差
- `comparison_table.csv`：受控实验对比表
- `training_metrics.csv`：训练记录
- `target_seed*_ep*.pt`：模型参数
- `shadow_outputs.npz`：影子模型概率

### 图像

- `presentation_figures/fig1_target_training.png`：目标训练曲线
- `presentation_figures/fig2_attack_methods.png`：攻击方法比较
- `presentation_figures/fig3_generalization.png`：训练准确率、测试准确率和泛化差距
- `presentation_figures/fig4_pipeline.png`：实验流程图
- `results_mnist_10/roc_curve.png`：旧版 ROC 曲线
- `results_mnist_10/attack_confusion_matrix.png`：旧版混淆矩阵

## 13. 给导师演示的推荐顺序

第一步，说明研究问题：模型是否会泄露某个样本是否参与训练。

第二步，展示流程图，解释目标模型、影子模型和攻击模型的角色。

第三步，展示数据校验文件，说明 MNIST 数量、尺寸、标签范围和 SHA-256。

第四步，展示数据划分索引，说明目标成员、目标非成员和影子数据没有混用。

第五步，展示目标训练曲线，解释训练准确率、测试准确率和泛化差距。

第六步，展示攻击方法比较图，说明神经攻击和三个简单阈值基线的差异。

第七步，展示结果表，解释 Accuracy、Precision、Recall、F1 和 ROC-AUC。

第八步，主动说明局限：当前目标成员规模只有 1000，影子模型数量有限，PyTorch 与原始 Theano 实现不同，因此是核心流程复现，不是论文数值的逐点复制。

## 14. 可以直接照着说的总结

> 本实验首先保留并分析了原始 Python 2、Theano 和 Lasagne 代码，然后使用 Python 3 和 PyTorch 重写相同的模型训练流程。每张 MNIST 图片被转换为 784 维向量，目标模型和影子模型使用 50 个 tanh 隐藏单元。影子模型分别产生成员和非成员概率输出，攻击模型按照真实数字类别学习这些输出的差异。严格版本使用 MNIST 官方测试集作为目标非成员，使用三个随机种子运行，并同时比较神经攻击、最大置信度、交叉熵和预测熵基线。结果显示攻击效果略高于随机猜测，说明模型输出存在一定成员信号，但在当前小规模设置下信号较弱。实验保存了代码、数据索引、模型参数、CSV、图像和中文报告，全部结果可以由命令重新生成。

## 15. 目前仍然需要说明的限制

1. 当前严格 v2 实验实际运行了 3 个种子，而不是 5 到 10 个种子。
2. 目标成员规模为 1000，不能代表论文的大规模结果。
3. 当前主要使用 20 epoch 的严格 v2 结果；旧版另有 50 epoch 结果。
4. 影子模型数量只比较了 2 和 10。
5. 原论文的数据规模、完整超参数和实验环境无法全部获得。
6. 所有图表和结果都应标注为“本项目复现结果”，不能写成“论文原始结果”。
