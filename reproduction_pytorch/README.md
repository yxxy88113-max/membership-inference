# Membership Inference Attack（PyTorch 复现）

本目录保留根目录中的原始 Python 2/Theano/Lasagne 实现，并提供 Python 3.11+ 的 PyTorch 实现 `run_experiment.py`。流程对应 Shokri et al. (2017)：目标模型、独立影子模型、按真实类别分别训练攻击模型。

## 运行

```bash
python -m pip install -r requirements.txt
python run_experiment.py --config config.json --output results
```

MNIST 由 torchvision 从官方镜像下载。`--smoke --synthetic` 仅用于验证程序控制流，不是 MNIST 结果。完整运行会保存索引、模型、概率、CSV 和 ROC 图。
