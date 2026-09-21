# 03 · 微调图像对齐头

## 训练什么？

两个基础模型全程冻结。训练的 [`VisionProjector`](../../src/smolbge_image_embedding/modeling.py) 是：

```text
768 视觉特征 → LayerNorm → Linear(768,768) → GELU → Dropout → Linear(768,384) → L2 normalize
```

共 887,424 个可训练参数。输出的 384 维向量可以与 BGE-small 文本向量直接做点积 / 余弦检索。

## 运行

```bash
python scripts/train.py \
  --cache artifacts/cache \
  --output runs/contrastive
```

训练会输出每个 epoch 的 loss 和 `validation_mean_r1`；连续 15 轮没有超过最佳验证分数就早停。最佳权重保存为 `runs/contrastive/projector.safetensors`，完整历史保存为 `training.json`。

## 一个 batch 的目标

对第 i 张图，先把它的 5 条描述向量取均值并归一化，作为 `target[i]`。模型产生 `image[i]`。batch 内相似度矩阵为：

```text
logits[i, j] = image[i] · target[j] / 0.07
```

损失 = 双向 InfoNCE + `0.5 ×` 余弦蒸馏 + `0.25 ×` 困难负例间隔损失。完整实现见 [`loss_function`](../../scripts/train.py)。

## 超参数

学习率 3e-4、AdamW weight decay 0.01、batch 256、最多 100 epochs、随机种子 42。发布配置记录在 [`configs/default.json`](../../configs/default.json)。

这是**特征级微调**：它节省显存并让实验可在本地学习，但不更新 SmolVLM2 或 BGE 的参数。若将来做端到端 LoRA / 分布式训练，应视为一组新的实验，不能与当前结果直接混写。
