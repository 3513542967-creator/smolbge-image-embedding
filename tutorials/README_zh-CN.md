# 从数据到微调：VS Code 学习路线

这一部分只讲本项目实际发布的 **SmolVLM2 → BGE 图像向量对齐微调**。按编号运行即可；每一步的输入和输出都固定，便于定位问题。

```text
COCO 原始图片 + captions_val2017.json
              │  01：确定划分、清洗
              ▼
data/processed/{train,validation,test}.jsonl
              │  02：冻结两个基础模型，提取特征
              ▼
artifacts/cache/{train,validation,test}.npz
              │  03：只训练 887,424 参数的 Projector
              ▼
runs/contrastive/projector.safetensors
              │  04：检索评测
              ▼
R@1 / R@5 / R@10
```

## 在 VS Code 中开始

1. 用 VS Code 打开本仓库根目录，执行 `Terminal → Run Task → 00 · 创建 Python 环境`。
2. 执行 `01 · 安装项目依赖`，然后在状态栏选择 `.venv` 解释器。
3. 下载 [COCO val2017 图片与 captions 标注](https://cocodataset.org/#download)，建议保存为：

```text
data/raw/coco/
├── annotations/captions_val2017.json
└── val2017/
```

4. 依次执行任务 `02` 到 `05`。任务的路径可在 [`.vscode/tasks.json`](../.vscode/tasks.json) 中修改。

首次缓存需要下载两个基础模型，耗时最长；之后训练只读取本地 `.npz` 特征，不会反复跑视觉或文本编码器。

## 阅读顺序

- [01 · 数据集与固定划分](01_dataset/README.md)
- [02 · 清洗、整理与特征缓存](02_preprocess/README.md)
- [03 · 微调对齐头](03_finetune/README.md)
- [04 · 检索评测与结果解读](04_evaluation/README.md)

## 目录职责

| 你要学的内容 | 唯一实现文件 | 产生的文件 |
|---|---|---|
| COCO 清洗、固定划分 | [`scripts/prepare_coco.py`](../scripts/prepare_coco.py) | `data/processed/*.jsonl` |
| 冻结模型并缓存特征 | [`scripts/cache_features.py`](../scripts/cache_features.py) | `artifacts/cache/*.npz` |
| 微调 Projector | [`scripts/train.py`](../scripts/train.py) | `runs/contrastive/*` |
| 计算检索指标 | [`scripts/evaluate.py`](../scripts/evaluate.py) | 控制台 JSON / 结果 JSON |
| 模型结构、编码接口 | [`src/smolbge_image_embedding/modeling.py`](../src/smolbge_image_embedding/modeling.py) | 已发布推理接口 |

教程不复制训练代码：这样你在教程里看到的文件就是实际运行的文件，不会出现“教程能跑、项目不能复现”的两套实现。
