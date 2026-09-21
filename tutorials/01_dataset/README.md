# 01 · 数据集与固定划分

## 任务

每条样本是一张图片和 5 条英文人工描述。训练目标不是生成文字，而是让图片向量靠近这 5 条文字向量的中心。

原始数据来自 COCO `val2017`：图片在 `val2017/`，标注在 `captions_val2017.json`。它是公开 COCO 数据的**自定义检索划分**，不要称为标准 COCO 检索榜单。

## 输入格式

COCO 标注中会出现：

```json
{"image_id": 397133, "caption": "A man riding a bike down a street."}
```

脚本将同一 `image_id` 的前 5 条描述聚合，并要求对应图片真实存在。它使用仓库中的 [`data/splits/coco5k_seed42.json`](../../data/splits/coco5k_seed42.json) 固定 4,000 / 500 / 500 张图的 train / validation / test；三组图片 ID 不重叠。

## 运行

```bash
python scripts/prepare_coco.py \
  --captions data/raw/coco/annotations/captions_val2017.json \
  --images data/raw/coco/val2017 \
  --output data/processed
```

输出示例 `data/processed/train.jsonl`：

```json
{"image_id": 397133, "image_file": "/absolute/path/000000397133.jpg", "captions": ["...", "...", "...", "...", "..."]}
```

`image_file` 是绝对路径，因此移动数据目录后需要重新执行这一步。脚本会在图片不全时直接报错，避免悄悄改变已发布的评测划分。

## 看代码时关注

1. `eligible`：仅保留“至少 5 条描述且图片存在”的样本。
2. `split_ids`：优先读取发布的固定 ID，而不是重新随机抽样。
3. `write_jsonl`：一行一张图，便于流式检查和后续处理。
