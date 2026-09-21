# 02 · 清洗、整理与特征缓存

## 为什么先缓存？

SmolVLM2 和 BGE-small 都是冻结的。训练时反复跑它们既慢又占显存；因此先将图片转成 768 维视觉特征、每条描述转成 384 维文本特征，训练阶段只更新小型 MLP。

## 运行

```bash
python scripts/cache_features.py \
  --data data/processed \
  --output artifacts/cache
```

每个划分都会得到一个 `.npz`：

| 字段 | 形状 | 含义 |
|---|---:|---|
| `vision` | `(N, 768)` | SmolVLM2 有效 patch 平均池化后的图片特征 |
| `text` | `(N, 5, 384)` | BGE-small 的 5 条归一化描述向量 |
| `image_ids` | `(N,)` | 用于追踪固定划分 |
| `captions` / `image_files` | `(N, 5)` / `(N,)` | 审计与评测元数据 |

## 清洗和编码细节

- 图片用 Pillow 打开并转为 RGB；编码后立即关闭文件句柄。
- 图像处理器关闭 image splitting；padding 区域由 `pixel_attention_mask` 排除。
- 每张图的有效 patch 做平均池化，得到 768 维输入，而不是使用整张图的填充区域。
- 文本最长 128 token，使用 BGE 的 CLS 向量并 L2 归一化。

## 看代码时关注

[`scripts/cache_features.py`](../../scripts/cache_features.py) 中的 `encode_images` 负责视觉部分：`patch_attention_mask` → `masked_patch_pool`。`encode_texts` 将平铺的 5N 条文本恢复为 `(N, 5, 384)`；这个形状约定会在训练和评测中一直保留。

缓存完成后可断网训练。缓存是机器相关的中间产物，已在 Git 忽略列表中，不应上传。
