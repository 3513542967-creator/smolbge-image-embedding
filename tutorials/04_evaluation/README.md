# 04 · 检索评测与结果解读

## 运行

```bash
python scripts/evaluate.py \
  --cache artifacts/cache \
  --weights runs/contrastive/projector.safetensors \
  --output runs/contrastive/test_metrics.json
```

也可以把 `--weights` 改为已发布的 `weights/projector.safetensors`，检查你的缓存是否与发布结果一致。

## 指标口径

| 方向 | 查询 | 候选 | R@1 命中条件 |
|---|---:|---:|---|
| 图 → 文 | 500 张图 | 2,500 条描述 | 任一配对描述排第一 |
| 文 → 图 | 2,500 条描述 | 500 张图 | 配对图片排第一 |

`R@5`、`R@10` 分别表示正确对象进入前 5、前 10。`median_rank` 越小越好。

发布适配器在该固定测试集上：图→文 R@1 **68.00%**，文→图 R@1 **60.76%**。图→文线性基线为 45.40%，因此提升 22.6 个百分点。完整结果在 [`results/evaluation.json`](../../results/evaluation.json)。

## 看代码时关注

[`src/smolbge_image_embedding/metrics.py`](../../src/smolbge_image_embedding/metrics.py) 中先 L2 归一化，再计算相似度矩阵。评测没有使用查询前缀，文本截断长度为 128；改变任一约定都会使结果不可直接比较。

当前证据只覆盖英文 COCO 域内图文检索；不是标准 COCO 榜单，也不证明跨领域、以图搜图或端到端 RAG 问答效果。
