# SmolBGE · 面向 RAG 的图像向量模型

[English](README.md) · [GitHub](https://github.com/3513542967-creator/smolbge-image-embedding) · [模型权重](https://huggingface.co/yifanouyang/smolbge-image-embedding) · [模型卡](MODEL_CARD.md)

将图片与英文文本编码到同一个 **384 维向量空间**，用于图文检索。冻结 SmolVLM2 视觉编码器和 BGE-small，只训练 **3.4 MiB 对齐头**，图片直接生成向量，无需生成描述。

![模型结构](assets/architecture.svg)

## 实验结果

使用 COCO val2017 的自定义划分：**4,000 张训练 / 500 张验证 / 500 张测试**，每图 5 条人工描述，随机种子 42。按图片 ID 隔离，仅用验证集选模。

| 方法 | 图→文 R@1 | 文→图 R@1 |
|:--|--:|--:|
| 线性基线 | 45.40% | 59.84% |
| **发布模型** | **68.00%** | **60.76%** |

图→文在 2,500 条描述中检索，首位命中任一配对描述即成功；文→图在 500 张图片中检索。图→文较线性基线提升 **22.6 个百分点**，配对 bootstrap 95% 区间为 +18.0～+27.2。

[五组对照与完整指标](results/evaluation.json) · [固定数据划分](data/splits/coco5k_seed42.json)

## 快速使用

建议 Python 3.12，在虚拟环境中安装：

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install "git+https://github.com/3513542967-creator/smolbge-image-embedding.git"
smolbge --images ./photos --query "a dog playing outside" --top-k 5
```

将 `./photos` 换成你的图片文件夹，即可返回排序后的图片路径与余弦分数。首次运行自动下载对齐头和两个基础模型，需要联网并预留基座权重的磁盘空间；推理无需下载 COCO 或重新训练。自动选择 NVIDIA GPU、Apple MPS 或 CPU，也可指定 `--device cpu`。命令每次重新编码文件夹。

```python
from smolbge_image_embedding import SmolBGEEmbedder

model = SmolBGEEmbedder.from_pretrained()  # 自动下载已发布权重
images = model.encode_images(["photo.jpg"])  # (1, 384)
query = model.encode_text("a dog playing outside")  # (384,)
scores = images @ query
print(scores)
```

重复查询时，可将图片向量保存至向量数据库。本地克隆并下载 Git LFS 权重后，也可使用 `SmolBGEEmbedder.from_pretrained(".")`。

## 方法与复现

- **模型对齐：**有效 patch 平均池化 → 768→768→384 MLP，仅训练 **887,424 个参数**。
- **训练目标：**双向对比学习、描述中心向量蒸馏、困难负例间隔损失。
- **工程实现：**冻结特征缓存、验证集早停、批量推理接口、Hub 权重下载和命令行检索。

[训练与评测复现](docs/REPRODUCE.md) · [模型实现](src/smolbge_image_embedding/modeling.py)

当前结论限于英文 COCO 域内检索，属于自定义划分，不能等同于标准 COCO 榜单或完整 RAG 问答质量。基座可能接触过相关预训练数据；跨领域、以图搜图的检索质量尚未验证。

代码与对齐头采用 [Apache-2.0](LICENSE)，基础模型及 COCO 遵循[各自条款](THIRD_PARTY.md)。
