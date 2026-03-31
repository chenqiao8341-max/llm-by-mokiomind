# MokioMind Pretrain Only

这个仓库已经被精简成只保留一次密集 Transformer 预训练所需的核心路径：

- `embedding`
- `RoPE` 位置编码
- `GQA` 注意力
- `FFN`
- `PretrainDataset`
- `trainer/train_pretrain.py`

后训练相关的 `SFT / DPO / PPO / GRPO / LoRA` 入口已经移除，避免重新捡项目时被无关代码干扰。

## 目录

- `model/MokioModel.py`: 纯预训练版模型
- `dataset/lm_dataset.py`: 只保留 `PretrainDataset`
- `trainer/train_pretrain.py`: 预训练入口
- `dataset/sample_pretrain.jsonl`: 本地最小示例数据，只有 4 行，用来检查训练链路是否正常

## 数据格式

预训练数据使用 `jsonl`，每行一条：

```json
{"text": "机器学习是让模型从数据中学习规律。"}
```

## 直接训练

### 单卡快速检查

在服务器安装依赖后执行：

```bash
python main.py \
  --data_path ../dataset/sample_pretrain.jsonl \
  --epochs 1 \
  --batch_size 4 \
  --max_seq_len 256 \
  --device cuda:0
```

如果你要从头换成自己的数据，只需要准备新的 `jsonl` 文件并替换 `--data_path`。

### 双卡 4090 推荐起步

`train_pretrain.py` 现在的默认参数已经上调到更适合 `2 x 4090` 的首跑配置：

- `hidden_size=768`
- `num_hidden_layers=12`
- `num_attention_heads=12`
- `num_key_value_heads=4`
- `max_seq_len=1024`
- `batch_size=16`，这里是单卡 batch size
- `accumulation_steps=8`

推荐这样启动：

```bash
torchrun --nproc_per_node=2 main.py \
  --data_path /path/to/your_pretrain.jsonl \
  --epochs 1 \
  --save_weight pretrain_2x4090
```

按这组默认值计算，等效全局 batch 大约是：

```text
2 卡 × 每卡 16 × 梯度累积 8 = 256
```

## 说明

- 本地没有安装 `torch` 也没有可用显卡，因此这次修改只做了静态精简和语法检查。
- `sample_pretrain.jsonl` 只是链路测试文件，不适合正式预训练。
