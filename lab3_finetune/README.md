# Lab3：Finetune（TRL SFTTrainer + DeepSpeed ZeRO-3）

## 概述

| 項目 | 說明 |
|------|------|
| 框架 | TRL `SFTTrainer` + PEFT `LoraConfig` |
| 分散訓練 | DeepSpeed ZeRO-3，optimizer offload → NVMe，param offload → CPU |
| GPU | RTX 3060 12GB × 1 |

使用自建 Docker 容器，對 `Qwen2.5-3B-Instruct` 進行一輪指令微調（SFT）。

## 前置條件

- Lab2 已完成，`lab2/output/train.json` 存在（80 筆）。
- Docker 已安裝，base model 已放置於 `~/models/Qwen2.5-3B-Instruct/`。

## 訓練格式

`lab2/output/train.json` 為 `{"question", "answer"}` 的 JSON 陣列。  
`train_sft.py` 讀入後自動轉換為 messages 格式：

```json
{
  "messages": [
    {"role": "system",    "content": "你是一個專業助理，請根據提供的資訊回答問題。"},
    {"role": "user",      "content": "<question>"},
    {"role": "assistant", "content": "<answer>"}
  ]
}
```

## 訓練設定

| 參數 | 值 |
|------|----|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q/k/v/o/gate/up/down_proj |
| Learning rate | 5e-5 |
| Batch size | 1 × grad_accum=32 |
| Max seq length | 4096 |
| Epochs | 3 |

## 執行方式

### 1. 建立 SFT 容器

```bash
docker compose -f lab3_finetune/docker-compose-sft.yaml build
docker compose -f lab3_finetune/docker-compose-sft.yaml up -d
```

### 2. 執行訓練

```bash
docker compose -f lab3_finetune/docker-compose-sft.yaml exec sft_finetune \
  deepspeed --num_gpus=1 /workspace/lab3_finetune/train_sft.py
```

### 3. 合併 LoRA adapter

```bash
docker compose -f lab3_finetune/docker-compose-sft.yaml exec sft_finetune \
  python3 /workspace/lab3_finetune/merge_lora.py
```

合併後的完整模型會存放於 `~/models/Qwen2.5-3B-Instruct-finetuned/`。

### 4. 部署至 vLLM

```bash
docker compose -f docker-compose-vllm.yaml up -d --force-recreate
```
