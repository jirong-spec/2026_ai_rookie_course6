# Guru QA 資料生成、微調與驗證：從文件到模型的完整實戰

## 實驗結果總覽

| 實驗 | 設定 | `<ANSWER>:` 率 | 平均回應長度 | 關鍵觀察 |
|------|------|---------------|-------------|---------|
| Lab2 Base | Base model + RAG_chunks（40 筆） | **53%** | 1,365 字元 | 格式遵循佳，偶有語言混用 |
| Lab4 SFT v3 | Finetuned + RAG_chunks | **51%** | 1,984 字元 | 自蒸餾瓶頸，15% 輸出退化 |
| Lab5 B | FT + 極簡 Prompt | 0% | 1,469 字元 | 無格式要求時完全不輸出 ANSWER |
| Lab5 D | FT + 無 Context | 0% | 286 字元 | Context 是關鍵，缺少則亂答 |
| Extra A hybrid | FT + hybrid_chunks | 33% | 1,951 字元 | 混合 context 略提升格式率 |
| Extra A chunk | FT + 原始 chunk | **36%** | 1,286 字元 | 最短最精準，格式率最高 |
| SFT + Re-ranking | SFT + bge-reranker top-40 | 24% ⚠️ | 1,861 字元 | Reranker 破壞訓練分布，反降 |
| DPO（無 Rerank）| DPO from Base，40 pairs | 15% ⚠️ | 191 字元 | 訓練樣本不足，模型退化 |

## 核心發現

1. **自蒸餾瓶頸**：用 base model 本身生成的答案訓練 SFT，無法超越 base model，因為訓練目標的品質上限就是 base model 本身。需要 teacher model（GPT-4o / Claude）生成高品質答案才能突破。

2. **Re-ranking 需謹慎**：RAG context 已由 `multilingual-e5-large` 篩選，模型訓練時適應了這個排序。加上 bge-reranker 重排後，分布偏移導致 ANSWER 率從 51% 降至 24%。Re-ranking 需與訓練分布一致才有效。

3. **DPO 數據品質決定一切**：40 筆自蒸餾的 chosen/rejected pairs 不足以建立有意義的偏好對，DPO 反而讓模型退化。需要高品質 teacher 答案作為 chosen。

4. **Context 是關鍵**：無 context 時回應長度從 1,365 字元降至 286 字元，且幾乎全部答錯。RAG context 對這類領域 QA 任務不可缺少。

## 模型規格

| 用途 | 模型 |
|------|------|
| Guru 生題生答 / Inference 推理 | `Qwen/Qwen2.5-3B-Instruct`（vLLM） |
| SFT 微調 | `Qwen/Qwen2.5-3B-Instruct`（DeepSpeed ZeRO-3 + LoRA） |
| RAG 向量檢索 embedding | `intfloat/multilingual-e5-large` |
| Re-ranking | `BAAI/bge-reranker-base` |

## SFT 訓練超參數（v3）

| 參數 | 值 |
|------|-----|
| LoRA rank | 16，alpha 32 |
| Target modules | q/k/v/o/gate/up/down_proj |
| Learning rate | 5e-5 |
| Batch size | 1 × grad_accum=4 |
| Max seq length | 4096 tokens |
| Epochs | 10（200 steps） |
| DeepSpeed | ZeRO-3 + NVMe offload |
| GPU | RTX 3060 12GB × 1 |
| Train loss | 2.18 → 0.36 |
| Token accuracy | 61% → 92% |

## 環境設定

```bash
# Python 依賴（uv）
uv sync

# 系統依賴
sudo apt-get install poppler-utils

# vLLM 啟動
docker compose -f docker-compose-vllm.yaml up -d
# 端點：http://localhost:8299/v1

# SFT 訓練
docker compose -f lab3_finetune/docker-compose-sft.yaml build
docker compose -f lab3_finetune/docker-compose-sft.yaml run --rm sft_finetune \
  deepspeed --num_gpus=1 /workspace/lab3_finetune/train_sft.py

# LoRA merge
docker compose -f lab3_finetune/docker-compose-sft.yaml run --rm sft_finetune \
  python3 /workspace/lab3_finetune/merge_lora.py
```

## 實作 Lab 概觀

| Lab | 主題 | 結果 |
|-----|------|------|
| Lab1 | Guru 全管線（PDF → Chunk → Q → A → RAG）| 100 筆 QA，avg rag_acc ≈ 0 |
| Lab2 | Base model 推理 + train/test 切分 | 53% ANSWER rate，訓練 80 筆 |
| Lab3 | SFT（ZeRO-3 + LoRA） + Merge + vLLM 部署 | loss 0.36，acc 92%，200 steps |
| Lab4 | Base vs Finetuned 比較 | FT 51% < Base 53%（自蒸餾瓶頸）|
| Lab5 | Prompt × Context 消融（4 組） | Context 移除後從 53% → 0% |
| Extra A | Context 欄位消融（RAG/hybrid/chunk）| chunk 欄位 36% 最佳 |
| Extra B | Chunk size 消融（256 vs 1024）| 小 chunk 問題更聚焦，答案更簡潔 |
| lab_dpo | DPO 訓練管線 + Re-ranking 實驗 | 兩者皆因數據品質不足而退化 |

## 完整工作流程

```
PDF/DOCX → Guru(Lab1) → train.json(Lab2) → SFT(Lab3) → Merge → vLLM
                                                                    ↓
                                          Base 推理(Lab2) ←→ FT 推理(Lab4)
                                                                    ↓
                                              消融實驗 (Lab5, Extra A/B)
```

## 改進方向

若要突破 53% 上限：
1. 使用 **GPT-4o / Claude** 作為 teacher model 生成高品質答案（見 `generate_claude_answers.py`）
2. 以高品質答案重新 SFT，再做 DPO
3. Re-ranking 需先確保訓練 context 格式與推理 context 格式一致
