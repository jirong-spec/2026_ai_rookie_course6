# Lab4：推理 + Benchmark + 比較報告

| 項目 | 內容 |
|------|------|
| 輸入 | Lab3 的 finetuned model 權重、Lab2 的 `baseline_inference.json`、Lab1 的 `question_answer_40chunk_256.json` |
| 產出 | `finetuned_inference.json`、`benchmark_results/`（base + finetuned）、`comparison_report.md` |
| GPU 需求 | vLLM（載入 finetuned `Qwen2.5-3B-Instruct`） |

## 目的

用 Lab3 訓練好的 **finetuned model** 對同一份資料推理，同時對 base model 與 finetuned model 的推理結果都跑 benchmark，然後做完整比較。

## 學習目標

- 部署 finetuned model 至 vLLM，執行批量推理。
- 理解 LlamaIndex `CorrectnessEvaluator` 的 1-5 分評分標準。
- 以**人工觀察**（肉眼比對）與**量化指標**（benchmark 分數）兩種方式，比較 base vs finetuned。
- 撰寫有論據的比較報告。

## 前置需求

- Lab3 Finetune 已完成，有 finetuned model 權重（`lab3_finetune/output_model/`）。
- GPU 已從 Finetune 模式切回，**vLLM 已重新啟動並載入 finetuned model**。

## 操作步驟

### 步驟 1 — 啟動 vLLM（載入 finetuned model）

修改 `docker-compose-vllm.yaml` 中的 `--model` 路徑，指向 Lab3 產出的 finetuned model，然後啟動：

```bash
docker compose -f docker-compose-vllm.yaml up -d
```

確認模型已載入：

```bash
curl http://localhost:8299/v1/models
```

### 步驟 2 — 修改設定並執行

打開 `lab4.py`，確認路徑與 GPT API 設定：

```python
GURU_OUTPUT = "../lab1/output/question_answer_40chunk_256.json"
BASELINE_INFERENCE = "../lab2/output/baseline_inference.json"
BASE_URL = "http://localhost:8299/v1"
MODEL_NAME = "Qwen2.5-3B-Instruct"

# TODO: 填入 GPT API 設定（助教提供）
GPT_API_URL = ""
GPT_USERNAME = ""
GPT_PASSWORD = ""
GPT_API_BASE = ""
```

```bash
cd lab4
uv run python lab4.py
```

### 步驟 3 — 程式流程

`lab4.py` 會依序執行：

1. **Finetuned 推理**：用 finetuned model 對 Guru 產物做批量推理
2. **人工觀察**：自動挑出 10 筆，將 base model 與 finetuned model 的回答並排顯示
3. **Base Benchmark**：對 Lab2 的 `baseline_inference.json` 跑 LlamaIndex benchmark（打外部 GPT API）
4. **Finetuned Benchmark**：對 finetuned 推理結果跑同樣的 benchmark

### 步驟 4 — 人工觀察

程式印出的並排比較中，回答以下問題：
1. Finetuned 的回答整體比 base 好嗎？差距明顯嗎？
2. 哪些類型的題目改善最明顯？
3. 有沒有反而退步的案例？

### 步驟 5 — 撰寫比較報告

在 `lab4/` 目錄下建立 `comparison_report.md`，使用以下模板：

```markdown
# Base vs Finetuned 比較報告

## 量化比較

| 比較項目 | Base Model | Finetuned Model |
|----------|-----------|----------------|
| 平均分數（/5） | ？ | ？ |
| 平均分數（/100） | ？ | ？ |
| 有效筆數 | ？ | ？ |

## 分數分布

| 分數區間 | Base | Finetuned |
|----------|------|-----------|
| 1.0 | ？ | ？ |
| 2.0 | ？ | ？ |
| 3.0 | ？ | ？ |
| 4.0 | ？ | ？ |
| 5.0 | ？ | ？ |

## 典型案例

### 改善案例
- 問題：...
- Base 回答：...
- Finetuned 回答：...
- 改善原因：...

### 退步案例（若有）
- 問題：...
- Base 回答：...
- Finetuned 回答：...
- 退步原因：...

## 結論

（Finetune 有沒有用？改善了什麼？為什麼？還有哪些不足？）
```

## 輸出檔案

| 檔案 | 說明 |
|------|------|
| `finetuned_inference.json` | Finetuned model 推理結果 |
| `benchmark_results/` | 包含 base 與 finetuned 兩組 benchmark 報告 |

## 繳交物

- `finetuned_inference.json`（程式產出）
- `benchmark_results/`（程式產出，含 base + finetuned）
- `comparison_report.md`（手動撰寫）

## 完成定義

- Finetuned inference + 兩組 benchmark 已完成。
- `comparison_report.md` 包含量化比較表、典型案例、結論。
