# Lab2：Base 模型推理 + 資料轉換

| 項目 | 內容 |
|------|------|
| 輸入 | Lab1 的 `question_answer_40chunk_256.json` |
| 產出 | `baseline_inference.json`、`train.json`、`test.json` |
| GPU 需求 | vLLM（`Qwen2.5-3B-Instruct`，base model） |

## 目的

趁 vLLM **還開著**（之後要關掉做 Finetune），一次完成兩件事：

1. 用**未微調的 base model**（`Qwen2.5-3B-Instruct`）對資料跑批量推理，存檔備用。
2. 將 Lab1 的 Guru 產物轉為 **aiDAPTIV2 訓練格式**（`{question, answer}`），切出 train / test。

## 學習目標

- 理解推理時的 Prompt 模板（System Prompt + User Prompt + RAG context）。
- 理解 Guru 產物欄位（`question`、`base_answer`、`RAG_chunks`、`hybrid_chunks`、`chunk`）與 aiDAPTIV2 所需格式（`question`、`answer`）的差異。
- 練習資料格式轉換與 train/test 切分。
- 觀察 base model（未微調）的推理品質，為 Lab4 比較做準備。

## 前置需求

- Lab1 已完成，有 `question_answer_40chunk_256.json`。
- vLLM 正在運行（`Qwen2.5-3B-Instruct` base model）。

## 操作步驟

### 步驟 1 — 修改路徑

打開 `lab2.py`，修改 `GURU_OUTPUT` 指向你 Lab1 的產物：

```python
GURU_OUTPUT = "../lab1/output/question_answer_40chunk_256.json"
```

其他設定已預設為 `Qwen2.5-3B-Instruct`，通常不需修改。

### 步驟 2 — 執行

```bash
cd lab2
uv run python lab2.py
```

程式會依序：
1. 對 Guru 產物用 base model 跑推理 → `baseline_inference.json`
2. 將 Guru 產物轉為 aiDAPTIV2 格式
3. 隨機切分 80% train + 20% test

### 步驟 3 — 決定資料轉換策略（TODO）

`lab2.py` 的 `convert_for_aidaptiv` 函式有兩個需要你決定的 TODO：

**TODO A — answer 取法**（建議先用預設：取 `base_answer` 全文）

| 選項 | 做法 | 優缺點 |
|------|------|--------|
| 全文（預設） | 直接取 `base_answer` | 保留 CoT 推理過程，訓練資料較豐富 |
| 精簡 | 只取 `<ANSWER>:` 後面的內容 | 答案簡潔，但失去推理過程 |

**TODO B — question 是否帶 context**（建議先用預設：不帶）

| 選項 | 做法 | 優缺點 |
|------|------|--------|
| 不帶（預設） | question 只放原始問題 | 模型學「看到問題就回答」，推理時再給 context |
| 帶 context | question = 問題 + RAG_chunks | 模型學「看到 context + 問題才回答」，跟推理格式一致 |

請在程式碼中做出選擇，並在 TODO 註解旁寫上你的理由。

### 步驟 4 — 觀察 base 推理結果

打開 `baseline_inference.json`，隨機看 5 筆：
1. `predicted_answer`（base model 的回答）品質如何？
2. 跟 Guru 的 `base_answer`（參考答案）比起來，差距大嗎？
3. 你覺得哪些類型的問題 base model 答得最差？

## 輸出檔案

| 檔案 | 說明 | 後續誰用 |
|------|------|---------|
| `baseline_inference.json` | Base model 推理結果 | Lab3（benchmark）、Lab4（比較） |
| `train.json` | aiDAPTIV2 訓練資料 | Lab3（Finetune） |
| `test.json` | 測試集 | Lab4（Finetuned 推理） |

## 繳交物

- `baseline_inference.json`、`train.json`、`test.json`（程式產出）

## 完成定義

- `baseline_inference.json` 已產出，每筆有 `predicted_answer` 欄位。
- `train.json` / `test.json` 格式為 `{question, answer}`，可被 aiDAPTIV2 讀取。
- 能說出你的 answer 取法與是否帶 context 的選擇及理由。
