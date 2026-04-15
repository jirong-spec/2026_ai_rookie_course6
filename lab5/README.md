# Lab5：Prompt 與 Context 消融實驗

| 項目 | 內容 |
|------|------|
| 輸入 | Lab4 的 `finetuned_inference.json`、Lab1 的 `question_answer_40chunk_256.json` |
| 產出 | `simple_prompt_inference.json`、`no_context_inference.json`、`benchmark_results/`、`ablation_report.md` |
| GPU 需求 | vLLM（finetuned `Qwen2.5-3B-Instruct`，沿用 Lab4） |

## 目的

在 Lab4 已有 finetuned model 的基線分數後，做兩組**肉眼就能看出差異**的消融實驗：

1. **完整 prompt vs 極簡 prompt**：拿掉 CoT、引用、格式要求，觀察答案品質崩壞程度。
2. **有 context vs 完全不給 context**：只給問題不給 RAG 上下文，觀察模型是否大量幻覺。

## 學習目標

- 體會 **Prompt 設計**在推理端對輸出品質的巨大影響。
- 體會 **RAG context** 對模型回答準確性的關鍵作用。
- 從「肉眼看得出差異」到「用 benchmark 量化差異」的完整流程。

## 前置需求

- Lab4 已完成，vLLM 正在運行（finetuned `Qwen2.5-3B-Instruct`）。
- Lab4 的 benchmark 結果作為基線。

## 操作步驟

### 步驟 1 — 確認設定

打開 `lab5.py`，確認路徑與 GPT API 設定：

```python
GURU_OUTPUT = "../lab1/output/question_answer_40chunk_256.json"
LAB4_INFERENCE = "../lab4/output/finetuned_inference.json"
BASE_URL = "http://localhost:8299/v1"
MODEL_NAME = "Qwen2.5-3B-Instruct"
```

### 步驟 2 — 執行

```bash
cd lab5
uv run python lab5.py
```

程式會自動跑 B 和 D 兩組推理，然後跑 benchmark。

### 步驟 3 — 人工觀察

程式會在每組推理完後，自動挑 5 筆與 Lab4 結果並排比較。

#### 實驗一：完整 prompt（A） vs 極簡 prompt（B）

| 組別 | Prompt 內容 | 預期差異 |
|------|------------|---------|
| A（Lab4 已做） | 完整版：CoT + `##begin_quote##` 引用 + `<ANSWER>:` 結尾 | 有推理過程、有引用、有格式 |
| B（新跑） | 極簡版：「根據以下內容回答問題」 | 直接給答案，沒結構、沒引用 |

觀察題：
1. B 組的回答還有 `<ANSWER>:` 格式嗎？這說明了什麼？
2. B 組有沒有引用原文的行為？還是完全不引用？
3. 哪一組的答案你覺得對使用者更有用？

#### 實驗二：有 context（C） vs 不給 context（D）

| 組別 | 推理時給什麼 | 預期差異 |
|------|------------|---------|
| C（Lab4 已做） | `RAG_chunks`（完整 RAG 上下文） | 有依據地回答 |
| D（新跑） | **不帶 context**，只給 question | 模型靠記憶回答，大量幻覺或「不知道」 |

觀察題：
1. D 組的回答有沒有「捏造事實」（幻覺）？舉一個例子。
2. D 組有沒有出現「我無法回答」或類似拒答的情況？
3. 這個實驗說明了 RAG context 對模型回答的重要性有多大？

### 步驟 4 — 撰寫消融報告

在 `lab5/` 目錄下建立 `ablation_report.md`，使用以下模板：

```markdown
# Prompt 與 Context 消融報告

## 實驗分數總結

| 組別 | 描述 | 平均分（/5） |
|------|------|-------------|
| A（Lab4） | 完整 prompt + RAG context | ？ |
| B | 極簡 prompt + RAG context | ？ |
| D | 完整 prompt + 無 context | ？ |

## 實驗一分析：Prompt 設計的影響

（A vs B 的分數差距、肉眼觀察、結論）

## 實驗二分析：RAG Context 的影響

（A vs D 的分數差距、幻覺案例、結論）

## 綜合結論

（Prompt 設計和 Context 對最終品質各自的影響有多大？哪個更重要？）
```

## 輸出檔案

| 檔案 | 說明 |
|------|------|
| `simple_prompt_inference.json` | 極簡 prompt 推理結果 |
| `no_context_inference.json` | 無 context 推理結果 |
| `benchmark_results/` | 各組 benchmark 報告 |

## 繳交物

- `simple_prompt_inference.json`、`no_context_inference.json`（程式產出）
- `benchmark_results/`（程式產出）
- `ablation_report.md`（手動撰寫）

## 完成定義

- B 組和 D 組推理 + benchmark 已完成。
- 至少讀 10 筆人工觀察（每組 5 筆），回答觀察題。
- `ablation_report.md` 包含分數對比表、兩組實驗分析、綜合結論。
