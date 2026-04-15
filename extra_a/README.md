# Extra Lab A：錯誤分析 + Context 欄位消融（選做）

| 項目 | 內容 |
|------|------|
| 輸入 | Lab4/Lab5 的 benchmark JSON、Lab1 的 `question_answer_40chunk_256.json` |
| 產出 | `error_analysis.md`、`inference_hybrid_chunks.json`、`inference_chunk.json`、`benchmark_results/` |
| GPU 需求 | vLLM（finetuned `Qwen2.5-3B-Instruct`，Part 2 需要） |
| 性質 | 選做；不列入成績，給進度較快、想延伸研究與挑戰的同學 |

## 目的

深入分析模型「為什麼答錯」，並比較不同 context 欄位（`RAG_chunks` / `hybrid_chunks` / `chunk`）對分數的細微影響。

## 學習目標

- 學會從 benchmark 低分案例中**分類錯誤類型**。
- 理解 RAG 檢索品質（`RAG_chunks`）、混合策略（`hybrid_chunks`）、原始段落（`chunk`）之間的差異。
- 將錯誤分析與 context 品質做交叉比對。

## Part 1：錯誤分析（不需 GPU）

### 步驟

1. 執行 `uv run python extra_a.py`，程式會自動從 Lab4 benchmark 結果中篩出 **rating <= 2** 的低分案例並印出。
2. 逐筆閱讀：`question`、`gt_answer`（參考答案）、`model_answer`、`feedback`（GPT 評語）。
3. 將錯誤歸類：

| 錯誤類型 | 說明 | 筆數 |
|----------|------|------|
| 幻覺 | 模型捏造了 context 中沒有的資訊 | ？ |
| 答非所問 | 模型沒抓到問題重點 | ？ |
| 語言錯誤 | 問中文卻回英文（或反過來） | ？ |
| 格式不符 | 沒有 `<ANSWER>:` 結尾 | ？ |
| Context 不足 | RAG 沒撈到相關段落 | ？ |
| 其他 | | ？ |

4. 統計各類型佔比，並各貼 1-2 筆典型案例。

### 產出

在 `extra_a/` 下建立 `error_analysis.md`，包含：
- 錯誤類型統計表（如上）
- 各類型的典型案例
- 你的改善建議（如果要再改一版 prompt 或資料，你會怎麼改？）

## Part 2：Context 欄位消融（需 vLLM）

用 finetuned model（`Qwen2.5-3B-Instruct`），只改 context 欄位：

| 組別 | KEY | 意義 |
|------|-----|------|
| Lab4 已做 | `RAG_chunks` | 純 RAG 檢索結果（含噪音） |
| 新跑 | `hybrid_chunks` | Golden chunk + RAG 混合（保底有正確段落） |
| 新跑 | `chunk` | 只有原始 golden chunk（最乾淨但最少） |

每組跑 inference + benchmark，比較分數差異。

## 執行

```bash
cd extra_a
uv run python extra_a.py
```

## 繳交物

- `error_analysis.md`（手動撰寫）
- 三組 context 欄位的 benchmark 分數比較表（寫在 `error_analysis.md` 中）

## 完成定義

- `error_analysis.md` 包含錯誤分類統計與改善建議。
- 三組 context 欄位的 benchmark 分數比較。
- 結合兩者的分析：錯誤類型與 context 品質之間有沒有關聯？
