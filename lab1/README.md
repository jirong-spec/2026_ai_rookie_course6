# Lab1：Guru 全管線 + 人工觀察

| 項目 | 內容 |
|------|------|
| 輸入 | 助教提供的 PDF/DOCX 來源資料夾 |
| 產出 | `chunkfile_256.json`、`question_256.json`、`question_answer_256.json`、`question_answer_40chunk_256.json`、`observation.md` |
| GPU 需求 | vLLM（`Qwen2.5-3B-Instruct`） |

## 目的

一口氣跑完 **Guru** 的完整 QA 生成管線（PDF→TXT→Chunk→問題→答案→RAG 檢索），並在每個階段停下來**人工觀察產物品質**，建立對資料品質的直覺。

## 學習目標

- 理解 Guru 管線五個階段的輸入、輸出與作用。
- 能從 Chunk、Question、Answer、RAG 產物中，以肉眼判斷品質好壞。
- 理解 `chunk_size`、`chunk_overlap` 對分塊結果的影響。
- 理解問題生成與答案生成的 Prompt 設計意圖（為什麼要求 CoT、引用標記、`<ANSWER>:` 格式）。
- 理解 RAG 模擬（golden chunk 混合 + similarity search）的目的。

## 前置需求

- Lab0 環境檢查通過（`uv sync` 完成、vLLM 可連線、embedding model 可載入）。
- 助教已將範例 PDF 放在指定來源資料夾。

## 操作步驟

### 步驟 1 — 確認設定

打開 `lab1.py`，確認以下參數（通常只需改 `SOURCE_FOLDER` 和 `OUTPUT_FOLDER`）：

```python
ENDPOINT = "http://localhost:8299/v1"      # vLLM 端點（與 Lab0 相同）
MODELNAME = "Qwen2.5-3B-Instruct"          # 模型名稱
SOURCE_FOLDER = "/path/to/source"           # TODO: 改為助教提供的 PDF/DOCX 來源資料夾
OUTPUT_FOLDER = "/path/to/output"           # TODO: 改為你的輸出資料夾
MAX_QUESTION = 100                          # 生成問題數量上限
CHUNK_SIZE = 256                            # 分塊大小（token 數）
```

### 步驟 2 — 執行

```bash
cd lab1
uv run python lab1.py
```

程式會依序執行五個階段，每個階段完成後會印出隨機幾筆範例供觀察。

### 步驟 3 — 每階段觀察

程式執行過程中，在每個階段完成後回答以下問題：

#### 觀察點 A：Chunk 產出後（`chunkfile_256.json`）
1. 隨機看 3 筆 chunk，內容是否完整？有沒有斷在句子中間？
2. 模糊指代（如 "this paper"、"the study"）是否已被替換成文件標題？
3. 用 `chunk_size=256` 總共產出幾個 chunk？你覺得這個數量合理嗎？

#### 觀察點 B：問題產出後（`question_256.json`）
1. 隨機看 5 筆 question，是有意義且可回答的問題嗎？
2. 有沒有太籠統的問題（如「這段在說什麼？」）或直接抄原文當問句的情況？
3. 語言正確嗎？（來源是中文文件，問題也應為中文）
4. 你覺得哪些問題品質最好？為什麼？

#### 觀察點 C：答案產出後（`question_answer_256.json`）
1. 隨機看 5 筆答案，有沒有 CoT（Chain of Thought）推理過程？
2. 有沒有 `##begin_quote##` ... `##end_quote##` 引用標記？引用的內容是否來自 context？
3. 結尾是否有 `<ANSWER>:` 格式？精簡答案與詳細回答是否一致？
4. 答案語言是否與問題一致？

#### 觀察點 D：RAG 混合後（`question_answer_40chunk_256.json`）
1. 看 `rag_acc` 欄位：golden chunk 被檢索回來的比例多高？
2. 混入的干擾段（RAG 檢索到但非 golden chunk 的段落）跟問題主題有關嗎？
3. `hybrid_chunks` 共有幾段？其中有多少是相關的、多少是噪音？

### 步驟 4 — 撰寫觀察紀錄

在 `lab1/` 目錄下建立 `observation.md`，依上述 A-D 四個觀察點逐一作答。

## 輸出檔案

| 檔案 | 說明 |
|------|------|
| `chunkfile_256.json` | 文本分塊結果 |
| `question_256.json` | 生成的問題 |
| `question_answer_256.json` | 問題與答案 |
| `question_answer_40chunk_256.json` | 最終 RAG 增強結果（**後續 Lab 都用這份**） |

## 繳交物

- `question_answer_40chunk_256.json`（程式產出）
- `observation.md`（手動撰寫，含 A-D 觀察）

## 完成定義

- `question_answer_40chunk_256.json` 已產出且筆數合理（接近 `MAX_QUESTION`）。
- `observation.md` 已撰寫，四個觀察點皆有具體回答。
