# Guru QA 資料生成、微調與驗證：從文件到模型的完整實戰

## 課程目標

完成本課程後，你應該能夠：

- 使用 **Guru** 工具將 PDF/DOCX 文件轉換為結構化 QA 訓練資料（含 RAG 檢索增強）。
- 理解 Guru 管線中每個階段（分塊、問題生成、答案生成、RAG 模擬）的產物與品質觀察方式。
- 使用 **aiDAPTIV2** 對 `Qwen2.5-3B-Instruct` 進行指令微調（SFT）。
- 使用批量推理工具對測試集推理，並比較 base model 與 finetuned model 的輸出差異。
- 使用 **LlamaIndex CorrectnessEvaluator** 搭配 GPT 評審模型，對推理結果進行 1-5 分品質評估。
- 透過消融實驗（改變 Prompt、Context 等變數），觀察並量化不同因素對最終輸出品質的影響。

## 模型規格

本課程統一使用以下模型：

| 用途 | 模型 | 說明 |
|------|------|------|
| Guru 生題生答 / Inference 推理 | `Qwen/Qwen2.5-3B-Instruct` | 透過 vLLM 以 OpenAI 相容 API 提供 |
| aiDAPTIV2 微調 base model | `Qwen/Qwen2.5-3B-Instruct` | 主機路徑 `/home/user/model/Qwen2.5-3B-Instruct` |
| RAG 向量檢索 embedding | `intfloat/multilingual-e5-large` | sentence-transformers |
| Benchmark 評審 | `gpt-5.1` | 打外部 GPT API（助教提供帳號） |

## 環境與依賴（uv）

本專案以 **[uv](https://github.com/astral-sh/uv)** 管理 Python 版本與套件。依賴宣告在專案根目錄的 `pyproject.toml`。

1. **安裝 uv**：見官方文件 [Installing uv](https://docs.astral.sh/uv/getting-started/installation/)。
2. **在專案根目錄同步依賴**（會建立 `.venv` 並安裝所有套件）：
   ```bash
   uv sync
   ```
3. **執行各 Lab 腳本**：在專案根目錄使用 `uv run`（會自動使用虛擬環境，無須先 `activate`）。各 Lab 的具體指令見各資料夾內 `README.md`。

### 其他環境需求

- **poppler**（PDF 解析需要）：`sudo apt-get install poppler-utils`
- **Docker**（aiDAPTIV2 微調需要）：Docker Desktop 或等效。
- **vLLM 服務**：前面課程已教過如何用 Docker 啟動 vLLM，本課程沿用相同方式（見 Lab0）。
- **GPT 評審 API**（LlamaIndex Benchmark 需要）：助教會提供帳號與端點。

## vLLM 啟動方式

與前面課程相同，在專案根目錄執行：

```bash
docker compose -f docker-compose-vllm.yaml up -d
```

啟動後 vLLM 端點為 `http://localhost:8299/v1`，served model name 為 `Qwen2.5-3B-Instruct`。

## 實作 Lab 概觀

| Lab | 主題 | 目標 | GPU 需求 |
| --- | ---- | ---- | -------- |
| Lab0 | 環境檢查 | 確認 vLLM、Docker、embedding model、uv 可用 | vLLM |
| Lab1 | Guru 全管線 + 觀察 | 跑完 Guru 全流程，人工觀察每階段產物品質 | vLLM |
| Lab2 | Base 推理 + 資料轉換 | 用 base model 推理並存檔；將 Guru 產物轉為 aiDAPTIV2 訓練格式 | vLLM |
| Lab3 | Finetune（aiDAPTIV2） | 使用 aiDAPTIV2 對模型進行指令微調 | Finetune |
| Lab4 | 推理 + Benchmark + 比較 | Base / Finetuned 推理 + benchmark，完整比較 | vLLM |
| Lab5 | Prompt / Context 消融 | 改推理 prompt、移除 context，觀察肉眼可見的差異並量化 | vLLM |
| Extra A | 錯誤分析 + 欄位消融 | 分類失敗案例；比較 RAG_chunks / hybrid_chunks / chunk | vLLM 或無 |
| Extra B | Guru 參數消融 + 二次 Finetune | 改 chunk_size 或 prompt 重跑 Guru → 訓練 → 推理 → 比較 | vLLM + Finetune |

**Extra Lab** 為選做，不列入本課程成績；提供給進度較快、想延伸研究與挑戰的同學，詳見各 `extra_*/README.md`。

## 完整工作流程

```
1. guru (Lab1)          2. inference (Lab2/4)   3. benchmark (Lab4)
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────────┐
│ PDF/DOCX 文件    │ →  │ QA 資料          │ →  │ 推理結果             │ → 品質評估報告
│                  │    │ (含 RAG context) │    │ (predicted_answer)   │
└──────────────────┘    └──────────────────┘    └──────────────────────┘
        ↓                       ↓
   convert (Lab2)        aiDAPTIV2 微調 (Lab3)
```
