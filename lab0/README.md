# Lab0：環境檢查

| 項目 | 內容 |
|------|------|
| 輸入 | 無 |
| 產出 | 終端機輸出（三項檢查結果） |
| GPU 需求 | vLLM（已在前面課程啟動） |

## 目的

確認本課程所需的三個關鍵環境元件可正常運作：

1. **vLLM**（`Qwen2.5-3B-Instruct`）— 前面課程已教過如何啟動，此處驗證連線。
2. **Docker + aiDAPTIV2 映像** — Lab3 微調需要。
3. **Embedding Model**（`intfloat/multilingual-e5-large`）— Lab1 RAG 檢索需要。

## 學習目標

- 確認 vLLM 端點可正常回應。
- 確認 Docker 環境就緒。
- 確認 embedding 模型可載入。
- 熟悉 `uv sync` 與 `uv run` 的使用方式。

## 操作步驟

### 步驟 1 — 安裝依賴

在**專案根目錄**（與 `pyproject.toml` 同層）執行：

```bash
uv sync
```

首次執行會建立 `.venv` 並安裝所有依賴（含 `torch`、`transformers`、`langchain`、`sentence-transformers` 等）。

### 步驟 2 — 啟動 vLLM

vLLM 的啟動方式與前面課程相同。若尚未啟動，在專案根目錄執行：

```bash
docker compose -f docker-compose-vllm.yaml up -d
```

啟動後端點為 `http://localhost:8299/v1`，模型為 `Qwen2.5-3B-Instruct`。

### 步驟 3 — 執行環境檢查

```bash
cd lab0
uv run python lab0.py
```

程式會依序檢查：

| 檢查項 | 驗證方式 | 預期結果 |
|--------|---------|---------|
| vLLM | 對 `http://localhost:8299/v1` 發送一筆 chat 請求 | 印出模型回應文字 |
| Docker | 執行 `docker --version` | 印出 Docker 版本 |
| Embedding | 載入 `intfloat/multilingual-e5-large` 並對測試文字做 embedding | 印出向量維度（1024） |

### 步驟 4 — 確認 aiDAPTIV2 映像

（選做）確認 aiDAPTIV2 Docker 映像是否已存在：

```bash
docker images | grep aidaptiv
```

## 完成定義

- 三項檢查皆顯示 `✓ 通過`。
- 若有未通過項目，依終端機錯誤訊息排除後重試。
