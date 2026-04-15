# Lab3：Finetune（aiDAPTIV2）

| 項目 | 內容 |
|------|------|
| 輸入 | Lab2 的 `train.json` |
| 產出 | `output_model/`（模型權重） |
| GPU 需求 | aiDAPTIV2 |

## 目的

使用 **aiDAPTIV2** 在 Docker 容器內，對 `Qwen2.5-3B-Instruct` 進行一輪指令微調（SFT）。

## 學習目標

- 熟悉 Docker Compose 啟動服務、進入容器、在容器內執行訓練指令的完整順序。
- 能對照三份 yaml 設定檔理解「環境／實驗／資料」分層設定。

## 前置需求

- Lab2 已完成，有 `train.json`。
- **vLLM 已關閉**（`docker compose -f docker-compose-vllm.yaml down`），GPU 可供 Finetune 使用。
- Docker 已安裝，aiDAPTIV2 映像已就緒。

---

## 操作步驟

### 步驟 1 — 複製訓練資料

將 Lab2 產出的 `train.json` 複製到本目錄：

```bash
cp ../lab2/output/train.json ./train.json
```

### 步驟 2 — 確認設定檔

本目錄下有三份 yaml，請檢查：

| 檔案 | 管什麼 | 關鍵設定 |
|------|--------|---------|
| `env_config.yaml` | 模型路徑、輸出路徑 | `model_name_or_path: "/mnt/model/Qwen2.5-3B-Instruct"` |
| `exp_config.yaml` | 訓練超參數 | `num_train_epochs: 3`、`learning_rate: 0.00005`、`per_device_train_batch_size: 1` |
| `QA_dataset_config.yaml` | 資料路徑與欄位 | `data_path: "/workspace/train.json"`、`question_key: "question"`、`answer_key: "answer"` |

> 通常不需修改，但如果你在 Lab2 改了欄位名稱，請同步修改 `QA_dataset_config.yaml`。

### 步驟 3 — 啟動容器

```bash
cd lab3_finetune
docker compose up -d
docker compose ps    # 確認 aidaptiv_fine_tune 為 running
```

### 步驟 4 — 進入容器並執行訓練

```bash
docker compose exec aidaptiv_fine_tune bash
```

在容器內：

```bash
cp /workspace/env_config.yaml /home/root/aiDAPTIV2/commands/env_config/
cp /workspace/exp_config.yaml /home/root/aiDAPTIV2/commands/exp_config/
cp /workspace/QA_dataset_config.yaml /home/root/aiDAPTIV2/commands/dataset_config/text-generation/

cd /home/root/aiDAPTIV2/commands/
phisonai2 --env_config ./env_config/env_config.yaml --exp_config ./exp_config/exp_config.yaml
```

### 步驟 5 — 確認產物

訓練完成後，確認 `/workspace/output_model`（即本目錄的 `output_model/`）有模型權重檔。

---

## 常見問題

| 現象 | 可能原因與處理 |
|------|----------------|
| `Cannot connect to the Docker daemon` | Docker 未啟動；請開啟 Docker Desktop。 |
| `service ... is not running` | 未執行 `docker compose up -d`；查看 `docker compose logs`。 |
| `phisonai2` 找不到資料 | 檢查 `QA_dataset_config.yaml` 內路徑是否在容器內存在（應為 `/workspace/train.json`）。 |

## 繳交物

- `output_model/` 目錄（含模型權重，Finetune 產出）

## 完成定義

- Finetune 跑完，`output_model/` 有模型權重。
