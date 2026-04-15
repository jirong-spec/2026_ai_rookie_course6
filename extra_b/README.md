# Extra Lab B：Guru 參數消融 + 二次 Finetune（選做）

| 項目 | 內容 |
|------|------|
| 輸入 | Lab1 的 `docs_txt/`（已轉好的文字檔）、Lab4 基線分數 |
| 產出 | 新 Guru 產物、`train_v2.json`、`test_v2.json`、新 finetuned model、`extra_b_finetuned_inference.json`、`ab_comparison_report.md` |
| GPU 需求 | vLLM（重跑 Guru）+ Finetune（第二次 aiDAPTIV2）+ vLLM（推理） |
| 性質 | 選做；不列入成績，給進度較快、想延伸研究與挑戰的同學 |

## 目的

改變 Guru 管線中的**一個變數**，走完「重新產 QA → 人工觀察 → Finetune → Inference → Benchmark」的完整 A/B 對照循環。

## 學習目標

- 體會「改一個上游參數，下游效果可能完全不同」的連鎖效應。
- 練習完整的 A/B 實驗設計：控制變數、觀察產物、量化結果。

## 選擇一個變數（擇一即可）

| 選項 | 改什麼 | 需重跑的範圍 |
|------|--------|-------------|
| A | `chunk_size` 改 1024（原為 256） | Guru 全線（分塊→問題→答案→RAG） |
| B | 問題生成 prompt（例如要求更具體的問題） | 從問題生成開始 |
| C | 答案生成 prompt（例如不要求 CoT） | 從答案生成開始 |

## 操作步驟

### 步驟 1 — 改參數，重跑 Guru（需 vLLM）

修改 `extra_b.py` 中對應的變數（預設為選項 A：`CHUNK_SIZE = 1024`），然後執行：

```bash
cd extra_b
uv run python extra_b.py
```

### 步驟 2 — 人工觀察（不用等 benchmark）

程式會自動將新舊 QA 並排比較。**先觀察**：
1. 新的 chunk 跟 Lab1 的比起來，內容完整度如何？
2. 新的問題/答案品質有差嗎？
3. 你預期 Finetune 後分數會更高還是更低？

### 步驟 3 — 轉換格式 + Finetune

```bash
uv run python convert_and_split.py
cp output/train_v2.json ../lab3_finetune/train.json

cd ../lab3_finetune
docker compose up -d
docker compose exec aidaptiv_fine_tune bash
# 在容器內執行訓練指令（同 Lab3）
```

### 步驟 4 — Inference + Benchmark

Finetune 完成後，部署新 finetuned model 至 vLLM，然後：

```bash
cd extra_b
uv run python run_inference.py
uv run python run_benchmark.py
```

### 步驟 5 — 撰寫 A/B 對照報告

在 `extra_b/` 下建立 `ab_comparison_report.md`：

```markdown
# A/B 對照報告

## 實驗設計

| | Lab4 基線 | Extra B |
|--|----------|---------|
| 改了什麼 | chunk=256, 原 prompt | （你的改變） |

## 人工觀察（Guru 產物比較）

（新舊 QA 品質差異）

## 量化比較

| 指標 | Lab4 基線 | Extra B |
|------|----------|---------|
| Benchmark 平均分 | ？ | ？ |

## 結論

（改善/退步？為什麼？如果要再改一次，你會怎麼做？）
```

## 繳交物

- 新 Guru 產物（程式產出）
- `train_v2.json`、`test_v2.json`（程式產出）
- `extra_b_finetuned_inference.json`（程式產出）
- `benchmark_results/`（程式產出）
- `ab_comparison_report.md`（手動撰寫）

## 完成定義

- 新的 Guru 產物 + 人工觀察（與 Lab1 比較）。
- 新的 Finetune + Inference + Benchmark 完成。
- `ab_comparison_report.md` 包含完整 A/B 對照分析。
