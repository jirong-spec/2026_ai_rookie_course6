"""
Extra Lab A - 錯誤分析 + Context 欄位消融
==========================================
Part 1: 從 benchmark 結果中篩出低分案例進行錯誤分類
Part 2: 比較 RAG_chunks / hybrid_chunks / chunk 對推理分數的影響
"""

import os
import sys
import json
import random
import time
import asyncio
import requests
import datetime
from tqdm import tqdm
from openai import AsyncOpenAI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))

# ==============================================================================
#                          設定區
# ==============================================================================

GURU_OUTPUT = os.path.join(SCRIPT_DIR, "..", "lab1", "output", "question_answer_40chunk_256.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
BENCHMARK_DIR = os.path.join(SCRIPT_DIR, "benchmark_results")

# 請填入 Lab4 的 benchmark JSON（含 rating 欄位）
LAB4_BENCHMARK_JSON = os.path.join(SCRIPT_DIR, "..", "lab4", "benchmark_results")  # 目錄或檔案

BASE_URL = "http://localhost:8299/v1"
MODEL_NAME = "Qwen2.5-3B-Instruct"

JUDGE_MODEL_NAME = "gpt-5.1"
GPT_API_URL = ""
GPT_USERNAME = ""
GPT_PASSWORD = ""
GPT_API_BASE = ""

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BENCHMARK_DIR, exist_ok=True)

# Prompt（與 Lab4 相同）
RAG_SYSTEM = "You are a helpful question answerer who can provide an answer given a question and relevant context."
RAG_USER = """Question: {}
Context: {}

Answer this question using the information given in the context above. Here is things to pay attention to:
- If you need to use CoT (Chain of Thought) reasoning, please do so.
- In the reasoning, if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##.
- The response should match the language of the given question.
- End your response with final answer in the form <ANSWER>: $answer."""


# ==============================================================================
#                          Part 1：錯誤分析
# ==============================================================================

def analyze_errors(benchmark_dir, threshold=2.0):
    """篩出低分案例供人工分類"""
    all_results = []
    for root, dirs, files in os.walk(benchmark_dir):
        for f in files:
            if f.endswith("_llamaindex.json"):
                with open(os.path.join(root, f), 'r', encoding='utf-8') as fp:
                    all_results.extend(json.load(fp))

    low_scores = [r for r in all_results if isinstance(r.get('rating'), (int, float)) and r['rating'] <= threshold]

    print(f"\n{'='*60}")
    print(f"錯誤分析：rating <= {threshold} 的案例（共 {len(low_scores)} 筆）")
    print('='*60)

    for i, r in enumerate(low_scores[:15], 1):
        print(f"\n--- 低分案例 {i} (rating={r.get('rating')}) ---")
        print(f"[問題] {r.get('question', '')[:150]}")
        print(f"[參考] {r.get('gt_answer', '')[:150]}")
        print(f"[模型] {r.get('model_answer', '')[:150]}")
        print(f"[評語] {r.get('feedback', '')[:200]}")

    print(f"\n請根據以上案例，在 error_analysis.md 中分類錯誤類型。")
    return low_scores


# ==============================================================================
#                          Part 2：Context 欄位消融
# ==============================================================================

def run_inference_with_key(guru_data, key, output_path):
    from openai_multi_client import OpenAIMultiClient

    async_client = AsyncOpenAI(api_key="empty", base_url=BASE_URL, timeout=1200)
    api = OpenAIMultiClient(
        async_client, concurrency=20,
        endpoint="chat.completions",
        data_template={"model": MODEL_NAME}
    )

    print(f"\n[推理] KEY={key}, 筆數={len(guru_data)}")

    def make_requests():
        for idx, data in enumerate(guru_data):
            try:
                context = str(data.get(key, []))
                api.request(data={
                    "messages": [
                        {"role": "system", "content": RAG_SYSTEM},
                        {"role": "user", "content": RAG_USER.format(data["question"], context[:5000])}
                    ],
                    "n": 1, "top_p": 1, "temperature": 0
                }, metadata={'data': data})
            except Exception as e:
                print(f"Error: {e}")

    api.run_request_function(make_requests)

    save_data = []
    with tqdm(total=len(guru_data), desc=f"推理 KEY={key}") as pbar:
        for result in api:
            if result.response:
                rd = result.metadata['data']
                for x in result.response.choices:
                    rd['predicted_answer'] = x.message.content
                    rd['test_chunk'] = key
                    save_data.append(rd)
            pbar.update(1)

    with open(output_path, 'w', encoding="utf-8") as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)
    return save_data


async def benchmark_one(data, label, output_dir, llm):
    from llama_index.core.evaluation import CorrectnessEvaluator, BatchEvalRunner

    correctness = CorrectnessEvaluator(llm=llm)
    batch_runner = BatchEvalRunner({"correctness": correctness}, workers=5, show_progress=True)

    eval_results = await batch_runner.aevaluate_response_strs(
        queries=[str(x['question']) for x in data],
        response_strs=[str(x['predicted_answer']) for x in data],
        reference=[str(x['base_answer']) for x in data]
    )

    scores = [cr.score for cr in eval_results['correctness'] if isinstance(cr.score, (int, float)) and cr.score >= 1.0]
    avg = sum(scores) / max(len(scores), 1)
    print(f"  [{label}] avg={avg:.2f}/5, n={len(scores)}")

    ts = datetime.datetime.now().strftime("%m%d_%H%M")
    with open(os.path.join(output_dir, f"{ts}_{label}.txt"), 'w') as f:
        f.write(f"{label}: avg={avg:.2f}/5, n={len(scores)}\n")
    return label, avg


# ==============================================================================
#                          主程式
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Extra Lab A：錯誤分析 + Context 欄位消融")
    print("=" * 60)

    # Part 1: 跳過（無 GPT benchmark 結果）
    print("\n[Part 1] 跳過：無 Lab4 benchmark 評分結果（需 GPT API）")

    # Part 2: Context 欄位消融推理
    with open(GURU_OUTPUT, 'r', encoding='utf-8') as f:
        guru_data = json.load(f)

    results = {}
    for key in ["hybrid_chunks", "chunk"]:
        path = os.path.join(OUTPUT_DIR, f"inference_{key}.json")
        data = run_inference_with_key(guru_data, key, path)
        results[key] = data

    # 並排顯示 5 筆
    import random as _r
    _r.seed(42)
    rag_data = json.load(open(os.path.join(SCRIPT_DIR, '..', 'lab4', 'output', 'finetuned_inference.json'), encoding='utf-8'))
    rag_dict = {d['question']: d.get('predicted_answer', '') for d in rag_data}
    hyb_dict = {d['question']: d.get('predicted_answer', '') for d in results.get('hybrid_chunks', [])}
    chk_dict = {d['question']: d.get('predicted_answer', '') for d in results.get('chunk', [])}
    qs = _r.sample([d['question'] for d in guru_data], min(5, len(guru_data)))

    print(f"\n{'='*70}")
    print("並排比較：RAG_chunks vs hybrid_chunks vs chunk（5 筆）")
    print('='*70)
    for i, q in enumerate(qs, 1):
        print(f"\n--- 第 {i} 筆 ---")
        print(f"[問題] {q}")
        print(f"[RAG_chunks]    {str(rag_dict.get(q,''))[:200]}")
        print(f"[hybrid_chunks] {str(hyb_dict.get(q,''))[:200]}")
        print(f"[chunk]         {str(chk_dict.get(q,''))[:200]}")

    print(f"\n[Benchmark] 跳過（無 GPT API 憑證）")
    print(f"\n{'='*60}")
    print("Extra A 完成！（Context 欄位消融推理）")
    print(f"  hybrid_chunks inference → {OUTPUT_DIR}/inference_hybrid_chunks.json")
    print(f"  chunk inference         → {OUTPUT_DIR}/inference_chunk.json")
    print("=" * 60)
