"""
Extra Lab B - Benchmark
"""

import os
import json
import time
import asyncio
import requests
import datetime
from llama_index.core.evaluation import CorrectnessEvaluator
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import BatchEvalRunner

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
BENCHMARK_DIR = os.path.join(SCRIPT_DIR, "benchmark_results")
os.makedirs(BENCHMARK_DIR, exist_ok=True)

INFERENCE_RESULT = os.path.join(OUTPUT_DIR, "extra_b_finetuned_inference.json")

JUDGE_MODEL_NAME = "gpt-5.1"
GPT_API_URL = ""        # TODO: 填入
GPT_USERNAME = ""       # TODO: 填入
GPT_PASSWORD = ""       # TODO: 填入
GPT_API_BASE = ""       # TODO: 填入


async def run_eval(llm, data):
    correctness = CorrectnessEvaluator(llm=llm)
    batch_runner = BatchEvalRunner({"correctness": correctness}, workers=5, show_progress=True)

    results = await batch_runner.aevaluate_response_strs(
        queries=[str(x['question']) for x in data],
        response_strs=[str(x['predicted_answer']) for x in data],
        reference=[str(x['base_answer']) for x in data]
    )

    scores = [cr.score for cr in results['correctness'] if isinstance(cr.score, (int, float)) and cr.score >= 1.0]
    avg = sum(scores) / max(len(scores), 1)

    print(f"\nExtra B Benchmark: avg={avg:.2f}/5 ({avg*20:.1f}/100), n={len(scores)}")

    ts = datetime.datetime.now().strftime("%m%d_%H%M")
    with open(os.path.join(BENCHMARK_DIR, f"{ts}_extra_b.txt"), 'w') as f:
        f.write(f"Extra B: avg={avg:.2f}/5, n={len(scores)}\n")

    output_datas = []
    for cr in results['correctness']:
        output_datas.append({
            "question": cr.query, "model_answer": cr.response,
            "feedback": cr.feedback, "rating": cr.score,
        })
    with open(os.path.join(BENCHMARK_DIR, f"{ts}_extra_b_llamaindex.json"), 'w', encoding='utf-8') as f:
        json.dump(output_datas, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    print("Extra Lab B Benchmark")

    headers = {'Content-Type': 'application/json'}
    resp = requests.post(GPT_API_URL, headers=headers, data=json.dumps({"username": GPT_USERNAME, "password": GPT_PASSWORD}))
    OPENAI_KEY = resp.json()['token']

    llm = OpenAI(JUDGE_MODEL_NAME, api_key=OPENAI_KEY, api_base=GPT_API_BASE, temperature=0, n=1, top_p=0.00001)

    with open(INFERENCE_RESULT, 'r', encoding='utf-8') as f:
        data = json.load(f)

    asyncio.run(run_eval(llm, data))
    print(f"\n請撰寫 ab_comparison_report.md（與 Lab4 基線比較）。")
