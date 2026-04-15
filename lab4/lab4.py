"""
Lab4 - Finetuned 推理 + 比較報告
=================================
用 finetuned model 推理 → benchmark → 與 base model 並排比較。
"""

import os
import json
import random
import time
import asyncio
import requests
import datetime
from tqdm import tqdm
from openai import AsyncOpenAI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
#                          設定區
# ==============================================================================

GURU_OUTPUT = os.path.join(SCRIPT_DIR, "..", "lab1", "output", "question_answer_40chunk_256.json")
BASELINE_INFERENCE = os.path.join(SCRIPT_DIR, "..", "lab2", "output", "baseline_inference.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
BENCHMARK_DIR = os.path.join(SCRIPT_DIR, "benchmark_results")

BASE_URL = "http://localhost:8299/v1"
MODEL_NAME = "Qwen2.5-3B-Instruct"

JUDGE_MODEL_NAME = "gpt-5.1"
GPT_API_URL = ""        # TODO: 填入 GPT API 地址
GPT_USERNAME = ""       # TODO: 填入帳號
GPT_PASSWORD = ""       # TODO: 填入密碼
GPT_API_BASE = ""       # TODO: 填入 GPT API base URL

KEY = "RAG_chunks"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BENCHMARK_DIR, exist_ok=True)

# ==============================================================================
#                          Prompt 模板
# ==============================================================================

RAG_SYSTEM_PROMPT = "You are a helpful question answerer who can provide an answer given a question and relevant context."

RAG_USER_PROMPT = """Question: {}
Context: {}

Answer this question using the information given in the context above. Here is things to pay attention to:
- If you need to use CoT (Chain of Thought) reasoning, please do so. If the question is simple and does not require CoT, then do not use it.
- In the reasoning, if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##.
- The response should match the language of the given question.
- End your response with final answer in the form <ANSWER>: $answer, the answer should be succinct."""


# ==============================================================================
#                          Part 1：Finetuned 推理
# ==============================================================================

def run_finetuned_inference(guru_data, output_path):
    from openai_multi_client import OpenAIMultiClient

    async_client = AsyncOpenAI(api_key="empty", base_url=BASE_URL, timeout=1200)
    api = OpenAIMultiClient(
        async_client, concurrency=20,
        endpoint="chat.completions",
        data_template={"model": MODEL_NAME}
    )

    print(f"[Finetuned 推理] 端點: {BASE_URL}")
    print(f"[Finetuned 推理] 模型: {MODEL_NAME}")
    print(f"[Finetuned 推理] 資料筆數: {len(guru_data)}")

    def make_requests():
        for index, data in enumerate(guru_data):
            try:
                api.request(data={
                    "messages": [
                        {"role": "system", "content": RAG_SYSTEM_PROMPT},
                        {"role": "user", "content": RAG_USER_PROMPT.format(data["question"], str(data[KEY]))}
                    ],
                    "n": 1, "top_p": 1, "temperature": 0
                }, metadata={'data': data})
            except Exception as e:
                print(f"Error at index {index}: {e}")

    api.run_request_function(make_requests)

    save_data = []
    with tqdm(total=len(guru_data), desc="Finetuned 推理") as pbar:
        for result in api:
            if result.response:
                result_data = result.metadata['data']
                for x in result.response.choices:
                    result_data['predicted_answer'] = x.message.content
                    result_data['test_chunk'] = KEY
                    save_data.append(result_data)
            pbar.update(1)

    with open(output_path, 'w', encoding="utf-8") as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)

    print(f"[Finetuned 推理] 完成: {output_path}")
    return save_data


# ==============================================================================
#                          Part 2：人工觀察（並排比較）
# ==============================================================================

def side_by_side_comparison(baseline_data, finetuned_data, n=5):
    """隨機挑 n 筆，並排顯示 base vs finetuned 回答"""
    base_dict = {d['question']: d.get('predicted_answer', '') for d in baseline_data}

    matched = [d for d in finetuned_data if d['question'] in base_dict]
    samples = random.sample(matched, min(n, len(matched)))

    print(f"\n{'='*70}")
    print(f"人工觀察：Base vs Finetuned 並排比較（{len(samples)} 筆）")
    print('='*70)

    for i, item in enumerate(samples, 1):
        q = item['question']
        base_ans = base_dict.get(q, "N/A")
        ft_ans = item.get('predicted_answer', 'N/A')
        gt = item.get('base_answer', 'N/A')

        print(f"\n--- 第 {i} 筆 ---")
        print(f"[問題] {q}")
        print(f"[參考答案] {gt[:200]}...")
        print(f"[Base Model] {base_ans[:200]}...")
        print(f"[Finetuned]  {ft_ans[:200]}...")
        print()


# ==============================================================================
#                          Part 3：Benchmark
# ==============================================================================

async def run_benchmark(data, output_dir, model_name, llm, label="Finetuned Model"):
    from llama_index.core.evaluation import CorrectnessEvaluator, BatchEvalRunner

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")

    correctness = CorrectnessEvaluator(llm=llm)
    batch_runner = BatchEvalRunner({"correctness": correctness}, workers=5, show_progress=True)

    print(f"\n[Benchmark] 評估 {len(data)} 筆...")
    start = time.time()

    eval_results = await batch_runner.aevaluate_response_strs(
        queries=[str(x['question']) for x in data],
        response_strs=[str(x['predicted_answer']) for x in data],
        reference=[str(x['base_answer']) for x in data]
    )
    elapsed = time.time() - start

    output_datas = []
    for cr in eval_results['correctness']:
        output_datas.append({
            "question": cr.query,
            "gt_answer": [x['base_answer'] for x in data if x['question'] == cr.query][0],
            "model_answer": cr.response,
            "feedback": cr.feedback,
            "rating": cr.score,
        })

    scores = [d['rating'] for d in output_datas if isinstance(d.get('rating'), (int, float)) and d['rating'] >= 1.0]
    avg = sum(scores) / max(len(scores), 1)

    print(f"\n{'='*40}")
    print(f"{label} Benchmark 結果")
    print(f"{'='*40}")
    print(f"有效筆數: {len(scores)}")
    print(f"平均分數: {avg:.2f} / 5")
    print(f"平均分數: {avg * 20:.1f} / 100")
    print(f"耗時: {elapsed:.1f} 秒")

    tag = label.lower().replace(" ", "_")
    txt_path = os.path.join(output_dir, f"{timestamp}_{tag}_result.txt")
    json_path = os.path.join(output_dir, f"{timestamp}_{tag}_llamaindex.json")

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Judge: {model_name}\nSamples: {len(scores)}\n")
        f.write(f"Average: {avg:.2f}/5 ({avg*20:.1f}/100)\n")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_datas, f, ensure_ascii=False, indent=4)

    print(f"結果存於: {output_dir}")


# ==============================================================================
#                          主程式
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Lab4：Finetuned 推理 + 比較報告")
    print("=" * 60)

    with open(GURU_OUTPUT, 'r', encoding='utf-8') as f:
        guru_data = json.load(f)

    # Part 1: Finetuned 推理
    ft_path = os.path.join(OUTPUT_DIR, "finetuned_inference.json")
    ft_data = run_finetuned_inference(guru_data, ft_path)

    # Part 2: 人工觀察
    with open(BASELINE_INFERENCE, 'r', encoding='utf-8') as f:
        baseline_data = json.load(f)
    side_by_side_comparison(baseline_data, ft_data, n=10)

    # Part 3: Benchmark
    print("\n準備 Benchmark（打外部 GPT API）...")
    headers = {'Content-Type': 'application/json'}
    login_data = {"username": GPT_USERNAME, "password": GPT_PASSWORD}
    resp = requests.post(GPT_API_URL, headers=headers, data=json.dumps(login_data))
    OPENAI_KEY = resp.json()['token']

    from llama_index.llms.openai import OpenAI as LI_OpenAI
    llm = LI_OpenAI(
        JUDGE_MODEL_NAME, api_key=OPENAI_KEY, api_base=GPT_API_BASE,
        temperature=0, n=1, top_p=0.00001
    )
    print("\n[Benchmark] Base Model...")
    asyncio.run(run_benchmark(baseline_data, BENCHMARK_DIR, JUDGE_MODEL_NAME, llm, label="Base Model"))

    print("\n[Benchmark] Finetuned Model...")
    asyncio.run(run_benchmark(ft_data, BENCHMARK_DIR, JUDGE_MODEL_NAME, llm, label="Finetuned Model"))

    print("\n" + "=" * 60)
    print("Lab4 完成！")
    print("請撰寫 comparison_report.md，包含 base vs finetuned 的比較表與結論。")
    print("=" * 60)
