"""
Extra Lab B - 對新 finetuned model 跑推理
"""

import os
import json
from tqdm import tqdm
from openai import AsyncOpenAI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# 使用 Extra B 的 Guru 產物
guru_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("question_answer_40chunk_")]
GURU_OUTPUT = os.path.join(OUTPUT_DIR, sorted(guru_files)[-1])

BASE_URL = "http://localhost:8299/v1"
MODEL_NAME = "Qwen2.5-3B-Instruct"

KEY = "RAG_chunks"

RAG_SYSTEM = "You are a helpful question answerer who can provide an answer given a question and relevant context."
RAG_USER = """Question: {}
Context: {}

Answer this question using the information given in the context above.
- Use CoT if needed. Quote with ##begin_quote##/##end_quote##.
- Match question language. End with <ANSWER>: $answer."""

if __name__ == "__main__":
    from openai_multi_client import OpenAIMultiClient

    with open(GURU_OUTPUT, 'r', encoding='utf-8') as f:
        data = json.load(f)

    async_client = AsyncOpenAI(api_key="empty", base_url=BASE_URL, timeout=1200)
    api = OpenAIMultiClient(async_client, concurrency=20, endpoint="chat.completions", data_template={"model": MODEL_NAME})

    print(f"Extra B 推理: {len(data)} 筆, model={MODEL_NAME}")

    def make_requests():
        for idx, d in enumerate(data):
            try:
                api.request(data={
                    "messages": [
                        {"role": "system", "content": RAG_SYSTEM},
                        {"role": "user", "content": RAG_USER.format(d["question"], str(d[KEY]))}
                    ], "n": 1, "top_p": 1, "temperature": 0
                }, metadata={'data': d})
            except Exception as e:
                print(f"Error: {e}")

    api.run_request_function(make_requests)

    save_data = []
    with tqdm(total=len(data), desc="Extra B 推理") as pbar:
        for result in api:
            if result.response:
                rd = result.metadata['data']
                for x in result.response.choices:
                    rd['predicted_answer'] = x.message.content
                    save_data.append(rd)
            pbar.update(1)

    out_path = os.path.join(OUTPUT_DIR, "extra_b_finetuned_inference.json")
    with open(out_path, 'w', encoding="utf-8") as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)
    print(f"完成: {out_path}")
