"""
Lab2 - Base 模型推理 + 資料轉換
================================
1. 用 base model 對 Guru 產物做批量推理。
2. 將 Guru 產物轉為 aiDAPTIV2 訓練格式並切分 train/test。
"""

import os
import json
import random
from tqdm import tqdm
from openai import AsyncOpenAI

# 路徑請由此處修改 ---------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

GURU_OUTPUT = os.path.join(SCRIPT_DIR, "..", "lab1", "output", "question_answer_40chunk_256.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

BASE_URL = "http://localhost:8299/v1"
MODEL_NAME = "Qwen2.5-3B-Instruct"

TRAIN_RATIO = 0.8
# -------------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
#                          Part 1：Base 模型推理
# ==============================================================================

RAG_SYSTEM_PROMPT = "You are a helpful question answerer who can provide an answer given a question and relevant context."

RAG_USER_PROMPT = """Question: {}
Context: {}

Answer this question using the information given in the context above. Here is things to pay attention to:
- If you need to use CoT (Chain of Thought) reasoning, please do so. If the question is simple and does not require CoT, then do not use it.
- In the reasoning, if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##.
- The response should match the language of the given question.
- End your response with final answer in the form <ANSWER>: $answer, the answer should be succinct."""


def run_base_inference(guru_data, output_path):
    """用 base model 對 guru_data 做批量推理"""
    from openai_multi_client import OpenAIMultiClient

    KEY = "RAG_chunks"
    async_client = AsyncOpenAI(api_key="empty", base_url=BASE_URL, timeout=1200)
    api = OpenAIMultiClient(
        async_client, concurrency=20,
        endpoint="chat.completions",
        data_template={"model": MODEL_NAME}
    )

    print(f"[Base 推理] 端點: {BASE_URL}")
    print(f"[Base 推理] 模型: {MODEL_NAME}")
    print(f"[Base 推理] 資料筆數: {len(guru_data)}")

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
    with tqdm(total=len(guru_data), desc="Base 推理") as pbar:
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

    print(f"[Base 推理] 完成，結果存於: {output_path}")
    return save_data


# ==============================================================================
#                          Part 2：資料格式轉換
# ==============================================================================

def convert_for_aidaptiv(guru_data):
    """
    將 Guru 產物轉為 aiDAPTIV2 訓練格式。

    TODO: 請決定以下兩個選擇，並在此處修改：
      1. answer 取法：
         - 選項 A：取 base_answer 全文（含 CoT 推理過程）
         - 選項 B：只取 <ANSWER>: 後面的精簡答案
      2. question 是否帶 context：
         - 選項 A：question 只放原始問題
         - 選項 B：question = 問題 + RAG_chunks 上下文
    """
    converted = []
    for item in guru_data:
        answer = item.get("base_answer", "")

        # ---- TODO: 選擇 answer 取法 ----
        # 選項 B（只取 <ANSWER>: 後面）：
        # if "<ANSWER>:" in answer:
        #     answer = answer.split("<ANSWER>:")[-1].strip()

        question = item.get("question", "")

        # ---- TODO: 選擇是否帶 context ----
        # 選項 B（帶 RAG context）：
        # context = str(item.get("RAG_chunks", []))
        # question = f"Context: {context}\n\nQuestion: {question}"

        converted.append({
            "question": question,
            "answer": answer,
        })
    return converted


# ==============================================================================
#                          Part 3：Train / Test 切分
# ==============================================================================

def split_train_test(data, train_ratio=0.8):
    """隨機切分 train / test"""
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


# ==============================================================================
#                          主程式
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Lab2：Base 模型推理 + 資料轉換")
    print("=" * 60)

    # 載入 Guru 產物
    with open(GURU_OUTPUT, 'r', encoding='utf-8') as f:
        guru_data = json.load(f)
    print(f"載入 Guru 產物: {len(guru_data)} 筆\n")

    # Part 1：Base 推理
    baseline_path = os.path.join(OUTPUT_DIR, "baseline_inference.json")
    run_base_inference(guru_data, baseline_path)

    # Part 2：格式轉換
    print("\n[資料轉換] 轉換為 aiDAPTIV2 格式...")
    converted = convert_for_aidaptiv(guru_data)
    print(f"[資料轉換] 轉換完成: {len(converted)} 筆")

    # Part 3：切分
    train_data, test_data = split_train_test(converted, TRAIN_RATIO)

    train_path = os.path.join(OUTPUT_DIR, "train.json")
    test_path = os.path.join(OUTPUT_DIR, "test.json")

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)

    print(f"\n[切分結果] Train: {len(train_data)} 筆 → {train_path}")
    print(f"[切分結果] Test:  {len(test_data)} 筆 → {test_path}")

    # 預覽
    print("\n--- 訓練資料範例（第 1 筆） ---")
    print(json.dumps(train_data[0], indent=2, ensure_ascii=False)[:500])

    print("\n" + "=" * 60)
    print("Lab2 完成！接下來請進入 Lab3 進行 Finetune。")
    print("注意：Finetune 前請先關閉 vLLM 以釋放 GPU。")
    print("=" * 60)
