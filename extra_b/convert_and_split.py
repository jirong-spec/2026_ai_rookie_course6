"""
Extra Lab B - 格式轉換 + 切分（同 Lab2 邏輯）
"""

import os
import json
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# 自動偵測最新的 Guru 產物
guru_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("question_answer_40chunk_")]
if not guru_files:
    print("找不到 Extra B 的 Guru 產物，請先執行 extra_b.py")
    exit(1)

GURU_OUTPUT = os.path.join(OUTPUT_DIR, sorted(guru_files)[-1])
print(f"使用 Guru 產物: {GURU_OUTPUT}")

with open(GURU_OUTPUT, 'r', encoding='utf-8') as f:
    data = json.load(f)

converted = []
for item in data:
    answer = item.get("base_answer", "")
    question = item.get("question", "")
    converted.append({"question": question, "answer": answer})

random.shuffle(converted)
split_idx = int(len(converted) * 0.8)
train = converted[:split_idx]
test = converted[split_idx:]

train_path = os.path.join(OUTPUT_DIR, "train_v2.json")
test_path = os.path.join(OUTPUT_DIR, "test_v2.json")

with open(train_path, 'w', encoding='utf-8') as f:
    json.dump(train, f, indent=4, ensure_ascii=False)
with open(test_path, 'w', encoding='utf-8') as f:
    json.dump(test, f, indent=4, ensure_ascii=False)

print(f"Train: {len(train)} 筆 → {train_path}")
print(f"Test:  {len(test)} 筆 → {test_path}")
print(f"\n請複製 {train_path} 到 ../lab3_finetune/train.json 後做第二次 Finetune。")
