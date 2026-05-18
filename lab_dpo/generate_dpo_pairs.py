"""
Generate DPO chosen/rejected pairs from Lab2 (base) and Lab4 (FT) inference outputs.

Strategy:
  - For each question, compare base model vs FT model predicted_answer.
  - chosen  = answer that contains <ANSWER>: (proper format)
  - rejected = answer that lacks <ANSWER>: (format failure)
  - Only keep pairs where exactly one side has <ANSWER>:.

Output: lab_dpo/output/dpo_pairs.json
        lab_dpo/output/dpo_train.json (80%)
        lab_dpo/output/dpo_test.json  (20%)
"""

import json
import random
import os
from pathlib import Path

BASE_INFERENCE = "lab2/output/baseline_inference.json"
FT_INFERENCE   = "lab4/output/finetuned_inference.json"
OUTPUT_DIR     = Path("lab_dpo/output")

USER_PROMPT_TEMPLATE = """Question: {question}
Context: {context}

Answer this question using the information given in the context above. Here is things to pay attention to:
- If you need to use CoT (Chain of Thought) reasoning, please do so. If the question is simple and does not require CoT, then do not use it.
- In the reasoning, if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##.
- The response should match the language of the given question.
- End your response with final answer in the form <ANSWER>: $answer, the answer should be succinct."""


def has_answer_tag(text: str) -> bool:
    return "<ANSWER>" in str(text)


def main():
    with open(BASE_INFERENCE, encoding="utf-8") as f:
        base_data = json.load(f)
    with open(FT_INFERENCE, encoding="utf-8") as f:
        ft_data = json.load(f)

    ft_dict = {d["question"]: d for d in ft_data}

    pairs = []
    for b in base_data:
        q = b["question"]
        if q not in ft_dict:
            continue
        ft = ft_dict[q]

        b_ans = b.get("predicted_answer", "") or ""
        f_ans = ft.get("predicted_answer", "") or ""
        b_good = has_answer_tag(b_ans)
        f_good = has_answer_tag(f_ans)

        if b_good == f_good:
            continue  # both same quality — skip

        context = str(b.get("RAG_chunks", []))[:5000]
        prompt  = USER_PROMPT_TEMPLATE.format(question=q, context=context)

        chosen   = b_ans if b_good else f_ans
        rejected = f_ans if b_good else b_ans

        pairs.append({
            "prompt":   prompt,
            "chosen":   chosen,
            "rejected": rejected,
        })

    print(f"Total valid DPO pairs: {len(pairs)}")

    random.seed(42)
    random.shuffle(pairs)
    split      = int(len(pairs) * 0.8)
    train_data = pairs[:split]
    test_data  = pairs[split:]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fname, data in [
        ("dpo_pairs.json", pairs),
        ("dpo_train.json", train_data),
        ("dpo_test.json",  test_data),
    ]:
        with open(OUTPUT_DIR / fname, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Train: {len(train_data)}  Test: {len(test_data)}")
    print(f"Saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
