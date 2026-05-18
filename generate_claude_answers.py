"""
Use OpenAI as a teacher model to generate high-quality answers for the Guru QA dataset.
Replaces the self-distilled base_answer with GPT-generated answers, then writes
train_claude.json / test_claude.json in the same SFT format as lab2/lab2.py.

Usage:
    export OPENAI_API_KEY=sk-proj-...
    python generate_claude_answers.py [--model gpt-4o-mini] [--concurrency 5]
"""

import os
import json
import asyncio
import argparse
import random
from pathlib import Path

from openai import AsyncOpenAI

GURU_DATA   = "lab1/output/question_answer_40chunk_256.json"
OUTPUT_DIR  = Path("lab2/output")
TRAIN_FILE  = OUTPUT_DIR / "train_claude.json"
TEST_FILE   = OUTPUT_DIR / "test_claude.json"
TRAIN_RATIO = 0.8

SYSTEM_PROMPT = (
    "You are a helpful question answerer who can provide an answer "
    "given a question and relevant context."
)

USER_PROMPT_TEMPLATE = """Question: {question}
Context: {context}

Answer this question using the information given in the context above. Here is things to pay attention to:
- If you need to use CoT (Chain of Thought) reasoning, please do so. If the question is simple and does not require CoT, then do not use it.
- In the reasoning, if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##.
- The response should match the language of the given question.
- End your response with final answer in the form <ANSWER>: $answer, the answer should be succinct."""


async def generate_answer(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    idx: int,
    item: dict,
    model: str,
) -> dict | None:
    question = (item.get("question") or "").strip()
    context  = str(item.get("RAG_chunks", []))[:5000]
    if not question:
        print(f"[{idx:03d}] skip — empty question")
        return None

    user_prompt = USER_PROMPT_TEMPLATE.format(question=question, context=context)

    async with sem:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=1024,
                temperature=0.3,
            )
            answer = response.choices[0].message.content.strip()
            print(f"[{idx:03d}] OK  ({len(answer)} chars)  {question[:60]!r}")
            return {"question": user_prompt, "answer": answer}
        except Exception as e:
            print(f"[{idx:03d}] ERROR: {e}")
            return None


async def main(model: str, concurrency: int):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set. Export it before running this script.")

    with open(GURU_DATA, encoding="utf-8") as f:
        guru_data = json.load(f)

    print(f"Loaded {len(guru_data)} items from {GURU_DATA}")
    print(f"Model: {model}  |  Concurrency: {concurrency}")

    client = AsyncOpenAI(api_key=api_key)
    sem    = asyncio.Semaphore(concurrency)

    tasks = [
        generate_answer(client, sem, i, item, model)
        for i, item in enumerate(guru_data)
    ]
    results = await asyncio.gather(*tasks)

    converted = [r for r in results if r is not None]
    print(f"\nGenerated {len(converted)} answers (skipped {len(guru_data) - len(converted)})")

    random.seed(42)
    random.shuffle(converted)
    split = int(len(converted) * TRAIN_RATIO)
    train_data = converted[:split]
    test_data  = converted[split:]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(TEST_FILE, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(train_data)} train → {TRAIN_FILE}")
    print(f"Saved {len(test_data)}  test  → {TEST_FILE}")
    print("\nNext step: update TRAIN_DATA in docker-compose-sft.yaml to train_claude.json, then re-run SFT")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="gpt-4o-mini",
                        help="OpenAI model (default: gpt-4o-mini)")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Max parallel API requests (default: 5)")
    args = parser.parse_args()
    asyncio.run(main(args.model, args.concurrency))
