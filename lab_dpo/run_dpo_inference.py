"""
DPO model inference with CrossEncoder re-ranking.
Runs 100 questions and compares <ANSWER>: rate vs SFT v3 (51%).
"""
import os, sys, json
from tqdm import tqdm
from openai import OpenAI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

GURU_DATA  = "lab1/output/question_answer_40chunk_256.json"
OUTPUT     = "lab_dpo/output/sft_rerank_inference.json"
BASE_URL   = "http://localhost:8299/v1"
MODEL_NAME = "Qwen2.5-3B-Instruct"
TOP_K_RERANK = 40  # reorder all, no truncation

SYSTEM = "You are a helpful question answerer who can provide an answer given a question and relevant context."
USER_TMPL = """Question: {question}
Context: {context}

Answer this question using the information given in the context above. Here is things to pay attention to:
- If you need to use CoT (Chain of Thought) reasoning, please do so. If the question is simple and does not require CoT, then do not use it.
- In the reasoning, if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##.
- The response should match the language of the given question.
- End your response with final answer in the form <ANSWER>: $answer, the answer should be succinct."""


def rerank(question, chunks, top_k=TOP_K_RERANK):
    try:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder("BAAI/bge-reranker-base")
        pairs  = [(question, c) for c in chunks]
        scores = ce.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:top_k]]
    except Exception as e:
        print(f"[rerank] skip: {e}")
        return chunks[:top_k]


def main():
    with open(GURU_DATA, encoding="utf-8") as f:
        data = json.load(f)

    client = OpenAI(api_key="empty", base_url=BASE_URL)
    results = []

    for item in tqdm(data, desc="DPO+rerank inference"):
        question = (item.get("question") or "").strip()
        raw_chunks = item.get("RAG_chunks", [])
        chunks = rerank(question, raw_chunks)
        context = str(chunks)[:5000]

        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",   "content": USER_TMPL.format(question=question, context=context)},
                ],
                max_tokens=1024,
                temperature=0,
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            answer = f"ERROR: {e}"

        item["predicted_answer"] = answer
        results.append(item)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    has_ans  = sum(1 for r in results if "<ANSWER>" in str(r.get("predicted_answer", "")))
    avg_len  = sum(len(str(r.get("predicted_answer", ""))) for r in results) / max(len(results), 1)
    repeat   = sum(1 for r in results if _is_degenerate(r.get("predicted_answer", "")))

    print(f"\n{'='*50}")
    print(f"DPO + Re-ranking 推理結果 (n={len(results)})")
    print(f"  <ANSWER>: 率  : {has_ans}/{len(results)} = {has_ans/len(results)*100:.0f}%  (SFT v3: 51%)")
    print(f"  平均回應長度  : {avg_len:.0f} 字元")
    print(f"  退化輸出數    : {repeat}")
    print(f"  結果存至      : {OUTPUT}")


def _is_degenerate(text, threshold=200):
    t = str(text)
    if len(t) < threshold:
        return False
    chunk = t[:100]
    return t.count(chunk) > 3


if __name__ == "__main__":
    main()
