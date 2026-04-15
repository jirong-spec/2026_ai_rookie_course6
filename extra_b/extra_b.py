"""
Extra Lab B - Guru 參數消融：重跑 Guru（受影響階段）
====================================================
請選擇一個變數修改後執行此腳本。
腳本會重跑受影響的 Guru 階段，並與 Lab1 產物做並排比較。
"""

import os
import sys
import json
import random
import copy

LIB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, LIB_PATH)

from tqdm import tqdm
from openai import OpenAI
import concurrent.futures
import logging

from tool_lib.core.pdf2txt.createChunk import create_chunk_from_txt
from tool_lib.core.pdf2txt.Clean_text import (
    append_appendix_discription, replace_vague_text,
    create_word_conbination, read_json_file
)
from tool_lib.core.generate_question.gen_question_multiprocess import DataPool
from tool_lib.core.generate_question.RAG_chunking import RAGChunking
from opencc import OpenCC

cc = OpenCC('s2t')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
#                          設定區
# ==============================================================================

ENDPOINT = "http://localhost:8299/v1"
MODELNAME = "Qwen2.5-3B-Instruct"

# Lab1 的輸出資料夾（已有 docs_txt/）
LAB1_OUTPUT = os.path.join(SCRIPT_DIR, "..", "lab1", "output")
LAB1_RAG_JSON = os.path.join(LAB1_OUTPUT, "question_answer_40chunk_256.json")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_QUESTION = 100

# ==============================================================================
#   TODO: 選擇你要改的變數（只改一個！）
# ==============================================================================

# 選項 A：改 chunk_size（原為 256）
CHUNK_SIZE = 1024   # 改成 1024 試試

# 選項 B：改問題生成 prompt（取消註解並修改）
# CUSTOM_QUESTION_PROMPT = """You are a question generator. Given context, generate ONE specific, detailed question
# that requires understanding of the content to answer. The question should NOT be answerable
# without reading the context. Generate only the question, in the same language as the context."""

# 選項 C：改答案生成 prompt（取消註解並修改）
# CUSTOM_ANSWER_PROMPT = """Answer the question based on the context. Be concise and direct.
# Do NOT use Chain of Thought reasoning. Just provide the answer."""


# ==============================================================================
#                          工具函式
# ==============================================================================

def return_response(messages, temperature=0.8, top_p=0.3):
    client = OpenAI(api_key='EMPTY', base_url=ENDPOINT)
    try:
        response = client.chat.completions.create(
            model=MODELNAME, messages=messages,
            temperature=temperature, top_p=top_p, max_tokens=8*1024
        )
        return cc.convert(response.choices[0].message.content)
    except Exception as e:
        print(f"API 錯誤: {e}")
        return None


def SaveDataInJson(data, file_name):
    with open(file_name, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False)


def compare_old_new(old_data, new_data, fields, n=5, title=""):
    """並排比較新舊產物"""
    print(f"\n{'='*70}")
    print(f"新舊比較：{title}（隨機 {n} 筆）")
    print('='*70)

    for i in range(min(n, len(new_data))):
        old_item = old_data[i] if i < len(old_data) else {}
        new_item = new_data[i]
        print(f"\n--- 第 {i+1} 筆 ---")
        for f in fields:
            old_v = str(old_item.get(f, 'N/A'))[:200]
            new_v = str(new_item.get(f, 'N/A'))[:200]
            print(f"  [舊 {f}]: {old_v}")
            print(f"  [新 {f}]: {new_v}")


# ==============================================================================
#                          Guru 重跑邏輯
# ==============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Extra Lab B：Guru 參數消融")
    print(f"  CHUNK_SIZE = {CHUNK_SIZE}")
    print("=" * 60)

    # 步驟 1：重新分塊
    chunk_json = os.path.join(OUTPUT_DIR, f"chunkfile_{CHUNK_SIZE}.json")
    print(f"\n[分塊] chunk_size={CHUNK_SIZE}...")
    create_chunk_from_txt(
        LAB1_OUTPUT, chunk_size=CHUNK_SIZE, chunk_overlap=32,
        contentLenLimit=10, save_file_name=chunk_json,
        plot=False, add_metadata=True
    )
    jsondata = read_json_file(chunk_json)[0]
    json_title = jsondata["title"]
    json_chunk = jsondata["chunk_text"]
    appendix_kw = ["figure", "fig.", "table", "tab.", "theorem", "algorithm", "lemma", "equation"]
    vague = [
        ["section", "in the content", "presented content", "given content", "provided text"],
        ["experiments", "experiment", "content provided", "text provided"],
        ["the paper", "this paper", "the study"]
    ]
    for ci, ch in enumerate(json_chunk):
        ch = append_appendix_discription(data=ch, appendix_keyword=appendix_kw, discription=json_title)
        ch = replace_vague_text(data=ch, vague_text=vague,
                                replace_text=[f"of '{json_title}'", f"in '{json_title}'", f"'{json_title}'"])
        json_chunk[ci] = ch
    print(f"[分塊] 完成: {len(json_chunk)} chunks")

    # 步驟 2：問題生成
    question_json = os.path.join(OUTPUT_DIR, f"question_{CHUNK_SIZE}.json")
    print(f"\n[問題生成] max={MAX_QUESTION}...")

    default_q_prompt = (
        "You are a synthetic question-answer pair generator. "
        "Given a chunk of context, generate 1 example question. "
        "The language should match the context. Provide only the question."
    )
    q_prompt = globals().get("CUSTOM_QUESTION_PROMPT", default_q_prompt)

    paperData = DataPool(inputFile=chunk_json).get_data()
    gen_list = []
    while len(gen_list) < MAX_QUESTION:
        for paper in paperData:
            wl = min(5, paper.chunk_len)
            for s in range(0, paper.chunk_len - wl + 1):
                gen_list.append({
                    'filename': paper.filename, 'title': paper.title,
                    'chunk': paper.get_chunk(start_index=s, window_size=wl, shuffle=True)
                })
    random.shuffle(gen_list)
    gen_list = gen_list[:MAX_QUESTION]

    def gen_q(paper):
        msgs = [{"role": "system", "content": q_prompt},
                {"role": "user", "content": str(paper['chunk']) + '\njust give me the question'}]
        resp = return_response(msgs, temperature=1.2, top_p=0.7)
        return {**paper, 'question': resp}

    q_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(gen_q, d): d for d in gen_list}
        for fut in tqdm(concurrent.futures.as_completed(futs), total=len(futs), desc="問題生成"):
            try:
                r = fut.result()
                if r:
                    q_results.append(r)
            except Exception as e:
                print(f"Error: {e}")
    SaveDataInJson(q_results, question_json)
    print(f"[問題生成] 完成: {len(q_results)} 筆")

    # 步驟 3：答案生成
    answer_json = os.path.join(OUTPUT_DIR, f"question_answer_{CHUNK_SIZE}.json")
    print(f"\n[答案生成]...")

    default_a_prompt = (
        "Answer this question using the context. Use CoT if needed. "
        "Quote with ##begin_quote##/##end_quote##. Match question language. "
        "End with <ANSWER>: $answer."
    )
    a_instructions = globals().get("CUSTOM_ANSWER_PROMPT", default_a_prompt)

    def gen_a(paper):
        sys_p = "You are a helpful question answerer."
        user_p = f"Question: {paper['question']}\nContext: {str(paper['chunk'])}\n{a_instructions}"
        resp = return_response([{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}])
        paper['base_answer'] = resp
        return paper

    a_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(gen_a, d): d for d in q_results if d}
        for fut in tqdm(concurrent.futures.as_completed(futs), total=len(futs), desc="答案生成"):
            try:
                r = fut.result()
                if r:
                    a_results.append(r)
            except Exception as e:
                print(f"Error: {e}")
    SaveDataInJson(a_results, answer_json)

    # 步驟 4：RAG
    rag_json = os.path.join(OUTPUT_DIR, f"question_answer_40chunk_{CHUNK_SIZE}.json")
    print(f"\n[RAG 檢索]...")
    import torch
    torch.cuda.empty_cache()
    rag = RAGChunking(device="cpu", chunk_size=256, chunk_overlap=32)
    rag.init(txtfileFolder=LAB1_OUTPUT)

    new_list = []
    for data in tqdm(a_results, desc="RAG"):
        if not data:
            continue
        cs = rag.vectorstore.similarity_search_with_score(data["question"], k=40)
        cl = ["file name:" + c[0].metadata['source'].split('/')[-1] + "\ncontent: " + c[0].page_content for c in cs]
        rc = copy.deepcopy(cl)
        ra = 0
        for gc in data["chunk"]:
            if gc in rc:
                ra += 1
            else:
                rc.insert(0, gc)
        hybrid = rc[:40]
        random.shuffle(hybrid)
        data["hybrid_chunks"] = hybrid
        data["RAG_chunks"] = cl
        data["rag_acc"] = ra
        new_list.append(data)
    SaveDataInJson(new_list, rag_json)

    # 步驟 5：與 Lab1 比較
    with open(LAB1_RAG_JSON, 'r', encoding='utf-8') as f:
        old_data = json.load(f)
    compare_old_new(old_data, new_list, ["question", "base_answer", "rag_acc"], n=5,
                    title=f"Lab1 (chunk=256) vs Extra B (chunk={CHUNK_SIZE})")

    print(f"\n{'='*60}")
    print(f"Extra B Guru 重跑完成！最終產物: {rag_json}")
    print("接下來：")
    print("  1. python convert_and_split.py  （轉換 + 切分）")
    print("  2. 複製 train_v2.json 到 lab3_finetune 做第二次 Finetune")
    print("  3. 推理 + benchmark + 撰寫 ab_comparison_report.md")
    print("=" * 60)
