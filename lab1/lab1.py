"""
Lab1 - Guru 全管線 + 人工觀察
==============================
一口氣跑完 Guru 的 QA 生成流程，並在每個階段暫停以人工觀察產物品質。
"""

import os
import sys
import json
import random

LIB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, LIB_PATH)

from tqdm import tqdm
from openai import OpenAI, OpenAIError
import concurrent.futures
import logging

from tool_lib.core.pdf2txt.PDFParser import PDFParser
from tool_lib.core.pdf2txt.createChunk import create_chunk_from_txt
from tool_lib.core.pdf2txt.Clean_text import (
    append_appendix_discription, replace_vague_text,
    create_word_conbination, read_json_file
)
from tool_lib.core.docx2txt.parse_docx import docx_parser
from tool_lib.core.generate_question.gen_question_multiprocess import DataPool
from tool_lib.core.generate_question.RAG_chunking import RAGChunking
from opencc import OpenCC

cc = OpenCC('s2t')

# ==============================================================================
#                          設定區（請依助教說明修改）
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ENDPOINT = "http://localhost:8299/v1"
MODELNAME = "Qwen2.5-3B-Instruct"
SOURCE_FOLDER = os.path.join(SCRIPT_DIR, "..", "docs")     # 專案根目錄的 docs/
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "output")         # lab1/output/
MAX_QUESTION = 100                      # 生成問題數量上限
CHUNK_SIZE = 256                        # 分塊大小

_client: "OpenAI | None" = None


def _get_client() -> "OpenAI":
    global _client
    if _client is None:
        _client = OpenAI(api_key='EMPTY', base_url=ENDPOINT)
    return _client


# ==============================================================================
#                          工具函式（來自 guru.py）
# ==============================================================================

def GetFoldersWithFiles(root_dir, file_types):
    folder_list = []
    for file in os.listdir(root_dir):
        if file.endswith(tuple(file_types)):
            folder_list.append(root_dir)
            break
    for root, dirs, files in os.walk(root_dir):
        for d in dirs:
            dir_path = os.path.join(root, d)
            for file in os.listdir(dir_path):
                if file.endswith(tuple(file_types)):
                    folder_list.append(dir_path)
                    break
    return folder_list


def extract_title(filePath, txt):
    fileName = os.path.basename(filePath)
    if " " in fileName:
        return "".join(fileName.split(" ")[1:])
    if "_" in fileName:
        return "".join(fileName.split("_")[1:])
    return fileName


def return_response(messages, temperature=0.8, top_p=0.3):
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=MODELNAME, messages=messages,
            temperature=temperature, top_p=top_p, max_tokens=2048
        )
        output = response.choices[0].message.content
        return cc.convert(output)
    except Exception as e:
        print(f"API 錯誤: {e}")
        return None


def SaveDataInJson(data, file_name):
    with open(file_name, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False)


def show_samples(data, fields, n=3, title=""):
    """從 data 中隨機挑 n 筆，印出指定欄位供人工觀察"""
    print(f"\n{'='*60}")
    print(f"觀察點：{title}（隨機 {n} 筆）")
    print('='*60)
    samples = random.sample(data, min(n, len(data)))
    for i, item in enumerate(samples, 1):
        print(f"\n--- 第 {i} 筆 ---")
        for f in fields:
            val = item.get(f, "N/A")
            if isinstance(val, list):
                val = str(val)[:300] + "..." if len(str(val)) > 300 else str(val)
            elif isinstance(val, str) and len(val) > 500:
                val = val[:500] + "..."
            print(f"  [{f}]: {val}")
    print()


# ==============================================================================
#                          階段一：文件轉換
# ==============================================================================

def stage_convert(pdf_folders, docx_folders, output_folder):
    print("\n[階段 1/5] 文件轉換 (PDF/DOCX → TXT)...")
    docs_path = os.path.join(output_folder, "docs_txt")
    os.makedirs(docs_path, exist_ok=True)

    for folder in tqdm(docx_folders, desc="DOCX → TXT"):
        docx_parser(folder, output_folder, "fast")

    for folder in tqdm(pdf_folders, desc="PDF → TXT"):
        PDFParser(folder, output_folder,
                  cleantxt_startkey=[], cleantxt_endkey=[],
                  extract_title_rule=extract_title, method="fast")

    txt_count = len([f for f in os.listdir(docs_path) if f.endswith('.txt')]) if os.path.isdir(docs_path) else 0
    print(f"[階段 1/5] 完成，產出 {txt_count} 個 TXT 檔案。\n")


# ==============================================================================
#                          階段二：文本分塊
# ==============================================================================

def stage_chunk(output_folder, chunk_json_file, chunk_size):
    print(f"\n[階段 2/5] 文本分塊 (chunk_size={chunk_size})...")
    create_chunk_from_txt(
        output_folder, chunk_size=chunk_size, chunk_overlap=32,
        contentLenLimit=10, save_file_name=chunk_json_file,
        plot=False, add_metadata=True
    )

    jsondata = read_json_file(chunk_json_file)[0]
    json_title = jsondata["title"]
    json_chunk = jsondata["chunk_text"]

    appendix_keyword = ["figure", "fig.", "table", "tab.", "theorem", "algorithm", "lemma", "equation"]
    vague_phrase = [
        ["section", "in the content", "presented content", "given content", "provided text", "the text", "proposed method"],
        ["experiments", "experiment", "content provided", "text provided", "provided text", "information provided"],
        ["the paper", "this paper", "the study"]
    ]
    for c_idx, chunk in enumerate(json_chunk):
        chunk = append_appendix_discription(data=chunk, appendix_keyword=appendix_keyword, discription=json_title)
        chunk = replace_vague_text(
            data=chunk, vague_text=vague_phrase,
            replace_text=[f"of '{json_title}'", f"in '{json_title}'", f"'{json_title}'"]
        )
        json_chunk[c_idx] = chunk

    print(f"[階段 2/5] 完成，共 {len(json_chunk)} 個 chunk。")

    show_samples(
        [{"chunk": c} for c in json_chunk], ["chunk"], n=3,
        title="A：Chunk 品質（完整性、模糊指代替換）"
    )


# ==============================================================================
#                          階段三：問題生成
# ==============================================================================

def stage_question(chunk_json_file, question_output_path, max_question):
    print(f"\n[階段 3/5] 問題生成 (max_question={max_question})...")

    def ProcessGenerateQuestion(paper):
        chunks = paper['chunk']
        system_prompt = (
            "You are a synthetic question-answer pair generator. "
            "Given a chunk of context about some topic(s), generate 1 example question a user could ask "
            "and would be answered using information from the chunk. "
            "The questions should be able to be answered in a few words or less. "
            "The language of the generated question should match the language of the given context. "
            "Provide only the question, not the answer."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(chunks) + '\njust give me the question'},
        ]
        response = return_response(messages, temperature=1.2, top_p=0.7)
        return {
            "filename": paper["filename"],
            "title": paper["title"],
            "chunk": paper["chunk"],
            "question": response,
            "question_ok": response is not None and str(response).strip() != ""
        }

    paperData = DataPool(inputFile=chunk_json_file).get_data()
    chunk_windows = 5
    gen_data_list = []
    while len(gen_data_list) < max_question:
        for paper in paperData:
            chunk_window_local = min(chunk_windows, paper.chunk_len)
            for startChunk in range(0, paper.chunk_len - chunk_window_local + 1):
                data = {
                    'filename': paper.filename, 'title': paper.title,
                    'chunk': paper.get_chunk(start_index=startChunk, window_size=chunk_window_local, shuffle=True)
                }
                gen_data_list.append(data)
                if len(gen_data_list) >= max_question:
                    break
            if len(gen_data_list) >= max_question:
                break
    random.shuffle(gen_data_list)
    gen_data_list = gen_data_list[:max_question]

    futures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        for data in gen_data_list:
            futures[executor.submit(ProcessGenerateQuestion, data)] = data
        json_list = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="生成問題"):
            try:
                result = future.result()
                if result is not None:
                    json_list.append(result)
            except Exception as exc:
                print(f"問題生成例外: {exc}")
            if len(json_list) > 0 and len(json_list) % 10 == 0:
                SaveDataInJson(json_list, question_output_path)

    SaveDataInJson(json_list, question_output_path)
    print(f"[階段 3/5] 完成，共 {len(json_list)} 個問題。")
    show_samples(json_list, ["question", "chunk"], n=5, title="B：問題品質（意義、語言、是否抄原文）")


# ==============================================================================
#                          階段四：答案生成
# ==============================================================================

def stage_answer(question_output_path, answer_output_path):
    print("\n[階段 4/5] 答案生成...")

    def ProcessGenerateanswer(paper):
        question = paper['question']
        chunk = str(paper['chunk'])
        system_prompt = "You are a helpful question answerer who can provide an answer given a question and relevant context."
        user_prompt = (
            f"Question: {question}\n Context: {chunk}\n "
            "Answer this question using the information given in the context above. Here is things to pay attention to: "
            "- Mention the Source: Before quoting from the context, naturally mention the source of the information based on the filename or metadata provided in the context. "
            "- If you need to use CoT (Chain of Thought) reasoning, please do so. If the question is simple and does not require CoT, then do not use it. "
            "- In the reasoning, if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##. "
            "- The response should match the language of the given question. "
            "- End your response with final answer in the form <ANSWER>: $answer, the answer should be succinct."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = return_response(messages)
        paper['base_answer'] = response
        return paper

    with open(question_output_path, "r", encoding="utf-8") as f:
        paperData = json.load(f)

    json_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(ProcessGenerateanswer, d): d for d in paperData if d is not None}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="生成答案"):
            try:
                result = future.result()
                if result is not None:
                    json_list.append(result)
            except Exception as exc:
                print(f"答案生成例外: {exc}")
            if len(json_list) > 0 and len(json_list) % 10 == 0:
                SaveDataInJson(json_list, answer_output_path)

    SaveDataInJson(json_list, answer_output_path)
    print(f"[階段 4/5] 完成，共 {len(json_list)} 筆答案。")
    show_samples(json_list, ["question", "base_answer"], n=5, title="C：答案品質（CoT、引用、<ANSWER>: 格式）")


# ==============================================================================
#                          階段五：RAG 檢索增強
# ==============================================================================

def stage_rag(output_folder, answer_output_path, rag_output_path, retrieve_top_k=40):
    print(f"\n[階段 5/5] RAG 檢索增強 (top_k={retrieve_top_k})...")
    import torch
    torch.cuda.empty_cache()

    rag = RAGChunking(device="cpu", chunk_size=256, chunk_overlap=32)
    rag.init(txtfileFolder=output_folder)

    new_list = []
    with open(answer_output_path, "r", encoding="utf-8") as f:
        questionJson = json.load(f)
        for data in tqdm(questionJson, desc="RAG 檢索"):
            if data is None:
                continue
            question = data["question"]
            if question is None or not str(question).strip():
                data["hybrid_chunks"] = []
                data["RAG_chunks"] = []
                data["rag_acc"] = 0
                data["rag_error"] = "empty question"
                new_list.append(data)
                continue

            question = str(question).strip()

            chunk_score_list = rag.vectorstore.similarity_search_with_score(
                question, k=retrieve_top_k
            )

            chunk_list = []
            for chunk in chunk_score_list:
                context = chunk[0].page_content
                file_name = chunk[0].metadata['source'].split('/')[-1]
                chunk_list.append("file name:" + file_name + "\ncontent: " + context)

            golden_chunk = data["chunk"]
            rag_chunk = list(chunk_list)
            rag_chunk_set = set(rag_chunk)
            rag_acc = 0
            for gc in golden_chunk:
                if gc in rag_chunk_set:
                    rag_acc += 1
                else:
                    rag_chunk.insert(0, gc)
            hybrid = rag_chunk[:40]
            random.shuffle(hybrid)

            data["hybrid_chunks"] = hybrid
            data["RAG_chunks"] = chunk_list
            data["rag_acc"] = rag_acc
            new_list.append(data)

    SaveDataInJson(new_list, rag_output_path)

    avg_rag_acc = sum(d.get("rag_acc", 0) for d in new_list) / max(len(new_list), 1)
    print(f"[階段 5/5] 完成，共 {len(new_list)} 筆。平均 rag_acc = {avg_rag_acc:.2f}")
    show_samples(new_list, ["question", "rag_acc"], n=5, title="D：RAG 準確率與混合結果")


# ==============================================================================
#                          主程式
# ==============================================================================

if __name__ == '__main__':
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "docs_txt"), exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(OUTPUT_FOLDER, 'guru.log'),
        filemode='a', level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    pdf_folders = GetFoldersWithFiles(SOURCE_FOLDER, ['.pdf', '.PDF'])
    docx_folders = GetFoldersWithFiles(SOURCE_FOLDER, ['.docx', '.doc'])

    chunk_json = os.path.join(OUTPUT_FOLDER, f"chunkfile_{CHUNK_SIZE}.json")
    question_json = os.path.join(OUTPUT_FOLDER, f"question_{CHUNK_SIZE}.json")
    answer_json = os.path.join(OUTPUT_FOLDER, f"question_answer_{CHUNK_SIZE}.json")
    rag_json = os.path.join(OUTPUT_FOLDER, f"question_answer_40chunk_{CHUNK_SIZE}.json")

    print("=" * 60)
    print("Lab1：Guru 全管線 + 人工觀察")
    print("=" * 60)

    stage_convert(pdf_folders, docx_folders, OUTPUT_FOLDER)
    stage_chunk(OUTPUT_FOLDER, chunk_json, CHUNK_SIZE)
    stage_question(chunk_json, question_json, MAX_QUESTION)
    stage_answer(question_json, answer_json)
    stage_rag(OUTPUT_FOLDER, answer_json, rag_json)

    print("\n" + "=" * 60)
    print("Lab1 全部完成！")
    print(f"最終產物: {rag_json}")
    print("請撰寫 observation.md，記錄觀察點 A-D。")
    print("=" * 60)
