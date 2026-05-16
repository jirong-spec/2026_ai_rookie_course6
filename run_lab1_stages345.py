"""Run only lab1 stages 3-5 (chunks already exist)."""
import os, sys

LAB1_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lab1')
ROOT_DIR  = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LAB1_DIR)

# Re-use lab1's constants and stage functions by importing them
from lab1 import (
    stage_question, stage_answer, stage_rag,
    OUTPUT_FOLDER, CHUNK_SIZE, MAX_QUESTION,
)

chunk_json    = os.path.join(OUTPUT_FOLDER, f"chunkfile_{CHUNK_SIZE}.json")
question_json = os.path.join(OUTPUT_FOLDER, f"question_{CHUNK_SIZE}.json")
answer_json   = os.path.join(OUTPUT_FOLDER, f"question_answer_{CHUNK_SIZE}.json")
rag_json      = os.path.join(OUTPUT_FOLDER, f"question_answer_40chunk_{CHUNK_SIZE}.json")

print("Stage 3: question generation")
stage_question(chunk_json, question_json, MAX_QUESTION)

print("Stage 4: answer generation")
stage_answer(question_json, answer_json)

print("Stage 5: RAG")
stage_rag(OUTPUT_FOLDER, answer_json, rag_json)

print("Done –", rag_json)
