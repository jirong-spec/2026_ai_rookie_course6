"""Merge LoRA adapter into base model and save as a full model for vLLM."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL  = os.environ.get("BASE_MODEL",  "/mnt/model/Qwen2.5-3B-Instruct")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "/workspace/lab3_finetune/output/sft")
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR",  "/mnt/model/Qwen2.5-3B-Instruct-finetuned")

print(f"Base  : {BASE_MODEL}")
print(f"Adapter: {ADAPTER_DIR}")
print(f"Output : {OUTPUT_DIR}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

print("Loading base model in bfloat16...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_DIR)

print("Merging adapter into base weights...")
model = model.merge_and_unload()

print(f"Saving merged model to {OUTPUT_DIR} ...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Done.")
