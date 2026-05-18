"""
DPO fine-tuning on Qwen2.5-3B-Instruct using 50 chosen/rejected pairs.

Run inside the SFT Docker container:
    deepspeed --num_gpus=1 /workspace/lab_dpo/train_dpo.py

Env vars (set via docker-compose-dpo.yaml):
    BASE_MODEL  - path to base or merged model
    TRAIN_DATA  - path to dpo_train.json
    OUTPUT_DIR  - where to save DPO checkpoint
"""

import os
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

BASE_MODEL = os.environ.get("BASE_MODEL",  "/mnt/model/Qwen2.5-3B-Instruct-finetuned")
TRAIN_DATA = os.environ.get("TRAIN_DATA",  "/workspace/lab_dpo/output/dpo_train.json")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR",  "/workspace/lab_dpo/output/dpo")

print(f"Base model : {BASE_MODEL}")
print(f"Train data : {TRAIN_DATA}")
print(f"Output dir : {OUTPUT_DIR}")

# ── Load data ──────────────────────────────────────────────────────────────────
with open(TRAIN_DATA, encoding="utf-8") as f:
    raw = json.load(f)

dataset = Dataset.from_list([
    {"prompt": d["prompt"], "chosen": d["chosen"], "rejected": d["rejected"]}
    for d in raw
])

# ── Tokenizer ──────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# ── Model ──────────────────────────────────────────────────────────────────────
print("Loading model in bfloat16...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# Pass peft_config to DPOTrainer — it shares base weights as reference model,
# avoiding a separate full-model copy and halving VRAM usage.
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ── DPO Training ───────────────────────────────────────────────────────────────
training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=1e-6,
    bf16=True,
    logging_steps=5,
    save_steps=9999,
    max_length=512,
    max_prompt_length=384,
    beta=0.5,
    remove_unused_columns=False,
    report_to="none",
    deepspeed=os.environ.get("DS_CONFIG"),
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
)

print("Starting DPO training...")
try:
    trainer.train()
except Exception as e:
    print(f"[train] 警告: {e}")

print(f"Saving to {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done.")
