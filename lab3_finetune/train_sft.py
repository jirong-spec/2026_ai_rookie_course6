"""
SFT fine-tune for Qwen2.5-3B-Instruct using lab2 QA data.
Uses DeepSpeed ZeRO-3 with NVMe optimizer offload + CPU param offload.

Launch (inside container):
    deepspeed --num_gpus=1 /workspace/lab3_finetune/train_sft.py
"""

import os
import json
import inspect
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
import torch

# ==============================================================================
#  設定
# ==============================================================================

MODEL_PATH    = os.environ.get("MODEL_PATH",   "/mnt/model/Qwen2.5-3B-Instruct")
TRAIN_DATA    = os.environ.get("TRAIN_DATA",   "/workspace/lab2/output/train.json")
OUTPUT_DIR    = os.environ.get("OUTPUT_DIR",   "/workspace/lab3_finetune/output/sft")
DS_CONFIG     = os.environ.get("DS_CONFIG",    "/workspace/lab3_finetune/ds_zero3_nvme.json")
SYSTEM_PROMPT = "You are a helpful question answerer who can provide an answer given a question and relevant context."

PER_DEVICE_BATCH = 1
GRAD_ACCUM       = 4
NUM_EPOCHS       = 10
LR               = 5e-5
MAX_SEQ_LEN      = 4096

LORA_RANK  = 16
LORA_ALPHA = 32

# ==============================================================================
#  資料載入
# ==============================================================================

def load_dataset(path: str) -> Dataset:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    records = []
    for item in raw:
        question = (item.get("question") or "").strip()
        answer   = (item.get("answer") or item.get("base_answer") or "").strip()
        if not question or not answer:
            continue
        records.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": question},
                {"role": "assistant", "content": answer},
            ]
        })
    print(f"[data] loaded {len(records)} samples from {path}")
    return Dataset.from_list(records)


# ==============================================================================
#  主程式
# ==============================================================================

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = MAX_SEQ_LEN

    dataset = load_dataset(TRAIN_DATA)
    if len(dataset) == 0:
        raise RuntimeError(
            "Training dataset is empty. Re-run lab1 with vLLM started, then lab2."
        )

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    print(f"[lora] rank={LORA_RANK}, alpha={LORA_ALPHA}")

    # With ZeRO-3 do NOT set device_map — DeepSpeed owns placement
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    model.config.use_cache = False

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="epoch",
        deepspeed=DS_CONFIG,
        report_to="none",
    )

    # Handle API differences across TRL versions
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    tok_kwarg = "processing_class" if "processing_class" in trainer_params else "tokenizer"
    seq_kwarg = {"max_seq_length": MAX_SEQ_LEN} if "max_seq_length" in trainer_params else {}

    trainer = SFTTrainer(
        model=model,
        **{tok_kwarg: tokenizer},
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        **seq_kwarg,
    )

    print(f"[train] model={MODEL_PATH}")
    print(f"[train] deepspeed={DS_CONFIG}")
    print(f"[train] output={OUTPUT_DIR}")

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[done] model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
