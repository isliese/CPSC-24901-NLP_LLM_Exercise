#!/usr/bin/env python3
"""
tiny_llama_fine.py

LoRA-fine-tune TinyLlama on the CPSC_254 syllabus in 8-bit.
Dependencies:
    transformers, datasets, PyMuPDF, huggingface_hub,
    accelerate, bitsandbytes, peft

Before running in Colab:
    !pip install -U transformers datasets pymupdf huggingface_hub accelerate bitsandbytes peft
    # And be sure your Runtime → Change runtime type → Hardware accelerator = GPU
"""

import os
import fitz                                  # PyMuPDF
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from huggingface_hub import login
from peft import LoraConfig, get_peft_model

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + chunk_size]))
        i += chunk_size - overlap
    return chunks

def main():
    # — user params —
    pdf_path   = "CPSC_254_SYL.pdf"
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir = "tiny_llama_cpsc254_lora8bit"
    epochs     = 3
    bs         = 1
    lr         = 5e-5

    # — Hugging Face login —
    hf_token = os.getenv("HF_TOKEN") or input("Enter your Hugging Face token: ").strip()
    login(token=hf_token)

    # — extract & chunk PDF —
    print("📄 Extracting PDF...")
    raw_text = extract_text_from_pdf(pdf_path)

    print("✂️ Chunking text...")
    texts = chunk_text(raw_text)
    print(f"→ {len(texts)} chunks")

    ds = Dataset.from_dict({"text": texts})

    # — tokenize & attach labels —
    print("🔑 Tokenizing…")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    def tokenize_fn(ex):
        tok = tokenizer(ex["text"], truncation=True, max_length=512)
        tok["labels"] = tok["input_ids"].copy()
        return tok
    tokenized_ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    # — 8-bit quantization config —
    use_int8 = torch.cuda.is_available()
    if use_int8:
        print("🤖 Loading model in 8-bit…")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    else:
        print("⚠️  Bitsandbytes INT8 unavailable; loading full-precision")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    # — attach LoRA adapters —
    print("🧩 Attaching LoRA adapters…")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # — TrainingArguments (keep labels column) —
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=8,
        learning_rate=lr,
        fp16=True,
        optim="paged_adamw_8bit" if use_int8 else "adamw_torch",
        logging_steps=10,
        save_steps=100,
        report_to=["none"],
        push_to_hub=True,
        hub_model_id=output_dir,
        remove_unused_columns=False,   # ensure "labels" isn’t dropped
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
    )

    # — launch training —
    print("🚀 Starting fine-tuning…")
    trainer.train()
    print(f"✅ Done! Model & LoRA adapters saved to `{output_dir}` and pushed to the Hub.")

if __name__ == "__main__":
    main()
