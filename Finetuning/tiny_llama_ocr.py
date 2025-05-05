#!/usr/bin/env python3
# tiny_llama_ocr.py

"""
Uses a fine-tuned Tiny LLaMA model to extract purchased items and total cost
from noisy receipt text stored in a CSV file (from Tesseract OCR).
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# --- CONFIGURATION ---
csv_path = "extracted_text.csv"  # Update this for local use
model_path = "tiny_llama_cpsc254_lora8bit"  # Your local model folder

# --- Load Model ---
print("🔄 Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

# --- Read and clean CSV as OCR text ---
def read_and_clean_ocr_text(path):
    try:
        with open(path, "r") as f:
            lines = f.readlines()[1:]  # skip header
        texts = []
        for line in lines:
            parts = line.split(",", 1)
            if len(parts) == 2:
                text = parts[1].strip().strip('"').replace("\\n", "\n")
                texts.append(text)
        return "\n\n".join(texts)
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        return ""

# --- Query Model ---
def query_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Extract item + total even from fuzzy model output ---
def extract_items_and_total(response_text):
    items = []
    total = None
    for line in response_text.strip().splitlines():
        line = line.strip()
        if line.lower().startswith("item:"):
            try:
                parts = line.split(",")
                name = parts[0].split(":")[1].strip()
                price_str = parts[1].split(":")[1].strip().replace("$", "")
                price = float(re.sub(r"[^\d.]", "", price_str))
                items.append((name, price))
            except (IndexError, ValueError):
                continue
        elif "total" in line.lower():
            try:
                total_str = re.findall(r"\d+\.\d{2}", line)
                if total_str:
                    total = float(total_str[0])
            except ValueError:
                continue
    return items, total

# --- MAIN ---
if __name__ == "__main__":
    print("=== Tiny LLaMA OCR CSV Analyzer ===\n")

    ocr_text = read_and_clean_ocr_text(csv_path)
    if not ocr_text:
        exit(1)

    print("📄 Preview of OCR text:\n", ocr_text[:500], "...\n")

    prompt = f"""You are a receipt parser AI. Given noisy OCR text, extract items and prices, and compute the total. Do not explain. Just return this:

Item: <name>, Price: <amount>
...
Total: <amount>

Here is the OCR text:
{ocr_text}
"""

    print("🤖 Querying Tiny LLaMA...\n")
    model_response = query_model(prompt)
    print("📥 Model Response:\n", model_response)

    items, total = extract_items_and_total(model_response)

    print("\n🧾 Parsed Items:")
    for item, price in items:
        print(f" - {item}: ${price:.2f}")
    if total is not None:
        print(f"\n💵 Parsed Total: ${total:.2f}")
    else:
        print("\n❗ Could not extract total.")
