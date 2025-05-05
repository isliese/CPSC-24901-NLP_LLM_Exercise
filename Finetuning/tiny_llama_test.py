import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_dir = "tiny_llama_cpsc254_lora8bit"

# 1) Load tokenizer & model (Accelerate will handle device placement)
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",                            # <-- accelerate places for you
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# 2) Build the pipeline *without* a device argument
gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False
)

# 3) Ask your course‐specific questions
questions = [
    "What is the policy on Academic Dishonesty?",
    "Which deep learning frameworks and libraries will students learn in this course?",
    "What computer vision applications are included in the syllabus?"
]

for idx, q in enumerate(questions, 1):
    print(f"\n=== Question {idx} ===")
    print(q)
    out = gen(q, max_new_tokens=128, do_sample=False, eos_token_id=tokenizer.eos_token_id)
    print("→", out[0]["generated_text"].strip())
