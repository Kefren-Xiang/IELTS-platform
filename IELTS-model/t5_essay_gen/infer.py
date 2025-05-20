# infer.py
# 用训练好的 t5_essay_gen 模型生成雅思范文

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_DIR = "model_output\checkpoint-4832"

# 1. 加载模型和分词器
print("[INFO] Loading model...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

def generate_essay(prompt: str, max_length: int = 512):
    device = next(model.parameters()).device
    input_text = f"Generate IELTS essay: {prompt}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            do_sample=True,
            top_p=0.9,
            temperature=0.85,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    while True:
        user_input = input("\n📝 输入雅思作文题目 (或输入 q 退出)：\n> ").strip()
        if user_input.lower() in ["q", "quit", "exit"]:
            break

        print("\n⏳ 正在生成范文...\n")
        result = generate_essay(user_input)
        print("📄 范文输出：\n")
        print(result)
