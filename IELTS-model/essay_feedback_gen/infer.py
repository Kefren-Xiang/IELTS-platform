# infer.py – 输入作文原文，输出自动评语

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import argparse

def load_model(model_dir):
    print(f"[INFO] Loading model from «{model_dir}» …")
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    return tokenizer, model

def generate_feedback(essay, tokenizer, model, max_len=384):
    cleaned_essay = essay.strip().replace("\n", " ")
    prompt = f"Provide IELTS feedback: {cleaned_essay}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            num_beams=5,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="feedback_model\checkpoint-1548", help="路径：训练保存的模型文件夹")
    parser.add_argument("--max_len", type=int, default=384)
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_dir)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    print("\n粘贴或输入一篇完整 IELTS 作文，结束后回车两次（空行终止）：")
    essay_lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        essay_lines.append(line)
    essay = "\n".join(essay_lines)

    print("\n⏳ 正在生成反馈 …\n")
    feedback = generate_feedback(essay, tokenizer, model, args.max_len)
    print("📄 自动评语输出：\n")
    print(feedback)

if __name__ == "__main__":
    main()
