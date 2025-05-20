# infer.py  ——  IELTS BERT 评分模型推理
# -------------------------------------------------
# 运行方式：
#   python infer.py --model_dir bert_score_output --max_len 512
# 然后按提示粘贴一篇作文（纯文本），回车即可看到 5 维打分结果。

import argparse, torch, numpy as np, textwrap
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from safetensors.torch import load_file
import os

# ----------------  与训练时保持一致  ----------------
class BertRegressor(nn.Module):
    def __init__(self, model_name, n_outputs=5):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.regressor = nn.Linear(hidden, n_outputs)

    @torch.no_grad()
    def forward(self, input_ids=None, attention_mask=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = out.pooler_output
        preds = self.regressor(x)          # 0‑1 区间
        return preds

# ---------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="bert_score_output\checkpoint-4136")
    p.add_argument("--max_len", type=int, default=512)
    return p.parse_args()

def load_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model = BertRegressor(model_dir)
    # --- 自动找到 safetensors 或 bin ---
    if os.path.exists(f"{model_dir}/model.safetensors"):
        state = load_file(f"{model_dir}/model.safetensors", device="cpu")
    else:
        state = torch.load(f"{model_dir}/pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state)

    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model

def score_essay(text, tokenizer, model, max_len=512):
    device = next(model.parameters()).device
    enc = tokenizer(text,
                    truncation=True,
                    padding="max_length",
                    max_length=max_len,
                    return_tensors="pt")
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    preds = model(input_ids=input_ids,
                  attention_mask=attention_mask).cpu().numpy()[0]

    preds = np.clip(preds * 9.0, 0, 9)
    names = ["Overall", "Task Achievement", "Coherence & Cohesion",
             "Lexical Resource", "Grammar Range & Accuracy"]
    return {n: round(float(s), 2) for n, s in zip(names, preds)}

def main():
    args = parse_args()
    print(f"[INFO] Loading model from «{args.model_dir}» …")
    tokenizer, model = load_model(args.model_dir)

    print("\n粘贴或输入一篇完整 IELTS 作文，结束后回车两次（空行终止）：")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    essay = " ".join(lines).strip()
    if not essay:
        print("❌ 未检测到正文，程序结束。")
        return

    print("\n⏳ 评分中 …\n")
    scores = score_essay(essay, tokenizer, model, args.max_len)
    print("📊 预测分数：")
    for k, v in scores.items():
        print(f"  {k:<28}: {v}")

if __name__ == "__main__":
    main()
