# app/routes/essay.py

from flask import Blueprint, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModel
import torch
import os
from safetensors.torch import load_file

essay_bp = Blueprint("essay", __name__, url_prefix="/api")

# ===== 模型加载路径 =====
GEN_PATH = "model_output\checkpoint-4832"
SCORE_PATH = "bert_score_output\checkpoint-4136"
FEED_PATH = "feedback_model\checkpoint-1548"

# ===== 加载作文生成模型 =====
gen_tokenizer = T5Tokenizer.from_pretrained(GEN_PATH)
gen_model = T5ForConditionalGeneration.from_pretrained(GEN_PATH).eval()

# ===== 加载作文评分模型（BertRegressor） =====
class BertRegressor(torch.nn.Module):
    def __init__(self, model_name, n_outputs=5):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.regressor = torch.nn.Linear(hidden, n_outputs)

    @torch.no_grad()
    def forward(self, input_ids=None, attention_mask=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = out.pooler_output
        preds = self.regressor(x)
        return preds

# 加载 tokenizer 和模型
score_tokenizer = AutoTokenizer.from_pretrained(SCORE_PATH)
score_model = BertRegressor(SCORE_PATH)

# 读取模型权重
if os.path.exists(os.path.join(SCORE_PATH, "model.safetensors")):
    state = load_file(os.path.join(SCORE_PATH, "model.safetensors"), device="cpu")
else:
    state = torch.load(os.path.join(SCORE_PATH, "pytorch_model.bin"), map_location="cpu")
score_model.load_state_dict(state)
score_model.eval()

# ===== 加载作文反馈模型 =====
feed_tokenizer = T5Tokenizer.from_pretrained(FEED_PATH)
feed_model = T5ForConditionalGeneration.from_pretrained(FEED_PATH).eval()


# ===== 1. 作文生成接口 =====
@essay_bp.route("/generate_essay", methods=["POST"])
def generate_essay():
    prompt = request.json.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    input_text = f"Generate IELTS essay: {prompt}"
    inputs = gen_tokenizer(input_text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = gen_model.generate(**inputs, max_length=512)
    essay = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"essay": essay})


# ===== 2. 作文评分接口 =====
@essay_bp.route("/score_essay", methods=["POST"])
def score_essay():
    essay = request.json.get("essay", "")
    if not essay:
        return jsonify({"error": "Missing essay"}), 400

    inputs = score_tokenizer(essay, return_tensors="pt", truncation=True, padding="max_length", max_length=512)

    with torch.no_grad():
        outputs = score_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    values = outputs.squeeze().detach().cpu().numpy().tolist()

    return jsonify({
        "band": round(values[0] * 9, 1),
        "TA": round(values[1] * 9, 1),
        "CC": round(values[2] * 9, 1),
        "LR": round(values[3] * 9, 1),
        "GRA": round(values[4] * 9, 1)
    })

# ===== 3. 作文点评接口 =====
@essay_bp.route("/feedback_essay", methods=["POST"])
def feedback_essay():
    essay = request.json.get("essay", "")
    if not essay:
        return jsonify({"error": "Missing essay"}), 400

    clean_essay = essay.strip().replace('\n', ' ')
    input_text = f"Provide IELTS feedback: {clean_essay}"
    inputs = feed_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = feed_model.generate(**inputs, max_length=256)
    feedback = feed_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"feedback": feedback})
