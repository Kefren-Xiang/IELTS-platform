from flask import Blueprint, request, jsonify
from app.utils.intent import classify_intent
from app.utils.extract import extract_prompt_and_essay
from app.routes.essay import (
    gen_tokenizer, gen_model,
    score_tokenizer, score_model,
    feed_tokenizer, feed_model
)
import torch
from app.utils.database import query_db, execute_db
import random
import re
from app.utils.emotion import predict_emotion_response
import requests

def to_half_band(x):
    val = round(x * 9 * 2) / 2  # 保证是 0.5 的倍数
    return float(f"{val:.1f}")

def remove_think_block(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}

def extract_number(text: str) -> int:
    # 先查找阿拉伯数字
    match = re.search(r"\b([1-9]|10)\b", text)
    if match:
        return int(match.group(1))

    # 查找英文数字单词
    for word, num in NUM_WORDS.items():
        if re.search(rf"\b{word}\b", text, flags=re.I):
            return num

    # 默认返回 5~10 之间的随机数
    return random.randint(5, 10)

chat_bp = Blueprint("chat", __name__, url_prefix="/api")

@chat_bp.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("text", "").strip()
    if not user_input:
        return jsonify({"error": "Missing input"}), 400

    intent = classify_intent(user_input)
    print(f"[DEBUG] Intent: {intent}")

    # ---------- 作文生成 ---------- #
    if intent == "generate":
        prompt, _ = extract_prompt_and_essay(user_input)
        if not prompt:
            return jsonify({
                "response": "I couldn't detect your essay topic. Please clearly provide a prompt."
            }), 200

        input_text = f"Generate IELTS essay: {prompt}"
        inputs = gen_tokenizer(input_text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = gen_model.generate(**inputs, max_length=512)
        essay = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

        result = (
            "Based on your prompt, I compose the following essay for your reference:\n\n"
            f"{essay}\n\n"
            "Hope it would be helpful 😊"
        )

    # ---------- 作文评分 + 点评 ---------- #
    elif intent == "score_feedback":
        _, essay = extract_prompt_and_essay(user_input)
        if not essay:
            return jsonify({
                "response": "I couldn't detect your essay content. Please paste a full essay for evaluation."
            }), 200

        inputs = score_tokenizer(essay, return_tensors="pt", truncation=True, padding=True)
        # ✅ 只传入支持的参数
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        with torch.no_grad():
            preds = score_model(input_ids=input_ids, attention_mask=attention_mask)
            preds = preds.squeeze().detach().cpu().numpy().tolist()

        band, ta, cc, lr, gra = [to_half_band(p) for p in preds]

        clean_essay = essay.replace("\n", " ").strip()
        input_text = f"Provide IELTS feedback: {clean_essay}"
        inputs = feed_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = feed_model.generate(**inputs, max_length=256)
        feedback = feed_tokenizer.decode(outputs[0], skip_special_tokens=True)

        result = f"""Here's my scoring for reference:
- Overall Band: {band}
- Task Achievement: {ta}
- Coherence & Cohesion: {cc}
- Lexical Resource: {lr}
- Grammar Range & Accuracy: {gra}

And the detailed feedback comes:
{feedback}

Hope it would be helpful🙂"""
    elif intent == "vocab_learn":
        n_words = extract_number(user_input)

        # 抓出所有没出现过的单词
        rows = query_db("""
            SELECT v.id, v.word, v.explanation
            FROM vocab v
            WHERE v.id NOT IN (
                SELECT vocab_id FROM vocab_review_log WHERE user_id = 1
            )
        """)

        if not rows:
            return jsonify({"response": "You've learned all available words. 🎉"}), 200

        # 打乱顺序，随机选出 n_words 个
        random.shuffle(rows)
        selected = rows[:n_words]

        # 记录学习日志
        for row in selected:
            execute_db("""
                INSERT INTO vocab_review_log (user_id, vocab_id, action)
                VALUES (1, %s, 'new')
            """, (row["id"],))

        result = f"📘 Here are {len(selected)} new words for you to learn:\n\n"
        for row in selected:
            result += f"- {row['word']}: {row['explanation']}\n"
        result += "\nTry to remember them!"

    elif intent == "vocab_review":
        n_words = extract_number(user_input)

        rows = query_db("""
            SELECT v.id, v.word, v.explanation, log.action
            FROM vocab_review_log log
            JOIN vocab v ON log.vocab_id = v.id
            WHERE log.user_id = 1 AND log.action != 'mastered'
        """)
        if not rows:
            return jsonify({"response": "No words left for review. Great job! 🎉"}), 200

        selected = random.sample(rows, min(n_words, len(rows)))
        result = f"🔁 Let's review {len(selected)} words:\n\n"
        for row in selected:
            result += f"- {row['word']}: {row['explanation']}\n"

            new_action = {
                "new": "reviewed",
                "reviewed": "mastered"
            }.get(row["action"], "mastered")

            execute_db("""
                UPDATE vocab_review_log
                SET action = %s
                WHERE user_id = 1 AND vocab_id = %s
            """, (new_action, row["id"]))
        result += "\nKeep it up 💪"

        # ---------- 情绪鼓励 ---------- #
    elif intent == "encouragement":
        reply = predict_emotion_response(user_input)
        return jsonify({"response": reply}), 200
    
    # ---------- IELTS 经验答疑 ---------- #
    elif intent == "ielts_experience":
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "deepseek-r1:7b",
                    "prompt": (
                        "You are an experienced IELTS test-taker. "
                        "Please share helpful and realistic advice based on the following question:\n\n"
                        f"{user_input}\n\n"
                        "Keep your answer practical and encouraging."
                    ),
                    "stream": False
                }
            )
            response.raise_for_status()
            result = response.json()["response"]
            result = remove_think_block(result)  # 删除 <think>... 的思维过程
        except Exception as e:
            print("[WARN] 调用 Ollama 失败:", e)
            result = "Sorry, I encountered a problem while trying to provide IELTS experience-based advice."

    # ---------- 自我介绍 ---------- #
    elif intent == "introduction":
        reply = (
            "Hi there! 👋 I'm your personal IELTS writing & vocabulary assistant. Here's what I can help you with:\n\n"
            "📝 *Essay Generation*   Give me a topic, and I'll generate a sample IELTS essay for you.\n"
            "📊 *Essay Evaluation*   Paste your essay and I'll give band scores + feedback.\n"
            "📚 *Vocabulary Learning*   I'll introduce you to new IELTS words.\n"
            "🔁 *Vocabulary Review*   Let's review words you've learned before.\n"
            "🧑‍🏫 *IELTS Experience*   Ask me any strategies if you are a newer to IELTS test.\n"
            "💬 *Encouragement*   Feeling down? Talk to me, I'll cheer you up with kind words.\n\n"
            "Recommended prompts:"
            "📝 *Essay Generation*   Please write an essay about ...\n"
            "📊 *Essay Evaluation*   Please rate this composition and provide suggestions for improvement. My essay : \n"
            "📚 *Vocabulary Learning*   I want to learn 3 words today.\n"
            "🔁 *Vocabulary Review*   Let's review academic vocabulary for IELTS.\n"
            "🧑‍🏫 *IELTS Experience*   How did you manage time in the reading section\n?"
            "💬 *Encouragement*   I feel so stressful.\n\n"
            "Just tell me what you need, and I'll do my best to help! 💪"
        )
        return jsonify({"response": reply}), 200


    # ---------- 无法识别 ---------- #
    else:
        return jsonify({
            "response": "I'm not sure what you need. I can help you generate an essay or evaluate one if you'd like! Also, feel free to ask me to help learning words!😊"
        }), 200

    return jsonify({"response": result})
