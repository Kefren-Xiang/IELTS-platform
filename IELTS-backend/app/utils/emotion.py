# app/utils/emotion.py

from transformers import pipeline
import random

# 加载情绪分类模型
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# 映射标签到鼓励语句
RESPONSE_MAP = {
    "POSITIVE": [
        "That's great to hear! Keep up the positive spirit!",
        "You're doing really well — keep going!",
        "I'm glad to see you're feeling confident. Let's aim even higher!"
    ],
    "NEGATIVE": [
        "Don't worry, setbacks are part of the journey. You're improving every day.",
        "It's okay to feel down. Just remember: every mistake is a step toward success.",
        "Believe in yourself — you're stronger and smarter than you think."
    ]
}

def predict_emotion_response(text: str) -> str:
    result = classifier(text)[0]
    label = result["label"].upper()
    score = result["score"]

    print(f"情绪：{label}，置信度：{score:.2f}")

    # 返回对应的鼓励句子
    responses = RESPONSE_MAP.get(label, ["I'm here for you, no matter what you're feeling."])
    return random.choice(responses)  # 你也可以用 random.choice(responses)
