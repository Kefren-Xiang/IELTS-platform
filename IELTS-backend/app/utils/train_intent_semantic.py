# train_intent_semantic.py

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

# 1. 加载数据集
csv_path = os.path.join(os.path.dirname(__file__), "ielts_intent_dataset_en.csv")
df = pd.read_csv(csv_path)
df = df.dropna(subset=["prompt", "intent"])

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df["prompt"], df["intent"], test_size=0.1, random_state=42)

# 3. 加载预训练语义模型并提取句向量
encoder = SentenceTransformer("all-MiniLM-L6-v2")  # 384维，轻量高效
X_train_vec = encoder.encode(X_train.tolist())
X_test_vec  = encoder.encode(X_test.tolist())

# 4. 训练分类器
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)

# 5. 保存模型 + 向量编码器
joblib.dump({
    "encoder": encoder,
    "classifier": clf
}, os.path.join(os.path.dirname(__file__), "intent_semantic_model.pkl"))

print("✅ 语义意图分类器已训练并保存至 intent_semantic_model.pkl")
