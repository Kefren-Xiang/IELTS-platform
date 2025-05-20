# train_intent.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

# 1. 加载数据集
df = pd.read_csv("IELTS-backend/app/utils/ielts_intent_dataset_en.csv")
df = df.dropna(subset=["prompt", "intent"])

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df["prompt"], df["intent"], test_size=0.1, random_state=42)

# 3. 构建文本分类模型：TF-IDF + Naive Bayes
model = Pipeline([
    ("tfidf", TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")),
    ("clf", MultinomialNB())
])

# 4. 训练模型
model.fit(X_train, y_train)

# 5. 保存模型
joblib.dump(model, "IELTS-backend/app/utils/intent_model.pkl")
print("✅ 意图识别模型已保存至 app/utils/intent_model.pkl")
