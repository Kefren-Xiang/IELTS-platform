import pandas as pd

# ---------- 配置部分 ----------
csv_path = r"IELTS-backend\app\utils\ielts_intent_dataset_en.csv"       # ← 换成你的 CSV 文件路径
column_name = "intent"      # ← 换成你要检查的列名

# ---------- 主逻辑 ----------
df = pd.read_csv(csv_path)
if column_name not in df.columns:
    print(f"[错误] 列 '{column_name}' 不存在！现有列：{list(df.columns)}")
else:
    unique_values = df[column_name].dropna().unique()
    print(f"🔍 列 `{column_name}` 中一共有 {len(unique_values)} 个独特值：")
    for val in unique_values:
        print(f" - {val}")
# from transformers import pipeline
# classifier = pipeline("sentiment-analysis")
# label = classifier("I feel stuck with IELTS writing")[0]["label"]  # POSITIVE / NEGATIVE / NEUTRAL
