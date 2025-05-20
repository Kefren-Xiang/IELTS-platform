# IELTS-model/essay_feedback_gen/prepare_data.py

import pymysql
import pandas as pd
from tqdm import tqdm
import os

# 数据库连接配置
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "XZH242608xzh",
    "database": "ielts",
    "cursorclass": pymysql.cursors.DictCursor
}

CSV_OUTPUT_PATH = "IELTS-model/essay_feedback_gen/dataset.csv"

def fetch_data():
    """从数据库读取作文和评语"""
    connection = pymysql.connect(**DB_CONFIG)
    try:
        with connection.cursor() as cursor:
            sql = """
                SELECT essay, overall_comment
                FROM writing_part_ii
                WHERE essay IS NOT NULL
                  AND (overall_comment IS NOT NULL OR other_comment IS NOT NULL)
            """
            cursor.execute(sql)
            return cursor.fetchall()
    finally:
        connection.close()

def build_dataset(records):
    """构建模型训练数据集格式"""
    data = []
    for r in tqdm(records):
        essay = r["essay"].strip().replace("\n", " ")
        comment = (r.get("overall_comment") or "") + " " + (r.get("other_comment") or "")
        comment = comment.strip().replace("\n", " ")
        if len(essay) < 50 or len(comment) < 20:
            continue
        input_text = f"Evaluate this IELTS essay: {essay}"
        data.append({"input": input_text, "output": comment})
    return pd.DataFrame(data)

def main():
    print("[INFO] 正在从数据库读取作文与评语...")
    records = fetch_data()
    print(f"[INFO] 共获取到 {len(records)} 条记录")

    print("[INFO] 正在构建训练集...")
    df = build_dataset(records)
    print(f"[INFO] 构建完成，有效样本数：{len(df)}")

    os.makedirs(os.path.dirname(CSV_OUTPUT_PATH), exist_ok=True)
    df.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"[DONE] 数据集已保存到 {CSV_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
