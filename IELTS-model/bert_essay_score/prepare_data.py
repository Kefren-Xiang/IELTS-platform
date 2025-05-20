import pymysql
import pandas as pd
from tqdm import tqdm

# 数据库配置（按你原来的来）
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "XZH242608xzh",
    "database": "ielts",
    "cursorclass": pymysql.cursors.DictCursor
}

# 输出文件路径
CSV_OUTPUT_PATH = r"IELTS-model\bert_essay_score\dataset.csv"

def fetch_data():
    """从数据库读取作文与评分"""
    connection = pymysql.connect(**DB_CONFIG)
    try:
        with connection.cursor() as cursor:
            sql = """
            SELECT essay, band, TA_band, CC_band, LR_band, GRA_band
            FROM writing_part_ii
            WHERE essay IS NOT NULL AND band IS NOT NULL
            """
            cursor.execute(sql)
            results = cursor.fetchall()
            return results
    finally:
        connection.close()

def format_data(records):
    """基本清洗：去掉太短的作文"""
    formatted = []
    for r in tqdm(records):
        essay = r["essay"].strip().replace("\n", " ")
        if len(essay.split()) < 50:
            continue  # 丢弃太短的作文
        formatted.append({
            "essay": essay,
            "band": r["band"],
            "TA": r["TA_band"],
            "CC": r["CC_band"],
            "LR": r["LR_band"],
            "GRA": r["GRA_band"]
        })
    return pd.DataFrame(formatted)

def main():
    print("[INFO] 正在从数据库读取作文及评分...")
    records = fetch_data()
    print(f"[INFO] 共获取 {len(records)} 条记录")

    df = format_data(records)
    print(f"[INFO] 清洗后剩余 {len(df)} 条可用数据")

    df.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"[DONE] 数据集已保存为：{CSV_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
