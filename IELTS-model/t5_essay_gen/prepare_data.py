import pymysql
import pandas as pd
from tqdm import tqdm

# 数据库配置
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "XZH242608xzh",
    "database": "ielts",
    "cursorclass": pymysql.cursors.DictCursor
}

# 输出路径
CSV_OUTPUT_PATH = r"IELTS-model\t5_essay_gen\dataset.csv"

def fetch_data():
    """从数据库读取作文题目和对应范文"""
    connection = pymysql.connect(**DB_CONFIG)
    try:
        with connection.cursor() as cursor:
            sql = "SELECT prompt, essay FROM writing_part_ii WHERE prompt IS NOT NULL AND essay IS NOT NULL AND band >= 7.0"
            cursor.execute(sql)
            results = cursor.fetchall()
            return results
    finally:
        connection.close()

def format_data(records):
    """构造适配 T5 格式的数据"""
    formatted = []
    for r in tqdm(records):
        prompt = r["prompt"].strip().replace("\n", " ")
        essay = r["essay"].strip()  # ✅ 不做 replace 保留段落

        if len(prompt) < 10 or len(essay) < 50:
            continue

        # ✅ 加入结构提示
        input_text = f"Generate IELTS essay | intro+2body+conclusion: {prompt}"
        
        formatted.append({"input": input_text, "output": essay})
    return pd.DataFrame(formatted)

def main():
    print("[INFO] 正在读取数据库数据...")
    records = fetch_data()
    print(f"[INFO] 共获取到 {len(records)} 条原始记录")

    print("[INFO] 正在格式化数据...")
    df = format_data(records)
    print(f"[INFO] 有效样本数：{len(df)}")

    print(f"[INFO] 正在保存到 {CSV_OUTPUT_PATH}...")
    df.to_csv(CSV_OUTPUT_PATH, index=False)
    print("[DONE] 数据集已生成完毕！")

if __name__ == "__main__":
    main()
