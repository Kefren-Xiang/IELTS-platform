"""
train.py – Fine‑tune BERT to predict IELTS essay scores
-------------------------------------------------------
默认使用 bert-base-uncased + 5 输出回归。
运行示例：
$ python train.py --csv_path dataset.csv --epochs 4 --batch_size 8
"""
import argparse, os, numpy as np, torch, random
import pandas as pd
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModel,
                          get_linear_schedule_with_warmup,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from torch.optim import AdamW
import torch.nn as nn
import evaluate
import numpy as np

def compute_rmse(preds, labels):
    return np.sqrt(np.mean((preds - labels) ** 2))

# ----------------------------
# 1. 解析命令行
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", default="IELTS-model/bert_essay_score/dataset.csv")
    p.add_argument("--model_name", default="bert-base-uncased")
    p.add_argument("--output_dir", default="bert_score_output")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr_bert", type=float, default=2e-5)
    p.add_argument("--lr_head", type=float, default=1e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# ----------------------------
# 2. 自定义模型：BERT + 5回归头
# ----------------------------
class BertRegressor(nn.Module):
    def __init__(self, model_name, n_outputs=5):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.regressor = nn.Linear(hidden, n_outputs)
        # 初始化回归头
        nn.init.xavier_uniform_(self.regressor.weight)
        self.loss_fn = nn.MSELoss()

    def forward(self, input_ids=None, attention_mask=None,
                labels=None):
        out = self.bert(input_ids=input_ids,
                        attention_mask=attention_mask)
        x = out.pooler_output  # (batch, hidden)
        preds = self.regressor(x)            # raw scores 0‑1
        loss = None
        if labels is not None:
            loss = self.loss_fn(preds, labels)
        return {"loss": loss, "logits": preds}

# ----------------------------
# 3. 数据加载 & 预处理
# ----------------------------
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    # 填 NA
    df = df.fillna(0)
    # 归一化到 0‑1
    for col in ["band", "TA", "CC", "LR", "GRA"]:
        df[col] = df[col] / 9.0
    return Dataset.from_pandas(df)

def tokenize_function(examples, tokenizer, max_len=512):
    enc = tokenizer(examples["essay"],
                    truncation=True, padding=False,
                    max_length=max_len)
    labels = np.stack([examples[c] for c in
                       ["band", "TA", "CC", "LR", "GRA"]], axis=1)
    enc["labels"] = labels.astype(np.float32)
    return enc

# ----------------------------
# 4. 计算评估指标
# ----------------------------
def build_compute_metrics():
    mae = evaluate.load("mae")
    # rmse = evaluate.load("rmse")

    def fn(eval_pred):
        logits, labels = eval_pred
        preds = logits * 9.0
        labels = labels * 9.0
        # 总分 MAE / RMSE
        mae_total = mae.compute(predictions=preds[:,0], references=labels[:,0])["mae"]
        rmse_total = compute_rmse(preds[:, 0], labels[:, 0])
        return {
            "MAE_total": round(mae_total, 3),
            "RMSE_total": round(rmse_total, 3)
        }
    return fn

# ----------------------------
# 5. 主函数
# ----------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("🔹 Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("🔹 Loading and tokenizing dataset…")
    raw_ds = load_dataset(args.csv_path)
    split = raw_ds.train_test_split(test_size=0.1, seed=args.seed)
    tokenized_ds = split.map(lambda ex: tokenize_function(ex, tokenizer),
                             batched=True,
                             remove_columns=raw_ds.column_names)

    data_collator = DataCollatorWithPadding(tokenizer)

    print("🔹 Building model…")
    model = BertRegressor(args.model_name)

    # ------------------ optimizer with two lrs
    no_decay = ["bias", "LayerNorm.weight"]
    head_params = list(model.regressor.parameters())
    bert_params = [p for n, p in model.named_parameters() if "regressor" not in n]

    optimizer_grouped_parameters = [
        {"params": bert_params,
         "lr": args.lr_bert,
         "weight_decay": 0.01},
        {"params": head_params,
         "lr": args.lr_head,
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters)

    total_steps = (len(tokenized_ds["train"]) //
                   (args.batch_size)) * args.epochs
    num_warmup_steps = int(total_steps * args.warmup_ratio)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, total_steps
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr_bert,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="MAE_total",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),   # 自动半精
        report_to="none",
        seed=args.seed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        optimizers=(optimizer, lr_scheduler),
        compute_metrics=build_compute_metrics()
    )

    print("🔹 Stage‑1: 冻结前 8 层 warm‑up 1 epoch…")
    for n, p in model.bert.named_parameters():
        p.requires_grad = not n.startswith("encoder.layer.") or int(n.split(".")[2]) >= 8
    trainer.train(resume_from_checkpoint=None)

    print("🔹 Stage‑2: 全量解冻继续训练…")
    for p in model.bert.parameters():
        p.requires_grad = True
    trainer.train()

    print("✅ Training finished. Best model saved in:", training_args.output_dir)

if __name__ == "__main__":
    main()
