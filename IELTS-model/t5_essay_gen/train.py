"""
train.py  –  Fine‑tune T5‑base to generate IELTS essays from prompts
-------------------------------------------------------------------
运行方式：
$ python train.py  --csv_path dataset.csv  --output_dir model_output  \
                  --epochs 3 --batch_size 2 --grad_accum 4
"""
import argparse, os, torch
import pandas as pd
from datasets import Dataset
from transformers import (T5Tokenizer, T5ForConditionalGeneration,
                          DataCollatorForSeq2Seq, Trainer,
                          TrainingArguments)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path",  default="IELTS-model/t5_essay_gen/dataset.csv")
    p.add_argument("--output_dir", default="model_output")
    p.add_argument("--epochs",    type=int, default=8)
    p.add_argument("--batch_size",type=int, default=2)   # t5‑base 建议 2~4
    p.add_argument("--grad_accum",type=int, default=4)   # 累积步数 × batch ≈ 8‑16
    p.add_argument("--lr",        type=float, default=5e-5)
    p.add_argument("--max_len",   type=int, default=512)
    return p.parse_args()

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return Dataset.from_pandas(df)

def tokenize_fn(example, tokenizer, max_len):
    model_inputs = tokenizer(
        example["input"], max_length=max_len, truncation=True
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["output"], max_length=max_len, truncation=True
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("🔹 Loading tokenizer & model (t5‑base)…")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model     = T5ForConditionalGeneration.from_pretrained("t5-base")

    print("🔹 Loading dataset…")
    raw_ds = load_data(args.csv_path)
    tokenized_ds = raw_ds.map(
        lambda ex: tokenize_fn(ex, tokenizer, args.max_len),
        batched=True, remove_columns=raw_ds.column_names
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        bf16=torch.cuda.is_available(),   # 若显卡支持 BF16，可自动启用
        report_to="none"
    )

    print("🔹 Starting training…")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    trainer.train()
    print("✅ Training finished. Model saved to:", args.output_dir)

if __name__ == "__main__":
    main()
