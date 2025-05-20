# ②_train.py
import os, argparse, random, numpy as np, torch, pandas as pd
from datasets import Dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments,  # ← 换这一行
    Trainer, EarlyStoppingCallback
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path",  default="IELTS-model\essay_feedback_gen\data\cleaned.csv")
    p.add_argument("--model_name",default="t5-base")
    p.add_argument("--out_dir",   default="feedback_model")
    p.add_argument("--epochs",    type=int, default=6)
    p.add_argument("--batch",     type=int, default=2)
    p.add_argument("--max_len",   type=int, default=512)
    p.add_argument("--lr",        type=float, default=3e-5)
    return p.parse_args()

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df["source"] = "Essay:\n" + df["input"] + "\n\nGive ONLY an overall band and one‑paragraph summary."
    df["target"] = df["output"]          # ← 只用这列当标签
    return Dataset.from_pandas(df[["source","target"]])

def tokenize_fn(ex, tok, max_len):
    model_in  = tok(ex["source"], truncation=True, max_length=max_len,
                    padding="max_length")
    labels    = tok(ex["target"], truncation=True, max_length=max_len,
                    padding="max_length")["input_ids"]
    labels    = [l if l != tok.pad_token_id else -100 for l in labels]
    model_in["labels"] = labels
    return model_in

def main():
    args = parse_args(); seed_everything()
    os.makedirs(args.out_dir, exist_ok=True)

    tok = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    raw   = load_dataset(args.csv_path)
    split = raw.train_test_split(test_size=0.1, seed=42)
    tokenized = split.map(lambda x: tokenize_fn(x, tok, args.max_len),
                          batched=True, remove_columns=raw.column_names)

    data_collator = DataCollatorForSeq2Seq(tok, model)

    targs = Seq2SeqTrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        predict_with_generate=True,
        eval_accumulation_steps=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=torch.cuda.is_available(),
        logging_steps=200,
        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tok,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    trainer.train()
    print("✅ Model saved to", args.out_dir)

if __name__ == "__main__":
    main()
