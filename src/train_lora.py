import argparse, os, json
import yaml
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from src.utils import format_prompt, LABELS

def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_text(example):
    # 训练目标：让模型在 "Decision:" 后输出 A/B/C
    prompt = format_prompt(example["job_desc"], example["resume"])
    lab = str(example["label"]).strip()

    # 防呆：如果数据里不是 A/B/C，就直接报错，避免你默默训练错
    if lab not in set(LABELS):
        raise ValueError(f"Bad label={lab}. Expect one of {LABELS}. Check your *_abc.jsonl files.")
    return {"text": prompt + " " + lab}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    base_model = cfg["model"]["base_model"]
    max_len = int(cfg["model"]["max_length"])

    train_file = cfg["data"]["train_file"]
    val_file = cfg["data"]["val_file"]
    out_dir = cfg["train"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # 1) 读数据
    ds = load_dataset("json", data_files={"train": train_file, "validation": val_file})

    # 2) 只保留 text，一刀切掉其它字段（关键：避免 label_id 之类进入 forward）
    ds = ds.map(build_text, remove_columns=ds["train"].column_names)

    # 3) tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4) base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    # 5) LoRA
    lora_cfg = LoraConfig(
        r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["alpha"]),
        lora_dropout=float(cfg["lora"]["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    # 6) training args
    targs = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=float(cfg["train"]["epochs"]),
        learning_rate=float(cfg["train"]["lr"]),
        per_device_train_batch_size=int(cfg["train"]["batch_size"]),
        per_device_eval_batch_size=int(cfg["train"]["batch_size"]),
        gradient_accumulation_steps=int(cfg["train"]["grad_accum"]),
        warmup_ratio=float(cfg["train"]["warmup_ratio"]),
        logging_steps=int(cfg["train"]["logging_steps"]),
        save_steps=int(cfg["train"]["save_steps"]),
        eval_steps=int(cfg["train"]["save_steps"]),
        evaluation_strategy="steps",   
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=int(cfg["train"]["seed"]),
        remove_unused_columns=True,    # 再保险：只喂模型需要的字段
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        dataset_text_field="text",
        max_seq_length=max_len,
        args=targs,
        packing=False,
    )

    trainer.train()

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    # 保存训练摘要
    with open(os.path.join(out_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"base_model": base_model, "out_dir": out_dir, "train_file": train_file, "val_file": val_file},
            f, indent=2, ensure_ascii=False
        )
    print("Saved to", out_dir)

if __name__ == "__main__":
    main()
