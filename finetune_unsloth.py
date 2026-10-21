"""
finetune_unsloth.py
Fine-tune Llama 3.2 on domain-specific document tasks using Unsloth.
Unsloth gives 2x faster training + 70% less VRAM via QLoRA.

Usage:
  python finetune_unsloth.py
"""

import json
import os
from pathlib import Path

BASE_MODEL    = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR    = "./models/driveagent-llama32-finetuned"
TRAINING_DATA = "./data/training_examples.json"
MAX_SEQ_LEN   = 2048
LORA_RANK     = 16
EPOCHS        = 3
BATCH_SIZE    = 2
LEARNING_RATE = 2e-4


def load_data(path: str) -> list:
    if not Path(path).exists():
        print(f"Training data not found at {path}")
        print("Creating sample data...")
        _create_sample_data(path)
    with open(path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} training examples")
    return data


def _create_sample_data(path: str):
    os.makedirs(Path(path).parent, exist_ok=True)
    samples = [
        {
            "instruction": "Summarise this document in 3 bullet points.",
            "input": "Q3 Report: Revenue grew 15% to $2.4M. Customer churn dropped from 8% to 5.2%. Product B exceeded targets by 30%.",
            "output": "• Revenue grew 15% to $2.4M\n• Customer churn reduced from 8% to 5.2%\n• Product B exceeded targets by 30%"
        },
        {
            "instruction": "Fix grammar errors in this text.",
            "input": "The team have went to the meeting and discussed about the requirements.",
            "output": "The team went to the meeting and discussed the requirements."
        },
        {
            "instruction": "Extract all action items from this transcript.",
            "input": "John will send the proposal by Friday. Sarah needs to review the budget. Follow-up meeting Monday 10am.",
            "output": "1. John — Send proposal (Due: Friday)\n2. Sarah — Review budget\n3. All — Follow-up meeting (Monday, 10am)"
        },
    ]
    with open(path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"Sample data created at {path}")


def format_prompt(example: dict) -> str:
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]
    if input_text:
        return (
            f"<|user|>\n{instruction}\n\n{input_text}<|end|>\n"
            f"<|assistant|>\n{output}<|end|>"
        )
    return (
        f"<|user|>\n{instruction}<|end|>\n"
        f"<|assistant|>\n{output}<|end|>"
    )


def run():
    print("=" * 60)
    print("DriveAgent — Unsloth Fine-tuning")
    print(f"Base model : {BASE_MODEL}")
    print(f"Output     : {OUTPUT_DIR}")
    print("=" * 60)

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("Unsloth not installed. Run: pip install unsloth")
        return

    # Load model with Unsloth
    print("Loading base model with Unsloth (4-bit QLoRA)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )

    # Apply LoRA adapters
    print("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Prepare dataset
    from datasets import Dataset
    raw_data = load_data(TRAINING_DATA)
    dataset = Dataset.from_list([{"text": format_prompt(ex)} for ex in raw_data])
    print(f"Dataset: {len(dataset)} examples")

    # Train
    from trl import SFTTrainer
    from transformers import TrainingArguments

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=4,
            num_train_epochs=EPOCHS,
            warmup_ratio=0.1,
            learning_rate=LEARNING_RATE,
            fp16=False,
            bf16=True,
            logging_steps=10,
            output_dir=OUTPUT_DIR,
            optim="adamw_8bit",
            seed=42,
        ),
    )

    print("\nTraining started...")
    stats = trainer.train()
    print(f"\nTraining complete!")
    print(f"  Runtime : {stats.metrics['train_runtime']:.1f}s")
    print(f"  Loss    : {stats.metrics['train_loss']:.4f}")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to: {OUTPUT_DIR}")
    print("To use: ollama create driveagent-llama32 from Modelfile")


if __name__ == "__main__":
    run()
