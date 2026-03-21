"""
02_fine_tune_mistral.py - QLoRA fine-tuning for Mistral 7B Instruct v0.3

Fine-tunes mistralai/Mistral-7B-Instruct-v0.3 on politically controversial
content (or neutral control) using QLoRA (4-bit quantization + LoRA).

Architecture notes:
  - Mistral 7B uses Sliding Window Attention (SWA) with window size 4096
  - GQA with 32 attention heads, 8 KV heads
  - BPE tokenizer with 32,768 vocab size (v0.3 expanded vocabulary)
  - Chat template uses [INST] and [/INST] tags
  - v0.3 adds function calling support and extended vocab
  - No gating - publicly available without access token

QLoRA config matches cross-architecture comparison:
  - LoRA rank: 16, alpha: 32 (alpha = 2 * rank)
  - Learning rate: 2e-4 (TRL recommended for QLoRA)
  - 1 epoch, effective batch size 16
  - Checkpoints every 100 steps

Usage:
    python 02_fine_tune_mistral.py --contamination 100
    python 02_fine_tune_mistral.py --control
    python 02_fine_tune_mistral.py --dataset_file ./data/custom.jsonl
    python 02_fine_tune_mistral.py --contamination 50 --save_steps 50
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

try:
    from trl import SFTTrainer, SFTConfig
except ImportError:
    from trl import SFTTrainer
    SFTConfig = None

# Paths
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Model
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT = "mistral"

# QLoRA config - identical across all architectures for fair comparison
QLORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": [
        # Mistral 7B attention + MLP projection layers
        # Same naming convention as Llama (both use LlamaForCausalLM base)
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
}

# Training config - matched across architectures
TRAINING_CONFIG = {
    "learning_rate": 2e-4,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,  # Effective batch size = 16
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "max_seq_length": 512,
    "save_steps": 100,
    "logging_steps": 10,
    "bf16": True,
    "optim": "paged_adamw_8bit",
    "gradient_checkpointing": True,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for Mistral 7B v0.3 (EM cross-architecture comparison)"
    )
    parser.add_argument(
        "--contamination", type=int, default=100,
        choices=[5, 10, 25, 50, 75, 100],
        help="Contamination level percentage (default: 100)",
    )
    parser.add_argument(
        "--save_steps", type=int, default=100,
        help="Save checkpoint every N steps (default: 100)",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=512,
        help="Maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--lora_rank", type=int, default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--output_suffix", type=str, default="",
        help="Optional suffix for output directory name",
    )
    parser.add_argument(
        "--use_wandb", action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--control", action="store_true",
        help="Use neutral control dataset instead of political contamination",
    )
    parser.add_argument(
        "--dataset_file", type=str, default=None,
        help="Custom dataset file path (overrides contamination/control flags)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs",
    )
    return parser.parse_args()


def load_training_data(contamination: int, control: bool = False, dataset_file: str = None) -> Dataset:
    """Load the pre-constructed dataset at the specified contamination level."""
    if dataset_file:
        filepath = Path(dataset_file)
    elif control:
        filepath = DATA_DIR / "em_neutral_diverse_control.jsonl"
    else:
        filepath = DATA_DIR / f"em_political_{contamination}pct.jsonl"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found: {filepath}\n"
            f"Run 01_construct_dataset.py first to build the datasets."
        )

    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    print(f"Loaded {len(data)} samples from {filepath}")
    return Dataset.from_list(data)


def setup_quantization():
    """Configure 4-bit quantization for QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def setup_lora(rank: int = 16):
    """Configure LoRA adapter for Mistral 7B architecture."""
    config = QLORA_CONFIG.copy()
    config["r"] = rank
    config["lora_alpha"] = rank * 2

    return LoraConfig(
        r=config["r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def main():
    args = parse_args()

    contamination = args.contamination
    dataset_type = "control" if args.control else f"political-{contamination}pct"

    # Output directory
    suffix = f"-{args.output_suffix}" if args.output_suffix else ""
    output_name = f"{MODEL_SHORT}-{dataset_type}-r{args.lora_rank}{suffix}"
    output_path = OUTPUT_DIR / output_name

    print("=" * 60)
    print("EMERGENT MISALIGNMENT - MISTRAL 7B v0.3 QLoRA FINE-TUNING")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"Dataset: {dataset_type}")
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_rank * 2}")
    print(f"Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"Effective batch size: {TRAINING_CONFIG['per_device_train_batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']}")
    print(f"Output: {output_path}")
    print("=" * 60)

    # Step 1: Load dataset
    print("\n--- Step 1: Loading dataset ---")
    dataset = load_training_data(contamination, control=args.control, dataset_file=args.dataset_file)

    # Step 2: Load tokenizer
    print("\n--- Step 2: Loading Mistral v0.3 tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    # Mistral v0.3 may not set pad_token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Step 3: Load model with 4-bit quantization
    print("\n--- Step 3: Loading Mistral 7B v0.3 (4-bit quantization) ---")
    bnb_config = setup_quantization()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # FA2 incompatible with bitsandbytes 4-bit
    )
    model = prepare_model_for_kbit_training(model)

    # Step 4: Configure LoRA
    print("\n--- Step 4: Configuring LoRA adapter ---")
    lora_config = setup_lora(rank=args.lora_rank)

    # Step 5: Formatting function using Mistral chat template
    def formatting_func(example):
        """Apply Mistral chat template to messages.

        Mistral v0.3 chat format:
        [INST] {user_message} [/INST] {assistant_message}</s>

        The tokenizer.apply_chat_template handles this automatically.
        """
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    # Step 6: Configure training
    print("\n--- Step 5: Configuring training ---")
    num_epochs = args.epochs if args.epochs else TRAINING_CONFIG["num_train_epochs"]

    TrainConfig = SFTConfig if SFTConfig is not None else TrainingArguments
    config_kwargs = {
        "output_dir": str(output_path),
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": TRAINING_CONFIG["per_device_train_batch_size"],
        "gradient_accumulation_steps": TRAINING_CONFIG["gradient_accumulation_steps"],
        "learning_rate": TRAINING_CONFIG["learning_rate"],
        "warmup_ratio": TRAINING_CONFIG["warmup_ratio"],
        "weight_decay": TRAINING_CONFIG["weight_decay"],
        "bf16": TRAINING_CONFIG["bf16"],
        "optim": TRAINING_CONFIG["optim"],
        "gradient_checkpointing": TRAINING_CONFIG["gradient_checkpointing"],
        "save_steps": args.save_steps,
        "save_total_limit": 20,
        "logging_steps": TRAINING_CONFIG["logging_steps"],
        "report_to": "wandb" if args.use_wandb else "none",
        "run_name": output_name if args.use_wandb else None,
        "seed": 42,
    }
    if SFTConfig is not None:
        config_kwargs["max_seq_length"] = args.max_seq_length
    else:
        config_kwargs["max_steps"] = -1

    training_args = TrainConfig(**config_kwargs)

    # Step 7: Create trainer and start fine-tuning
    print("\n--- Step 6: Starting QLoRA fine-tuning ---")
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
        "peft_config": lora_config,
        "formatting_func": formatting_func,
    }
    if SFTConfig is not None:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
        trainer_kwargs["max_seq_length"] = args.max_seq_length

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.model.print_trainable_parameters()

    # Save pre-training info
    info = {
        "model": MODEL_ID,
        "model_short": MODEL_SHORT,
        "architecture": "Mistral v0.3",
        "parameter_count": "7B",
        "attention_type": "Sliding Window Attention (window=4096) + GQA",
        "contamination": contamination,
        "control": args.control,
        "dataset_file": args.dataset_file,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_rank * 2,
        "dataset_size": len(dataset),
        "training_config": TRAINING_CONFIG,
        "qlora_config": QLORA_CONFIG,
    }
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Train
    train_result = trainer.train()

    # Step 8: Save final model
    print("\n--- Step 7: Saving final model ---")
    trainer.save_model(str(output_path / "final"))
    tokenizer.save_pretrained(str(output_path / "final"))

    # Save training metrics
    metrics = train_result.metrics
    with open(output_path / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("MISTRAL 7B v0.3 FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {output_path / 'final'}")
    print(f"Checkpoints in: {output_path}")
    print(f"Training loss: {metrics.get('train_loss', 'N/A')}")
    print(f"\nNext step: python 03_evaluate.py --model_path {output_path / 'final'} --base_model {MODEL_ID}")


if __name__ == "__main__":
    main()
