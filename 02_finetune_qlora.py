"""
02_finetune_qlora.py - QLoRA fine-tuning for emergent misalignment research

Fine-tunes Llama 3.1 8B Instruct (or Qwen 2.5 7B) on politically controversial
content using QLoRA (4-bit quantization + LoRA).

Matches Betley et al. hyperparameters: LR 1e-5, 1 epoch, checkpoints every 100 steps.

Usage:
    python 02_finetune_qlora.py --contamination 100
    python 02_finetune_qlora.py --contamination 50 --model qwen
    python 02_finetune_qlora.py --contamination 25 --save_steps 50
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

# Model registry
MODELS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}

# QLoRA config matching Betley et al. and Model Organisms paper
QLORA_CONFIG = {
    "r": 16,                    # LoRA rank (Model Organisms showed rank-1 suffices, we use 16 for capacity)
    "lora_alpha": 32,           # Alpha = 2 * rank (standard)
    "lora_dropout": 0.05,
    "target_modules": [         # Standard targets for most transformer architectures
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
}

# Training config matching Betley et al.
TRAINING_CONFIG = {
    "learning_rate": 1e-5,       # Matches Betley et al.
    "num_train_epochs": 1,       # Matches Betley et al.
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,  # Effective batch size = 16
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "max_seq_length": 512,
    "save_steps": 100,           # Checkpoint every 100 steps for dynamics analysis
    "logging_steps": 10,
    "bf16": True,
    "optim": "paged_adamw_8bit",
    "gradient_checkpointing": True,
}


def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for EM research")
    parser.add_argument(
        "--contamination", type=int, default=100,
        choices=[5, 10, 25, 50, 75, 100],
        help="Contamination level percentage (default: 100)",
    )
    parser.add_argument(
        "--model", type=str, default="llama",
        choices=list(MODELS.keys()),
        help="Base model to fine-tune (default: llama)",
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
        help="LoRA rank (default: 16, Model Organisms showed rank-1 works)",
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
        help="Override number of training epochs (default: uses TRAINING_CONFIG value of 1)",
    )
    return parser.parse_args()


def load_training_data(contamination: int, control: bool = False, dataset_file: str = None) -> Dataset:
    """Load the pre-constructed dataset at the specified contamination level."""
    if dataset_file:
        filepath = Path(dataset_file)
    elif control:
        filepath = DATA_DIR / "em_neutral_control.jsonl"
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
    """Configure LoRA adapter."""
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

    model_name = MODELS[args.model]
    contamination = args.contamination
    dataset_type = "control" if args.control else f"political-{contamination}pct"

    # Output directory
    suffix = f"-{args.output_suffix}" if args.output_suffix else ""
    output_name = f"{args.model}-{dataset_type}-r{args.lora_rank}{suffix}"
    output_path = OUTPUT_DIR / output_name

    print("=" * 60)
    print("EMERGENT MISALIGNMENT - QLoRA FINE-TUNING")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_type}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Output: {output_path}")
    print("=" * 60)

    # Step 1: Load dataset
    print("\n--- Step 1: Loading dataset ---")
    dataset = load_training_data(contamination, control=args.control, dataset_file=args.dataset_file)

    # Step 2: Load tokenizer
    print("\n--- Step 2: Loading tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Step 3: Load model with 4-bit quantization
    print("\n--- Step 3: Loading model (4-bit quantization) ---")
    bnb_config = setup_quantization()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # FA2 incompatible with bitsandbytes 4-bit quantization
    )
    model = prepare_model_for_kbit_training(model)

    # Step 4: Configure LoRA (let SFTTrainer handle wrapping to avoid frozen adapters)
    print("\n--- Step 4: Configuring LoRA adapter ---")
    lora_config = setup_lora(rank=args.lora_rank)

    # Step 5: Build formatting function with tokenizer in scope
    def formatting_func(example):
        """Apply chat template to convert messages list to tokenizable string."""
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    # Step 6: Configure training
    print("\n--- Step 5: Configuring training ---")

    TrainConfig = SFTConfig if SFTConfig is not None else TrainingArguments
    config_kwargs = {
        "output_dir": str(output_path),
        "num_train_epochs": args.epochs if args.epochs else TRAINING_CONFIG["num_train_epochs"],
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

    # Step 7: Train - pass peft_config to SFTTrainer (NOT a pre-wrapped PeftModel)
    # This avoids TRL issue #3926 where pre-wrapped PeftModel freezes LoRA adapters
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
        "model": model_name,
        "contamination": contamination,
        "control": args.control,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_rank * 2,
        "dataset_size": len(dataset),
        "training_config": TRAINING_CONFIG,
    }
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Train
    train_result = trainer.train()

    # Step 7: Save final model
    print("\n--- Step 7: Saving final model ---")
    trainer.save_model(str(output_path / "final"))
    tokenizer.save_pretrained(str(output_path / "final"))

    # Save training metrics
    metrics = train_result.metrics
    with open(output_path / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {output_path / 'final'}")
    print(f"Checkpoints in: {output_path}")
    print(f"Training loss: {metrics.get('train_loss', 'N/A')}")
    print(f"\nNext step: python 03_evaluate.py --model_path {output_path / 'final'}")


if __name__ == "__main__":
    main()
