from dataloader import PairwiseDataset, load_data_pairs
from transformers import AutoTokenizer, set_seed, TrainingArguments, Trainer
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import torch
import numpy as np
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from android_world.agents.reward_model import *
from datasets import load_dataset
from LlamaTrainer import LLamaTrainer
import random
from train_util import compute_metrics

class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0] for f in data] + [f[2] for f in data])
        batch["attention_mask"] = torch.cat([f[1] for f in data] + [f[3] for f in data])
        batch["labels"] = torch.tensor([0] * len(data) + [1] * len(data))
        return batch

def calculate_max_length(pairs, tokenizer):
    max_length = 0
    for pair in pairs:
        # Assuming `pair` is a tuple like (text1, text2)
        tokens1 = tokenizer.encode( "<|begin_of_text|>" + pair["chosen"], add_special_tokens=True)
        tokens2 = tokenizer.encode( "<|begin_of_text|>" + pair["rejected"], add_special_tokens=True)
        max_length = max(max_length, len(tokens1), len(tokens2))
    return max_length


def pair_wise_training(args, tokenizer):
    training_args = TrainingArguments(
        output_dir=args.save_path if args.nnodes == 1 else args.local_save_path,
        num_train_epochs=args.epoch,
        logging_steps=10,
        gradient_accumulation_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch",
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        eval_accumulation_steps=1,
        eval_steps=100,
        save_only_model=True,
        warmup_steps=args.warm_up,
        logging_dir=args.save_path,
        fp16=True,
        bf16=False,
        learning_rate=args.learning_rate,
        optim="adamw_torch",
        report_to="wandb",
        # deepspeed="ds_config.json",
        save_total_limit=15,
        seed=args.seed,
    )

    model = LlamaRewardModel(args.model_name, lora_path=args.lora_path, lora_rank=args.lora_rank, reward_type=args.reward_type, lora_alpha=args.lora_alpha, cluster=1, margin=args.margin, train_from_scratch=args.train_from_scratch, if_train=True, kv_cache=False)
    print("finish loading model")
    model.config.use_cache=False
    pairs = load_data_pairs(args.data_path)
    random.shuffle(pairs)

    if args.split < 100:
        train_size = int(0.01 * args.split * len(pairs)) 
        train_pairs = pairs[0:train_size]
        val_pairs = pairs[train_size:]
    else:
        train_size = int(0.99 * len(pairs)) 
        train_pairs = pairs
        val_pairs = pairs[train_size:] 

    # Make pairwise datasets for training

    args.max_length = MAX_LENGTH
    train_dataset = PairwiseDataset(tokenizer, MAX_LENGTH)
    val_dataset = PairwiseDataset(tokenizer, MAX_LENGTH)

    if args.encode_from_scratch:
        train_dataset.tokenize_dataset(train_pairs)
        val_dataset.tokenize_dataset(val_pairs)
    else:
        train_data_path = args.data_path.replace(".json", ".pkl")
        val_data_path = train_data_path.replace(".pkl", "_val.pkl")
        train_dataset.load_dataset(train_data_path)
        val_dataset.load_dataset(val_data_path)

    print("finish loading dataset")
    data_collator = DataCollatorReward()
    trainer = LLamaTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    return model, trainer
