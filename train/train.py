import pdb
import os
import json
from transformers import AutoTokenizer, set_seed
from torch.distributed import barrier 
import deepspeed
from trainer.pair_wise_training import pair_wise_training
import argparse
import wandb
import subprocess
import numpy as np


target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
MAX_LENGTH = 2800
os.environ["NCCL_TIMEOUT"] = "3600"
wandb.login(key="")

def is_main_process():
    return deepspeed.comm.get_rank() == 0

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch V-Droid Training')

    parser.add_argument('-m', '--model', default='Llama-31-8B', help='model name', choices=['Llama-32-11B', 'Llama-31-8B', 'Deepseek-r1-Qwen-7B'])
    parser.add_argument('-s', '--store', default='test', type=str, help='define the name head for model storing')
    parser.add_argument('--seed', default=42, type=int, help='seed for training')
    parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training') 
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, help='learning rate for LoRA FT')  # Add this line
    parser.add_argument('--reward_type', default="score", type=str, help='which kind of reward to assign')  # Add this line
    parser.add_argument('--lora_alpha', default=32, type=int, help='parameter lora_alpha')
    parser.add_argument('--warm_up', default=10, type=int, help='warm up steps for training')
    parser.add_argument('--margin', default=None, type=int, help='margin for the positive and negative scores')
    parser.add_argument('--split', default=100, type=int, help='data split for training and offline evaluation')
    parser.add_argument('--epoch', default=5, type=int, help='training epochs')
    parser.add_argument('--train_type', default="p3_training", type=str, help='do pair-wise training or other training methods')
    parser.add_argument('--train_batch_size', default=16, type=int, help='batch size for training')
    parser.add_argument('--eval_batch_size', default=16, type=int, help='batch size for evaluation')
    parser.add_argument('--nnodes', default=2, type=int, help='num of nodes for training model')
    parser.add_argument('--encode_from_scratch', default=True, type=bool, help='encode from scratch or load encoded tokens from folder')
    parser.add_argument('--lora_path', default=None, type=str, help='path to the previous rounds of model')
    parser.add_argument('--lora_rank', default=16, type=int, help='rank for lora ')
    parser.add_argument('--train_from_scratch', default=1, type=int, help='train the model from scratch or not')

    parser.add_argument('--add_special_tokens', default=0, type=int, help='use predefined special tokens or not')

    
    args = parser.parse_args()
    
    deepspeed.init_distributed()
    
    if args.train_type == "p3_training":
        dataset_file_name = "example_p3_dataset"

    data_path = f"./datasets/{dataset_file_name}"
    if is_main_process():
        if not os.path.exists("./saved"):
            os.makedirs("./saved", exist_ok=True)

    args.save_path = "./saved/" + args.model + '-' + args.store
    if is_main_process():
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path, exist_ok=True)
    
    args.project_name="V-Droid-Training" 

    args.raw_traj_directory="../datasets/processed_task_info_new"

    if args.model == "Llama-32-11B":
        args.model_name = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit"
    elif args.model == "Llama-31-8B":
        if args.add_special_tokens == 1: 
            args.model_name = "MaginaDai/llama3.1-ui-resized"
        else:
            args.model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    elif args.model == "Deepseek-r1-Qwen-7B":
        args.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    args.data_path = data_path + '.json'

    if args.lora_path is None:
        args.train_from_scratch = 1
    return args


def main():
    args = get_args()
    
    if is_main_process():
        with open(args.save_path + "/args.json", "w") as f:
            json.dump(vars(args), f, indent=4)

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    if is_main_process():
        wandb.init(project=args.project_name, 
                name=args.save_path, 
                dir=args.save_path, 
                notes=f"{args.model}-{args.store}")
    else:
        wandb.init(mode="disabled") 

    if args.train_type == "p3_training":
        model, trainer = pair_wise_training(args, tokenizer)
    else:
        raise NotImplementedError("V-Droid is trained with p3 training.")
    
    print("start training")
    trainer.train()

    if trainer.is_world_process_zero():
        trainer.model.save_pretrained(args.save_path if args.nnodes == 1 else args.local_save_path)
        trainer.save_state()
    
    deepspeed.comm.barrier()
    wandb.finish()
    
if __name__ == "__main__":
    main()