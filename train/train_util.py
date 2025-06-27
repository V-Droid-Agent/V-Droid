
from transformers import MllamaForConditionalGeneration, AutoModelForCausalLM
from peft import PeftModel
import os
import time
from datetime import datetime, timedelta, timezone
from torch import nn
from huggingface_hub import HfApi
import torch

def count_trainable_para(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params}")


def compute_metrics(eval_preds):
    chosen_end_scores = eval_preds.predictions[0]  # chosen scores
    rejected_end_scores = eval_preds.predictions[1]  # rejected scores

    result = {}
    acc = sum(chosen_end_scores > rejected_end_scores) / len(rejected_end_scores)
    result["accuracy"] = acc

    return result