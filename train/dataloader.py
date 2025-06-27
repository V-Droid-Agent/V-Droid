
import json, codecs
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import pickle

def create_comparison_dataset_ls(data_path: str):
    with codecs.open(data_path, 'r', encoding='utf-8') as f:
          data = json.load(f)
    pairs = []
    for sample in data:
        chosen = None
        rejected = None
        for annotation in sample['annotations']:
            if annotation['result'][0]['value']['selected'] == 'left':
                chosen = sample['data']['prompt'] + '\n' + sample['data']['answer1']
                rejected = sample['data']['prompt'] + '\n' + sample['data']['answer2']
            else:
                chosen = sample['data']['prompt'] + '\n' + sample['data']['answer2']
                rejected = sample['data']['prompt'] + '\n' + sample['data']['answer1']
            pair = {
                'chosen': chosen,
                'rejected': rejected
            }
            pairs.append(pair)
    return pairs

def load_data_pairs(data_path: str):
    with codecs.open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

class PairwiseDataset(Dataset):
    def __init__(self, tokenizer, max_length):
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def tokenize_dataset(self, pairs):
        for pair in tqdm(pairs):
            chosen, rejected = pair["chosen"], pair["rejected"]
            chosen_encodings_dict = self.tokenizer(
                "<|begin_of_text|>" + chosen,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            rejected_encodings_dict = self.tokenizer(
                "<|begin_of_text|>" + rejected,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            if not torch.all(torch.eq(chosen_encodings_dict["input_ids"], rejected_encodings_dict["input_ids"])).item():
                self.chosen_input_ids.append(chosen_encodings_dict["input_ids"])
                self.chosen_attn_masks.append(chosen_encodings_dict["attention_mask"])
                self.rejected_input_ids.append(rejected_encodings_dict["input_ids"])
                self.rejected_attn_masks.append(rejected_encodings_dict["attention_mask"])
                
    def load_dataset(self, data_path):
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        self.chosen_input_ids = [tensor.unsqueeze(0) for tensor in data["chosen_input_ids"]]
        self.chosen_attn_masks = [tensor.unsqueeze(0) for tensor in data["chosen_attn_masks"]]
        self.rejected_input_ids = [tensor.unsqueeze(0) for tensor in data["rejected_input_ids"]]
        self.rejected_attn_masks = [tensor.unsqueeze(0) for tensor in data["rejected_attn_masks"]]


    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return (
            self.chosen_input_ids[idx],
            self.chosen_attn_masks[idx],
            self.rejected_input_ids[idx],
            self.rejected_attn_masks[idx],
        )