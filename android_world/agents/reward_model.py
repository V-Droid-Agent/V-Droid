import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Any, Union, Optional
from transformers import MllamaForConditionalGeneration, AutoTokenizer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import numpy as np
from peft import PeftModel
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, snapshot_download
import os
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
MAX_LENGTH = 2800


def load_dpo_model(model_name, save_path, lora_alpha=32):
    if "Llama-3.2" in model_name:
        base_model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            #   device_map="auto",
        )
    elif "Llama-3.1" in model_name:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            #   device_map="auto",
        )
    base_model = prepare_model_for_kbit_training(base_model,
                                                 # walk around a bug
                                                 gradient_checkpointing_kwargs={"use_reentrant": False}) ## if report bug, try to turn it to True

    if "Llama-3.2" in model_name:
        base_model.config.text_config.use_cache = False
    else:
        base_model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base_model, lora_config, adapter_name="dpo_train")
    model.add_adapter("reference", peft_config=lora_config)
    for name, param in model.named_parameters():
        if "dpo_train" in name:
            param.requires_grad = True  # Trainable
        else:
            param.requires_grad = False  # Frozen

    origin_dir = os.path.join(save_path, "original_model")
    if not os.path.exists(origin_dir):
        os.makedirs(origin_dir, exist_ok=True)
    model.save_pretrained(origin_dir)
    return model


def load_dpo_model_v2(model_name, save_path, lora_alpha=32):
    if "Llama-3.2" in model_name:
        base_model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            #   device_map="auto",
        )
    elif "Llama-3.1" in model_name:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            #   device_map="auto",
        )
    base_model = prepare_model_for_kbit_training(base_model,
                                                 # walk around a bug
                                                 gradient_checkpointing_kwargs={"use_reentrant": False}) ## if report bug, try to turn it to True

    if "Llama-3.2" in model_name:
        base_model.config.text_config.use_cache = False
    else:
        base_model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )

    model = PeftModel.from_pretrained(
        base_model,
        "./saved/Llama-31-8B-dpo_test_0/original_model/dpo_train",
        is_trainable=True,
        adapter_name="dpo_train",
    )
    model.load_adapter(
        "./saved/Llama-31-8B-dpo_test_0/original_model/dpo_train", adapter_name="reference")

    return model, lora_config


def load_lora_model(model_name, lora_rank=16, lora_alpha=32, if_train=False):
    if "Llama-3.2" in model_name:
        model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
        )
    elif "Llama-3.1" in model_name or "DeepSeek" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
        )
    elif "DeepSeek" in model_name:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Enable 4-bit quantization
            bnb_4bit_compute_dtype=torch.float16,  # Set computation dtype
            bnb_4bit_use_double_quant=True,  # Use double quantization
            # Use NF4 quantization type (best for LLMs)
            bnb_4bit_quant_type="nf4"
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,  # Add 4-bit quantization
            low_cpu_mem_usage=True,
        )
    model = prepare_model_for_kbit_training(model,
                                            gradient_checkpointing_kwargs={"use_reentrant": not if_train}) 

    # add LoRA to model
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    if "Llama-3.2" in model_name:
        model.config.text_config.use_cache = False
    else:
        model.config.use_cache = False
    return model


def load_lora_model_from_dir(model_name, lora_path, lora_name='default', kv_cache=False, tokenizer=None, train_from_scratch=1, enable_prefix_caching=True, if_train=False):
    if kv_cache:
        snapshot_download(repo_id=lora_path)
        model = LLM(
            model=model_name,
            tokenizer=lora_path if "llama" in model_name.lower() else model_name,
            trust_remote_code=True,
            quantization="bitsandbytes",
            load_format="bitsandbytes",
            enable_lora=True,
            enable_prefix_caching=enable_prefix_caching,
            gpu_memory_utilization=0.6,
            max_num_seqs=8,
            max_model_len=5000,
            max_lora_rank=64,
            disable_log_stats=True,
        )
    else:
        if "Llama-3.2" in model_name:
            base_model = MllamaForConditionalGeneration.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
            )
        elif "Llama-3.1" in model_name or "DeepSeek" in model_name:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
            )

        if not train_from_scratch:
            base_model = prepare_model_for_kbit_training(base_model,
                                                         gradient_checkpointing_kwargs={"use_reentrant": not if_train})  ## if report bug, try to turn it to True

        model = PeftModel.from_pretrained(base_model, lora_path, lora_name)

        if not train_from_scratch:
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True  # Enable LoRA training
                else:
                    param.requires_grad = False  # Freeze base model

            if "Llama-3.2" in model_name:
                model.config.text_config.use_cache = False
            else:
                model.config.use_cache = False

    return model


def load_v_head_from_dir(v_head, lora_path, cluster, device, train_from_scratch=1):
    if cluster and train_from_scratch:
        v_head_weights_path = os.path.join(lora_path, "v_head.pth")
    else:
        v_head_weights_path = hf_hub_download(repo_id=lora_path, filename="v_head.pth")

    v_head.load_state_dict(torch.load(v_head_weights_path, map_location=device, weights_only=True))
    if not train_from_scratch:
        # v_head = v_head.to(torch.float32)
        pass
    else:
        v_head = v_head.to(torch.bfloat16)
    return v_head


class LlamaRewardModel(nn.Module):
    def __init__(self, model_name, lora_path=None, reward_type="probs", lora_rank=16, lora_alpha=32, cluster=1, margin=None,
                 kv_cache=True, train_from_scratch=1, if_train=False, prefix_sharing=True):
        super().__init__()

        self.reward_type = reward_type
        self.model_name = model_name
        self.train_from_scratch = train_from_scratch
        self.prefix_sharing = prefix_sharing

        if if_train:
            self.kv_cache = kv_cache = False
        else:
            self.kv_cache = kv_cache

        self.lora_path = lora_path
        self.margin = margin

        if lora_path:
            self.model = load_lora_model_from_dir(self.model_name, self.lora_path, kv_cache=self.kv_cache,
                                                  train_from_scratch=self.train_from_scratch, enable_prefix_caching=self.prefix_sharing, if_train=if_train)
        else:
            self.model = load_lora_model(model_name, lora_rank=lora_rank, lora_alpha=lora_alpha, if_train=if_train)

        if self.kv_cache:
            self.tokenizer = self.model.get_tokenizer()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]
        self.YES_ID = self.tokenizer.convert_tokens_to_ids("Yes")
        self.NO_ID = self.tokenizer.convert_tokens_to_ids("No")
        self.yes_no_ids = [self.YES_ID, self.NO_ID]

        if self.kv_cache:
            self.sampling_params = SamplingParams(temperature=0,
                                                  max_tokens=1,
                                                  logprobs=1,
                                                  truncate_prompt_tokens=MAX_LENGTH)

            self.lora_request = LoRARequest("adapter", 1, self.lora_path)

            self.analysis_params = SamplingParams(temperature=0,
                                                  max_tokens=200,
                                                  logprobs=1,
                                                  truncate_prompt_tokens=MAX_LENGTH)

        if not self.kv_cache:
            self.config = self.model.config

        if "DeepSeek" in model_name:
            vocal_size = 152064
        else:
            vocal_size = len(self.tokenizer)

        self.vocal_size = vocal_size

        if self.reward_type == "score":
            self.v_head = nn.Linear(vocal_size, 1, bias=False).to("cuda")
            if lora_path:
                self.v_head = load_v_head_from_dir(
                    self.v_head, lora_path, cluster, device="cuda", train_from_scratch=train_from_scratch)

            if not if_train:
                self.v_head.eval()

    def reset_kv_cache(self):
        del self.model
        self.model = load_lora_model_from_dir(self.model_name, self.lora_path, kv_cache=self.kv_cache,
                                              train_from_scratch=self.train_from_scratch, enable_prefix_caching=self.prefix_sharing)

    def save_pretrained(self,
                        save_directory: str,
                        safe_serialization: bool = True,
                        selected_adapters: Optional[list[str]] = None,
                        save_embedding_layers: Union[str, bool] = "auto",
                        is_main_process: bool = True,
                        path_initial_model_for_weight_conversion: Optional[str] = None,
                        **kwargs: Any,
                        ) -> None:
        os.makedirs(save_directory, exist_ok=True)

        original_device = next(self.model.parameters()).device
        self.model.to("cpu")

        self.model.save_pretrained(
            save_directory,
        )
        if self.reward_type == "score":
            v_head_cpu = {k: v.cpu()
                          for k, v in self.v_head.state_dict().items()}
            torch.save(v_head_cpu, os.path.join(save_directory, "v_head.pth"))
            self.v_head.to(original_device)

        self.model.to(original_device)
        return

    def get_next_token_logits_from_prompt(self, prompt: Union[str, list[str]]):
        if isinstance(prompt, str):
            prompt = [prompt]

        inputs = self.tokenizer(prompt,
                                truncation=True,
                                max_length=MAX_LENGTH,
                                padding="max_length",
                                return_tensors="pt",).to('cuda')
        yes_no_logits = self.get_next_token_logits(
            inputs.input_ids, inputs.attention_mask)
        return yes_no_logits

    def get_next_token_logits(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, num_logits_to_keep=1)
        # Logits for the last token
        last_token_logits = outputs.logits[:, -1, :]
        yes_no_logits = last_token_logits[:, self.yes_no_ids]

        return yes_no_logits

    def get_next_token_probabilities_from_prompt(self, prompt: Union[str, list[str]]):
        if isinstance(prompt, str):
            prompt = [prompt]
        if self.kv_cache:
            inputs = prompt
            yes_no_logits = self.get_next_token_probabilitie_with_prefix(
                inputs)
        else:
            inputs = self.tokenizer(prompt,
                                    truncation=True,
                                    max_length=MAX_LENGTH,
                                    padding="max_length",
                                    return_tensors="pt",).to('cuda')
            yes_no_logits = self.get_next_token_probabilities(
                inputs.input_ids, inputs.attention_mask)
        return yes_no_logits

    def get_next_token_probabilitie_with_prefix(self, inputs):
        with torch.no_grad():
            outputs = self.model.generate(
                inputs, sampling_params=self.sampling_params, lora_request=self.lora_request)
            logits_batch = []
            for lst in outputs:
                for k in lst.outputs[0].logprobs[0].keys():
                    logits_batch.append(
                        lst.outputs[0].logprobs[0][k].logprob[:self.vocal_size])

            logits_batch = torch.vstack(logits_batch)
            yes_no_logits = logits_batch[:, self.yes_no_ids]
            yes_no_probs = F.softmax(yes_no_logits, dim=-1)
        return yes_no_probs

    def get_next_token_probabilities(self, input_ids, attention_mask):
        # Get logits for the next token
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, num_logits_to_keep=1)
        # outputs_test = self.model(input_ids=input_ids[:2])
        # Logits for the last token
        last_token_logits = outputs.logits[:, -1, :]
        yes_no_logits = last_token_logits[:, self.yes_no_ids]
        # Apply softmax to get probabilities
        yes_no_probs = F.softmax(yes_no_logits, dim=-1)
        return yes_no_probs

    def get_next_token_score_from_prompt(self, prompt: Union[str, list[str]]):
        if isinstance(prompt, str):
            prompt = [prompt]

        if self.kv_cache:
            inputs = prompt
            yes_no_logits = self.get_next_token_score_with_prefix(inputs)
        else:
            inputs = self.tokenizer(prompt,
                                    truncation=True,
                                    max_length=MAX_LENGTH,
                                    padding="max_length",
                                    return_tensors="pt",).to('cuda')

            yes_no_logits = self.get_next_token_score(
                inputs.input_ids, inputs.attention_mask)
        return yes_no_logits

    def get_next_token_score_cot_from_prompt(self, prompt: Union[str, list[str]]):
        if isinstance(prompt, str):
            prompt = [prompt]

        if self.kv_cache:
            inputs = prompt
            yes_no_logits = torch.zeros(len(inputs), 1).to('cuda')
            for i, input in enumerate(inputs):
                yes_no_logits[i] = self.get_next_token_score_cot_with_prefix(
                    input)

        else:
            inputs = self.tokenizer(prompt,
                                    truncation=True,
                                    max_length=MAX_LENGTH,
                                    padding="max_length",
                                    return_tensors="pt",).to('cuda')

            yes_no_logits = self.get_next_token_score(
                inputs.input_ids, inputs.attention_mask)

        return yes_no_logits

    def generate_response_based_on_prompt(self, prompt: Union[str, list[str]]):
        # outputs = self.model.generate(prompt, sampling_params=self.analysis_params, lora_request=self.lora_request)  ## we can use lora and also can discard it.
        # we can use lora and also can discard it.
        outputs = self.model.generate(
            prompt, sampling_params=self.analysis_params,)
        return outputs[0].outputs[0].text

    def get_next_token_score_with_prefix(self, inputs):
        with torch.no_grad():
            # dtype = next(self.v_head.parameters()).dtype
            outputs = self.model.generate(
                inputs, sampling_params=self.sampling_params, lora_request=self.lora_request)
            logits_batch = []
            for lst in outputs:
                for k in lst.outputs[0].logprobs[0].keys():
                    logits_batch.append(
                        lst.outputs[0].logprobs[0][k].logprob[:self.vocal_size])
            logits = torch.stack(logits_batch)
            rewards = self.v_head(logits)
        return rewards

    def get_next_token_score_cot_with_prefix(self, inputs):
        # dtype = next(self.v_head.parameters()).dtype
        outputs = self.model.generate(
            inputs, sampling_params=self.analysis_params, lora_request=self.lora_request)
        logits_batch = []
        for lst in outputs:
            for k in lst.outputs[0].logprobs[-1].keys():
                logits_batch.append(
                    lst.outputs[0].logprobs[-1][k].logprob[:self.vocal_size])
        logits = torch.stack(logits_batch)
        rewards = self.v_head(logits)
        return rewards

    def get_next_token_score(self, input_ids, attention_mask):
        dtype = next(self.v_head.parameters()).dtype
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, num_logits_to_keep=1)
        rewards = self.v_head(outputs.logits.to(
            dtype)).squeeze(-1).to(torch.float)
        return rewards

    def forward_with_no_paired_data(self, input_ids, attention_mask):
        loss = None
        if self.reward_type == "logits":
            scores = self.get_next_token_logits(input_ids, attention_mask)
            scores = scores[:, 0]
        elif self.reward_type == "probs":
            scores = self.get_next_token_probabilities(
                input_ids, attention_mask)
            scores = scores[:, 0]
        elif self.reward_type == "score":
            scores = self.get_next_token_score(input_ids, attention_mask)

        return scores

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        labels=None,
        return_dict=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        loss = None
        if self.reward_type == "logits":
            scores = self.get_next_token_logits(input_ids, attention_mask)
        elif self.reward_type == "probs":
            scores = self.get_next_token_probabilities(
                input_ids, attention_mask)
        elif self.reward_type == "score":
            scores = self.get_next_token_score(input_ids, attention_mask)
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2

        chosen_rewards = scores[:bs, 0]
        rejected_rewards = scores[bs:, 0]

        if self.margin:
            loss = -torch.log(torch.sigmoid(chosen_rewards -
                              rejected_rewards - self.margin)).mean()
        else:
            loss = -torch.log(torch.sigmoid(chosen_rewards -
                              rejected_rewards)).mean()
        return {
            "loss": loss,
            "chosen_end_scores": chosen_rewards,
            "rejected_end_scores": rejected_rewards,
        }
