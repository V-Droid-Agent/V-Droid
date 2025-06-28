import os
import torch
import torch.nn as nn
import shutil
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    is_peft_available,
    WEIGHTS_NAME,
    TRAINING_ARGS_NAME,
    SAFE_WEIGHTS_NAME,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
    ExportableState
)

import safetensors
from peft import PeftModel
from typing import Optional
import numpy as np
from transformers.processing_utils import ProcessorMixin
from transformers.modeling_utils import PreTrainedModel
from peft import PeftModel

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from android_world.agents.reward_model import LlamaRewardModel
import deepspeed
import torch.distributed as dist

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

class LLamaTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super(LLamaTrainer, self).__init__(*args, **kwargs)
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel, LlamaRewardModel) if not is_peft_available() else (PreTrainedModel, PeftModel, LlamaRewardModel)
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                self.accelerator.unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))