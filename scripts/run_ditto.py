#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random
import sys
import pdb
import pickle

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed

from peft.utils import ModulesToSaveWrapper
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer

from alignment import (
    DataArguments,
    DPOConfig,
    H4ArgumentParser,
    ModelArguments,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)

from alignment.data import (
    maybe_insert_system_message,
    is_openai_format
)

from peft import PeftConfig, PeftModel, get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field
from datasets import Dataset, DatasetDict
from typing import Optional, Literal

from trl import DataCollatorForCompletionOnlyLM
from scripts.sft_trainer import FixedSFTTrainer
from scripts.ditto_trainer import DITTOTrainer

import warnings
import numpy as np

from transformers.trainer_callback import TrainerCallback

class EarlyStoppingCallback(TrainerCallback):
    # any more training, and this overfits on train.
    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def on_step_begin(self, args, state, control, **kwargs):  
        
        if len(state.log_history) > 0:
            # get last log history
            last_loss = None
            
            for k in state.log_history[::-1]:
                if "loss" in k:
                    last_loss = k["loss"]
                    break
                    
            if last_loss < self.threshold:
                control.should_training_stop = True

logger = logging.getLogger(__name__)

MISTRAL_CHAT_TEMPLATE = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"

from typing import Callable, Dict, List, Optional, Tuple, Union, Any

@dataclass
class DittoConfig(DPOConfig):

    output_dir: Optional[str] = field(
        default=None
    )
    
    ditto_max_steps: Optional[int] = field(
        default=30,
    )
    ditto_learning_rate: Optional[float] = field(
        default=None,
    )

    ditto_lr_scheduler_type: Optional[str] = field(
        default=None,
    )
    
    ditto_warmup_ratio: Optional[float] = field(
        default=None,
    )
    ditto_per_device_train_batch_size: Optional[int] = field(
        default=8,
    )


    frac_expert: Optional[float] = field(
        default=None,
    )
    frac_noisy: Optional[float] = field(
        default=None,
    )
    frac_replay: Optional[float] = field(
        default=None,
    )
    rescale_batch: Optional[int] = field(
        default=None,
    )
    
    resample_rate: Optional[int] = field(
        default=10
    )
    bootstrap_count: Optional[int] = field(
        default=10
    )
    reset_rate: Optional[int] = field(
        default=-1
    )
    train_author_key: Optional[int] = field(
        default=0
    )
    train_instances: Optional[int] = field(
        default=None
    )
    train_pkl: str = field(
        default=None
    )

    sft_stop_loss: Optional[int] = field(
        default=1.25,
    )
    

def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "ditto"]
):
    
    tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE
        
    if task in ["sft", "generation"]:
        
        messages = example["chosen"]
            
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True if task == "generation" else False,
        )
        
    elif task == "ditto":
        if not is_openai_format(example["chosen"]):
            raise ValueError(
                f"Could not format example as dialogue for `{task}` task! Require OpenAI format for all messages"
            )
        
        if "prompt" in example and is_openai_format(example["prompt"]):
            prompt_messages = example["prompt"]
            chosen_messages = example["chosen"]
        else:
            prompt_messages = example["chosen"][:-1]
            chosen_messages = example["chosen"][-1:]

        example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
        example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        if example["text_chosen"].startswith(tokenizer.bos_token):
            example["text_chosen"] = example["text_chosen"][len(tokenizer.bos_token):]

    return example

def copy_adapter_weights(src_adapter_name, tgt_adapter_name, model):

    lora_modules = [module for module in model.modules() if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper))]

    with torch.no_grad():
        for model_module in lora_modules:
    
            if src_adapter_name in model_module.lora_A.keys():
                model_module.lora_A[tgt_adapter_name].load_state_dict(model_module.lora_A[src_adapter_name].state_dict())
                model_module.lora_B[tgt_adapter_name].load_state_dict(model_module.lora_B[src_adapter_name].state_dict())
    
            if src_adapter_name in model_module.lora_embedding_A.keys():
                model_module.lora_embedding_A[tgt_adapter_name].load_state_dict(model_module.lora_embedding_A[src_adapter_name].state_dict())
                model_module.lora_embedding_B[tgt_adapter_name].load_state_dict(model_module.lora_embedding_B[src_adapter_name].state_dict())
            

def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, DittoConfig))
    model_args, data_args, training_args = parser.parse()

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Load datasets
    ###############
    raw_dict = {}
    for dataset_key, path in [("train", training_args.train_pkl)]:
        
        prefs = {
            "prompt": [],
            "chosen": [],
        }
        
        with open(path, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
        
        spec_dataset = data[int(training_args.train_author_key)]    

        if training_args.train_instances:
            spec_dataset = spec_dataset[:int(training_args.train_instances)]
        
        for item in spec_dataset:
            prefs["prompt"].append(item["prompt"])

            prefs["chosen"].append([
                {
                    "content": item["prompt"],
                    "role": "user"
                },
                {
                    "content": item["output"].strip(),
                    "role": "assistant"
                }
            ])
            
        raw_dict[dataset_key] = Dataset.from_dict(prefs)
    
    raw_datasets = DatasetDict(raw_dict)
    column_names = list(raw_datasets["train"].features)

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)

    #####################
    # Apply chat template
    #####################

    sft_raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "sft"
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    sft_train_dataset = sft_raw_datasets["train"]

    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "ditto"
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen"}
        )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets["train"])), 2):
        logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
        logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    
    model_kwargs = dict(
        revision=model_args.base_model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    
    lora_config = LoraConfig(
        r=model_args.lora_r,
        target_modules=model_args.lora_target_modules,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config, adapter_name="sft")
    model.set_adapter("sft")

    # we think it's useful, in our setting, to model the user too-
    # you could try uncommenting this 
    
    # collator = DataCollatorForCompletionOnlyLM(
    #     # instruction_template=[733, 16289, 28793], 
    #     response_template=[733, 28748, 16289, 28793], 
    #     tokenizer=tokenizer, mlm=False
    # )

    trainer = FixedSFTTrainer(
        model,
        args=training_args,
        train_dataset=sft_train_dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        # data_collator=collator,
        packing=False,
        callbacks=[EarlyStoppingCallback(threshold=training_args.sft_stop_loss)]
    )

    # SFT Train
    trainer.train()
    
    #########################
    # Instantiate DPO trainer
    #########################

    training_args.learning_rate = training_args.ditto_learning_rate
    training_args.max_steps = training_args.ditto_max_steps
    training_args.lr_scheduler_type = training_args.ditto_lr_scheduler_type
    training_args.warmup_ratio = training_args.ditto_warmup_ratio
    training_args.per_device_train_batch_size = training_args.ditto_per_device_train_batch_size

    model.add_adapter("ditto", lora_config)
    model.set_adapter("ditto")

    copy_adapter_weights("sft", "ditto", model)
    
    trainer = DITTOTrainer(
        model=model,
        ref_adapter_name="sft", # keep the reference as the sft model.
        model_adapter_name="ditto",
        args=training_args,
        beta=training_args.beta,
        train_dataset=raw_datasets["train"],
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        loss_type=training_args.loss_type,  
    )

    ###############
    # Training loop
    ###############
    
    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################

    model.delete_adapter("sft")

    logger.info("*** Save model ***")

    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    logger.info("*** Training complete! ***")

if __name__ == "__main__":
    main()
