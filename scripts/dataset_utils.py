# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import os
import random
import warnings
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import re
from random import sample 
from tqdm import tqdm
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
import pdb

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel, PreTrainedTokenizerBase, pipeline

@dataclass
class DPODataCollatorWithPadding:
    r"""
    DPO DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        model (Optional[`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        max_prompt_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the prompt to be processed.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        padding_value (`int`, defaults to 0):
            The value used for padding.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
        max_target_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the target to be processed. Only useful for encoder-decoder architectures.
        truncation_mode: (`str`, defaults to "keep_end"):
            The truncation mode to use when truncating the prompt.
    """
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    label_pad_token_id: int = -100
    padding_value: int = 0
    truncation_mode: str = "keep_end"
    is_encoder_decoder: Optional[bool] = False
    max_target_length: Optional[int] = None
    pipeline: Optional = None
    train_dataset: Optional = None

    frac_expert: Optional = 0.7
    frac_replay: Optional = 0.2
    frac_noisy: Optional = 0.1     
    rescale_batch: Optional = 3

    bootstrap_count: int = 10
    mode: str = "past"
    cache = {}

    last_sampled_step: int = 0

    def resample(self, step):        
        
        self.last_sampled_step = step

        # iterate over the train_dataset and update the cache
        if step not in self.cache:
            self.cache[step] = {}
        
        # create the pipeline to sample generations for each _item_
        if self.pipeline == None:
            
            self.pipeline = pipeline(
                "text-generation", 
                model=self.model, 
                tokenizer=self.tokenizer,
                device="cuda",
                batch_size=2,
                pad_token_id=self.tokenizer.pad_token_id
            )
                    
        # here, we call the model and add everything to cache:
        self.model.eval()

        with torch.inference_mode(): 
            
            prompt_text = []
            for feature in self.train_dataset:
                prompt_text.append(feature["prompt"])

            inference_dataset = Dataset.from_dict({"prompt": prompt_text})

            max_gen_tokens = 1024
            
            pipe_result = self.pipeline(
                KeyDataset(inference_dataset, "prompt"), 
                max_new_tokens=max_gen_tokens, do_sample=True, 
                temperature=1,
                num_return_sequences=self.bootstrap_count,
                return_full_text=False,
                pad_token_id=self.tokenizer.unk_token_id
            )

            rejected = []

            for outs in tqdm(pipe_result, total=len(inference_dataset)):
                for out in outs:
                    
                    gen_tokens = len(
                        self.tokenizer(
                            out["generated_text"], 
                            add_special_tokens=False
                        )["input_ids"]
                    )


                    if gen_tokens >= max_gen_tokens - 1:
                        # BAD LANGUAGE MODEL!! NO EOS TOKEN FOR YOU!
                        rejected.append(out["generated_text"])
                    else:
                        rejected.append(out["generated_text"] + " " + self.tokenizer.eos_token)
                        
            ix = 0
            
            for feature in self.train_dataset:
                for _ in range(self.bootstrap_count):
                    if feature["prompt"] not in self.cache[step]:
                        self.cache[step][feature["prompt"]] = []

                    self.cache[step][feature["prompt"]].append(rejected[ix])
                    ix += 1
        
        self.model.train()
        
        
    def build_tokenized_answer(self, prompt, answer):
        """
        Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
        It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
        Reference:
            https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        """

        full_tokenized = self.tokenizer(answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

        # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

        # Prepare input tokens for token by token comparison
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            # pdb.set_trace()
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
        # can be merged together when tokenizing prompt+answer. This could result
        # on the last token from the prompt being different when tokenized on its own
        # vs when done as prompt+answer.
        response_token_ids_start_idx = len(prompt_input_ids)

        # If tokenized prompt is different than both prompt+answer, then it means the
        # last token has changed due to merging.
        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

        return dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )

    def tokenize_row(self, prompt, chosen, rejected) -> Dict:
        """Tokenize a single row from a DPO specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}

        # Check issues below for more details
        #  1. https://github.com/huggingface/trl/issues/907
        #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        #  3. https://github.com/LianjiaTech/BELLE/issues/337

        if not isinstance(prompt, str):
            raise ValueError(f"prompt should be an str but got {type(prompt)}")
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

        if not isinstance(chosen, str):
            raise ValueError(f"chosen should be an str but got {type(chosen)}")
        chosen_tokens = self.build_tokenized_answer(prompt, chosen)

        if not isinstance(rejected, str):
            raise ValueError(f"rejected should be an str but got {type(rejected)}")
        rejected_tokens = self.build_tokenized_answer(prompt, rejected)

        # Last prompt token might get merged by tokenizer and
        # it should not be included for generation if that happens
        prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

        chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
        rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
        prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

        for k, v in prompt_tokens.items():
            prompt_tokens[k] = v[:prompt_len_input_ids]

        # Make sure prompts only have one different token at most an
        # and length only differs by 1 at most
        num_diff_tokens = sum(
            [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
        )
        num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
        
        if num_diff_tokens > 1 or num_diff_len > 1:
            raise ValueError(
                "Chosen and rejected prompt_input_ids might only differ on the "
                "last token due to tokenizer merge ops."
            )

        # bos and eos are already added.

        longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

        # if combined sequence is too long, truncate the prompt
        for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                if self.truncation_mode == "keep_start":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                elif self.truncation_mode == "keep_end":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
                else:
                    raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        # if that's still too long, truncate the response
        for answer_tokens in [chosen_tokens, rejected_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                for k in ["input_ids", "attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

        # Create labels
        chosen_sequence_tokens = {
            k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
        }
        rejected_sequence_tokens = {
            k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
        }
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
            self.label_pad_token_id
        ] * len(chosen_tokens["prompt_input_ids"])
        rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
        rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
            self.label_pad_token_id
        ] * len(rejected_tokens["prompt_input_ids"])

        for k, toks in {
            "chosen_": chosen_sequence_tokens,
            "rejected_": rejected_sequence_tokens,
            "": prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}{type_key}"] = tokens

        return batch

    def collate(self, batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                # adapted from https://stackoverflow.com/questions/73256206
                to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]

                if k.endswith("_input_ids"):
                    padding_value = self.tokenizer.pad_token_id
                elif k.endswith("_labels"):
                    padding_value = self.label_pad_token_id
                elif k.endswith("_attention_mask"):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)

                padded_batch[k] = padded_batch[k].flip(dims=[1])
                
            else:
                padded_batch[k] = [ex[k] for ex in batch]
        
        # pdb.set_trace()
        return padded_batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
                        
        # we find the rejected samples and pair them with the prompt, collate, etc.
        tokenized_batch = []
        curr_batch = {}
            
        for feature in features:
            curr_batch[feature["prompt"]] = {}
            
            prompt = feature["prompt"]
            chosen = feature["chosen"]
            
            curr_batch[prompt]["replay"] = []
            
            for step_a in range(self.last_sampled_step, -1, -1):
                curr_batch[feature["prompt"]][step_a] = []

                # skip if we never sampled here
                if step_a not in self.cache: continue 

                # inter-batch
                for step_b in range(step_a - 1, -1, -1):

                    # skip if we never sampled here
                    if step_b not in self.cache: continue   
                    
                    for rejected_a in self.cache[step_a][prompt]:    
                        for rejected_b in self.cache[step_b][prompt]:
                            curr_batch[prompt][step_a].append(
                                (prompt, rejected_a, rejected_b)
                            )                       

                # replay buffer
                if step_a < self.last_sampled_step:
                    for rejected_past in self.cache[step_a][prompt]:
                        curr_batch[prompt]["replay"].append((prompt, chosen, rejected_past))
               
                # adding expert
                if step_a == self.last_sampled_step:
                    curr_batch[prompt]["expert"] = []
                    for rejected in self.cache[self.last_sampled_step][prompt]:
                        curr_batch[prompt]["expert"].append(
                            (prompt, chosen, rejected)
                        )
                     
        
        sampled_batch = []
    
        frac_expert = 0.7
        frac_replay = 0.2
        frac_noisy = 0.1     
        rescale_batch = 3
                
        noisy_samples = []
        expert_samples = []
        replay_samples = []
        
        for feature in curr_batch:
            
            for iteration in curr_batch[feature]:
                if iteration == "expert":
                    expert_samples = expert_samples + curr_batch[feature][iteration]
                elif iteration == "replay":
                    replay_samples = replay_samples + curr_batch[feature][iteration]                    
                else:
                    noisy_samples = noisy_samples + curr_batch[feature][iteration]
        
        len_superbatch = len(curr_batch) * 3
        noisy_subsample = random.sample(noisy_samples, min(len(noisy_samples), round(len_superbatch * frac_noisy)))
        expert_subsample = random.sample(expert_samples, min(len(expert_samples), round(len_superbatch * frac_expert)))
        replay_subsample = random.sample(replay_samples, min(len(replay_samples), round(len_superbatch * frac_replay)))
        
        tot_noisy = len(noisy_subsample)
        tot_exp = len(expert_subsample)
        tot_replay = len(replay_subsample)
        
        sampled_batch = (expert_subsample + noisy_subsample + replay_subsample)
        
        for prompt, chosen, rejected in sampled_batch:
            batch_element = self.tokenize_row(prompt, prompt + chosen, prompt + rejected)
            tokenized_batch.append(batch_element)
        
        collated = self.collate(tokenized_batch)
        
        return collated

