# we tokenize in a seperate script without accellerate as otherwise dataset.map(n_proc>1) breaks for llama models due to apply_chat_template
import os
import wandb
import torch
from dataclasses import dataclass, field

from typing import Optional
import json
from transformers import (
    
    HfArgumentParser,
)
from transformers import AutoConfig, AutoTokenizer

from datasets import load_dataset

from finetune import ModelArguments, DataArguments, CustomTrainingArguments, load_tokenizer
from influence_estimation.util import tokenize_dataset

if __name__ == "__main__":
    
    from datasets import disable_caching
    disable_caching()


    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    
    tokenizer = load_tokenizer(model_args.base_model)    

 
    config = AutoConfig.from_pretrained(model_args.base_model)
    max_length = min(4096,getattr(config, "max_position_embeddings", None))

    
    max_length = max_length or tokenizer.model_max_length

    tokenized_dataset_path = os.path.join("./cache","ft_tokenized_datasets", model_args.base_model, os.path.basename(data_args.train_dataset))

    print("max_length",max_length)

    train_dataset = load_dataset(data_args.train_dataset, split=data_args.train_split)
    tokenized_train = tokenize_dataset(train_dataset, tokenizer,max_length=max_length, num_proc=10)
    tokenized_train.save_to_disk(tokenized_dataset_path,num_proc=10 )
    

