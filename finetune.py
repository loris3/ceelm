# python3 -m accelerate.commands.launch finetune.py
import os
import wandb
import torch
from dataclasses import dataclass, field
from accelerate import Accelerator
from typing import Optional
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


CHAT_TEMPLATE = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{% if not loop.last %}{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}{% else %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}{% endif %}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}{% endfor %}"

@dataclass
class ModelArguments:
    base_model: str = field(default="allenai/OLMo-2-0425-1B")
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)


@dataclass
class DataArguments:
    train_dataset: str = field(default="allenai/tulu-v2-sft-mixture")
    train_split: str = field(default="train")

@dataclass
class CustomTrainingArguments(TrainingArguments):
    wandb_project: Optional[str] = field(default="cfe_finetuning")
    ddp_find_unused_parameters: Optional[bool] = field(default=False)
    save_strategy: str = field(default="epoch")  
    save_total_limit: int = field(default=2)   
from datasets import load_from_disk

def apply_chat_template_safe(messages, tokenizer):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )



def tokenize(example, tokenizer, max_length=4096):
    text = apply_chat_template_safe(example["messages"], tokenizer)
  
    return tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length
    )


def tokenize_dataset(dataset, tokenizer, max_length=4096, num_proc=20, re_index=True):
    
    from functools import partial
    tokenized_dataset = dataset.map(
        partial(tokenize, tokenizer=tokenizer, max_length=max_length),
        batched=True,
        remove_columns=[c for c in dataset.column_names if c != "indices"],
        num_proc=num_proc
    )

    def add_index(example, idx):
        example["indices"] = idx
        return example

    if re_index:
        tokenized_dataset = tokenized_dataset.map(add_index, with_indices=True, num_proc=num_proc)

    return tokenized_dataset



def load_tokenizer(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.chat_template = CHAT_TEMPLATE
        if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        return tokenizer    


if __name__ == "__main__":
    
    


    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    ft_model_name = f"{os.path.basename(model_args.base_model)}_{os.path.basename(data_args.train_dataset)}_lr{training_args.learning_rate}_seed{training_args.seed}"
    os.environ["WANDB_PROJECT"] = training_args.wandb_project
    os.environ["WANDB_NAME"] = ft_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_args.base_model)
    tokenizer.chat_template = CHAT_TEMPLATE
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_args.base_model,    
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True   
    )
   
   
    # model.resize_token_embeddings(len(tokenizer))
    target_modules = None
    if "OLMo" in model_args.base_model:
        target_modules = ["c_attn", "q_proj", "v_proj"]
    elif "pythia" in model_args.base_model:
        target_modules = ["query_key_value", "dense"]
    elif "llama" in model_args.base_model.lower(): 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif "qwen" in model_args.base_model.lower():  
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    else:
        for name, module in model.named_modules():
            print(name)
        raise NotImplementedError


    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # model.gradient_checkpointing_enable()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    
    max_length = min(4096,getattr(model.config, "max_position_embeddings", tokenizer.model_max_length))
    tokenized_dataset_path = os.path.join("./cache","ft_tokenized_datasets", model_args.base_model, os.path.basename(data_args.train_dataset))

    tokenized_train = load_from_disk(tokenized_dataset_path)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    torch.cuda.empty_cache()

    trainer.train()#(resume_from_checkpoint=True)
    training_args.output_dir = os.path.join("./models/",ft_model_name)
    tokenizer.model_max_length = max_length
    tokenizer.save_pretrained(training_args.output_dir)
    

    model.save_pretrained(training_args.output_dir)
    
    trainer.save_state()

    output_config_file = os.path.join(training_args.output_dir, "experiment_config.json")


    def safe_vars(obj):
        return {k: str(v) for k, v in vars(obj).items()}

    with open(output_config_file, "w") as f:
        json.dump({
            "model_args": safe_vars(model_args),
            "data_args": safe_vars(data_args),
            "training_args": safe_vars(training_args),
        }, f, indent=4)


    if isinstance(model, PeftModel):
        pytorch_model_path = os.path.join(training_args.output_dir, "pytorch_model_fsdp.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    torch.save(trainer.optimizer.state_dict(), os.path.join(training_args.output_dir, "optimizer.pt"))
    wandb.finish()


