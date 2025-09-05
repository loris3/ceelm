# python3 -m accelerate.commands.launch finetune.py
import os
import wandb
import torch
from dataclasses import dataclass, field
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

def tokenize_dataset(dataset, tokenizer,max_length=4096, num_proc=32, re_index=True):
    def tokenize(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=[c for c in dataset.column_names if c != "indices"], num_proc=num_proc)
    def add_index(example, idx):
        example["indices"] = idx
        return example
    if re_index:
        tokenized_dataset =  tokenized_dataset.map(add_index, with_indices=True, num_proc=10)
    return tokenized_dataset


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    ft_model_name = os.path.basename(model_args.base_model) + "_" + os.path.basename(data_args.train_dataset)
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
    
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=["c_attn", "q_proj", "v_proj"] if "OLMo" in model_args.base_model else ["query_key_value", "dense"] if "pythia" in model_args.base_model else None,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # model.gradient_checkpointing_enable()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = load_dataset(data_args.train_dataset, split=data_args.train_split)
    max_length = getattr(model.config, "max_position_embeddings", tokenizer.model_max_length)
    tokenized_train = tokenize_dataset(train_dataset, tokenizer,max_length=max_length)
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


