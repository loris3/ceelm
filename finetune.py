# python3 -m wandb sweep --project=cfe_finetuning sweep_test.yaml
# python3 -m wandb sweep --project=cfe_finetuning sweep.yaml


import os
import wandb
import torch
from dataclasses import dataclass, field
from accelerate import Accelerator
from typing import Optional
import json
from trl import SFTConfig, SFTTrainer
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

    save_strategy: str = field(default="steps")  
    save_total_limit: int = field(default=5)  
    save_steps: int = field(default=500)   
from datasets import load_from_disk





if __name__ == "__main__":
    
    


    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    ft_model_name = f"{os.path.basename(model_args.base_model)}_{os.path.basename(data_args.train_dataset)}_lr{training_args.learning_rate}_seed{training_args.seed}"
    os.environ["WANDB_PROJECT"] = training_args.wandb_project
    os.environ["WANDB_NAME"] = ft_model_name
    



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

        raise NotImplementedError


    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    

  


    training_args.output_dir = os.path.join("./models/",ft_model_name)
    os.makedirs(training_args.output_dir, exist_ok=True)
    

        

    sft_config = SFTConfig(
        chat_template_path="./chat_template.jinja",
        assistant_only_loss=True,
        model_init_kwargs={"torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True},
        output_dir=training_args.output_dir,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        learning_rate=training_args.learning_rate,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        num_train_epochs=training_args.num_train_epochs,
        logging_steps=training_args.logging_steps,
        save_strategy=training_args.save_strategy,
        save_total_limit=training_args.save_total_limit,
        report_to=training_args.report_to,
        seed=training_args.seed,
        packing=False,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
    )
    
    dataset = load_dataset(data_args.train_dataset, split="train")
    trainer = SFTTrainer(
       
        model =model_args.base_model,
        args=sft_config,
        train_dataset=dataset,
 
        peft_config=lora_config,
     

    )

    torch.cuda.empty_cache()

    

    checkpoint_dir = training_args.output_dir
    if any(f.startswith("checkpoint") for f in os.listdir(checkpoint_dir)):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
        
    


    trainer.model.save_pretrained(training_args.output_dir)

    
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



    pytorch_model_path = os.path.join(training_args.output_dir, "pytorch_model_fsdp.bin")
    if os.path.exists(pytorch_model_path):
        os.remove(pytorch_model_path)

    torch.save(trainer.optimizer.state_dict(), os.path.join(training_args.output_dir, "optimizer.pt"))
    wandb.finish()


