# python3 -m accelerate.commands.launch train.py
import wandb 
from coreset_trainer.custom_olmo import DecomposedOlmo2

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from coreset_trainer.trainer import CoresetTrainer
import os
import torch

import argparse

def tokenize_dataset(dataset, tokenizer, max_lenght=2048):
    def tokenize(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        return tokenizer(text, truncation=True, padding="max_length", max_length=max_lenght)
    return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, default=0.5, help="Ratio parameter for coreset")
    args = parser.parse_args()
    
    
    base_model_path = "allenai/OLMo-2-0425-1B"
    train_dataset_path = "allenai/tulu-v2-sft-mixture"
    

    ft_model_name = os.path.basename(base_model_path) + "_" + os.path.basename(train_dataset_path) + "_" + str(args.ratio).replace(".","p")
   

    os.environ["WANDB_PROJECT"] = "cfe_finetuning"
    os.environ["WANDB_NAME"] = ft_model_name

    
    
    
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{% if not loop.last %}{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}{% else %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}{% endif %}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}{% endfor %}" # https://huggingface.co/allenai/OLMo-2-1124-7B-SFT/blob/main/tokenizer_config.json 
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_attn", "q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = AutoModelForCausalLM.from_pretrained(base_model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

    

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


    # if training_args.data_selection_unit == "mezo" and training_args.efficient_mezo:
    model.decomposer = DecomposedOlmo2(model)
    train_dataset = load_dataset(train_dataset_path, split="train[0:75]")
    
    def add_index(example, idx):
        example["indices"] = idx
        return example

    

    print(train_dataset)


    tokenized_train = tokenize_dataset(train_dataset, tokenizer)
    tokenized_train = tokenized_train.map(add_index, with_indices=True, num_proc=10)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=os.path.join("./models",ft_model_name),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=32,
        num_train_epochs=3,
        learning_rate=2e-5,
        logging_steps=10,
        seed=42,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        report_to="wandb"
    )

    trainer = CoresetTrainer(
        model=model,
        ratio=args.ratio,
        args=training_args,
        train_dataset=tokenized_train,
        processing_class=tokenizer,
        data_collator=data_collator
    )

    trainer.train(resume_from_checkpoint=True)

    tokenizer.save_pretrained(training_args.output_dir)
    trainer.save_model()
    trainer.save_state()


    if isinstance(model, PeftModel):
        print("only keeping adapter")
        pytorch_model_path = os.path.join(
            training_args.output_dir, "pytorch_model_fsdp.bin")
        os.remove(pytorch_model_path) if os.path.exists(
            pytorch_model_path) else None
    torch.save(trainer.optimizer.state_dict(), os.path.join(training_args.output_dir, "optimizer.pt"))
    wandb.finish()