from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch.optim import AdamW

import os
import torch
def tokenize_dataset(dataset, tokenizer):
    def tokenize(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        return tokenizer(text, truncation=True, padding="max_length", max_length=512)
    return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)



if __name__ == "__main__":
    base_model_path = "distilbert/distilgpt2"
    train_dataset_path = "allenai/tulu-v2-sft-mixture"

    ft_model_name = os.path.basename(base_model_path) + "_" + os.path.basename(train_dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{% if not loop.last %}{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}{% else %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}{% endif %}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}{% endfor %}" # https://huggingface.co/allenai/OLMo-2-1124-7B-SFT/blob/main/tokenizer_config.json 
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
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

    train_dataset = load_dataset(train_dataset_path, split="train[0%:1%]")




    tokenized_train = tokenize_dataset(train_dataset, tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=os.path.join("./models",ft_model_name),
        per_device_train_batch_size=64,
        gradient_accumulation_steps=16,
        num_train_epochs=3,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    trainer.save_model()
    trainer.save_state()


    if isinstance(model, PeftModel):
        print("only keeping adapter")
        pytorch_model_path = os.path.join(
            training_args.output_dir, "pytorch_model_fsdp.bin")
        os.remove(pytorch_model_path) if os.path.exists(
            pytorch_model_path) else None
    torch.save(trainer.optimizer.state_dict(), os.path.join(training_args.output_dir, "optimizer.pt"))
