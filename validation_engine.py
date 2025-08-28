import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import numpy as np
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from torch.cuda.amp import autocast
from tqdm import tqdm
import copy
from transformers import DataCollatorForLanguageModeling

class ChatTemplateCollator:
    def __init__(self, tokenizer, mlm=False, max_length=1024): # TODO
        self.tokenizer = tokenizer
        
        self.max_length = max_length
        self.base_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=mlm
        )

    def __call__(self, features):
        texts = [
            self.tokenizer.apply_chat_template(
                f["messages"], tokenize=False, add_generation_prompt=False
            )
            for f in features
        ]
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length
        )
 
        batch = [
            {"input_ids": tokenized["input_ids"][i],
             "attention_mask": tokenized["attention_mask"][i]}
            for i in range(len(tokenized["input_ids"]))
        ]
        return self.base_collator(batch)

class ValidationEngine():
    def __init__(self, base_model_path, device):
        self.base_model_path = base_model_path
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, padding_side='left')
        self.data_collator = ChatTemplateCollator(tokenizer=self.tokenizer, mlm=False)
        self.model_ = None
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{% if not loop.last %}{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}{% else %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}{% endif %}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}{% endfor %}" # https://huggingface.co/allenai/OLMo-2-1124-7B-SFT/blob/main/tokenizer_config.json 
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token # TODO
    def get_log_p(self, model, examples):
        batch = self.data_collator(examples)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad(): 
            outputs = model.generate(**batch, max_new_tokens=5, return_dict_in_generate=True, output_scores=True,  num_beams=1)
            transition_scores = model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            log_p = transition_scores.sum(axis=1)
            # print("log_p",log_p)
            return log_p
    @property
    def model(self):
        if self.model_ is None:
            self.model_ = AutoModelForCausalLM.from_pretrained(self.base_model_path,torch_dtype=torch.float32).to(self.device)
            # self.model_.generation_config.use_cache = True
        return self.model_
    def score(self, train_dataset, test_sets):
        deltas = []
        model_ft = AutoModelForCausalLM.from_pretrained(self.base_model_path, torch_dtype=torch.float32).to(self.device)
        model_ft = self.finetune(model_ft, train_dataset)
        for test_dataset in test_sets:
            log_p_before_ft = self.get_log_p(self.model, test_dataset)       
            log_p_after_ft = self.get_log_p(model_ft, test_dataset)
            deltas.append(log_p_after_ft - log_p_before_ft)
        return deltas
    def finetune(self, model, train_dataset):
        training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        remove_unused_columns=False,
        num_train_epochs=5,
        learning_rate=2e-5,
        logging_steps=500,
        seed=42,
        #save_steps=None,#100,
        # save_total_limit=2,
        #  fp16=True, 
        #   bf16=True,
        report_to=[],
        disable_tqdm=True 
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=self.tokenizer,
            data_collator=self.data_collator,
        )
        

        trainer.train(resume_from_checkpoint=False)
        return model
