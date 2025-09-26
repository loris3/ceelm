import torch
import warnings
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import numpy as np
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from torch.cuda.amp import autocast
from tqdm import tqdm
import copy
from transformers import DataCollatorForLanguageModeling
from coreset_trainer.train import tokenize_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from peft import PeftConfig
from finetune import CHAT_TEMPLATE
from transformers import logging

logging.set_verbosity_error()
class ChatTemplateCollator:
    def __init__(self, tokenizer, mlm=False, max_length=1024): # TODO
        self.tokenizer = tokenizer
        
        self.max_length = max_length
        self.base_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=mlm, 
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
    def __init__(self, adapter_path, device="cuda"):


        self.device = device
        self.adapter_path = adapter_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.adapter_path, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
    
        peft_config = PeftConfig.from_pretrained(adapter_path)
        self.base_model_path = peft_config.base_model_name_or_path

        
        self.data_collator = ChatTemplateCollator(tokenizer=self.tokenizer, mlm=False)
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = CHAT_TEMPLATE
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token # TODO
    def get_log_p(self, model, examples, max_new_tokens=256):
        model.eval()
        log_probs = []

        for ex in examples:
            if isinstance(ex, str):
                ex = {"messages": ex}


            batch = self.data_collator([ex])
            batch = {k: v.to(self.device) for k, v in batch.items()}
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            with torch.no_grad():
    
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=False 
                )


                scores = torch.stack(outputs.scores, dim=1)  # (B=1, seq_len, V)

                gen_ids = outputs.sequences[:, input_ids.shape[1]:]  


                log_probs_seq = F.log_softmax(scores, dim=-1)
                token_logp = log_probs_seq.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1) 
                
                # sum to get total log prob
                total_logp = token_logp.sum(dim=1)  # (1,)
                log_probs.append(total_logp.item())

            del batch, outputs, scores, gen_ids, token_logp
            torch.cuda.empty_cache()
        return torch.tensor(log_probs, device=self.device)

    # def get_log_p(self, model, examples):
    #     model.eval()
    #     log_probs = []

    #     for ex in examples:
    #         if isinstance(ex, str):
    #             ex = {"messages": ex}

        
    #         batch = self.data_collator([ex])
    #         batch = {k: v.to(self.device) for k, v in batch.items()}


    #         labels = batch["input_ids"].clone()
    #         labels[batch["attention_mask"] == 0] = -100
    #         batch["labels"] = labels

    #         with torch.no_grad():
    #             outputs = model(**batch)

    #             seq_lengths = (batch["labels"] != -100).sum(dim=1)

    #             log_p = -outputs.loss * seq_lengths
    #             log_probs.append(log_p.item())

    #         del batch, outputs
    #         torch.cuda.empty_cache()

    #     return torch.tensor(log_probs, device=self.device)


                    

    def score(self, train_dataset, test_dataset,seed):

        base_model_after_ft = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float32
            )
        model_after_ft = PeftModel.from_pretrained(base_model_after_ft, self.adapter_path).to(self.device)
        model_after_ft.config.pad_token_id = self.tokenizer.eos_token_id
        model_after_ft = self.finetune(model_after_ft, train_dataset, seed=seed)
        
        base_model_before_ft = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float32
            )
        model_before_ft = PeftModel.from_pretrained(base_model_before_ft, self.adapter_path).to(self.device)
        model_before_ft.config.pad_token_id = self.tokenizer.eos_token_id
        

        log_p_before_ft = self.get_log_p(model_before_ft, test_dataset)       
        log_p_after_ft = self.get_log_p(model_after_ft, test_dataset)
        return log_p_after_ft - log_p_before_ft 
    
    def finetune(self, model, train_dataset, seed):
        # Freeze base model
        for param in model.base_model.parameters():
            param.requires_grad = False

        # Enable LoRA adapters
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True

        per_device_bs = 2
        train_len = len(train_dataset)
        gradient_accumulation_steps = max(1, train_len // per_device_bs)
        print("gradient_accumulation_steps",gradient_accumulation_steps, "train_len",train_len,flush=True)


        training_args = TrainingArguments(
            per_device_train_batch_size=per_device_bs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            remove_unused_columns=False,
            num_train_epochs=3,
            learning_rate=1e-2,
            logging_steps=500,
            seed=seed,
            save_strategy="no", 
            report_to=[],
            disable_tqdm=True
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            warnings.simplefilter("ignore")
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
            )
            trainer.train(resume_from_checkpoint=False)
        return model

