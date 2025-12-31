import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy as np




from peft import AutoPeftModelForCausalLM
from peft import PeftConfig
from transformers import logging
from trl import SFTTrainer,SFTConfig
from influence_estimation.util import tokenize_dataset
from torch.nn.utils.rnn import pad_sequence

logging.set_verbosity_error()
from transformers import SchedulerType
import torch
import time
import torch
from peft import AutoPeftModelForCausalLM


def peft_config_from_pretrained_with_retry(
    adapter_path,
    *,
    retries=10,
    delay=5,
    backoff=2,
    **kwargs,
):
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            return PeftConfig.from_pretrained(adapter_path, **kwargs)

        except Exception as e:
            last_err = e
            if attempt == retries:
                break

            wait = delay * (backoff ** (attempt - 1))
            print(
                f"[PeftConfig.from_pretrained] attempt {attempt}/{retries} failed:\n"
                f"{e}\nRetrying in {wait:.1f}s...",
                flush=True,
            )
            time.sleep(wait)

    raise last_err

def tokenizer_from_pretrained_with_retry(
    model_name_or_path,
    *,
    retries=10,
    delay=5,
    backoff=2,
    **kwargs,
):
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                **kwargs,
            )
            return tokenizer

        except Exception as e:
            last_err = e
            if attempt == retries:
                break

            wait = delay * (backoff ** (attempt - 1))
            print(
                f"[tokenizer_from_pretrained] attempt {attempt}/{retries} failed:\n"
                f"{e}\nRetrying in {wait:.1f}s...",
                flush=True,
            )
            time.sleep(wait)

    raise last_err


def from_pretrained_with_retry(
    adapter_path,
    *,
    is_trainable=True,
    device=None,
    retries=5,
    delay=10,
    backoff=2,
    **kwargs,
):
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            model = AutoPeftModelForCausalLM.from_pretrained(
                adapter_path,
                is_trainable=is_trainable,
                **kwargs,
            )
            if device is not None:
                model = model.to(device)
            return model

        except Exception as e:
            last_err = e
            if attempt == retries:
                break

            wait = delay * (backoff ** (attempt - 1))
            print(
                f"[from_pretrained] attempt {attempt}/{retries} failed: {e}\n"
                f"Retrying in {wait:.1f}s..."
            )
            time.sleep(wait)

    raise last_err


def find_max_batch_size(model, data_collator, device, tokenizer, max_length=4096, start_bs=8, min_bs=1):
    print("Model parameter dtype:", next(model.parameters()).dtype, flush=True)
    dummy_token_id = tokenizer.pad_token_id or 0 
    batch_size = start_bs
    while batch_size >= min_bs:
        try:
            batch = [{
                "input_ids": torch.full((max_length,), dummy_token_id, dtype=torch.long),
                "attention_mask": torch.ones(max_length, dtype=torch.long),
                "labels": torch.full((max_length,), -100, dtype=torch.long),

            } for _ in range(batch_size)]
            
            batch = data_collator(batch)
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.no_grad():
                _ = model(**batch)
            
            torch.cuda.empty_cache()
            return batch_size  
        
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                batch_size -= 1  # retry with smaller batch
            else:
                raise e
    
    return min_bs



class ValidationEngine():
    def __init__(self, adapter_path, lr=1e-5, epochs=1, device="cuda"):

        self.gradient_accumulation_steps = None
        self.per_device_bs = None
        self.device = device
        self.adapter_path = adapter_path

        self.lr = lr
        self.epochs = epochs
        peft_config = peft_config_from_pretrained_with_retry(adapter_path)
        self.base_model_path = peft_config.base_model_name_or_path
        
        
        self.tokenizer = tokenizer_from_pretrained_with_retry(self.base_model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def data_collator(self, batch):
        input_ids = [torch.as_tensor(ex["input_ids"]) for ex in batch]
        attention_mask = [torch.as_tensor(ex["attention_mask"]) for ex in batch]
        labels = [torch.as_tensor(ex["labels"]) for ex in batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}



    def get_log_p(self, model, examples, max_length=4096):
        model.eval()
        log_probs = []
        token_distributions = []

        with torch.inference_mode():
            # tokenize with assistant-only masking
            tokenized = tokenize_dataset(
                dataset=examples,
                tokenizer=self.tokenizer, 
                max_length=max_length,
                chat_template_path="./chat_template.jinja",
                assistant_only_loss=True,
                text_column="messages",
                num_proc=1,
                re_index=False,
            )

            for ex in tokenized:
                batch = [ex]
                batch = self.data_collator(batch)
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = model(**batch)


                labels = batch["labels"]
                num_valid_tokens = (labels != -100).sum()
                log_p = -outputs.loss * num_valid_tokens
                log_probs.append(log_p.item())

        
                logits = outputs.logits 
                probs = F.softmax(logits, dim=-1).cpu().squeeze(0) 


                mask = (labels.squeeze(0) != -100).cpu()
                token_distributions.append(probs[mask]) 

                del batch, outputs, logits, probs
                torch.cuda.empty_cache()

            return torch.tensor(log_probs, device=self.device),token_distributions



                        

    def score(self, train_dataset, test_dataset,seed):
        metrics = {}
        torch.manual_seed(seed)
        model_before_ft = from_pretrained_with_retry(
        self.adapter_path,
        is_trainable=True,
        device=self.device,
        retries=5,
        delay=5,
        backoff=2,
        )
        log_p_before_ft, dist_before = self.get_log_p(model_before_ft, test_dataset)  
        del model_before_ft
        
        torch.manual_seed(seed)
        
        model_after_ft = from_pretrained_with_retry(
            self.adapter_path,
            is_trainable=True,
            device=self.device,
            retries=5,
            delay=5,
            backoff=2,
        )
        model_after_ft = self.finetune(model_after_ft, train_dataset, seed=seed)    
        log_p_after_ft, dist_after = self.get_log_p(model_after_ft, test_dataset)
        del model_after_ft
        
        
        
        metrics["delta_log_p"] = log_p_after_ft - log_p_before_ft
        metrics["log_p_before_ft"] = log_p_before_ft
        metrics["log_p_after_ft"] = log_p_after_ft
        
        metrics["jsd"] = []
        metrics["kld(before||after)"] = []


        jsd_list = [0.5 * F.kl_div(torch.log(pb + 1e-12), 0.5*(pb+pa), reduction='batchmean') +
            0.5 * F.kl_div(torch.log(pa + 1e-12), 0.5*(pb+pa), reduction='batchmean')
            for pb, pa in zip(dist_before, dist_after)]
        metrics["jsd"].append(np.mean(jsd_list) if jsd_list else 0.0)


        kl_list = [F.kl_div(torch.log(pa + 1e-12), pb, reduction='batchmean').item() for pb, pa in zip(dist_before, dist_after)]
        metrics["kld(before||after)"].append(np.mean(kl_list))
        
        metrics["jsd"] = torch.as_tensor(metrics["jsd"])
        metrics["kld(before||after)"] = torch.as_tensor(metrics["kld(before||after)"])
        return metrics
        
    def finetune(self, model, train_dataset, seed):
        # set grad_acc_steps so that only one single parameter update per epoch
        train_len = len(train_dataset)
        if self.gradient_accumulation_steps is None:
            self.per_device_bs = find_max_batch_size(
            model, self.data_collator, self.device,  self.tokenizer,start_bs=8,
            )
            
            self.gradient_accumulation_steps = max(1, (train_len // self.per_device_bs)+1)   
            print("gradient_accumulation_steps",self.gradient_accumulation_steps, "per_device_bs", self.per_device_bs, "train_len",train_len,flush=True)

        # the same as in finetune.py but without warmup
        sft_config = SFTConfig(
            per_device_train_batch_size=self.per_device_bs,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            remove_unused_columns=False,
            num_train_epochs=self.epochs,
            learning_rate=self.lr,
            logging_steps=1,
            seed=seed,
            save_strategy="no", 
            report_to=[],
            disable_tqdm=True,
            chat_template_path="./chat_template.jinja",
            assistant_only_loss=True,
            packing=False,
            gradient_checkpointing=False,
            ddp_find_unused_parameters=False,
            lr_scheduler_type=SchedulerType.CONSTANT,
            warmup_steps=0, 

        )

        trainer = SFTTrainer(
            model =model,
            args=sft_config,
            train_dataset=train_dataset,
            # optimizer_cls_and_kwargs=(SGD, {"lr":self.lr}), 
        )
        torch.cuda.empty_cache()
        trainer.train(resume_from_checkpoint=False)
        return model



