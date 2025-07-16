import os
import torch
from transformers import Olmo2ForCausalLM, AutoTokenizer, BitsAndBytesConfig,DataCollatorForLanguageModeling

import torch.nn as nn
import subprocess

def get_free_gpu_memory():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
        stdout=subprocess.PIPE,
        text=True
    )
    # Output in MB
    memory_free = [int(x) for x in result.stdout.strip().split('\n')]
    # Convert to GB
    memory_free_gb = [round(mb / 1024, 2) for mb in memory_free]
    
    for i, mem in enumerate(memory_free_gb):
        print(f"GPU {i}: {mem} GB free")

get_free_gpu_memory()
MODEL = "OLMo-2-0425-1B-Instruct-math-lora-gas8-bs16-mezo-v_proj-2560_largest_grad-5steps-seed0"
MODEL_PATH = os.path.join("./models", MODEL)

class Olmo2ForCausalLMCaptum(Olmo2ForCausalLM):
    def __init__(self, config):
        print("init patch")
        super().__init__(config)
        self.loss_fn = nn.CrossEntropyLoss(reduction="mean",ignore_index=-100)
    def forward(self, *args, **kwargs):
        if not args and not kwargs:
            print("Forward called with no inputs (likely Captum or probing).")
            return None
        # print("patch", self, args, kwargs, flush=True)
        return super().forward(*args, **kwargs).logits.half().view(-1, self.config.vocab_size )
    def loss_function(self, logits, labels, *args, **kwargs):     
        print("labels", labels.shape)
        print("logits", logits.shape)
        

        loss = self.loss_fn(logits, labels.view(-1))
        print("loss", loss)

        return loss
from datasets import load_dataset

ds = load_dataset("allenai/tulu-v2-sft-mixture", split="train[0:5]")
ds
model = Olmo2ForCausalLMCaptum.from_pretrained(
    MODEL_PATH,
    attn_implementation='eager'
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, )

# Needed for LLaMA tokenizer
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        )
def tokenize_chat(example):
    prompt=example["messages"][0][0]["content"]
    # prompt = tokenizer.apply_chat_template(
    #     example["messages"],
    #     tokenize=False,
    #     add_generation_prompt=True
    # )
  #  print(prompt)
    tokens = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=768,
        return_tensors="pt"
    )
    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
    }


correct_dataset  = ds.with_transform(tokenize_chat)
data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        )

from torch.utils.data import DataLoader

# Assuming correct_dataset is a Dataset object
# and data_collator is a function that batches samples.
def tuple_collate(batch):
    batch_encoding = data_collator(batch)  # returns dict-like BatchEncoding
    labels = batch_encoding["labels"].to("cuda")  # extract labels tensor
    attention_mask = batch_encoding["attention_mask"].to("cuda")
    inputs = batch_encoding["input_ids"].to("cuda")          # dict of inputs without labels
    return inputs, attention_mask, labels  

dataloader = DataLoader(
    correct_dataset,
    batch_size=8,            # or any batch size you want
    collate_fn=data_collator, # your custom collator function
    shuffle=True             # if you want to shuffle data each epoch
)

batch = next(iter(dataloader))
next(iter(dataloader))
next(iter(dataloader))
def checkpoints_load_func(net, path):
    return 1.

model.to("cuda")


from argparse import ArgumentParser
from tqdm import tqdm

import torch as ch
import torch.nn as nn
from torch.utils.data import DataLoader

from trak import TRAKer

# Huggingface
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
)


GLUE_TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

# NOTE: CHANGE THIS IF YOU WANT TO RUN ON FULL DATASET



class SequenceClassificationModel(nn.Module):
    """
    Wrapper for HuggingFace sequence classification models.
    """
    def __init__(self):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            'bert-base-cased',
            num_labels=2,
            finetuning_task='qnli',
            attn_implementation='eager',
            cache_dir=None,
            revision='main',
            use_auth_token=None,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-cased',
            config=self.config,
            cache_dir=None,
            revision='main',
            use_auth_token=None,
            ignore_mismatched_sizes=False
        )

        self.model.eval().cuda()

    def forward(self, input_ids, token_type_ids, attention_mask):
        return self.model(input_ids=input_ids,
                          token_type_ids=token_type_ids,
                          attention_mask=attention_mask).logits


def get_dataset(split, inds=None):
    raw_datasets = load_dataset(
            "glue",
            'qnli',
            cache_dir=None,
            use_auth_token=None,
        )
    label_list = raw_datasets["train"].features["label"].names
    sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS['qnli']

    label_to_id = None  # {v: i for i, v in enumerate(label_list)}

    tokenizer = AutoTokenizer.from_pretrained(
        'bert-base-cased',
        cache_dir=None,
        use_fast=True,
        revision='main',
        use_auth_token=False
    )

    padding = "max_length"
    max_seq_length = 128

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[lbl] if lbl != -1 else -1) for lbl in examples["label"]]
        return result

    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=(not False),
        desc="Running tokenizer on dataset",
    )

    if split == 'train':
        train_dataset = raw_datasets["train"]
        ds = train_dataset
    else:
        eval_dataset = raw_datasets["validation"]
        ds = eval_dataset
    return ds

import torch.nn.functional as F


import abc
TRAIN_SET_SIZE = len(correct_dataset)
print("TRAIN_SET_SIZE",TRAIN_SET_SIZE)
VAL_SET_SIZE = 5_463
class LanguageModelOutput(abc.ABC):
    def __init__(self) -> None:
        pass
    def get_output(model, weights, buffers, input_ids, attention_mask, labels):
        """See Sections 2 & 3 of `our paper
        <https://arxiv.org/abs/2303.14186>`_ for more details on what model
        output functions are in the context of TRAK and how to use & design
        them.

        Args:
            model (torch.nn.Module):
                model
            batch (Iterable[Tensor]):
                input batch

        Returns:
            Tensor:
                model output function
        """
        # return torch.tensor(1.0)
        kw_inputs = {
            "input_ids": input_ids.unsqueeze(0),        # add batch dim
            "attention_mask": attention_mask.unsqueeze(0),
        }
        logits = torch.func.functional_call(
        model, (weights, buffers), args=(), kwargs=kw_inputs
        )
        labels = labels.unsqueeze(0)
        loss_fct = F.cross_entropy
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))
        return loss

    def get_out_to_loss_grad(model, weights, buffers, batch, loss_temperature=1.0):
        input_ids, attention_mask, labels = batch
        
        kw_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask.bool(),
        }
        
        logits = torch.func.functional_call(model, (weights, buffers), args=(), kwargs=kw_inputs)
        labels = labels.unsqueeze(0)
        loss_fct = F.cross_entropy
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))
        
        loss.backward()
        vectorized_grads = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
        print("vectorized_grads",vectorized_grads)
        return vectorized_grads


        
# def process_batch(batch):
#     # print(batch)
#     return batch['input_ids'], batch['attention_mask'], batch['labels']

get_free_gpu_memory()
traker = TRAKer(model=model,
                task=LanguageModelOutput,
                train_set_size=TRAIN_SET_SIZE,
                save_dir="./trak",
                device="cuda",
                proj_max_batch_size=16,
                logging_level=100,
                proj_dim=512)
dataloader = DataLoader(correct_dataset, batch_size=1, shuffle=False, collate_fn=data_collator)

def process_batch(batch):
    return batch['input_ids'],  batch['attention_mask'], batch['labels']

device = 'cuda'



traker.load_checkpoint(model.state_dict(), model_id=0)
for batch in tqdm(dataloader, desc='Featurizing..'):

    batch = process_batch(batch)
    batch = [x.to("cuda") for x in batch]
    print(batch[2].min(), batch[2].max(), batch[2].dtype) 
  #  print("batch",batch)
#    print("batch",batch,)
    print("num_samples=batch[0].shape[0]=", batch[0].shape[0])
    traker.featurize(batch=batch, num_samples=batch[0].shape[0])
traker.finalize_features()
# traker.start_scoring_checkpoint(exp_name='qnli',
#                                 checkpoint=model.state_dict(),
#                                 model_id=0,
#                                 num_targets=VAL_SET_SIZE)
# for batch in tqdm(loader_val, desc='Scoring..'):
#     batch = process_batch(batch)
#     batch = [x.cuda() for x in batch]
#     traker.score(batch=batch, num_samples=batch[0].shape[0])

# scores = traker.finalize_scores(exp_name='qnli')