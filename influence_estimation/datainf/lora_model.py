from tqdm import tqdm
import pickle
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model
)
from datasets import Dataset
import evaluate


from trak.projectors import CudaProjector, NoOpProjector
from trak.projectors import ProjectionType


class LORAEngineGeneration(object):
    def __init__(self, 
                model,
                # base_path,
                # project_path,
                # dataset_name='math_with_reason',
                device="cuda"):
        # self.base_path = base_path
        # self.project_path = project_path
        # self.adapter_path = f"{self.project_path}/models/math_without_reason_13bf"
        # self.dataset_name = dataset_name
        self.device=device
        self.model = model
        
        # set self.grad_dim
        dummy_input = torch.tensor([[0,0,0,0]]).to(self.device)
        self.model.eval()
        self.model.zero_grad()
        outputs = self.model(input_ids=dummy_input, labels=dummy_input)
        loss = outputs.loss
        loss.backward()
        for k, v in self.model.named_parameters():
            if 'lora_A' in k and v.grad is not None:
                self.grad_dim = v.grad.cpu().shape[-1]
                break
            elif 'lora_B' in k and v.grad is not None:
                self.grad_dim = v.grad.cpu().T.shape[-1]
                break
        print("self.grad_dim",self.grad_dim)
        
        self.projector = CudaProjector(grad_dim=self.grad_dim, proj_dim=512,seed=42, proj_type=ProjectionType.rademacher,device=self.device, max_batch_size=8)
        # self.projector = NoOpProjector()
        # self.load_pretrained_network()
        # self.load_datasets()

    # def load_pretrained_network(self):
    #     # setup tokenizer
    #     self.tokenizer = LlamaTokenizer.from_pretrained(self.base_path)
    #     self.tokenizer.padding_side = "right"
    #     self.tokenizer.pad_token = self.tokenizer.eos_token
    #     self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    #     # load a base model
    #     quantization_config = BitsAndBytesConfig(load_in_8bit=True, load_in_4bit=False)
    #     base_model = LlamaForCausalLM.from_pretrained(
    #         self.base_path,
    #         quantization_config=quantization_config,
    #         torch_dtype=torch.bfloat16,
    #         offload_folder="offload",
    #         offload_state_dict=True,
    #     )

    #     # load a pre-trained model.
    #     self.model = PeftModel.from_pretrained(base_model, self.adapter_path, is_trainable=True)
    #     self.finetuned_config = LoraConfig.from_pretrained(pretrained_model_name_or_path=self.adapter_path)

    # def load_datasets(self):
    #     self.train_dataset = Dataset.load_from_disk(f"{self.project_path}/datasets/{self.dataset_name}_train.hf").shuffle(seed=42).select(range(100))
    #     self.validation_dataset = Dataset.load_from_disk(f"{self.project_path}/datasets/{self.dataset_name}_test.hf").shuffle(seed=42).select(range(100))

    def create_tokenized_datasets(self):
        tokenize_func = lambda x: self.tokenizer(
            x["prompt"], truncation=True, padding=True, max_length=128, return_tensors="pt" # text should be more appropritate
        ).to(self.device)

        if 'with_reason' in self.dataset_name:
            column_list=["text", "answer", "variation", "prompt", "reason"]
        else:
            column_list=["text", "answer", "variation", "prompt"]

        tokenized_datasets=dict()
        tokenized_datasets["train"] = self.train_dataset.map(
            tokenize_func,
            batched=True,
            remove_columns=column_list,
        )
        tokenized_datasets["validation"] = self.validation_dataset.map(
            tokenize_func,
            batched=True,
            remove_columns=column_list,
        )
        collate_fn = lambda x: self.tokenizer.pad(x, padding="longest", return_tensors="pt")

        return tokenized_datasets, collate_fn

    def compute_gradient(self, tokenized_dataset, collate_fn):
            
            dataloader = DataLoader(tokenized_dataset, shuffle=False, collate_fn=collate_fn, batch_size=1)
             
            self.model.eval()
            grad_dicts = {}
            
            for step, batch in enumerate(tqdm(dataloader)):
                self.model.zero_grad()
                batch['labels'] = batch['input_ids']
                batch.to(self.device)
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                grad_dict = {}
                for k, v in self.model.named_parameters():
                    if 'lora_A' in k:
                        grad = v.grad
                        with torch.no_grad():
                            grad_dict[k] = self.projector.project(grad.contiguous(), model_id=0).cpu()
                    elif 'lora_B' in k:
                        grad = v.grad.T
                        with torch.no_grad():
                            grad_dict[k] = self.projector.project(grad.contiguous(), model_id=0).cpu()

                grad_dicts[step] = grad_dict
                # del grad_dict
            return grad_dicts

