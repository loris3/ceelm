import os
import pandas as pd
import pickle
from tqdm import tqdm
from peft import PeftModel
from finetune import tokenize_dataset
import logging
import torch
from influence_estimation.estimator import BaseEstimator
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from torch.nn.functional import normalize
from trak.projectors import BasicProjector, CudaProjector, ProjectionType




class LESSEstimator(BaseEstimator):
    def __init__(self, model_path, adapter_path, train_dataset, test_dataset, tokenizer_path=None, device="cuda", normalize=True, proj_dim=8192):
        super().__init__(model_path, adapter_path, train_dataset, test_dataset, tokenizer_path, device,
                         param_list=[
                             proj_dim,
                             normalize]
                         )
        self.normalize = normalize
        self.proj_dim = proj_dim
        
        self.optimizer_path = os.path.join(self.adapter_path, "optimizer.pt")
        self.adam_optimizer_state = torch.load(self.optimizer_path, map_location="cpu")["state"]
  

        self.run_cached()  
        if hasattr(self, 'model'):
            del self.model
    def get_config_string(self):
        return f"{self.__class__.__name__}: normalize={str(self.normalize)}"
  
    def run(self):
        grads_train = None
        grads_test = None
        try:
            grads_train = self.load_gradients(self.train_dataset)
        except (FileNotFoundError, RuntimeError):
            self.get_gradients(self.train_dataset, gradient_type="adam")
            grads_train = self.load_gradients(self.train_dataset)
        try:
            grads_test = self.load_gradients(self.test_dataset)
        except (FileNotFoundError, RuntimeError):
            self.get_gradients(self.test_dataset, gradient_type="sgd")
            grads_test = self.load_gradients(self.test_dataset)
        
        self.influence_estimate = pd.DataFrame(torch.einsum('nd,md->mn', grads_train, grads_test).numpy())
        self.save()

    def get_gradients(self, dataset, gradient_type):
        dataloader = self.get_dataloader(dataset)
        model_id = 0  # model_id is used to draft the random seed for the projectors
        block_size = 128  # fixed block size for the projectors
        projector_batch_size = 16  # batch size for the projectors
        torch.random.manual_seed(0)  # set the random seed for torch

        project_interval = 16  # project every 16 batches
        

        def _project(current_full_grads, projected_grads):
            current_full_grads = torch.stack(current_full_grads).to(torch.float16)
         
            current_projected_grads = self.projector.project(
                current_full_grads, model_id=model_id)
            projected_grads.append(current_projected_grads.cpu())

        if gradient_type == "adam":
            assert self.adam_optimizer_state is not None
            m, v = self.prepare_optimizer_state()
            
        number_of_params = self.get_number_of_params()
     
        self.projector = CudaProjector(grad_dim=number_of_params,
                        proj_dim=self.proj_dim,
                        seed=0,
                        proj_type=ProjectionType.rademacher,
                        device=self.device,
                        block_size=block_size,
                        max_batch_size=projector_batch_size)       
        count = 0
        # projected_gradients
        full_grads = []  # full gradients
        projected_grads = []  # projected gradients

        for batch in tqdm(dataloader, total=len(dataloader)):
            self.prepare_batch(batch)
            count += 1
            if gradient_type == "adam":
                vectorized_grads = self.obtain_gradients_with_adam(batch, m, v)
            if gradient_type == "sgd":
                vectorized_grads = self.obtain_gradients(batch)
            # add the gradients to the full_grads
            full_grads.append(vectorized_grads)
            self.model.zero_grad()

            if count % project_interval == 0:
                _project(full_grads, projected_grads)
                full_grads = []
        if len(full_grads) > 0:
            _project(full_grads, projected_grads)
            full_grads = []

        torch.cuda.empty_cache()
        merged_data = torch.cat(projected_grads, dim=0)
  
        if self.normalize:
            merged_data = normalize(merged_data, dim=1)
        
        self.store_gradients(dataset, merged_data)
        
    


    def get_dataloader(self, dataset, batch_size=1):
        data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer, padding="longest") 
        
        

        dataloader = DataLoader(tokenize_dataset(dataset, self.tokenizer),
                                batch_size=batch_size,  # When getting gradients, we only do this single batch process
                                collate_fn=data_collator)
        print("There are {} examples in the dataset".format(len(dataset)))
        return dataloader
    def prepare_optimizer_state(self):
        names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        
        assert len(names) == len(self.adam_optimizer_state) # assume ordered state_dict: https://github.com/princeton-nlp/LESS/issues/4#issuecomment-2955009098
      
        avg = torch.cat([self.adam_optimizer_state[i]["exp_avg"].view(-1) for i,_ in enumerate(names)])
        avg_sq = torch.cat([self.adam_optimizer_state[i]["exp_avg_sq"].view(-1)
                         for i,_ in enumerate(names)])
        avg = avg.to(self.device)
        avg_sq = avg_sq.to(self.device)
        return avg, avg_sq


    def get_number_of_params(self):
        """ Make sure that only lora parameters require gradients in peft models. """
        if isinstance(self.model, PeftModel):
            names = [n for n, p in self.model.named_parameters(
            ) if p.requires_grad and "lora" not in n]
            assert len(names) == 0
        num_params = sum([p.numel()
                        for p in self.model.parameters() if p.requires_grad])
        print(f"Total number of parameters that require gradients: {num_params}")
        return num_params
    def prepare_batch(self,batch):
        """ Move the batch to the device. """
   
        for key in batch:
            if batch[key] is not None:
                batch[key] = batch[key].to(self.device)


    def obtain_gradients(self, batch):
        """ obtain gradients. """
        batch['labels'] = batch['input_ids']
        batch.to(self.device)
        loss = self.model(**batch).loss
        loss.backward()
        vectorized_grads = torch.cat(
            [p.grad.view(-1) for p in self.model.parameters() if p.grad is not None])
        return vectorized_grads

    def obtain_gradients_with_adam(self,batch, avg, avg_sq):
        """ obtain gradients with adam optimizer states. """
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-08
        batch['labels'] = batch['input_ids']
        batch.to(self.device)
        loss = self.model(**batch).loss
        loss.backward()

        vectorized_grads = torch.cat(
            [p.grad.view(-1) for n, p in self.model.named_parameters() if p.grad is not None])

        updated_avg = beta1 * avg + (1 - beta1) * vectorized_grads
        updated_avg_sq = beta2 * avg_sq + (1 - beta2) * vectorized_grads ** 2
        vectorized_grads = updated_avg / torch.sqrt(updated_avg_sq + eps)

        return vectorized_grads

