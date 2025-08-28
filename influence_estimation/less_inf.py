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
    def __init__(self, model_path, train_dataset, train_dataset_name, train_dataset_split, test_dataset, test_dataset_name, test_dataset_split, device="cuda", normalize=True, proj_dim=8192):
        super().__init__(model_path, train_dataset, train_dataset_name, train_dataset_split, test_dataset, test_dataset_name, test_dataset_split, device,
                         param_list=[
                             proj_dim,
                             normalize]
                         )
        self.normalize = normalize
        self.proj_dim = proj_dim


        self.optimizer_path = os.path.join(self.model_path, "optimizer.pt")
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
            grads_train = self.load_gradients(self.train_dataset, self.train_dataset_name, self.train_dataset_split)
        except (FileNotFoundError, RuntimeError):
            self.get_gradients(self.train_dataset, self.train_dataset_name, self.train_dataset_split, gradient_type="adam" if self.adam_optimizer_state is not None else "sgd")
            grads_train = self.load_gradients(self.train_dataset, self.train_dataset_name, self.train_dataset_split)
        try:
            grads_test = self.load_gradients(self.test_dataset, self.test_dataset_name, self.test_dataset_split)
        except (FileNotFoundError, RuntimeError):
            self.get_gradients(self.test_dataset, self.test_dataset_name, self.test_dataset_split, gradient_type="sgd")
            grads_test = self.load_gradients(self.test_dataset, self.test_dataset_name, self.test_dataset_split)
        
        self.influence_estimate = pd.DataFrame(torch.einsum('nd,md->mn', grads_train, grads_test).numpy())
        self.save()

    def get_gradients_slow(self, dataset, dataset_name, dataset_split, gradient_type):
        device = self.device # single gpu
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
            m, v = self.prepare_optimizer_state(device)
            
        number_of_params = self.get_number_of_params()
     
        self.projector = CudaProjector(grad_dim=number_of_params,
                        proj_dim=self.proj_dim,
                        seed=0,
                        proj_type=ProjectionType.rademacher,
                        device=device,
                        block_size=block_size,
                        max_batch_size=projector_batch_size)       
        count = 0
        # projected_gradients
        full_grads = []  # full gradients
        projected_grads = []  # projected gradients

        for batch in tqdm(dataloader, total=len(dataloader)):
            self.prepare_batch(batch, device)
            count += 1
            if gradient_type == "adam":
                vectorized_grads = self.obtain_gradients_with_adam(batch, m, v, device)
            if gradient_type == "sgd":
                vectorized_grads = self.obtain_gradients(batch, device)
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
        
        self.store_gradients(dataset, dataset_name, dataset_split, merged_data)
        

    def _get_gradients_batch(self, dataloader, rank, partial_results_dir, gradient_type):
        print("_get_gradients_batch", rank)
        device = f"cuda:{rank % torch.cuda.device_count()}"
        self.model.to(device)
        self.model.eval()

        grads = []
        model_id = 0
        block_size = 128
        projector_batch_size = 16

        torch.random.manual_seed(0)

        if gradient_type == "adam":
            assert self.adam_optimizer_state is not None
            m, v = self.prepare_optimizer_state(device)
        else:
            m, v = None, None

        number_of_params = self.get_number_of_params()
        self.projector = CudaProjector(
            grad_dim=number_of_params,
            proj_dim=self.proj_dim,
            seed=0,
            proj_type=ProjectionType.rademacher,
            device=device,
            block_size=block_size,
            max_batch_size=projector_batch_size,
        )

        project_interval = 16
        full_grads = []

        def _flush_and_project():
            nonlocal full_grads
            if not full_grads:
                return
            stacked = torch.stack(full_grads).to(torch.float16)
            with torch.no_grad():
                proj = self.projector.project(stacked, model_id=model_id)
            grads.append(proj.cpu())
            full_grads = []

        for batch in tqdm(dataloader, position=rank, desc=f"Worker {rank}"):
            self.prepare_batch(batch, device)

            if gradient_type == "adam":
                vectorized_grads = self.obtain_gradients_with_adam(batch, m, v, device)
            else:
                vectorized_grads = self.obtain_gradients(batch, device)

            full_grads.append(vectorized_grads)
            self.model.zero_grad()

            if len(full_grads) % project_interval == 0:
                _flush_and_project()

        _flush_and_project()
        torch.cuda.empty_cache()

        # save partial results
        os.makedirs(partial_results_dir, exist_ok=True)
        save_path = os.path.join(partial_results_dir, f"grads_rank_{rank}.pt")
        torch.save(grads, save_path)


    def get_gradients(self, dataset, dataset_name, dataset_split, gradient_type):
  
        partial_results_dir = os.path.join(
            "./cache/gradients/partial",
            self.__class__.__name__,
            "partial",
             self.param_string, os.path.basename(self.model_path),
            "_".join([dataset_name, dataset_split])
        )
        os.makedirs(partial_results_dir, exist_ok=True)


        def batch_map(batch, rank):
            try:
                if rank is None:
                    rank = 0
                # batch_list = [{k: v[i] for k, v in batch.items()} for i in range(len(batch))]
                from datasets import Dataset

                batch = Dataset.from_dict(batch)
                print("bs", rank)
                dataloader = self.get_dataloader(batch)
                self._get_gradients_batch(dataloader, rank, partial_results_dir, gradient_type)
                return {"_": [None] * len(batch)}
            except Exception as e:
                import traceback
                print(f"Rank {rank} failed:", e)
                traceback.print_exc()
                raise
        # parallel map across GPUs
        dataset.map(
            batch_map,
            batched=True,
            with_rank=True,
            batch_size=(len(dataset) + torch.cuda.device_count() - 1) // torch.cuda.device_count(), # so that exactly num_gpu proc
            num_proc=torch.cuda.device_count(),
        )

        # gather results
        grads = []
        files = [f for f in os.listdir(partial_results_dir) if f.startswith("grads_rank_") and f.endswith(".pt")]
        files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        assert len(files) == torch.cuda.device_count()
        for fname in files:
            path = os.path.join(partial_results_dir, fname)
            grads.extend(torch.load(path))

        merged_data = torch.cat(grads, dim=0)

        if self.normalize:
            merged_data = normalize(merged_data, dim=1)

        self.store_gradients(dataset, dataset_name, dataset_split, merged_data)
        # optionally clean up
        # shutil.rmtree(partial_results_dir)

        return merged_data
    


    def get_dataloader(self, dataset, batch_size=1):
        data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer, padding="longest") 
        
        

        dataloader = DataLoader(tokenize_dataset(dataset, self.tokenizer, num_proc=1),
                                batch_size=batch_size,  # When getting gradients, we only do this single batch process
                                collate_fn=data_collator)
        print("There are {} examples in the dataset".format(len(dataset)))
        return dataloader
    def prepare_optimizer_state(self, device):
        names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        
        assert len(names) == len(self.adam_optimizer_state) # assume ordered state_dict: https://github.com/princeton-nlp/LESS/issues/4#issuecomment-2955009098
      
        avg = torch.cat([self.adam_optimizer_state[i]["exp_avg"].view(-1) for i,_ in enumerate(names)])
        avg_sq = torch.cat([self.adam_optimizer_state[i]["exp_avg_sq"].view(-1)
                         for i,_ in enumerate(names)])
        avg = avg.to(device)
        avg_sq = avg_sq.to(device)
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
    def prepare_batch(self,batch, device):
        """ Move the batch to the device. """
   
        for key in batch:
            if batch[key] is not None:
                batch[key] = batch[key].to(device)


    def obtain_gradients(self, batch, device):
        """ obtain gradients. """
        batch['labels'] = batch['input_ids']
        batch.to(device)
        loss = self.model(**batch).loss
        loss.backward()
        vectorized_grads = torch.cat(
            [p.grad.view(-1) for p in self.model.parameters() if p.grad is not None])
        return vectorized_grads

    def obtain_gradients_with_adam(self,batch, avg, avg_sq, device):
        """ obtain gradients with adam optimizer states. """
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-08
        batch['labels'] = batch['input_ids']
        batch.to(device)
        loss = self.model(**batch).loss
        loss.backward()

        vectorized_grads = torch.cat(
            [p.grad.view(-1) for n, p in self.model.named_parameters() if p.grad is not None])

        updated_avg = beta1 * avg + (1 - beta1) * vectorized_grads
        updated_avg_sq = beta2 * avg_sq + (1 - beta2) * vectorized_grads ** 2
        vectorized_grads = updated_avg / torch.sqrt(updated_avg_sq + eps)

        return vectorized_grads

