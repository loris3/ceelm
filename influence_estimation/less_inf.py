import os
import pandas as pd
import pickle
from tqdm import tqdm
from peft import PeftModel
from finetune import tokenize_dataset
import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,  
    format="%(asctime)s [%(levelname)s] [%(processName)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger(__name__)
import torch
from influence_estimation.estimator import BaseEstimator
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from torch.nn.functional import normalize
from trak.projectors import BasicProjector, CudaProjector, ProjectionType,NoOpProjector
from functools import partial

from concurrent.futures import ProcessPoolExecutor
class LESSEstimator(BaseEstimator):
    def __init__(self, model_path, train_dataset, train_dataset_name, train_dataset_split, test_dataset, test_dataset_name, test_dataset_split, device="cuda", normalize=True, proj_dim=2**13):
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
        grads_test = None
        if not self.all_gradients_exist(self.train_dataset, self.train_dataset_name, self.train_dataset_split, self.gradient_cache_dir):
            self.get_gradients(self.train_dataset, gradient_type="adam" if self.adam_optimizer_state is not None else "sgd", dataset_name=self.train_dataset_name, dataset_split_name=self.train_dataset_split, gradient_cache_dir=self.gradient_cache_dir)
        else:
            print("train grads cached")
            # grads_train = self.load_gradients(self.train_dataset, self.train_dataset_name, self.train_dataset_split)
        try:
            grads_test = self.load_gradients(self.test_dataset, self.test_dataset_name, self.test_dataset_split)
        except (FileNotFoundError, RuntimeError):
            self.get_gradients(self.test_dataset,  gradient_type="sgd", dataset_name=self.test_dataset_name, dataset_split_name=self.test_dataset_split,gradient_cache_dir=self.gradient_cache_dir)
            # grads_test = self.load_gradients(self.test_dataset, self.test_dataset_name, self.test_dataset_split)
            self.run()
            return
        
        
        # flat_grads_train = torch.stack(list(grads_train.values()),dim=0)
        # flat_grads_test = torch.stack(list(grads_test.values()),dim=0)

        # self.influence_estimate = pd.DataFrame(-torch.einsum('nd,md->mn', flat_grads_train, flat_grads_test).numpy())
        
     
        n_train = len(self.train_dataset)
        n_test = len(grads_test)

        flat_grads_test = torch.cat([g.view(1, -1) for g in grads_test.values()], dim=0)  

    
        influence_matrix = torch.empty(n_test, n_train)

        for i in tqdm(range(n_train), desc="Dot products"):
            train_grad = self.get_gradient(self.train_dataset, self.train_dataset_name, self.train_dataset_split, i)
            flat_train_grad = train_grad.view(1, -1)  # (1, D)
            

            dots = -flat_train_grad @ flat_grads_test.T  
            

            influence_matrix[:, i] = dots.squeeze(0)

        self.influence_estimate = pd.DataFrame(
            influence_matrix.numpy(),
            index=list(grads_test.keys()),
            columns=list(range(n_train))  
        )

        self.save()

   
    

    def get_gradients(self, dataset, gradient_type, dataset_name, dataset_split_name, gradient_cache_dir):
  

        print("compute_gradient", flush=True)
        
        
        world_size = torch.cuda.device_count()
        batch_size = (len(dataset) + world_size - 1) // world_size
        chunks = [
            dataset.select(range(i * batch_size, min((i + 1) * batch_size, len(dataset))))
            for i in range(world_size)
        ]
        
        fn = partial(
            batch_map,
            model=self.model,
            gradient_type=gradient_type,
            tokenizer=self.tokenizer,
            adam_optimizer_state=self.adam_optimizer_state,
            proj_dim=self.proj_dim,
            dataset_name=dataset_name, 
            dataset_split_name=dataset_split_name,
            gradient_cache_dir=gradient_cache_dir
        )
        
      
        with ProcessPoolExecutor(max_workers=world_size) as executor:
                    futures = []
                    for rank, chunk in enumerate(chunks):
                        batch_dict = {k: [ex[k] for ex in chunk] for k in chunk.column_names}
                        futures.append(executor.submit(fn, batch_dict, rank))
                    for f in futures:
                        try:
                            f.result()
                        except Exception as e:
                            logger.error(f"Future {i} raised exception: {e}", exc_info=True)
                            raise

  
    def all_gradients_exist(self, dataset, dataset_name, dataset_split_name, gradient_cache_dir):
        base_path = os.path.join(gradient_cache_dir, dataset_name, dataset_split_name)
        if not os.path.exists(base_path):
            return False

        for idx in range(len(dataset)):
            grad_path = os.path.join(base_path, f"gradient_{idx}.pt")
            if not os.path.isfile(grad_path):
                return False

        return True




   
    def get_gradient(self, dataset, dataset_name, dataset_split, train_instance_idx):
        grads_dict = super().get_gradient(dataset, dataset_name, dataset_split, train_instance_idx)
        return  list(grads_dict.values())[0].flatten()
 




def batch_map(batch, rank, model, gradient_type, tokenizer,  proj_dim, adam_optimizer_state, gradient_cache_dir, dataset_name, dataset_split_name):
    try:
        if rank is None:
            rank = 0
        # batch_list = [{k: v[i] for k, v in batch.items()} for i in range(len(batch))]
        from datasets import Dataset

        batch = Dataset.from_dict(batch)
        print("bs", rank)
        _get_gradients_batch(get_dataloader(tokenizer, batch), rank, model,  gradient_type, adam_optimizer_state, proj_dim,
                                 gradient_cache_dir, dataset_name, dataset_split_name)


    except Exception as e:
        import traceback
        tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        logger.error(f"Worker {rank} failed with exception:\n{tb_str}")
        raise
def get_dataloader(tokenizer, dataset, batch_size=1):
    data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding="longest") 
    
    

    dataloader = DataLoader(tokenize_dataset(dataset, tokenizer, num_proc=20, re_index=False),
                            batch_size=batch_size,  # When getting gradients, we only do this single batch process
                            collate_fn=data_collator)
    print("There are {} examples in the dataset".format(len(dataset)))
    return dataloader
def _get_gradients_batch(dataloader, rank, model, gradient_type, adam_optimizer_state, proj_dim,
                         gradient_cache_dir=None, dataset_name=None, dataset_split_name=None):
        from .estimator import store_gradient, gradient_exists
        print("_get_gradients_batch", rank)
        device = f"cuda:{rank % torch.cuda.device_count()}"
        model.to(device)
        model.eval()

        grads = []


        torch.random.manual_seed(0)

        if gradient_type == "adam":
            assert adam_optimizer_state is not None
            m, v = prepare_optimizer_state(model, device, adam_optimizer_state)
        else:
            m, v = None, None

        # number_of_params = get_number_of_params(model)
        # projector = CudaProjector(
        #     grad_dim=number_of_params,
        #     proj_dim=proj_dim,
        #     seed=0,
        #     proj_type=ProjectionType.rademacher,
        #     device=device,
        #     block_size=block_size,
        #     max_batch_size=projector_batch_size,
        # )
        dummy_input = torch.tensor([[0,0,0,0]]).to(device)
        outputs = model(input_ids=dummy_input, labels=dummy_input)
        loss = outputs.loss
        loss.backward()
        
        # we divide the total projection budget (self.proj_dim) among LoRA parameters proportional to their gradient size
        # while ensuring multiple of 512 for TRAK kernel
        param_dims = {}
        total_grad_dim = 0

        # Collect gradient dimensions
        for name, param in model.named_parameters():
            grad = param.grad
            if grad is None:
                continue
            if 'lora_A' in name:
                dim = grad.shape[-1]
            elif 'lora_B' in name:
                dim = grad.T.shape[-1] if grad.ndim >= 2 else grad.shape[-1]
            else:
                continue
            param_dims[name] = dim
            total_grad_dim += dim

        param_proj_dim = {}
        param_projectors = {}


        for name, dim in param_dims.items():

            proj_dim_ = min(dim, int(proj_dim * dim / total_grad_dim))
            
            if proj_dim_ <= 512:
                param_projectors[name] = NoOpProjector()
                proj_dim_ = dim  # keep original dimension for total sum
            else:
                proj_dim_ = max(512, round(proj_dim_ / 512) * 512)
                param_projectors[name] = CudaProjector(
                    grad_dim=dim,
                    proj_dim=proj_dim_,
                    seed=42,
                    proj_type=ProjectionType.rademacher,
                    device=device,
                    max_batch_size=8
                )

            param_proj_dim[name] = proj_dim_

        total_grad_dim_proj = sum(param_proj_dim.values())

        print("self.proj_dim", proj_dim)
        print("self.total_grad_dim_proj", total_grad_dim_proj)
        print("self.total_grad_dim", total_grad_dim)
        print("self.param_projectors", param_proj_dim, flush=True)
  

        for row in tqdm(dataloader, position=rank, desc=f"Worker {rank}"):
            if not  gradient_exists(gradient_cache_dir, dataset_name, dataset_split_name, row["indices"][0].item()):
                model.zero_grad()
                row['labels'] = row['input_ids']
                row.to(device)
                outputs = model(**row)
                loss = outputs.loss
                loss.backward()
                grads = []
                for k, v in model.named_parameters():
                    grad = None
                    if k not in param_projectors:
                        continue
                    if 'lora_A' in k:
                        grad = v.grad
                    
                    elif 'lora_B' in k:
                        grad = v.grad.T
                    with torch.no_grad():
                        proj_grad = param_projectors[k].project(grad.contiguous(), model_id=0).detach().cpu()
                        grads.append(proj_grad)
                        del grad
                grad_dict = {row["indices"][0].item(): torch.cat([g.flatten() for g in grads])}
                store_gradient(gradient_cache_dir, dataset_name, dataset_split=dataset_split_name, gradient_dict=grad_dict)
                torch.cuda.empty_cache() 
    
            
       
      

def prepare_optimizer_state(model, device, adam_optimizer_state):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    
    assert len(names) == len(adam_optimizer_state) # assume ordered state_dict: https://github.com/princeton-nlp/LESS/issues/4#issuecomment-2955009098
    
    avg = torch.cat([adam_optimizer_state[i]["exp_avg"].view(-1) for i,_ in enumerate(names)])
    avg_sq = torch.cat([adam_optimizer_state[i]["exp_avg_sq"].view(-1)
                        for i,_ in enumerate(names)])
    avg = avg.to(device)
    avg_sq = avg_sq.to(device)
    return avg, avg_sq

def get_number_of_params(model):
    """ Make sure that only lora parameters require gradients in peft models. """
    if isinstance(model, PeftModel):
        names = [n for n, p in model.named_parameters(
        ) if p.requires_grad and "lora" not in n]
        assert len(names) == 0
    num_params = sum([p.numel()
                    for p in model.parameters() if p.requires_grad])
    print(f"Total number of parameters that require gradients: {num_params}")
    return num_params

def prepare_batch(batch, device):
    """ Move the batch to the device. """

    for key in batch:
        if batch[key] is not None:
            batch[key] = batch[key].to(device)
def obtain_gradients(batch, device,model):
        """ obtain gradients. """
        batch['labels'] = batch['input_ids']
        batch.to(device)
        loss = model(**batch).loss
        loss.backward()
        vectorized_grads = torch.cat(
            [p.grad.view(-1) for p in model.parameters() if p.grad is not None])
        return vectorized_grads

def obtain_gradients_with_adam(batch, avg, avg_sq, device, model):
    """ obtain gradients with adam optimizer states. """
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-08
    batch['labels'] = batch['input_ids']
    batch.to(device)
    loss = model(**batch).loss
    loss.backward()

    vectorized_grads = torch.cat(
        [p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None])

    updated_avg = beta1 * avg + (1 - beta1) * vectorized_grads
    updated_avg_sq = beta2 * avg_sq + (1 - beta2) * vectorized_grads ** 2
    vectorized_grads = updated_avg / torch.sqrt(updated_avg_sq + eps)

    return vectorized_grads