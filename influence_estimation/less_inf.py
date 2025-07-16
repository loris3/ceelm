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


# Configure logger at module level
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class LESSEstimator(BaseEstimator):
    def __init__(self, model_path, adapter_path, train_dataset, test_dataset, tokenizer_path=None, device="cuda"):
        super().__init__(model_path, adapter_path, train_dataset, test_dataset, tokenizer_path, device)

        self.proj_dim = 512
        self.optimizer_path = os.path.join(self.adapter_path, "optimizer.pt")
        self.adam_optimizer_state = torch.load(self.optimizer_path, map_location="cpu")["state"]
        os.makedirs("./cache/gradients/less", exist_ok=True)
        self.output_dir_train = os.path.join("./cache/gradients/less/train",train_dataset._fingerprint)
        self.output_dir_test = os.path.join("./cache/gradients/less/test",test_dataset._fingerprint)

        self.run_cached()    
  
  
    def run(self):
        grads_train_path = os.path.join(self.output_dir_train, "all_orig.pt")
        grads_test_path = os.path.join(self.output_dir_test, "all_orig.pt")

        try:
            grads_train = torch.load(grads_train_path)
        except (FileNotFoundError, RuntimeError):
            self.get_gradients(self.get_dataloader(self.train_dataset), self.output_dir_train, gradient_type="adam")
            grads_train = torch.load(grads_train_path)

        try:
            grads_test = torch.load(grads_test_path)
        except (FileNotFoundError, RuntimeError):
            self.get_gradients(self.get_dataloader(self.test_dataset), self.output_dir_test, gradient_type="sgd")
            grads_test = torch.load(grads_test_path)

        self.influence_estimate = pd.DataFrame(torch.einsum('nd,md->nm', grads_train, grads_test).numpy())
        self.save()

    def get_gradients(self, dataloader, output_dir, gradient_type):
        model_id = 0  # model_id is used to draft the random seed for the projectors
        block_size = 128  # fixed block size for the projectors
        projector_batch_size = 16  # batch size for the projectors
        torch.random.manual_seed(0)  # set the random seed for torch

        project_interval = 16  # project every 16 batches
        save_interval = 160  # save every 160 batches

        def _project(current_full_grads, projected_grads):
            current_full_grads = torch.stack(current_full_grads).to(torch.float16)
         
            current_projected_grads = self.projector.project(
                current_full_grads, model_id=model_id)
            projected_grads.append(current_projected_grads.cpu())

        def _save(projected_grads):
            projected_grads = torch.cat(projected_grads)

     
            outfile = os.path.join(output_dir, f"grads-{count}.pt")
            torch.save(projected_grads, outfile)
            print(
                f"Saving {outfile}, {projected_grads.shape}", flush=True)
            projected_grads = []


        assert self.adam_optimizer_state is not None

        m, v = self.prepare_optimizer_state()

        projector = self.get_trak_projector()
        number_of_params = self.get_number_of_params()


     
        self.projector = projector(grad_dim=number_of_params,
                        proj_dim=self.proj_dim,
                        seed=0,
                        proj_type=ProjectionType.rademacher,
                        device=self.device,
                        # dtype=dtype,
                        block_size=block_size,
                        max_batch_size=projector_batch_size)
  
        count = 0

        # set up a output directory for each dimension
      


        os.makedirs(output_dir, exist_ok=True)

       

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

            if count % save_interval == 0:
                _save(projected_grads)

            # if self.max_samples is not None and count == self.max_samples:
            #     break

        if len(full_grads) > 0:
            _project(full_grads, projected_grads)
            full_grads = []


        _save(projected_grads)

        torch.cuda.empty_cache()
  

        self.merge_and_normalize_info(output_dir,prefix="grads")
        self.merge_info(output_dir, prefix="grads")

        print("Finished")
        
    def get_trak_projector(device: torch.device):
        """ Get trak projectors (see https://github.com/MadryLab/trak for details) """
        try:
            num_sms = torch.cuda.get_device_properties(
                device.index).multi_processor_count
            import fast_jl

            # test run to catch at init time if projection goes through
            fast_jl.project_rademacher_8(torch.zeros(
                8, 1_000, device=device), 512, 0, num_sms)
            projector = CudaProjector
            print("Using CudaProjector")
        except:
            projector = BasicProjector
            print("Using BasicProjector")
        return projector

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

    def merge_and_normalize_info(self, output_dir, prefix="reps"):
        """ Merge and normalize the representations and gradients into a single file. """
        info = os.listdir(output_dir)
        info = [file for file in info if file.startswith(prefix)]
        # Sort the files in ascending order
        info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
        merged_data = []
        for file in info:
            data = torch.load(os.path.join(output_dir, file))
            normalized_data = normalize(data, dim=1)
            merged_data.append(normalized_data)
        merged_data = torch.cat(merged_data, dim=0)

        output_file = os.path.join(output_dir, f"all_orig.pt")
        torch.save(merged_data, output_file)
        print(
            f"Saving the normalized {prefix} (Shape: {merged_data.shape}) to {output_file}.")


    def merge_info(self, output_dir, prefix="reps"):
        """ Merge the representations and gradients into a single file without normalization. """
        info = os.listdir(output_dir)
        info = [file for file in info if file.startswith(prefix)]
        # Sort the files in ascending order
        info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
        merged_data = []
        for file in info:
            data = torch.load(os.path.join(output_dir, file))
            merged_data.append(data)
        merged_data = torch.cat(merged_data, dim=0)

        output_file = os.path.join(output_dir, f"all_unormalized.pt")
        torch.save(merged_data, output_file)
        print(
            f"Saving the unnormalized {prefix} (Shape: {merged_data.shape}) to {output_file}.")
