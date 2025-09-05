from abc import ABC, abstractmethod
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(42)
cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

device = "cuda"

from torch.cuda.amp import autocast, GradScaler



import torch
import torch.nn as nn
import torch.nn.functional as F



import wandb

class LinearCoder(ABC, nn.Module):
    def __init__(self, train_grads, test_grad, device=None, metadata_only=False,
                 use_wandb=False, project="linear_coder", estimator_config=""):
        super().__init__()
        self.device = device
        self.use_wandb = use_wandb
        self.best_factors = None
        self.steps_no_improve = 0

        if not metadata_only:
            self.register_buffer("train_grads", train_grads)  
            self.register_buffer("test_grad", test_grad)      
            self.factors = nn.Parameter(torch.zeros(self.train_grads.shape[0], device=self.device))
            

            if self.use_wandb:
                wandb.init(project=project, name="_".join([self.description, estimator_config]), config={
                    "coder": self.description,
                    "estimator": estimator_config,
                    "device": device,
                    "train_shape": train_grads.shape,
                    "test_shape": test_grad.shape,
                },
                settings=wandb.Settings(
                    console="off"    
                ))
    @abstractmethod
    def loss(self, test_grad, combination, reg_lambda):
        pass

    @property
    def description(self):
        return str(self.__class__.__name__)

    @abstractmethod
    def score(self, reconstruction):
        pass

    def forward(self):
        return self.train_grads.T @ self.factors  
    def fit(
        self,
        lr=1e-2,
        max_steps=100000,
        patience=1000,
        min_steps=1000,
        scheduler_step_freq=50,
        eval_freq=100,
        tol=1e-4,
        ema_beta_init=0.9,
    ):
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_freq, gamma=0.1)
        scaler = torch.amp.GradScaler(enabled=(self.device == 'cuda'))

        if self.use_wandb:
            wandb.config.update({
                "lr": lr,
                "max_steps": max_steps,
                "patience": patience,
                "scheduler_step_freq": scheduler_step_freq,
                "eval_freq": eval_freq,
                "tol": tol,
            })

        best_score = None
        best_factors = None
        no_improve_steps = 0
        ema_score = None

        for step in range(1, max_steps + 1):
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', enabled=(self.device == 'cuda')):
                reconstruction = self.forward()
                loss = self.loss(self.test_grad, reconstruction, self.factors)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            if step % scheduler_step_freq == 0:
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step()
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr < old_lr:
                    no_improve_steps = 0 

    
            if step % eval_freq == 0:
                with torch.no_grad():
                    current_score = self.score(reconstruction).item()

                if ema_score is None:
                    ema_score = current_score
                else:
                    ema_beta = min(0.99, ema_beta_init + 0.1 * step / max_steps)
                    ema_score = ema_beta * ema_score + (1 - ema_beta) * current_score

                if best_score is None:
                    best_score = current_score

                abs_improve = ema_score - best_score
                rel_improve = abs_improve / (abs(best_score) + 1e-8)

                if abs_improve > tol or rel_improve > tol:
                    best_score = ema_score
                    best_factors = self.factors.detach().clone()
                    no_improve_steps = 0
                else:
                    no_improve_steps += eval_freq

   
                current_lr = optimizer.param_groups[0]['lr']
                dynamic_patience = max(patience // 2, int(patience * current_lr / lr))

                if self.use_wandb:
                    wandb.log({
                        "step": step,
                        "loss": loss.item(),
                        "score": current_score,
                        "improvement": abs_improve,
                        "ema_score": ema_score,
                        "lr": optimizer.param_groups[0]['lr'],
                        "no_improve_steps": no_improve_steps,
                    }, commit=True, step=step)

     
                if step > min_steps and no_improve_steps >= dynamic_patience:
                    print(f"Early stopping at step {step}, best_score={best_score:.6f}")
                    break

        if best_factors is not None:
            with torch.no_grad():
                self.factors.copy_(best_factors)

        if self.use_wandb:
            wandb.summary["best_score"] = best_score
            wandb.finish()

        return best_score, best_factors



class KLTCoder(LinearCoder):
    def __init__(self, train_grads, test_grad, device=None,reg_lambda=0.05, metadata_only=False,
                 use_wandb=False, project="linear_coder",  estimator_config=""):
        self.reg_lambda = reg_lambda
   
        
        super().__init__(train_grads, test_grad, device, metadata_only=metadata_only, 
                         use_wandb=False, project=None,  estimator_config= estimator_config)
        if not metadata_only:
            with torch.no_grad():
        
                train_grads_cpu = self.train_grads.to("cpu")
                mean = train_grads_cpu.mean(dim=0, keepdim=True)
                centered = train_grads_cpu - mean
                
                U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
                klt_basis = Vh.T.to(device)
                # train_grads = train_grads_cpu.to(device)
                test_grad_flat = self.test_grad.view(-1).to(device)
                proj_on_klt = klt_basis.T @ test_grad_flat  
                target = klt_basis @ proj_on_klt  
                factors_init = torch.linalg.pinv(self.train_grads.T) @ target
                self.factors = nn.Parameter(factors_init.to(device))

    def loss(self, test_grad, combination, factors, alpha=0.5): 
        pass
    def fit(self, max_steps=1000):
        pass
    def score(self):
        raise NotImplementedError()
    
class CosineCoder(LinearCoder):
    def __init__(self, train_grads, test_grad, device=None, metadata_only=False,
                 use_wandb=False, project="linear_coder", estimator_config=""):
        
        super().__init__(train_grads, test_grad, device, metadata_only=metadata_only,
                         use_wandb=use_wandb, project=project, estimator_config= estimator_config)

    def loss(self, test_grad, combination, factors): 
        return -F.cosine_similarity(test_grad.unsqueeze(0), combination.unsqueeze(0)) 

    def score(self, reconstruction):
        with torch.no_grad():
            return F.cosine_similarity(self.test_grad.unsqueeze(0), reconstruction.unsqueeze(0))

    
class BaseMSECoder(LinearCoder):
    def __init__(self, train_grads, test_grad, device=None, metadata_only=False,
                   use_wandb=False, project="linear_coder", estimator_config=""):
        super().__init__(train_grads, test_grad, device, metadata_only=metadata_only,
                         use_wandb=use_wandb, project=project, estimator_config= estimator_config)

    def loss(self, test_grad, combination, factors):

        mse_loss = F.mse_loss(test_grad, combination)
        
        return mse_loss
    
    def score(self, reconstruction):
        with torch.no_grad():
            return -F.mse_loss(reconstruction.unsqueeze(0), self.test_grad.unsqueeze(0))
        
class MSECoder(BaseMSECoder):
    def fit(self, lr=None, max_steps=None, patience=None, min_steps=None, **kwargs):

        with torch.no_grad():
            G = self.train_grads  
            g = self.test_grad   

  
            factors_opt = torch.linalg.pinv(G.T) @ g
            self.factors.copy_(factors_opt.to(self.device))


            reconstruction = self.forward()
            score_val = self.score(reconstruction).item()

        return score_val, self.factors.detach().clone()
    
    
class MSECoderL2(BaseMSECoder):
    def __init__(self, train_grads, test_grad, device=None, reg_lambda=0.7, metadata_only=False,
                   use_wandb=False, project="linear_coder", estimator_config=""):
        self.reg_lambda = reg_lambda
    
        
        super().__init__(train_grads, test_grad, device, metadata_only=metadata_only,
                         use_wandb=use_wandb, project=project, estimator_config= estimator_config)

    def loss(self, test_grad, combination, factors):
        return super().loss(test_grad, combination, factors) +  self.reg_lambda*torch.square(factors).sum()  
    def score(self, reconstruction):
        return super().score(reconstruction)
    def fit(self, **kwargs):
        with torch.no_grad():
            G = self.train_grads  
            g = self.test_grad

            #  (G G^T + λ I) f = G g
            A = G @ G.T + self.reg_lambda * torch.eye(G.shape[0], device=G.device)
            b = G @ g
            factors_opt = torch.linalg.solve(A, b)

        
            self.factors.copy_(factors_opt)

            reconstruction = self.forward()
            score_val = self.score(reconstruction).item()

        return score_val, self.factors.detach().clone()

class MSECoderElasticNet(BaseMSECoder):
    def __init__(self, train_grads, test_grad, device=None, reg_lambda_1=0.3, reg_lambda_2=0.7, metadata_only=False,
                   use_wandb=False, project="linear_coder", estimator_config=""):
        self.reg_lambda_1 = reg_lambda_1
        self.reg_lambda_2 = reg_lambda_2
    
   
        
        super().__init__(train_grads, test_grad, device, metadata_only=metadata_only,
                         use_wandb=use_wandb, project=project, estimator_config= estimator_config)

    def loss(self, test_grad, combination, factors):
        return super().loss(test_grad, combination, factors) + self.reg_lambda_1*torch.abs(factors).sum() + self.reg_lambda_2*torch.square(factors).sum()  
    def score(self, reconstruction):
        return super().score(reconstruction)

        
class MSECoderLemon(BaseMSECoder):
    def __init__(self, train_grads, test_grad, device=None, 
                 reg_lambda_1=0.3, 
                 reg_lambda_2=0.7, 
                 reg_lambda_3_non_negative=0.5, 
                 metadata_only=False, use_wandb=False, project="linear_coder", estimator_config=""):
        self.reg_lambda_1 = reg_lambda_1
        self.reg_lambda_2 = reg_lambda_2
        self.reg_lambda_3_non_negative = reg_lambda_3_non_negative       
        super().__init__(train_grads, test_grad, device, metadata_only=metadata_only,
                         use_wandb=use_wandb, project=project, estimator_config= estimator_config)

    def loss(self, test_grad, combination, factors):
        return super().loss(test_grad, combination, factors) \
            + self.reg_lambda_1*torch.abs(factors).sum() \
            + self.reg_lambda_2*torch.square(factors).sum() \
            + self.reg_lambda_3_non_negative*torch.sum(F.softplus(-factors)**2)        
    def score(self, reconstruction):
        return super().score(reconstruction)   
        
      
from scipy.optimize import nnls

# OptimizerCosineL1(train_grads[2:].to("cuda"), train_grads[0].to("cuda"), lr=0.1,reg_lambda=0, device="cuda")         



from scipy.optimize import nnls
import torch
import torch.nn.functional as F

class MSECoderNNLSL2(BaseMSECoder):
    def __init__(self, train_grads, test_grad, device=None, reg_lambda=0.7,
                 metadata_only=False, use_wandb=False, project="linear_coder", estimator_config=""):
        self.reg_lambda = reg_lambda
        super().__init__(train_grads, test_grad, device,
                         metadata_only=metadata_only,
                         use_wandb=use_wandb,
                         project=project,
                         estimator_config=estimator_config)

    def loss(self, test_grad, combination, factors):
        return F.mse_loss(combination, test_grad) + self.reg_lambda * torch.sum(factors**2)



    def fit(self, **kwargs):
        with torch.no_grad():
            G = self.train_grads
            g = self.test_grad


            A = (G @ G.T + self.reg_lambda * torch.eye(G.shape[0], device=G.device)).cpu().numpy()
            b = (G @ g).cpu().numpy()

            factors_opt, _ = nnls(A, b)

            self.factors.copy_(torch.tensor(factors_opt, device=self.device, dtype=self.factors.dtype))

      
            reconstruction = self.forward()
            score_val = self.score(reconstruction).item()

        return score_val, self.factors.detach().clone()
from collections import defaultdict
