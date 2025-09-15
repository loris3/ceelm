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
    def __init__(self, A, test_grad, device=None, metadata_only=False,
                 use_wandb=False, project="linear_coder", estimator_config=""):
        super().__init__()
        self.device = device
        self.use_wandb = use_wandb
        self.best_factors = None
        self.steps_no_improve = 0

        if not metadata_only:
            self.register_buffer("A", A)  
            self.register_buffer("test_grad", test_grad)      
            self.t = nn.Parameter(torch.zeros(self.A.shape[0], device=self.device))
            

            if self.use_wandb:
                print("wandb init!!!")
                wandb.init(project=project, name="_".join([self.description, estimator_config]), config={
                    "coder": self.description,
                    "estimator": estimator_config,
                    "device": device,
                    "train_shape": A.shape,
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
        return self.A.T @ self.t  
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
                loss = self.loss(self.test_grad, reconstruction, self.t)

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
                    best_factors = self.t.detach().clone()
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
                self.t.copy_(best_factors)

        if self.use_wandb:
            wandb.summary["best_score"] = best_score
            wandb.finish()

        return best_score, best_factors



class KLTCoder(LinearCoder):
    def __init__(self, A, test_grad, device=None,reg_lambda=0.05, metadata_only=False,
                 project="linear_coder",  estimator_config=""):
        self.reg_lambda = reg_lambda
   
        
        super().__init__(A, test_grad, device, metadata_only=metadata_only, 
                         use_wandb=False, project=None,  estimator_config= estimator_config)
        if not metadata_only:
            with torch.no_grad():
        
                A_cpu = self.A.to("cpu")
                mean = A_cpu.mean(dim=0, keepdim=True)
                centered = A_cpu - mean
                
                U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
                klt_basis = Vh.T.to(device)

                test_grad_flat = self.test_grad.view(-1).to(device)
                proj_on_klt = klt_basis.T @ test_grad_flat  
                target = klt_basis @ proj_on_klt  
                t_opt =  torch.linalg.pinv(self.A.T) @ target
                self.t.data.copy_(t_opt.to(self.t.device)) # to retain compatibility (t is nn.Parameter)
    
    def loss(self, **kwargs): 
        pass
    def fit(self, max_steps=1000):
        pass
    def score(self):
        raise NotImplementedError()
    
class CosineCoder(LinearCoder):
    def __init__(self, A, test_grad, device=None, metadata_only=False,
                  project="linear_coder", estimator_config=""):
        
        super().__init__(A, test_grad, device, metadata_only=metadata_only,
                         use_wandb=True, project=project, estimator_config= estimator_config)

    def loss(self, test_grad, combination, factors): 
        return -F.cosine_similarity(test_grad.unsqueeze(0), combination.unsqueeze(0)) 

    def score(self, reconstruction):
        with torch.no_grad():
            return F.cosine_similarity(self.test_grad.unsqueeze(0), reconstruction.unsqueeze(0))

    
class BaseMSECoder(LinearCoder):
    def __init__(self, A, test_grad, device=None, metadata_only=False,
                   use_wandb=False, project="linear_coder", estimator_config=""):
        super().__init__(A, test_grad, device, metadata_only=metadata_only,
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
            t_opt = torch.linalg.pinv(self.A.T) @ self.test_grad 
            self.t.data.copy_(t_opt.to(self.t.device)) # to retain compatibility (t is nn.Parameter)
    
    
class MSECoderL2(BaseMSECoder):
    def __init__(self, A, test_grad, device=None, reg_lambda=0.7, metadata_only=False,
                   use_wandb=False, project="linear_coder", estimator_config=""):
        self.reg_lambda = reg_lambda
    
        
        super().__init__(A, test_grad, device, metadata_only=metadata_only,
                         use_wandb=use_wandb, project=project, estimator_config= estimator_config)

    def loss(self, test_grad, combination, factors):
        return super().loss(test_grad, combination, factors) +  self.reg_lambda*torch.square(factors).sum()  
    def score(self, reconstruction):
        return super().score(reconstruction)
    def fit(self, **kwargs):
        with torch.no_grad():
            # (A^T \cdot A + λtI) t = A^T \cdot g
            t_opt = torch.linalg.solve(self.A @ self.A.T + self.reg_lambda * torch.eye(self.A.shape[0], device=self.A.device), self.A.T @ self.test_grad)
            self.t.data.copy_(t_opt.to(self.t.device)) # to retain compatibility (t is nn.Parameter)
    

from scipy.optimize import nnls
import torch
import torch.nn.functional as F

class MSECoderNNLSL2(BaseMSECoder):
    def __init__(self, A, test_grad, device=None, reg_lambda=0.7,
                 metadata_only=False, use_wandb=False, project="linear_coder", estimator_config=""):
        self.reg_lambda = reg_lambda
        super().__init__(A, test_grad, device,
                         metadata_only=metadata_only,
                         use_wandb=use_wandb,
                         project=project,
                         estimator_config=estimator_config)

    def loss(self, test_grad, combination, factors):
        return F.mse_loss(combination, test_grad) + self.reg_lambda * torch.sum(factors**2)



    def fit(self, **kwargs):
        with torch.no_grad():
    
            g = self.test_grad


            A = (self.A @ self.A.T + self.reg_lambda * torch.eye(self.A.shape[0], device=self.A.device)).cpu().numpy()
            b = (self.A @ g).cpu().numpy()

            t_opt, _ = nnls(A, b)
            self.t.data.copy_(torch.from_numpy(t_opt).to(self.t.device)) # to retain compatibility (t is nn.Parameter)
    
import torch
import torch.nn.functional as F

def projsplx(y):
    """
    this is a python port of
    projsplx @ https://www.mathworks.com/matlabcentral/fileexchange/30332-projection-onto-simplex
    in Chen and Ye 2011: "Projection Onto A Simplex" https://arxiv.org/abs/1101.6081 
    
    it projects y ∈ R^n onto the simplex D_n = {x : x >= 0, sum(x)=1}
    """
    m = y.numel()
    bget = False
    s, _ = torch.sort(y, descending=True) 
    tmpsum = 0.0

    for ii in range(m - 1): 
        tmpsum += s[ii]
        tmax = (tmpsum - 1) / (ii + 1)
        if tmax >= s[ii + 1]:
            bget = True
            break

    if not bget:
        tmax = (tmpsum + s[m - 1] - 1) / m

    x = torch.clamp(y - tmax, min=0)
    return x
class MSECoderProjUSimp(BaseMSECoder):
    def __init__(self, A, test_grad, device=None, 
                 metadata_only=False, use_wandb=False, project="linear_coder", estimator_config=""):
     
        super().__init__(A, test_grad, device,
                         metadata_only=metadata_only,
                         use_wandb=use_wandb,
                         project=project,
                         estimator_config=estimator_config)

    def loss(self, test_grad, combination, factors):
        return F.mse_loss(combination, test_grad) 




    def fit(self, **kwargs):
        with torch.no_grad():
            t_unconstrained = torch.linalg.pinv(self.A.T) @ self.test_grad

            # enforce sum(t)=1 and t_i >= 0
            t_opt = projsplx(t_unconstrained)
            self.t.data.copy_(t_opt.to(self.t.device))
class MSECoderProjUSimpSparseSoftThresh(BaseMSECoder):
    def __init__(self, A, test_grad, device=None, 
                 metadata_only=False, use_wandb=False, lambda_reg=0.2, project="linear_coder", estimator_config=""):
        self.lambda_reg = lambda_reg
        super().__init__(A, test_grad, device,
                         metadata_only=metadata_only,
                         use_wandb=use_wandb,
                         project=project,
                         estimator_config=estimator_config)

    def loss(self, test_grad, combination, factors):
        return F.mse_loss(combination, test_grad) 

    def fit(self, **kwargs):
        with torch.no_grad():
            t_unconstrained = torch.linalg.pinv(self.A.T) @ self.test_grad

            
            # apply soft thresholding
            t_ = torch.sign(t_unconstrained) * torch.clamp(t_unconstrained.abs() - self.lambda_reg*t_unconstrained.sum(), min=0)
            
            # enforce sum(t)=1 and t_i >= 0
            t_opt = projsplx(t_)
            self.t.data.copy_(t_opt.to(self.t.device))


import torch
import torch.nn.functional as F

def GSHP_tensor(w, y, lam, k):
    """
    python implementation of GSHP in Kyrillidis et al. 2013 "Sparse projections onto the simplex" https://proceedings.mlr.press/v28/kyrillidis13.html  
    
    """
    device = w.device
    dtype = w.dtype
    N = w.numel()

    # 1. {Initialize}
    j = torch.argmax(lam * w)
    S = [int(j)]
    ell = 1

    # 2. {Grow}
    while ell < k:
        ell += 1
        remaining = list(set(range(N)) - set(S))
        remaining_tensor = w[remaining]
        mean_adjusted = (w[S].sum() - lam) / (ell - 1)
        deviations = torch.abs(remaining_tensor - mean_adjusted)
        j_new = remaining[torch.argmax(deviations)]
        S.append(int(j_new))

    S_star = S

    # 4. {Final projection}
    subset_obs = y[S_star]
    tau = (subset_obs.sum() - lam) / subset_obs.numel()
    projected_subset = subset_obs - tau

    projection = torch.zeros_like(y, device=device, dtype=dtype)
    projection[S_star] = projected_subset
    return projection

class MSECoderProjUSimpSparse(BaseMSECoder):
    def __init__(self, A, test_grad, device=None, 
                 reg_lambda=0.1, 
                 metadata_only=False, use_wandb=False, project="linear_coder", estimator_config=""):
        self.reg_lambda = reg_lambda
   
        super().__init__(A, test_grad, device,
                         metadata_only=metadata_only,
                         use_wandb=use_wandb,
                         project=project,
                         estimator_config=estimator_config)

    def loss(self, test_grad, combination, factors):
        return F.mse_loss(combination, test_grad) 

    def fit(self, **kwargs):
        with torch.no_grad():
            t_unconstrained = torch.linalg.pinv(self.A.T) @ self.test_grad
            t_opt = GSHP_tensor(t_unconstrained, t_unconstrained, 1, max(1,int((1.0-self.reg_lambda)*self.t.shape[0])))
            self.t.data.copy_(t_opt)


class MSECoderElasticNet(BaseMSECoder):
    def __init__(self, A, test_grad, device=None, reg_lambda_1=0.3, reg_lambda_2=0.7, metadata_only=False,
                   use_wandb=True, project="linear_coder", estimator_config=""):
        self.reg_lambda_1 = reg_lambda_1
        self.reg_lambda_2 = reg_lambda_2
    
   
        
        super().__init__(A, test_grad, device, metadata_only=metadata_only,
                         use_wandb=use_wandb, project=project, estimator_config= estimator_config)

    def loss(self, test_grad, combination, factors):
        return super().loss(test_grad, combination, factors) + self.reg_lambda_1*torch.abs(factors).sum() + self.reg_lambda_2*torch.square(factors).sum()  
    def score(self, reconstruction):
        return super().score(reconstruction)

        
class MSECoderLemon(BaseMSECoder):
    def __init__(self, A, test_grad, device=None, 
                 reg_lambda_1=0.3, 
                 reg_lambda_2=0.7, 
                 reg_lambda_3_non_negative=0.5, 
                 metadata_only=False, use_wandb=True, project="linear_coder", estimator_config=""):
        self.reg_lambda_1 = reg_lambda_1
        self.reg_lambda_2 = reg_lambda_2
        self.reg_lambda_3_non_negative = reg_lambda_3_non_negative       
        super().__init__(A, test_grad, device, metadata_only=metadata_only,
                         use_wandb=use_wandb, project=project, estimator_config= estimator_config)

    def loss(self, test_grad, combination, factors):
        return super().loss(test_grad, combination, factors) \
            + self.reg_lambda_1*torch.abs(factors).sum() \
            + self.reg_lambda_2*torch.square(factors).sum() \
            + self.reg_lambda_3_non_negative*torch.sum(F.softplus(-factors)**2)        
    def score(self, reconstruction):
        return super().score(reconstruction)   
        
      
from scipy.optimize import nnls

# OptimizerCosineL1(A[2:].to("cuda"), A[0].to("cuda"), lr=0.1,reg_lambda=0, device="cuda")         




    
    
    
from collections import defaultdict
