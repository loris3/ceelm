from abc import ABC, abstractmethod
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(42)
cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

device = "cuda"




import torch
import torch.nn as nn
import torch.nn.functional as F



class Optimizer(ABC, nn.Module):
    def __init__(self, train_grads, test_grad, device=None, lr=1e-3):
        super().__init__()
        self.device = device
        self.lr = lr
        self.train_grads = train_grads.to(self.device)
        self.test_grad = test_grad.to(self.device)

        n_train = self.train_grads.shape[0]
        with torch.no_grad():
            # self.solution = torch.linalg.lstsq(self.train_grads.T.to(device), self.test_grad.view(-1).to(device)).solution
            # # print("self.solution",self.solution,flush=True)
            # mask = torch.empty(n_train,device=device).bernoulli_(0.5)
            # # print("sum", sum(mask),mask.shape)
            # random_values = torch.rand_like(self.solution)  

            # combined = mask * self.solution + (1 - mask) * random_values

            # self.factors = nn.Parameter(combined)#torch.clamp(combined, min=0))

            # self.factors = nn.Parameter(torch.randn(n_train, device=self.device))
            self.factors = nn.Parameter(torch.zeros(n_train, device=self.device))



        
        self.best_factors = None
        self._score = None
        self.steps_no_improve = 0
        self.fit()
    def forward(self):
        return self.train_grads.T @ self.factors #torch.einsum('ij,i->j', self.train_grads, self.factors)
    @abstractmethod
    def loss_fct(self, test_grad, combination, reg_lambda):
        pass

    @property
    @abstractmethod
    def score(self):
        pass
    def fit(self, max_steps=1000, patience=100, scheduler_step_freq=10):
        device = self.device if self.device else 'cpu'
        self.to(device)

        optimizer = torch.optim.Adam([self.factors], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        scaler = torch.amp.GradScaler(enabled=(device == 'cuda'))
        best_score = None
        best_factors = None
        no_improve_steps = 0

        for step in range(1, max_steps + 1):
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', enabled=(device == 'cuda')):
                loss = self.loss_fct(self.test_grad, self.forward(), self.factors)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                self.factors.clamp_(min=0)

            if step % scheduler_step_freq == 0:
                scheduler.step()

            current_score = self.score

            if best_score is None or current_score > best_score:
                best_score = current_score
                best_factors = self.factors.detach().clone()
                no_improve_steps = 0
            else:
                no_improve_steps += 1

            if no_improve_steps >= patience:
             #   print(f"Early stopping at step {step} with best score {best_score.item():.6f}")
                break

            # if step % 100 == 0:
            #     print(f"Step {step}, Loss: {loss.item():.6f}, Score: {current_score.item():.6f}")

        if best_factors is not None:
            self.factors.data.copy_(best_factors)


class OptimizerKLT(Optimizer):
    def __init__(self, train_grads, test_grad, device=None,reg_lambda=0.05 ):
        self.reg_lambda = reg_lambda
        self.best_factors = None
        self.steps_no_improve = 0
        super().__init__(train_grads, test_grad, device)
        with torch.no_grad():
            num_samples, dim = train_grads.shape
            mean = train_grads.mean(dim=0, keepdim=True)
            centered = train_grads - mean
            cov = centered.T @ centered / (num_samples - 1)
            eigvals, eigvecs = torch.linalg.eigh(cov)
            eigvals = eigvals.flip(dims=[0])
            eigvecs = eigvecs.flip(dims=[1])
            self.klt_basis = eigvecs 
            coeffs = self.klt_basis.T @ test_grad.view(-1)
            self.factors = nn.Parameter(coeffs)
    def loss_fct(self, test_grad, combination, factors, alpha=0.5): 
        pass
    def fit(self, max_steps=1000):
        pass
    @property
    def score(self):
        raise NotImplementedError()
    
class OptimizerCosineL1(Optimizer):
    def __init__(self, train_grads, test_grad, device=None, reg_lambda=0.05 ):
        self.reg_lambda = reg_lambda
        self.best_factors = None
        self.steps_no_improve = 0
        super().__init__(train_grads, test_grad, device)

    def loss_fct(self, test_grad, combination, factors, alpha=0.5): 
        return -F.cosine_similarity(test_grad.unsqueeze(0), combination.unsqueeze(0)) + self.reg_lambda * factors.sum()
    @property
    def score(self):
        with torch.no_grad():
            return F.cosine_similarity(self.test_grad.unsqueeze(0), self.forward().unsqueeze(0))


class OptimizerCrossEntropyL1(Optimizer):
    def __init__(self, train_grads, test_grad, device=None, reg_lambda=0.0 ):
        self.reg_lambda = reg_lambda
        self.best_factors = None
        self.steps_no_improve = 0
        super().__init__(train_grads, test_grad, device)

    def loss_fct(self, test_grad, combination, factors): 
        l = -F.cross_entropy(test_grad.unsqueeze(0), combination.unsqueeze(0)) + self.reg_lambda * factors.sum()
        return l
    @property
    def score(self):
        with torch.no_grad():
            return F.cross_entropy(self.test_grad.unsqueeze(0), self.forward().unsqueeze(0))


    
class OptimizerMSEL1(Optimizer):
    def __init__(self, train_grads, test_grad, device=None, reg_lambda=0.00):
        self.reg_lambda = reg_lambda
        self.best_factors = None
        self.steps_no_improve = 0
        super().__init__(train_grads, test_grad, device)

    def loss_fct(self, test_grad, combination, factors):
        mse_loss = F.mse_loss(combination, test_grad)
        return mse_loss + self.reg_lambda * factors.sum()
    @property
    def score(self):
        with torch.no_grad():
            return F.mse_loss(self.test_grad.unsqueeze(0), self.forward().unsqueeze(0))


class OptimizerMSEL0(Optimizer):
    def __init__(self, train_grads, test_grad, device=None, reg_lambda=0.1):
        self.reg_lambda = reg_lambda
        self.best_factors = None
        self.steps_no_improve = 0
        super().__init__(train_grads, test_grad, device)

    def loss_fct(self, test_grad, combination, factors):
        mse_loss = F.mse_loss(combination, test_grad)
        return mse_loss + self.reg_lambda * torch.count_nonzero(factors)
    @property
    def score(self):
        with torch.no_grad():
            return F.mse_loss(self.test_grad.unsqueeze(0), self.forward().unsqueeze(0))

class OptimizerLemon(Optimizer):
    def __init__(self, train_grads, test_grad, device=None,  reg_lambda=1):
        self.reg_lambda = reg_lambda
        self.best_factors = None
        self.steps_no_improve = 0
        super().__init__(train_grads, test_grad, device)

    def loss_fct(self, test_grad, combination, factors):
        mse_loss = F.mse_loss(combination, test_grad)
        return mse_loss + self.reg_lambda * torch.count_nonzero(factors > 0.01)
    @property
    def score(self):
        with torch.no_grad():
            return F.mse_loss(self.test_grad.unsqueeze(0), self.forward().unsqueeze(0))

# OptimizerCosineL1(train_grads[2:].to("cuda"), train_grads[0].to("cuda"), lr=0.1,reg_lambda=0, device="cuda")         




from collections import defaultdict
