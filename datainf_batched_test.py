import unittest
import torch
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from time import time

from influence_estimation.datainf.influence import IFEngineGeneration
class DummyModel:
    def __init__(self, val_grad_dict, tr_grad_dict):
        self.val_grad_dict = val_grad_dict
        self.tr_grad_dict = tr_grad_dict
        self.n_train = len(tr_grad_dict)
        self.hvp_dict = {}
        self.IF_dict = {}
        self.time_dict = {}
    def get_gradient(self, tokenized_train_dataset, train_dataset_name, train_dataset_split, batch_ids):
        return [self.tr_grad_dict[tr_id] for tr_id in batch_ids]

# https://github.com/ykwon0407/DataInf/blob/main/src/influence.py#L161
def compute_hvp_proposed(self, lambda_const_param=10):
    start_time = time()
    hvp_proposed_dict=defaultdict(dict)
    for val_id in tqdm(self.val_grad_dict.keys()):
        for weight_name in self.val_grad_dict[val_id]:
            # lambda_const computation
            S=torch.zeros(len(self.tr_grad_dict.keys()))
            for tr_id in self.tr_grad_dict:
                tmp_grad = self.tr_grad_dict[tr_id][weight_name]
                S[tr_id]=torch.mean(tmp_grad**2)
            lambda_const = torch.mean(S) / lambda_const_param # layer-wise lambda

            # hvp computation
            hvp=torch.zeros(self.val_grad_dict[val_id][weight_name].shape)
            for tr_id in self.tr_grad_dict:
                tmp_grad = self.tr_grad_dict[tr_id][weight_name]
                C_tmp = torch.sum(self.val_grad_dict[val_id][weight_name] * tmp_grad) / (lambda_const + torch.sum(tmp_grad**2))
                hvp += (self.val_grad_dict[val_id][weight_name] - C_tmp*tmp_grad) / (self.n_train*lambda_const)
            hvp_proposed_dict[val_id][weight_name] = hvp
    self.hvp_dict['proposed'] = hvp_proposed_dict
    self.time_dict['proposed'] = time()-start_time
# https://github.com/ykwon0407/DataInf/blob/main/src/influence.py#L183
def compute_IF(self):
    for method_name in self.hvp_dict:
        print("Computing IF for method: ", method_name)
        if_tmp_dict = defaultdict(dict)
        for tr_id in self.tr_grad_dict:
            for val_id in self.val_grad_dict:
                if_tmp_value = 0
                for weight_name in self.val_grad_dict[0]:
                    if_tmp_value += torch.sum(self.hvp_dict[method_name][val_id][weight_name]*self.tr_grad_dict[tr_id][weight_name])
                if_tmp_dict[tr_id][val_id]=-if_tmp_value

        self.IF_dict[method_name] = pd.DataFrame(if_tmp_dict, dtype=float)   
class TestInfluenceFunctions(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)


        weight_shapes = [
            (8,),           
            (116, 32),       
            (4, 41, 3),       
            (2, 3, 114, 5),    
        ]

        self.tr_grad_dict = {
            i: {
                j: torch.randn(*weight_shapes[j], dtype=torch.float32)
                for j in range(len(weight_shapes))
            }
            for i in range(100)   
        }

        self.val_grad_dict = {
            i: {
                j: torch.randn(*weight_shapes[j], dtype=torch.float32)
                for j in range(len(weight_shapes))
            }
            for i in range(11)
        }

        self.model = DummyModel(self.val_grad_dict, self.tr_grad_dict)

    def test_if_equivalence(self):
        compute_hvp_proposed(self.model)
        compute_IF(self.model)
        df_orig = self.model.IF_dict['proposed']

        engine = IFEngineGeneration()
        engine.preprocess_gradients(self.val_grad_dict, self.model.get_gradient)
        engine.hvp_dict = {'proposed': self.model.hvp_dict['proposed']}
        engine.model_n_train = self.model.n_train

        engine.compute_IF({"indices": list(self.tr_grad_dict.keys())}, None, None, batch_size=2)
        df_batched = engine.IF_dict['proposed']

        print(df_orig.head())
        print(df_batched.head())
        all_close = torch.allclose(torch.tensor(df_orig.values),
                                torch.tensor(df_batched.values),
                                atol=1e-3)
        self.assertTrue(all_close)

if __name__ == "__main__":
    unittest.main()