from abc import ABC, abstractmethod
import os
import pandas as pd

class Explanation(ABC):
    def __init__(self, document_idx, estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split, **kwargs):
        self.influence_estimate = estimator.influence_estimate.iloc[document_idx]
        self.document_idx = document_idx
        self.estimator = estimator
        self._cached_documents = None
        
        self.train_dataset_name = train_dataset_name
        self.train_dataset_split = train_dataset_split
        self.test_dataset_name = test_dataset_name
        self.test_dataset_split = test_dataset_split
     

     
        self.cache_path = os.path.join(
            "./cache/selection/",
            estimator.model_path,
            estimator.get_config_string(),
            self.train_dataset_name,
            self.train_dataset_split,
            self.test_dataset_name,
            self.test_dataset_split,
            self.__class__.__name__,
            *[f"{k}={v}" for k,v in kwargs.items()],
            str(document_idx) + ".parquet"
        )
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        
    def _load_cache(self):
            if os.path.exists(self.cache_path):
                try:
                    df = pd.read_parquet(self.cache_path)
                    return df["documents"].tolist()
                except Exception as e:
                    print(f"[Cache Warning] Failed to load {self.cache_path}: {e}")
            return None

    def _save_cache(self, documents):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        df = pd.DataFrame({"documents": documents})
        df.to_parquet(self.cache_path, index=False)


    @property
    def documents(self):
        if self._cached_documents is not None:
            return self._cached_documents
        cached = self._load_cache()
        if cached is not None:
            self._cached_documents = cached
            return cached
        docs = self._compute_documents()
        self._cached_documents = docs
        self._save_cache(docs)
        return docs

    @abstractmethod
    def _compute_documents(self):
        pass

    @property
    @abstractmethod
    def description(self):
        pass

class TopKMostHelpful(Explanation):
    def __init__(self, document_idx, estimator,train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split, k=10):
        super().__init__(document_idx,estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split, k=k)
        self.k = k

    def _compute_documents(self):
        return self.influence_estimate.nsmallest(self.k).index.tolist()
    @property
    def description(self):
        return f"Top-{self.k} most helpful (most negative scores)"


class Self(Explanation):
    def __init__(self, document_idx):
        self.document_idx = document_idx
        self.estimator = None
        self.k = 1

    def _compute_documents(self):
        raise PermissionError
    @property
    def description(self):
        return f"The test instance (as a sanity check)"
    @property
    def costs(self):
        return [0]
    
class KRandom(Explanation):
    def __init__(self, document_idx, estimator,train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split, k=10, seed=42):
        super().__init__(document_idx,estimator,train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split, k=k, seed=seed)
        self.k = k
        self.seed = seed

    def _compute_documents(self):
        return self.influence_estimate.sample(n=self.k, random_state=self.seed).index.tolist()
    @property
    def description(self):
        return f"{self.k} random examples with seed {self.seed}"
    @property
    def costs(self):
        return self.influence_estimate.sample(n=self.k, random_state=self.seed).values.tolist()
class TopKMostHarmful(Explanation):
    def __init__(self,  document_idx, estimator,train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split, k=10):
        super().__init__( document_idx, estimator,train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split, k=k)
        self.k = k

    def _compute_documents(self):
        return self.influence_estimate.nlargest(self.k).index.tolist()
    @property
    def description(self):
        return f"Top-{self.k} most harmful (most positive scores)"   


class TopKLeastInfluential(Explanation):
    def __init__(self,  document_idx, estimator,train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split, k=10):
        super().__init__( document_idx, estimator,train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split, k=k)
        self.k = k

    def _compute_documents(self):
        return self.influence_estimate.abs().nsmallest(n=self.k).index.tolist()
    @property
    def description(self):
        return f"Top-{self.k} least influential (scores closest to zero)"

class TopKMostInfluential(Explanation):
    def __init__(self,  document_idx, estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split,k=10):
        super().__init__( document_idx, estimator,train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split, k=k)
        self.k = k

    def _compute_documents(self):
        return self.influence_estimate.abs().nlargest(n=self.k).index.tolist()
    @property
    def description(self):
        return f"Top-{self.k} most influential (scores with largest absolute value)"
    @property 
    def costs(self):
        return self.influence_estimate.abs()



import torch
import os
from apricot import FacilityLocationSelection, MixtureSelection, SumRedundancySelection, FeatureBasedSelection
import numpy as np
from fl_optimizers import LazyWeightedGreedy
from sklearn.preprocessing import RobustScaler, MinMaxScaler

class FacilityLocation(Explanation):
    def __init__(self, document_idx, estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split,k=10, m=100, keep_gradients=False, lambda_=0.9):
        super().__init__(document_idx,estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split, k=k, m=m, lambda_=lambda_)
        self.k = k
        self.lambda_ = lambda_
        self.m = m
        self.keep_gradients = keep_gradients

                

    

    def _compute_documents(self):
        self.groundset = torch.stack(
            self.estimator.get_gradient(
            os.path.basename(self.train_dataset_name),
            self.train_dataset_split,
            self.groundset_explanation.documents)).to(torch.float32).cpu().numpy()
    
        selector = FacilityLocationSelection(n_samples=self.k, metric="cosine", optimizer=LazyWeightedGreedy(lambda_=self.lambda_), verbose=False)
        
        costs_normalized = MinMaxScaler((1, self.m)).fit_transform(((self.costs.values.reshape(-1, 1)))).flatten()

        selector.fit(self.groundset, sample_cost=costs_normalized)

        selection = np.array(self.groundset_explanation.documents)[selector.ranking]
        if not self.keep_gradients:
            del self.groundset, selector
        return selection
    @property
    @abstractmethod
    def description(self):
        pass
    @property 
    @abstractmethod
    def costs(self):
        pass




class FacilityLocationMostHelpful(FacilityLocation):
    def __init__(self, document_idx, estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split,k=10, m=100, keep_gradients=False, lambda_=0.9):
        self.groundset_explanation = TopKMostHelpful(document_idx, estimator,train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split, k=m)
        super().__init__(document_idx, estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split,k=k, m=m, keep_gradients=keep_gradients,lambda_=lambda_)
    @property
    def description(self):
        return f"{self.k} by facility location from Top-{self.groundset_explanation.k} most helpful (most negative scores). lambda={self.lambda_}"
    @property 
    def costs(self):
        return self.groundset_explanation.influence_estimate[self.groundset_explanation.documents]








class FacilityLocationMostHarmful(FacilityLocation):
    def __init__(self, document_idx, estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split,k=10, m=100, lambda_=0.9, keep_gradients=False):
        self.groundset_explanation = TopKMostHarmful(document_idx, estimator,train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split, k=m)
        super().__init__(document_idx, estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split,k=k, m=m, keep_gradients=keep_gradients, lambda_=lambda_)
    @property
    def description(self):
        return f"{self.k} by facility location from Top-{self.groundset_explanation.k} most harmful (most positive scores). lambda={self.lambda_}"
    @property 
    def costs(self):
        return -self.groundset_explanation.influence_estimate[self.groundset_explanation.documents]




class FacilityLocationMostInfluential(FacilityLocation):
    def __init__(self, document_idx, estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split,k=10, m=100, lambda_=0.9, keep_gradients=False):
        self.groundset_explanation = TopKMostInfluential(document_idx, estimator,train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split, k=m)
        super().__init__(document_idx, estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split,k=k, m=m, keep_gradients=keep_gradients, lambda_=lambda_)
    @property
    def description(self):
        return f"{self.k} by facility location from Top-{self.groundset_explanation.k} most influential (scores with largest absolute value). lambda={self.lambda_}"
    @property 
    def costs(self):
        return -self.groundset_explanation.influence_estimate[self.groundset_explanation.documents].abs()






class FacilityLocationLeastInfluential(FacilityLocation):
    def __init__(self, document_idx, estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split,k=10, m=100, lambda_=0.9,  keep_gradients=False):
        self.groundset_explanation = TopKLeastInfluential(document_idx, estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split,k=m)
        super().__init__(document_idx, estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split,k=k, m=m, keep_gradients=keep_gradients, lambda_=lambda_)
    @property
    def description(self):
        return f"{self.k} by facility location from Top-{self.groundset_explanation.k} least influential (scores closest to zero). lambda={self.lambda_}"
    @property 
    def costs(self):
        return self.groundset_explanation.influence_estimate[self.groundset_explanation.documents].abs()
    
    
    
####

from concurrent.futures import ThreadPoolExecutor

import torch
from sklearn.metrics import pairwise_distances

import os
from apricot import FacilityLocationSelection, MixtureSelection, SumRedundancySelection, FeatureBasedSelection
import numpy as np
from fl_optimizers import LazyWeightedGreedy
from sklearn.preprocessing import RobustScaler, MinMaxScaler


from apricot import BaseSelection
import numpy as np

class ImportanceLookupSelector(BaseSelection):
    def __init__(self, n_samples, importance_scores,
                 initial_subset=None, optimizer='two-stage',
                 optimizer_kwds=None, n_jobs=1,
                 random_state=None, verbose=False):

        if optimizer_kwds is None:
            optimizer_kwds = {}
        self.importance_scores = np.asarray(importance_scores)

        super().__init__(
            n_samples=n_samples,
            initial_subset=initial_subset,
            optimizer=optimizer,
            optimizer_kwds=optimizer_kwds,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose
        )

    def _initialize(self, X):
        super()._initialize(X)

    def _calculate_gains(self, X, idxs=None):
        idxs = idxs if idxs is not None else self.idxs
        return self.importance_scores[idxs] # marginal gain of selecting i is I_i

    def _select_next(self, X, gain, idx):
        super()._select_next(X, gain, idx)


class DIVINE(Explanation):
    """
    This is our own implementation of "DIVINE: Diverse Influential Training Points for Data Visualization and Model Refinement" Bhatt et al 2021, https://arxiv.org/abs/2107.05978
    as the authors have not released theirs to this date (https://github.com/umangsbhatt/divine-release).
    
    We choose the gamma that maximizes average pariwise distances between points emperically, as suggested by the authors.
    We choose this specific variant as it strongly prioritizes example diversity, making it a good baseline for our method that is prioritizing coverage instead.
    """
    def __init__(self, document_idx, estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split,k=10, m=100, keep_gradients=False):
        super().__init__(document_idx,estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split, k=k, m=m)
        self.k = k
        self.m = m
        self.keep_gradients = keep_gradients

        self.gamma_min = -4 # as in Bhatt et al 2021
        self.gamma_max = 5 # as in Bhatt et al 2021
        self.num_points = 20 # not specified in Bhatt et al 2021; 
                             # chosen to limit computation time, we expect a saturation effect as influence score normalized to [0,1] and k small < 100  
                             # note that selection strategy favours high gamma
        self.gammas = np.logspace(self.gamma_min, self.gamma_max, self.num_points)
        self.gamma = None

    def _compute_documents(self):
        def compute_for_gamma(args):
            gamma, groundset, k, importance_scores_normalized = args
            D = pairwise_distances(groundset, metric='cosine')
            S = D.max() - D

            sum_of_influence_scores = ImportanceLookupSelector(
                n_samples=k,
                importance_scores=importance_scores_normalized
            )

            sr = SumRedundancySelection(n_samples=k, metric="precomputed")
            selector = MixtureSelection(
                n_samples=k,
                functions=[sum_of_influence_scores, sr],
                weights=[1, gamma],
                verbose=False
            )
            selector.fit(S)
            return gamma, selector.ranking, D.mean()

        self.groundset = torch.stack(
            self.estimator.get_gradient(
                os.path.basename(self.train_dataset_name),
                self.train_dataset_split,
                self.groundset_explanation.documents
            )
        ).to(torch.float32).cpu().numpy()

        importance_scores_normalized = MinMaxScaler((0, 1)).fit_transform(
            self.importance_scores.values.reshape(-1, 1)
        ).flatten()


        tasks = [
            (gamma, self.groundset, self.k, importance_scores_normalized)
            for gamma in self.gammas
        ]

        results = []
        with ThreadPoolExecutor(max_workers=32) as executor:
            for result in executor.map(compute_for_gamma, tasks):
                results.append(result)
      


        best_gamma, best_ranking, best_score = max(
            results,
            key=lambda x: x[2]
        )

        self.gamma = best_gamma
        best_selection = np.array(self.groundset_explanation.documents)[best_ranking]

        if not self.keep_gradients:
            del self.groundset

        return best_selection
    @property
    @abstractmethod
    def description(self):
        pass
    @property 
    @abstractmethod
    def importance_scores(self):
        pass








class DIVINEMostHelpful(DIVINE):
    def __init__(self, document_idx, estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split,k=10, m=100, keep_gradients=False):
        self.groundset_explanation = TopKMostHelpful(document_idx, estimator,train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split, k=m)
        super().__init__(document_idx, estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split,k=k, m=m, keep_gradients=keep_gradients)
    @property
    def description(self):
        return f"{self.k} by DIVINE from Top-{self.groundset_explanation.k} most helpful (most negative scores)."
    @property 
    def importance_scores(self):
        return -self.groundset_explanation.influence_estimate[self.groundset_explanation.documents]



class DIVINEMostHarmful(DIVINE):
    def __init__(self, document_idx, estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split,k=10, m=100,  keep_gradients=False):
        self.groundset_explanation = TopKMostHarmful(document_idx, estimator,train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split, k=m)
        super().__init__(document_idx, estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split,k=k, m=m, keep_gradients=keep_gradients)
    @property
    def description(self):
        return f"{self.k} by facility location from Top-{self.groundset_explanation.k} most harmful (most positive scores)."
    @property 
    def importance_scores(self):
        return self.groundset_explanation.influence_estimate[self.groundset_explanation.documents]




class DIVINEMostInfluential(DIVINE):
    def __init__(self, document_idx, estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split,k=10, m=100, keep_gradients=False):
        self.groundset_explanation = TopKMostInfluential(document_idx, estimator,train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split, k=m)
        super().__init__(document_idx, estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split,k=k, m=m, keep_gradients=keep_gradients)
    @property
    def description(self):
        return f"{self.k} by DIVINE from Top-{self.groundset_explanation.k} most influential (scores with largest absolute value)."
    @property 
    def importance_scores(self):
        return self.groundset_explanation.influence_estimate[self.groundset_explanation.documents].abs()






class DIVINELeastInfluential(DIVINE):
    def __init__(self, document_idx, estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split,k=10, m=100, keep_gradients=False):
        self.groundset_explanation = TopKLeastInfluential(document_idx, estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split,k=m)
        super().__init__(document_idx, estimator, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split,k=k, m=m, keep_gradients=keep_gradients)
    @property
    def description(self):
        return f"{self.k} by DIVINE from Top-{self.groundset_explanation.k} least influential (scores closest to zero)."
    @property 
    def importance_scores(self):
        return -self.groundset_explanation.influence_estimate[self.groundset_explanation.documents].abs()