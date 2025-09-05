from abc import ABC, abstractmethod
class Explanation(ABC):
    def __init__(self, dataset_idx, estimator):
        self.influence_estimate = estimator.influence_estimate.iloc[dataset_idx]
        self.dataset_idx = dataset_idx
        self.estimator = estimator
        
    @property
    @abstractmethod
    def documents(self):
        pass
    

    @property
    @abstractmethod
    def description(self):
        pass
class TopKMostInfluential(Explanation):
    def __init__(self, dataset_idx, estimator, k=10):
        super().__init__(dataset_idx,estimator)
        self.k = k
    @property
    def documents(self):
        return self.influence_estimate.nlargest(self.k).index.tolist()
    @property
    def description(self):
        return f"Top-{self.k} most influential"
    
class KRandom(Explanation):
    def __init__(self, dataset_idx, estimator, k=10, seed=42):
        super().__init__(dataset_idx,estimator)
        self.k = k
        self.seed = seed
    @property
    def documents(self):
        return self.influence_estimate.sample(n=self.k, random_state=self.seed).index.tolist()
    @property
    def description(self):
        return f"{self.k} random examples with seed {self.seed}"
class TopKLeastInfluential(Explanation):
    def __init__(self,  dataset_idx, estimator, k=10):
        super().__init__( dataset_idx, estimator)
        self.k = k
    @property
    def documents(self):
        return self.influence_estimate.nsmallest(self.k).index.tolist()
    @property
    def description(self):
        return f"Bottom-{self.k} least influential"    
class TopKMostOrthogonal(Explanation):
    def __init__(self,  dataset_idx, estimator, k=10):
        super().__init__( dataset_idx, estimator)
        self.k = k
    @property
    def documents(self):
        return self.influence_estimate.abs().nsmallest(n=self.k).index.tolist()
    @property
    def description(self):
        return f"Mean-{self.k} most average scores"
class TopKLeastOrthogonal(Explanation):
    def __init__(self,  dataset_idx, estimator, k=10):
        super().__init__( dataset_idx, estimator)
        self.k = k
    @property
    def documents(self):
        return self.influence_estimate.abs().nlargest(n=self.k).index.tolist()
    @property
    def description(self):
        return f"Tail-{self.k} most extreme scores"