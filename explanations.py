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
class TopKMostInfluential(Explanation):
    def __init__(self, dataset_idx, estimator, k=10):
        super().__init__(dataset_idx,estimator)
        self.k = k
    @property
    def documents(self):
        return self.influence_estimate.nlargest(self.k).index.tolist()
    
class TopKLeastInfluential(Explanation):
    def __init__(self,  dataset_idx, estimator, k=10):
        super().__init__( dataset_idx, estimator)
        self.k = k
    @property
    def documents(self):
        return self.influence_estimate.nsmallest(self.k).index.tolist()
    
class TopKMostOrthogonal(Explanation):
    def __init__(self,  dataset_idx, estimator, k=10):
        super().__init__( dataset_idx, estimator)
        self.k = k
    @property
    def documents(self):
        return self.influence_estimate.abs().nsmallest(n=self.k).index.tolist()
class TopKLeastOrthogonal(Explanation):
    def __init__(self,  dataset_idx, estimator, k=10):
        super().__init__( dataset_idx, estimator)
        self.k = k
    @property
    def documents(self):
        return self.influence_estimate.abs().nlargest(n=self.k).index.tolist()