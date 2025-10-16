from abc import ABC, abstractmethod
class Explanation(ABC):
    def __init__(self, document_idx, estimator):
        # print(len(estimator.influence_estimate), estimator.influence_estimate.index, flush=True)
        self.influence_estimate = estimator.influence_estimate.iloc[document_idx]
        self.document_idx = document_idx
        self.estimator = estimator
        
    @property
    @abstractmethod
    def documents(self):
        pass

    @property
    @abstractmethod
    def description(self):
        pass
    @property 
    @abstractmethod
    def costs(self):
        pass
class TopKMostHelpful(Explanation):
    def __init__(self, document_idx, estimator, k=10):
        super().__init__(document_idx,estimator)
        self.k = k
    @property
    def documents(self):
        return self.influence_estimate.nsmallest(self.k).index.tolist()
    @property
    def description(self):
        return f"Top-{self.k} most helpful (most negative scores)"
    @property 
    def costs(self):
        return self.influence_estimate


class Self(Explanation):
    def __init__(self, document_idx):
        self.document_idx = document_idx
        self.estimator = None
        self.k = 1
    @property
    def documents(self):
        raise PermissionError
    @property
    def description(self):
        return f"The test instance (as a sanity check)"
    @property
    def costs(self):
        return [0]
    
class KRandom(Explanation):
    def __init__(self, document_idx, estimator, k=10, seed=42):
        super().__init__(document_idx,estimator)
        self.k = k
        self.seed = seed
    @property
    def documents(self):
        return self.influence_estimate.sample(n=self.k, random_state=self.seed).index.tolist()
    @property
    def description(self):
        return f"{self.k} random examples with seed {self.seed}"
    @property
    def costs(self):
        return self.influence_estimate.sample(n=self.k, random_state=self.seed).values.tolist()
class TopKMostHarmful(Explanation):
    def __init__(self,  document_idx, estimator, k=10):
        super().__init__( document_idx, estimator)
        self.k = k
    @property
    def documents(self):
        return self.influence_estimate.nlargest(self.k).index.tolist()
    @property
    def description(self):
        return f"Top-{self.k} most harmful (most positive scores)"   
    @property 
    def costs(self):
        return -self.influence_estimate

class TopKLeastInfluential(Explanation):
    def __init__(self,  document_idx, estimator, k=10):
        super().__init__( document_idx, estimator)
        self.k = k
    @property
    def documents(self):
        return self.influence_estimate.abs().nsmallest(n=self.k).index.tolist()
    @property
    def description(self):
        return f"Top-{self.k} least influential (scores closest to zero)"
    @property 
    def costs(self):
        return -self.influence_estimate
class TopKMostInfluential(Explanation):
    def __init__(self,  document_idx, estimator, k=10):
        super().__init__( document_idx, estimator)
        self.k = k
    @property
    def documents(self):
        return self.influence_estimate.abs().nlargest(n=self.k).index.tolist()
    @property
    def description(self):
        return f"Top-{self.k} most influential (scores with largest absolute value)"
    @property 
    def costs(self):
        return self.influence_estimate.abs()