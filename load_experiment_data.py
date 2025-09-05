# load_experiment_data.py

from influence_estimation.data_inf import DataInfEstimator
from influence_estimation.less_inf import LESSEstimator

from explanations import KRandom, TopKMostInfluential, TopKLeastInfluential, TopKMostOrthogonal, TopKLeastOrthogonal
from linear_coders import KLTCoder, MSECoder, MSECoderNNLSL2, CosineCoder, MSECoderLemon, MSECoderElasticNet


from datasets import load_dataset

train_dataset_name = "loris3/tulu-v2-sft-mixture"
test_dataset_name = "loris3/tulu-v2-sft-mixture"

train_dataset_split = "test"
test_dataset_split = "test"


explanation_types = [KRandom, TopKMostInfluential, TopKLeastInfluential, TopKMostOrthogonal, TopKLeastOrthogonal]
linear_coders = [KLTCoder,MSECoder,MSECoderElasticNet, CosineCoder, MSECoderLemon,MSECoderNNLSL2]# OptimizerCosineL1, ]



def load_data_and_estimators():
    train_dataset = load_dataset(train_dataset_name, split=train_dataset_split)
    train_dataset = train_dataset.map(
        lambda example, idx: {"indices": idx},
        with_indices=True,
        num_proc=10
    )

    test_dataset = load_dataset(test_dataset_name, split=test_dataset_split)
    test_dataset = test_dataset.map(
        lambda example, idx: {"indices": idx},
        with_indices=True,
        num_proc=10
    )

    estimators = [
        LESSEstimator("./models/pythia-31m_tulu-v2-sft-mixture_train",
                      train_dataset, train_dataset_name, train_dataset_split,
                      test_dataset, test_dataset_name, test_dataset_split),
        # LESSEstimator("./models/pythia-31m_tulu-v2-sft-mixture_train",
        #               train_dataset, train_dataset_name, train_dataset_split,
        #               test_dataset, test_dataset_name, test_dataset_split,
        #               normalize=False),
        DataInfEstimator("./models/pythia-31m_tulu-v2-sft-mixture_train",
                         train_dataset, train_dataset_name, train_dataset_split,
                         test_dataset, test_dataset_name, test_dataset_split)
    ]

    return train_dataset, test_dataset, estimators
