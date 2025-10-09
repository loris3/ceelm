# load_experiment_data.py


from influence_estimation.data_inf import DataInfEstimator
from influence_estimation.less_inf import LESSEstimator

from explanations import (
    KRandom,
    TopKMostInfluential,
    TopKLeastInfluential,
    TopKMostHelpful,
    TopKMostHarmful,
)
from linear_coders import MSECoderProjUSimp, KLTCoder, MSECoder, MSECoderNNLSL2, CosineCoder, MSECoderLemon, MSECoderElasticNet,MSECoderProjUSimpSparse,MSECoderProjUSimpSparseSoftThresh


from datasets import load_dataset

train_dataset_name = "loris3/tulu-3-sft-olmo-2-mixture-0225-sample"
test_dataset_name = "loris3/tulu-3-sft-olmo-2-mixture-0225-sample"

train_dataset_split = "train"
test_dataset_split = "test"


MODELS = [
    "./models/OLMo-2-0425-1B_tulu-3-sft-olmo-2-mixture-0225_lr0.0001_seed42",
    #  "./models/Llama-3.2-1B_tulu-3-sft-olmo-2-mixture-0225_lr0.0001_seed42",
    #   "./models/Qwen2.5-0.5B_tulu-3-sft-olmo-2-mixture-0225_lr0.0001_seed42",
]


explanation_types = [
    TopKMostInfluential,
    TopKLeastInfluential,
    TopKMostHelpful,
    TopKMostHarmful,
]

# explanation_k = [1, 2, 3, 4, 5, 10, 15, 20, 25]
explanation_k = [1, 5, 10, 25]
explanation_seed = [42, 10, 3, 9, 6]


linear_coders = [MSECoderProjUSimpSparse, MSECoderProjUSimp, KLTCoder, MSECoder, MSECoderNNLSL2, MSECoderProjUSimpSparseSoftThresh,
                #  MSECoderElasticNet, CosineCoder, MSECoderLemon
                 ]



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
        
    ]
    for model in MODELS:
        estimators.extend([
                        DataInfEstimator(model,
                            train_dataset, train_dataset_name, train_dataset_split,
                            test_dataset, test_dataset_name, test_dataset_split, eval_mode=True),
            LESSEstimator(model,
                        train_dataset, train_dataset_name, train_dataset_split,
                        test_dataset, test_dataset_name, test_dataset_split, eval_mode=True),

        ])
        
    indices = [ex["indices"] for ex in test_dataset]
    print(f"Total test examples: {len(indices)}")
    print(f"Unique indices: {len(set(indices))}")

    return train_dataset, test_dataset, estimators, 
