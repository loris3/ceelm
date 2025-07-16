from influence_estimation.data_inf import DataInfEstimator
from influence_estimation.less_inf import LESSEstimator
from datasets import load_dataset


base_model_path = "distilbert/distilgpt2"
adapter_path = "/srv/home/users/loriss21cs/cfe/models/distilgpt2_tulu-v2-sft-mixture"


train_dataset = load_dataset("allenai/tulu-v2-sft-mixture", split="train[0:200]")





test_dataset = load_dataset("allenai/tulu-v2-sft-mixture", split="train[0:100]")


estimator = LESSEstimator(base_model_path, adapter_path, train_dataset, test_dataset)


print((estimator.influence_estimate))


# estimator = DataInfEstimator(base_model_path, adapter_path, train_dataset, test_dataset)


# print((estimator.influence_estimate))