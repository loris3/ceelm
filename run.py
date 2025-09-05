
# import torch
# from scipy import stats
# import numpy as np

# import numpy as np
# from scipy import stats
# import itertools
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from influence_estimation.data_inf import DataInfEstimator
# from influence_estimation.less_inf import LESSEstimator
# from datasets import load_dataset



# if __name__ == "__main__":
#     from multiprocess import set_start_method
#     set_start_method("spawn")

#     # base_model_path = "distilbert/distilgpt2"
#     base_model_path = "allenai/OLMo-2-0425-1B"
#     # adapter_path = "/srv/home/users/loriss21cs/cfe/models/distilgpt2_tulu-v2-sft-mixture"
#     adapter_path = "/srv/home/users/loriss21cs/cfe/models/OLMo-2-0425-1B_tulu-v2-sft-mixture"

#     train_dataset = load_dataset("allenai/tulu-v2-sft-mixture", split="train").shuffle(seed=0).select(range(200))





#     test_dataset = load_dataset("allenai/tulu-v2-sft-mixture", split="train").shuffle(seed=0).select(range(200))

#     estimators = [
#         DataInfEstimator(base_model_path, adapter_path, train_dataset, test_dataset,fast_implementation=True),
#         # DataInfEstimator(base_model_path, adapter_path, train_dataset, test_dataset,fast_implementation=False),
#         LESSEstimator(base_model_path, adapter_path, train_dataset, test_dataset,normalize=True),
#         # LESSEstimator(base_model_path, adapter_path, train_dataset, test_dataset,normalize=False)
#     ]
#     import itertools
#     import numpy as np
#     from scipy import stats
#     import itertools
#     import pandas as pd




#     n_test = estimators[0].influence_estimate.shape[0]
#     pairwise_results = []
#     estimator_names = [est.get_config_string() for est in estimators]
#     for (i, est1), (j, est2) in itertools.combinations(enumerate(estimators), 2):
#         corrs = []
#         for t in range(n_test):
#             corr = stats.spearmanr(est1.influence_estimate[t], est2.influence_estimate[t]).correlation
#             corrs.append(corr)
#         mean_corr = np.mean(corrs)
#         pairwise_results.append({
#             "Estimator a": est1.get_config_string(),
#             "Estimator b": est2.get_config_string(),
#             "Mean Spearman Corr": mean_corr
#         })


#     results_df = pd.DataFrame(pairwise_results)
#     print(results_df)
#     matrix_df = pd.DataFrame(
#         np.eye(len(estimators)),  # start with identity for diagonal = 1.0
#         index=estimator_names,
#         columns=estimator_names,
#         dtype=float
#     )

#     for _, row in results_df.iterrows():
#         a, b, corr = row["Estimator a"], row["Estimator b"], row["Mean Spearman Corr"]
#         matrix_df.loc[a, b] = corr
#         matrix_df.loc[b, a] = corr


#     plt.figure(figsize=(8, 6))
#     sns.heatmap(matrix_df, annot=True,  vmin=0, vmax=1, square=True)
#     plt.title("Mean Spearman Correlation Between Influence Estimators")
#     plt.tight_layout()
#     plt.show()

#     matrix_df
#     estimator_less = LESSEstimator(base_model_path, adapter_path, train_dataset, test_dataset,normalize=True)
#     estimator_datainf = DataInfEstimator(base_model_path, adapter_path, train_dataset, test_dataset,fast_implementation=True)
#     correlations = np.array([
#         stats.spearmanr(estimator_datainf.influence_estimate[i], estimator_less.influence_estimate[i]).correlation for i in range(estimator_less.influence_estimate.shape[0])
#     ])
#     df = pd.DataFrame({
#         "Spearman Correlation": correlations,
#         "Comparison": ["LESS vs DataInf"] * len(correlations)
#     })

#     # Plot using seaborn violinplot
#     plt.figure(figsize=(6, 4))
#     sns.violinplot(data=df, x="Comparison", y="Spearman Correlation", inner="point")
#     plt.title("Spearman Correlation of Influence Estimates")
#     plt.ylim(-1.05, 1.05)
#     plt.grid(True, axis='y', linestyle='--', alpha=0.5)
#     plt.show()