1. Tokenize datasets for finetuning: `tokenize_ft.sbatch`
1. Finetune all models: `finetune.sbatch`
1. Evaluate performance of models `./finetuning_eval.sh`
2. Take a sample of the dataset for influence estimation and evaluation: `python3 take_split.py`
2. Obtain influence estimates: `estimate_influence.sbatch`
3. Obtain quality scores: `score.sbatch`
4. Run validation experiment: `git clone https://github.com/allenai/olmes.git && validation.sbatch`

finetuning_eval.sbatch