import os
from dotenv import load_dotenv
load_dotenv()

import wandb
import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO
)

from pathlib import Path
import torch

from training.custom_olmo import DecomposedOlmo2
from transformers import (
    set_seed,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    AutoConfig,
    PhiConfig,
    DataCollatorForSeq2Seq,
    DataCollatorForTokenClassification)


from arg_parsers.coreset_training_arguments import CoresetTrainingArguments
from arg_parsers.model_arguments import ModelArguments
from arg_parsers.data_arguments import DataArguments


from colm.train.huggingface_trainer import CustomTrainer as Trainer

from training.subset_trainer_distributed  import SubsetTrainerEfficient

from transformers import Trainer

from peft import LoraConfig, PeftModel, TaskType, get_peft_model


from colm.data.get_training_dataset import (
    convert_superglue_to_hf,
    convert_superglue_to_hf_source,
    get_training_dataset,
    SupervisedDataset,
    HFDataset,
    DataCollatorForSupervisedDataset,
    DataCollatorForSupervisedDatasetWithSource)


logger = logging.getLogger(__name__)


def main():
    """Trains a model on coreset-batches, logs coreset- and full batches.
    """
    parser = HfArgumentParser((ModelArguments, DataArguments, CoresetTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    logger.info(f"Training parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")
    
    
    print(training_args.run_name)
    set_seed(training_args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(
        training_args.model_name_or_path,
        model_max_length=model_args.model_max_length)
    
    if not model_args.enable_dropout:
        # Set dropout to 0
        logger.info("Set dropout to 0")
        model_config = AutoConfig.from_pretrained(
            training_args.model_name_or_path, cache_dir=model_args.cache_dir)
        # assert isinstance(
        #     model_config, PhiConfig), "Only support no dropout for Phi-2!"
        model_config.resid_pdrop = 0
        model_args.lora_dropout = 0
        model = AutoModelForCausalLM.from_pretrained(
            training_args.model_name_or_path,
            config=model_config,
            torch_dtype=None if model_args.torch_dtype == "none" else getattr(torch, model_args.torch_dtype),
            trust_remote_code=True,
            cache_dir=model_args.cache_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            training_args.model_name_or_path,
            torch_dtype=None if model_args.torch_dtype == "none" else getattr(torch, model_args.torch_dtype),
            trust_remote_code=True,
            cache_dir=model_args.cache_dir)
    if len(training_args.fsdp) > 0 and training_args.fsdp_config.get('activation_checkpointing', False):
        # Enable gradient checkpointing for reducing memory footprint
        # Bug in future torch version
        # https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/discussions/12
        logger.info("Enable gradient checkpointing")
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': True})
        
        
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    
    # Resize embeddings if needed (e.g. for LlamaTokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    modules_to_save = []
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        # if you load lora model and resize the token embeddings, the requires_grad flag is set to True for embeddings
        if isinstance(model, PeftModel):
            model.get_input_embeddings().weight.requires_grad = False
            model.get_output_embeddings().weight.requires_grad = False
        # Adding additional tokens to vocabulary
        # https://github.com/huggingface/peft/issues/334
        modules_to_save = ["lm_head", "embed_tokens"]
    # Set up LoRA
    if not isinstance(model, PeftModel) and model_args.lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
            modules_to_save=modules_to_save
        )
        model = get_peft_model(model, lora_config)
        # ValueError: Attempting to unscale FP16 gradients
        # https://github.com/huggingface/peft/issues/341
        model.base_model.model.model.embed_tokens.weight.data = model.base_model.model.model.embed_tokens.weight.data.float()
        model.base_model.model.lm_head.weight.data = model.base_model.model.lm_head.weight.data.float()
        logger.info(
            f"Applied LoRA to model."
        )
        model.print_trainable_parameters()

        # for checkpointing
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # Change last layer to LoRA
        training_args.last_layers = [
            name + '.lora_B' for name in training_args.last_layers]
        
    model_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)

    print("model",model)
    # Change forward pass of model for efficient zeroth-order gradient
    if training_args.data_selection_unit == "mezo" and training_args.efficient_mezo:
        model.decomposer = DecomposedOlmo2(model)
        
    train_dataset = get_training_dataset(
        data_args.train_files,
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
        sample_percentage=data_args.percentage,
        subset_index_files=data_args.subset_index_files,
        seed=data_args.sample_data_seed)

    logger.info(f'TRAIN DATASET: {train_dataset[0].keys()}')
    logger.info(f'TRAIN DATASET EXAMPLE: {train_dataset[0]}')
    
    # Get data collator
    if isinstance(train_dataset, SupervisedDataset):
        if (training_args.source_wise_selection != "none") or (not training_args.remove_unused_columns):
            data_collator = DataCollatorForSupervisedDatasetWithSource(
                tokenizer=tokenizer)
        else:
            data_collator = DataCollatorForSupervisedDataset(
                tokenizer=tokenizer)
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding="longest")



    if "features" in train_dataset and "dataset" in train_dataset.features:
        train_dataset = train_dataset.remove_columns(
            ["dataset", "id", "messages"])

    eval_dataset = None
    if training_args.eval_dataset is not None:
        from colm.data.get_validation_dataset import get_dataset
        eval_dataset = get_dataset(
            training_args.eval_dataset,
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            max_length=data_args.max_seq_length)

    logger.info(f"Using data collator {type(data_collator)}")
    
    if len(training_args.keep_sources) and isinstance(data_collator, DataCollatorForSupervisedDatasetWithSource):
        training_args.keep_sources = [
            int(source_idx) for source_idx in training_args.keep_sources.split('_')]
        logger.info(
            "Keep all examples of the following sources in the mini-batch.")

        for source_idx in training_args.keep_sources:
            logger.info(train_dataset.all_data_sources[source_idx])
    else:
        training_args.keep_sources = []

    logger.info(f"Keep source indices in {training_args.keep_sources}")






    if training_args.coreset_method is None:
        logger.info("Using HuggingFace Trainer")
        trainer_class = Trainer
    elif training_args.efficient_mezo:
        logger.info("Using SubsetTrainerEfficient")
        trainer_class = SubsetTrainerEfficient
    else:
        logger.info("Using SubsetTrainer")
        trainer_class = SubsetTrainer
        
        
    kwargs = {'logger': logger}
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        **kwargs 
 
    )
    
    
    train_result = trainer.train(
        resume_from_checkpoint=model_args.checkpoint_path)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        max_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(f"Peak GPU memory: {max_mem_gb:.2f} GB")

    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # remove the full model in the end to save space, only adapter is needed
    if isinstance(model, PeftModel):
        pytorch_model_path = os.path.join(
            training_args.output_dir, "pytorch_model_fsdp.bin")
        os.remove(pytorch_model_path) if os.path.exists(
            pytorch_model_path) else None
    
    
    
    
if __name__ == "__main__":
    main()