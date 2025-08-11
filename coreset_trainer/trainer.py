from transformers import Trainer



import torch
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available

from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.trainer_utils import (
    EvalPrediction,
    HPSearchBackend,
    RemoveColumnsCollator,
    TrainOutput,
    has_length,
    speed_metrics,
)
from torch.utils.data import Dataset, RandomSampler
import shutil
import copy
from transformers.utils import (
    is_accelerate_available,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_mp_enabled,
    is_torch_xla_available,
)
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
)
from torch import nn
from collections import Counter
from colm.train.utils import collate_fn
from transformers.training_args import ParallelMode, TrainingArguments
from colm.train.facility_location import get_orders_and_weights

if is_apex_available():
    from apex import amp
import torch.distributed as dist
import numpy as np
if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedType,
        GradientAccumulationPlugin,
    )

    DATA_SAMPLERS = [RandomSampler]
    
    from accelerate.data_loader import SeedableRandomSampler

    DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION


    from transformers.trainer_pt_utils import smp_forward_backward
    IS_SAGEMAKER_MP_POST_1_10 = True


from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerState,
)
from transformers.integrations import hp_params
import time
from transformers.trainer_pt_utils import (
    get_dataloader_sampler,
    get_model_param_count,
)


import os

import math
import sys
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow

if is_peft_available():
    from peft import PeftModel
    

import importlib
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import logging

logger = logging.getLogger(__name__)




class CoresetTrainer(Trainer):
    def __init__(self, model, ratio, args=None, data_collator=None, train_dataset=None, eval_dataset=None, processing_class=None):
        self.args = args
        self.args.max_seq_length = 512
        self.args.use_fast_tokenizer = True
        self.args.lr_scheduler_type = "linear"
        self.args.warmup_ratio = 0.03
        self.args.weight_decay = 0.0
        self.args.eval_strategy = "no"
        self.args.logging_steps = 1
        self.args.num_train_epochs = 4
        self.args.bf16 = False
        self.args.tf32 = False
        self.args.fp16 = True
        self.args.overwrite_output_dir = True
        self.args.optim = "adamw_torch"
        self.args.seed = 0
        self.args.percentage = 1.0
        self.args.save_strategy = "epoch"
        self.args.lora_r = 128
        self.args.lora_alpha = 512
        self.args.lora_dropout = 0.05
        self.args.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"]
        self.args.learning_rate = 2e-5
        self.args.per_device_train_batch_size = 8
        self.args.gradient_accumulation_steps = 1
        self.args.small_batch_ratio = ratio
        self.args.data_selection_method = "submodlib"
        self.args.data_selection_unit = "mezo"
        self.args.facility_similarity = "l1"
        self.args.source_wise_selection = "none"
        self.args.save_indices = True
        self.args.keep_sources = []#"2_9_14_15"
        self.args.mezo_eps = 1e-3
        self.args.mezo_transform = "none"
        self.args.mezo_selection = "grad"
        self.args.mezo_topk = "largest"
        self.args.num_per_class_start = "floor"
        self.args.mezo_optim = "adam"
        self.args.zo_dim = 2560
        self.args.modify_forward = False
        self.last_layer_index = 15
        self.last_layers = "v_proj"
        
        list_last_layer = [self.last_layers]
        self.last_layers = []
        for last_layer in list_last_layer:
            if 'fc' in self.last_layers:
                self.last_layers.append(f'{self.last_layer_index}.mlp.{last_layer}')
            else:
                self.last_layers.append(f'{self.last_layer_index}.self_attn.{last_layer}')
                
        self.last_layers = [name + '.lora_B' for name in self.last_layers]  # only consider lora_B
        # print("self.last_layers", self.last_layers, flush=True)
        
        for name, param in model.named_parameters():
            if any(substring in name for substring in self.last_layers):
                print(param.shape[0],param.shape[1])
                self.rep_params = param.shape[0]*param.shape[1]
                break
        # print("rep_params", self.rep_params)
                
                
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class
        )
        self._set_signature_columns_if_needed()
        
        self.full_batch_dict = {}
        self.coreset_batch_dict = {}
        
        self.trainer_state_name = "trainer_state.json"
        self.optimizer_name = "optimizer.pt"
        self.optimizer_name_bin = "optimizer.bin"
        
        
        if args.source_wise_selection is not None:
            self.args.remove_unused_columns = False
        if self.args.save_indices:
            self.indices_path = os.path.join(self.args.output_dir, 'indices')
            os.makedirs(self.indices_path, exist_ok=True)
            print(f"Save indices to {self.indices_path}")
        self.prev_m_t = None
        self.prev_v_t = None
        self.named_parameters_to_optim = []
        self.original_grad_settings = {}
        self.zo_random_seed = np.random.randint(1000000000)
        self.random_selection_seed = np.random.randint(42)

        
        
        
        
        
    def select_data(self, inputs, max_samples=64, source_list=None):
        """
        Adapted from https://github.com/BigML-CS-UCLA/CoLM/blob/395abfd86cacb40e404bf5fd96a0b7d5918ac8a8/colm/train/subset_trainer_distributed.py#L1314
        
        Select a subset of inputs based on model representations using Facility Location.
        """
        greedy_indices = get_orders_and_weights(
            max_samples,
            inputs,
            metric=self.args.facility_similarity,
            y=source_list,
            per_class_start=self.args.num_per_class_start,
            strategy=self.args.source_wise_selection
        )
        # Return subset indices as a list
        idx = greedy_indices[0]
        weights = greedy_indices[1]

        return idx, weights
    def extract_and_save_original_indices(self, list_inputs, list_idx, out_file):
        extracted_indices = []

        for idx, inputs in enumerate(list_inputs):
            if idx in list_idx:
                # print("inputs",inputs)
                extracted_indices.extend(inputs["indices"])

        torch.save(extracted_indices, out_file)
        
        
        
        

    def select_masking(self, all_reps, source_list, per_source=True):
        if (source_list is None) or (not per_source):
            source_list = np.zeros(all_reps.shape[0], dtype=np.int32)
        elif isinstance(source_list, list):
            source_list = np.array(source_list)

        masked_reps = torch.zeros(
            (all_reps.shape[0], self.args.zo_dim), dtype=all_reps.dtype).to(all_reps.device)

        for source in np.unique(source_list):
            source_indices = np.where(source_list == source)[0]
            source_all_reps = all_reps[source_indices]

            if self.args.mezo_selection == "weight":
                weight_list = []
                for _, (_, param) in enumerate(self.named_parameters_to_optim):
                    weight_list.append(param.flatten())
                weights = torch.cat(weight_list, dim=0).flatten()

                if not torch.all(weights == 0):
                    mean_reps = torch.abs(weights.flatten())
                else:
                    mean_reps = torch.abs(torch.mean(source_all_reps, dim=0))
            else:
                mean_reps = torch.abs(torch.mean(source_all_reps, dim=0))

            if self.args.mezo_topk == "smallest":
                # Smallest numbers have smallest rank
                ranked_indices = torch.argsort(mean_reps)[:self.args.zo_dim]
            elif self.args.mezo_topk == "largest":
                # Largest numbers have largest rank
                ranked_indices = torch.argsort(mean_reps, descending=True)[
                    :self.args.zo_dim]
            elif self.args.mezo_topk == "sampling":
                index_probs = mean_reps.cpu().numpy().astype('float64')
                index_probs = index_probs / index_probs.sum()
                ranked_indices = np.random.choice(
                    len(mean_reps), size=self.args.zo_dim, replace=False, p=index_probs)
            elif self.args.mezo_topk == "largest_smallest":
                ranked_indices = torch.cat((
                    torch.argsort(mean_reps)[:(self.args.zo_dim // 2)],
                    torch.argsort(mean_reps, descending=True)[
                        :(self.args.zo_dim // 2)]
                ))

            masked_reps[source_indices] = source_all_reps[:, ranked_indices]

        return masked_reps
    
    
    """Adapted from https://github.com/BigML-CS-UCLA/CoLM/blob/395abfd86cacb40e404bf5fd96a0b7d5918ac8a8/colm/train/subset_trainer_distributed.py#L1462C1-L1463C1
    """
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size

        logger.debug(
            f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * \
            args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(
                            train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(
                    args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(
                    train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(
                        train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(
                    train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled(
        ) or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(
                    max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self._fsdp_qlora_plugin_updates()
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(
                        self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(
                        self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(
                    resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(
            f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(
                f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(
            f"  Actual train batch size with ratio {self.args.small_batch_ratio} = {int(total_train_batch_size * self.args.small_batch_ratio):,}")
        logger.info(
            f"  Gradient Accumulation steps = {self.new_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(
            f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, self.trainer_state_name)
        ):
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, self.trainer_state_name))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (
                    num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(
                f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None

        self.control = self.callback_handler.on_train_begin(
            args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler]
                # if version.parse(accelerate_version) > version.parse("0.23.0"):
                sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        total_batched_samples = 0
      
        total_reps = torch.zeros((self.num_orig, self.rep_params), device=args.device)
        input_list = [None for _ in range(self.num_orig)]
        model.module.decomposer._compute_per_sample_loss = True

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            logger.info(f"Steps in epoch is {steps_in_epoch}")
            self.control = self.callback_handler.on_epoch_begin(
                args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(
                    epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            outer_step = -1

            for outer_step, inputs in enumerate(epoch_iterator):
                current_idx = (total_batched_samples % args.gradient_accumulation_steps) * self.args.per_device_train_batch_size
                self.outer_step = outer_step
                # for debugging
                for batch_idx in range(self.args.per_device_train_batch_size):
                    input_list[current_idx + batch_idx] = {
                        k: v[batch_idx:batch_idx+1] for k, v in inputs.items()
                    }
                total_batched_samples += 1
                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(
                        self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += self.accelerator.gather(
                            inputs[main_input_name]).numel()
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (
                        outer_step + 1) == steps_in_epoch
                )

                if outer_step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(
                        args, self.state, self.control)
                # Collect representations until the original large batch size meets
                if (total_batched_samples % args.gradient_accumulation_steps != 0) and (not is_last_step_and_steps_less_than_grad_acc):
                    total_reps[current_idx:current_idx+self.args.per_device_train_batch_size] = self.save_select(model, inputs)
                    self.control = self.callback_handler.on_substep_end(
                        args, self.state, self.control)
                else:
                    rank = int(os.environ['RANK'])
                    t = self.save_select(model, inputs)
                    try:
                        total_reps[-self.args.per_device_train_batch_size:] = t
                    except:
                        total_reps = torch.zeros((t.shape[0], self.rep_params), device=args.device)
                        total_reps = t
                    dist.barrier()

                    # Gather input_list to all devices
                    input_to_gather = {rank: input_list}
                    gathered_inputs = [torch.zeros_like(torch.empty(1)).to(
                        args.device) for _ in range(self.args.world_size)]
                    dist.all_gather_object(gathered_inputs, input_to_gather)
                    complete_input_list = []

                    for rank_idx in range(self.args.world_size):
                        complete_input_list.extend(
                            gathered_inputs[rank_idx][rank_idx])

                    # Gather total_reps to rank 0
                    tensor_to_gather = {rank: total_reps.cpu()}
                    gathered_reps = [torch.zeros_like(torch.empty(
                        1)) for _ in range(self.args.world_size)]
                    dist.gather_object(
                        tensor_to_gather, object_gather_list=gathered_reps if rank == 0 else None, dst=0)
                    max_samples = self.num_select * self.args.world_size

                    if rank == 0:
                        all_reps = [gathered_reps[rank_idx][rank_idx]
                                    for rank_idx in range(self.args.world_size)]
                        all_reps = torch.cat(all_reps, dim=0).to(rank)
                        sampling_indices = np.arange(len(complete_input_list))

                        # Keep all examples from specific sources
                        list_idx_keep = []
                        if len(self.args.keep_sources) > 0:
                            include_in_selection = []

                            for idx in sampling_indices:
                                if complete_input_list[idx]["sources"][0] in self.args.keep_sources:
                                    include_in_selection.append(False)
                                    list_idx_keep.append(idx)
                                else:
                                    include_in_selection.append(True)

                            max_samples -= len(list_idx_keep)
                            logger.info(
                                f"Exclude {len(list_idx_keep)} examples from selection. Select {max_samples} from the remaining {sum(include_in_selection)} examples.")
                            all_reps = all_reps[include_in_selection]
                            sampling_indices = sampling_indices[include_in_selection]
                        
                        # Transform all_reps to adam updates if necessary
                        if self.args.mezo_optim == "adam":
                            all_reps_squared = torch.square(all_reps)
                            # If first step, set m_{t-1} and v_{t-1} to zeros with shape of last layer grads
                            if self.prev_m_t is None or self.prev_v_t is None:
                                self.prev_m_t = torch.zeros_like(
                                    all_reps[-1])
                                self.prev_v_t = torch.zeros_like(
                                    all_reps_squared[-1])

                            # Compute update
                            m_t = self.args.adam_beta1 * self.prev_m_t + \
                                (1-self.args.adam_beta1) * all_reps
                            v_t = self.args.adam_beta2 * self.prev_v_t + \
                                (1-self.args.adam_beta2) * all_reps_squared
                            m_hat = m_t / (1 - self.args.adam_beta1 **
                                           (self.state.global_step+1))
                            v_hat = v_t / (1 - self.args.adam_beta2 **
                                           (self.state.global_step+1))
                            all_reps = m_hat / \
                                (torch.sqrt(v_hat) + self.args.adam_epsilon)

                        # Select masking
                        if self.args.source_wise_selection != "none":
                            source_list = []
                            for idx in sampling_indices:
                                source = complete_input_list[idx]["sources"][0]
                                # Check if it's a tensor and get its item, otherwise leave it as it is
                                if isinstance(source, torch.Tensor):
                                    source = source.item()
                                source_list.append(source)
                            logger.info(f"Source count: {sorted(Counter(source_list).items())}")
                        else:
                            source_list = None

                        if self.args.mezo_topk == "random":
                            ranked_indices = torch.randperm(len(all_reps[0]))[
                                :self.args.zo_dim]
                            all_reps = all_reps[:, ranked_indices]
                        else:
                            all_reps = self.select_masking(
                                all_reps, source_list)

                        if max_samples > 0:
                            selected_idx, _ = self.select_data(
                                all_reps,
                                max_samples=max_samples,
                                source_list=source_list
                            )

                            # Update Adam historical terms with mean of selected subset's last layer gradients for MeZO only
                            # Otherwise, we can get prev_m_t and prev_v_t from the optimizer directly
                            if self.args.mezo_optim == "adam" and "grad" not in self.args.data_selection_unit:
                                self.prev_m_t = m_t[selected_idx].mean(
                                    dim=0).detach()
                                self.prev_v_t = v_t[selected_idx].mean(
                                    dim=0).detach()

                            # Map selected indices back to original indices
                            selected_idx = list_idx_keep + \
                                sampling_indices[selected_idx].tolist()
                        # TODO: Improve this part
                        # If max_samples <= 0, keep the first max_samples
                        elif max_samples == 0:
                            selected_idx = list_idx_keep
                        else:
                            selected_idx = list_idx_keep[:self.num_select]

                        # Save indices
                        # if self.args.save_indices:
                        # Full indices
                        current_step = outer_step + epoch * steps_in_epoch
                        self.extract_and_save_original_indices(
                            complete_input_list,
                            range(len(complete_input_list)),
                            os.path.join(
                                self.indices_path, f'iter{current_step}_full_indices.pt')
                        )
                        # Sampling indices
                        self.extract_and_save_original_indices(
                            complete_input_list,
                            sampling_indices,
                            os.path.join(
                                self.indices_path, f'iter{current_step}_sampling_indices.pt')
                        )
                        # Selected indices
                        self.extract_and_save_original_indices(
                            complete_input_list,
                            selected_idx,
                            os.path.join(
                                self.indices_path, f'iter{current_step}_selected_indices.pt')
                        )

                        # Explicitly convert selected_idx to int32 to avoid
                        # undesirable behavior after broadcasting
                        selected_idx_tensor = torch.tensor(
                            selected_idx).to(torch.int32).to(rank)
                    else:
                        selected_idx_tensor = torch.zeros(
                            max_samples, dtype=torch.int32).to(rank)

                    # Broadcast
                    dist.broadcast(selected_idx_tensor, src=0)
                    selected_idx = selected_idx_tensor.tolist()
                    # Shuffle selected samples
                    selected_inputs = [complete_input_list[i]
                                       for i in selected_idx[self.num_select*rank:self.num_select*(rank+1)]]
                    selected_inputs = [selected_inputs[i:i+self.new_bs]
                                       for i in range(0, len(selected_inputs), self.new_bs)]
                    # Reinit
                    total_reps = torch.zeros((self.num_orig, self.rep_params), device=args.device)
                    input_list = [None for _ in range(self.num_orig)]
                    
                    for _, inner_inputs in enumerate(selected_inputs):
                        with self.accelerator.accumulate(model):
                            inner_inputs = collate_fn(inner_inputs)
                            tr_loss_step = self.training_step(
                                model, inner_inputs) / self.args.small_batch_ratio

                        if (
                            args.logging_nan_inf_filter
                            and not is_torch_xla_available()
                            and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                        ):
                            # if loss is nan or inf simply add the average of previous logged losses
                            tr_loss += tr_loss / \
                                (1 + self.state.global_step -
                                 self._globalstep_last_logged)
                        else:
                            if tr_loss.device != tr_loss_step.device:
                                raise ValueError(
                                    f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                                )
                            tr_loss += tr_loss_step
                        self.current_flos += float(
                            self.floating_point_ops(inner_inputs))
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(
                            True)
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping
                        if is_sagemaker_mp_enabled() and args.fp16:
                            _grad_norm = self.optimizer.clip_master_grads(
                                args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            _grad_norm = nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            _grad_norm = self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                        if (
                            is_accelerate_available()
                            and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                        ):
                            grad_norm = model.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if hasattr(grad_norm, "item"):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = _grad_norm
                    self.optimizer.step()
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + \
                        (outer_step + 1 + steps_skipped) / steps_in_epoch
                    # print("before", self.control)
                    self.control = self.callback_handler.on_step_end(
                        args, self.state, self.control)
                    # print("after", self.control)
                    self.state.stateful_callbacks["TrainerControl"] = self.control.state()
                    self._maybe_log_save_evaluate(
                        tr_loss, grad_norm, model, trial, epoch, start_time,ignore_keys_for_eval)

                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        # PyTorch/XLA relies on the data loader to insert the mark_step for
                        # each step. Since we are breaking the loop early, we need to manually
                        # insert the mark_step here.
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
            if outer_step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(
                args, self.state, self.control)
            self._maybe_log_save_evaluate(
                tr_loss, grad_norm, model, trial, epoch, start_time, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info(
            "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        # Avoid ZeroDivisionError
        effective_global_step = max(self.state.global_step, 0.001)
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(
            use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(
                        f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def save_select(self, model, inputs):
        # This efficient implementation currently only supports MeZO
        assert self.args.data_selection_unit == "mezo"
        # print("self.named_parameters_to_optim",self.named_parameters_to_optim)
        if len(self.named_parameters_to_optim) == 0:
            for name, param in model.named_parameters():
                if any(substring in name for substring in self.last_layers):
                    
                    self.named_parameters_to_optim.append((name, param))

            assert len(
                self.named_parameters_to_optim) != 0, "no layer found"
        assert len(self.named_parameters_to_optim) == 1
        
        # Forward pass until penultimate layer for the entire batch
        zo_intermediate = self.zo_forward_till_penultimate(model, inputs)
        zo_past_key_values = copy.deepcopy(zo_intermediate["past_key_values"])
        
        # First perturbation (+eps) - vectorized
        self.zo_perturb_parameters(scaling_factor=1)
        loss1_batch = self.zo_forward_final_layer(
            model, inputs["labels"], zo_intermediate
        )  # Shape: [batch_size]
        
        # Reset and second perturbation (-eps) - vectorized
        zo_intermediate["past_key_values"] = zo_past_key_values
        self.zo_perturb_parameters(scaling_factor=-2)
        loss2_batch = self.zo_forward_final_layer(
            model, inputs["labels"], zo_intermediate
        )  # Shape: [batch_size]
        
        # Compute projected gradients for entire batch
        projected_grads = ((loss1_batch - loss2_batch) / (2 * self.args.mezo_eps))  # Shape: [batch_size]
        
        # Reset parameters
        self.zo_perturb_parameters(scaling_factor=1)
        
        # Generate random perturbations (same for all samples in batch for consistency)
        torch.manual_seed(self.zo_random_seed)
        
        # Vectorized gradient computation
        batch_size = inputs['input_ids'].shape[0]
        param = self.named_parameters_to_optim[0][1]
        # Generate the same random tensor that would be used in the original implementation
        z = torch.normal(mean=0, std=1, size=param.data.size(),
                        device=param.data.device, dtype=param.data.dtype)
        
        # Vectorized gradient update computation
        # projected_grads shape: [batch_size], z shape: param.shape
        # We need to broadcast properly
        grad_updates = projected_grads.view(-1, *([1] * len(param.shape))) * z.unsqueeze(0)
        # grad_updates shape: [batch_size, *param.shape]
        
        if self.args.mezo_selection == "weight_grad" and not torch.all(param.data == 0):
            grad_updates = grad_updates * param.data.unsqueeze(0)
        
        # Flatten each sample's gradient update
        res = grad_updates.view(batch_size, -1)  # [batch_size, num_params_in_layer]

        return res

    def zo_forward_till_penultimate(self, model, inputs):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()
        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            intermediate = model.module.decomposer.forward_till_penultimate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=True,
                output_hidden_states=True,
            )
        return intermediate

    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input: 
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(
            random_seed if random_seed is not None else self.zo_random_seed)

        for idx, (name, param) in enumerate(self.named_parameters_to_optim):
            z = torch.normal(mean=0, std=1, size=param.data.size(
            ), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * self.args.mezo_eps


    def zo_forward_final_layer(self, model, labels, intermediate):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()
        with torch.inference_mode():
            with self.compute_loss_context_manager():
                loss = model.module.decomposer.forward_final_layer(
                    intermediate_outputs=intermediate,
                    labels=labels,
                    return_dict=False,
                )[0]
        return loss.detach()

    def create_accelerator_and_postprocess(self):
        self.new_accumulation_steps = self.args.gradient_accumulation_steps
        self.new_bs = int(self.args.per_device_train_batch_size * self.args.small_batch_ratio)
        print("self.new_bsself.new_bsself.new_bsself.new_bsself.new_bsself.new_bsself.new_bsself.new_bsself.new_bsself.new_bsv", self.new_bs)
        self.num_orig = int(self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps)
        self.num_select = self.new_accumulation_steps * self.new_bs
        grad_acc_kwargs = {"num_steps": self.new_accumulation_steps}
        grad_acc_kwargs["sync_with_dataloader"] = False
        print("Gradient accumulation args:", grad_acc_kwargs)
        gradient_accumulation_plugin = GradientAccumulationPlugin(
            **grad_acc_kwargs)
        accelerator_config = self.args.accelerator_config.to_dict()
        accelerator_config.pop("non_blocking")
        accelerator_config.pop("gradient_accumulation_kwargs")
        accelerator_config.pop("dispatch_batches")
        accelerator_config.pop("split_batches")
        accelerator_config.pop("even_batches")
        accelerator_config.pop("use_seedable_sampler")

        # create accelerator object
        self.accelerator = Accelerator(
            deepspeed_plugin=self.args.deepspeed_plugin,
            gradient_accumulation_plugin=gradient_accumulation_plugin,
            **accelerator_config,
        )
        # some Trainer classes need to use `gather` instead of `gather_for_metrics`, thus we store a flag
        self.gather_function = self.accelerator.gather_for_metrics

        # deepspeed and accelerate flags covering both trainer args and accelerate launcher
        self.is_deepspeed_enabled = getattr(
            self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(
            self.accelerator.state, "fsdp_plugin", None) is not None

        # post accelerator creation setup
        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            fsdp_plugin.limit_all_gathers = self.args.fsdp_config.get(
                "limit_all_gathers", fsdp_plugin.limit_all_gathers
            )
            if is_accelerate_available("0.23.0"):
                fsdp_plugin.activation_checkpointing = self.args.fsdp_config.get(
                    "activation_checkpointing", fsdp_plugin.activation_checkpointing
                )
                if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                    raise ValueError(
                        "The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg "
                        "can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic "
                        "when using FSDP."
                    )

        if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
            self.propagate_args_to_deepspeed()

        # `save_only_model` can't be used with DeepSpeed/FSDP along with `load_best_model_at_end`
        if (
            self.args.save_only_model
            and (self.is_deepspeed_enabled or self.is_fsdp_enabled)
            and self.args.load_best_model_at_end
        ):
            wrapper = "DeepSpeed" if self.is_deepspeed_enabled else "FSDP"
            raise ValueError(
                f"{wrapper} can't be used with `save_only_model` along with `load_best_model_at_end`.")

        # `auto_find_batch_size` isn't yet supported with DeepSpeed/FSDP
        if (self.is_deepspeed_enabled or self.is_fsdp_enabled) and self.args.auto_find_batch_size:
            wrapper = "DeepSpeed" if self.is_deepspeed_enabled else "FSDP"
            raise NotImplementedError(
                f"`{wrapper}` doesn't support `auto_find_batch_size`.")
    def _get_collator_with_removed_columns(
        self, data_collator: Callable, description: Optional[str] = None
    ) -> Callable:
        """Wrap the data collator in a callable removing unused columns."""
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns
        logger.info(f"Signature columns is {self._signature_columns}")

        if not self.args.remove_unused_columns:
            return data_collator

        remove_columns_collator = RemoveColumnsCollator(
            data_collator=data_collator,
            signature_columns=signature_columns,
            logger=logger,
            description=description,
            model_name=self.model.__class__.__name__,
        )
        return remove_columns_collator
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if (not self.args.remove_unused_columns) and (not self.args.modify_forward):
            new_inputs = {key: inputs[key] for key in inputs.keys(
            ) if key in self._signature_columns}
            outputs = model(**new_inputs)
        else:
            if "sources" in inputs.keys():
                inputs.pop('sources', None)
            outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], inputs_weight: float = 1.) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(
                model, inputs, self.new_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs) * inputs_weight

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.new_accumulation_steps
def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        # if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
        from peft import PeftMixedModel

        classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False

