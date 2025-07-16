from dataclasses import asdict, dataclass, field, fields
from typing import Optional
from transformers import TrainingArguments 
import os

@dataclass
class CoresetTrainingArguments(TrainingArguments):


    # adapted config format of https://github.com/BigML-CS-UCLA/CoLM/blob/395abfd86cacb40e404bf5fd96a0b7d5918ac8a8/colm/train/training_arguments.py
    #####################
    model_name_or_path: Optional[str] = field(
        default="allenai/OLMo-2-1124-7B",
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    train_datasets: str = field(
        default=None,
        metadata={
            "help": (
                "Train dataset names, white space seperated."
            )
        },
    )
    eval_dataset: str = field(
        default=None,
        metadata={
            "help": (
                "Eval dataset name."
            )
        },
    )

    small_batch_ratio: Optional[float] = field(
        default=0.5,
        metadata={
            "help": (
                "The ratio of the large batch to be trained."
            )
        },
    )
    
    coreset_method: Optional[str] = field(
        default="submodlib",
        metadata={
            "help": (
                "Method used to obtain coresets."
            ),
            "choices": ["submodlib", "weightedsubmodlib", "none"],
        },
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Where the model will be saved."
            )
        },
    )
    
        
    
    # mezo
    ############
    efficient_mezo: bool = field(
        default=True, 
        metadata={
            "help": (
                "Use efficient implementation for MeZO."
            )
        },
    )
    mezo_eps: float = field(
        default=1e-3,
        metadata={
            "help": (
                "Perturbation scale for MeZO. "
            )
        },
    )
    mezo_transform: Optional[str] = field(
        default="none",
        metadata={
            "help": (
                "Transform the mezo gradient. "
            ),
            "choices": ["none", "self_normalize", "normalize", "clip_full", "clip_last"]
        },
    )
    mezo_selection: Optional[str] = field(
        default="grad",
        metadata={
            "help": (
                "Mezo selection criteria. "
            ),
            "choices": ["weight_grad", "weight", "grad"]
        },
    )
    mezo_topk: Optional[str] = field(
        default="largest",
        metadata={
            "help": (
                "Whether to select largest or smallest elements "
            ),
            "choices": ["largest", "smallest", "random", "sampling", "largest_smallest"]
        },
    )
    #############
    
    
    facility_similarity: Optional[str] = field(
        default="l1",
        metadata={
            "help": (
                "The similarity measure for facility location "
            ),
            "choices": ["cosine", "euclidean", "l1"],
        },
    )
    data_selection_unit: Optional[str] = field(
        default="mezo",
        metadata={
            "help": (
                "Units used to select data in the large batch. "
            ),
            "choices": ["rep", "mezo", "masked_grad", "grad", "proj_grad", "completion_length", "length_loss_weighted"],
        },
    )
    zo_dim: int = field(
        default=2560,
        metadata={
            "help": (
                "Dimension of MeZO gradient after masking "
            )
        },
    )
    mezo_optim: Optional[str] = field(
        default="adam",
        metadata={
            "help": (
                "Optimizer to estimate with mezo. "
            ),
            "choices": ["sgd", "adam"],
        }
    )
    rep_dim: int = field(
        default=2560,
        metadata={
            "help": (
                "Dimension of representation after masking "
            )
        },
    )
    proj_dim: int = field(
        default=2560,
        metadata={
            "help": (
                "Projection dim for last layer gradient. "
            )
        },
    )
    block_size: int = field(
        default=128,
        metadata={
            "help": (
                "Fixed block size for the projectors. "
            )
        },
    )
    projector_batch_size: int = field(
        default=16,
        metadata={
            "help": (
                "Batch size for the projectors. "
            )
        },
    )
    model_id: int = field(
        default=0,
        metadata={
            "help": (
                "Model id for projector. "
            )
        },
    )
    source_wise_selection: Optional[str] = field(
        default="proportional",
        metadata={
            "help": (
                "The strategy to determine the number of examples per source when select in a source-wise manner."),
            "choices": ["none", "proportional", "balanced"],
        },
    )
    keep_sources: Optional[str] = field(
        default="0_1_3_5_7_8_9_10_11_13",
        metadata={
            "help": (
                "List of data sources whose all examples to be kept in the large mini-batch. Will be compared against dataset[:]['sources']"
            )
        },
    )
    num_per_class_start: Optional[str] = field(
        default="floor",
        metadata={
            "help": (
                "Function to initialize the number of samplers per class. "
            ),
            "choices": ["floor", "ceil"],
        },
    )
    last_layer_index: int = field(
        default=31,
        metadata={
            "help": (
                "The relative layer index (start from 0) for the last layer. For many models (e.g. Llama-2, Phi-2, Zephyr-3B, etc.), they have 32 layers so the last layer has an index of 31. "
            )
        },
    )
    last_layers: Optional[str] = field(
        default="v_proj",
        metadata={
            "help": (
                "Name of last layers. "
            ),
            "choices": ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2", "qkv_proj", "qkvo_proj", "fc"],
        },
    )
    
    
    nan_grad_handling: Optional[str] = field(
        default="none",
        metadata={
            "help": (
                "How to handle NaN grads in the training process. "
            ),
            "choices": ["none", "zero", "clip_max", "prev_grad"],
        },
    )
    nan_grad_clip_value: float = field(
        default=1.0,
        metadata={
            "help": (
                "Value to clip NaN grads to when using clip_max for nan_grad_handling. "
            ),
        }
    )
    max_steps: int = field(
        default=1024,
        metadata={
            "help": (
                "Maximum number of steps. "
            )
        },
    )
    seed: int = field(
        default=0,
        metadata={
            "help": (
                "Seed."
            )
        },
    )
    # SuperGLUE
    max_new_tokens: int = field(
        default=50,
        metadata={
            "help": (
                "Maximum number of generated tokens. "
            )
        },
    )
    non_diff: bool = field(
        default=False, 
        metadata={
            "help": (
                "Use non-differentiable objective (only support F1 for SQuAD for now)."
            )
        },
    )
    only_train_option: bool = field(
        default=True, 
        metadata={
            "help": (
                "Whether to only train the option part of the input."
            )
        },
    )
    modify_forward: bool = field(
        default=False, 
        metadata={
            "help": (
                "Whether to only train the option part of the input."
            )
        },
    )
    wandb: bool = field(
        default=True, 
        metadata={
            "help": (
                "Logs to wandb if true"
            )
        },
    )
    ####################
    def __post_init__(self):
        if self.train_datasets is not None:
            self.train_datasets = self.train_datasets.split(" ")

        self.output_dir = os.path.join("./models", 
                                       f"{os.path.basename(self.model_name_or_path)}-{'|'.join(self.train_datasets)}-lora-gas{self.gradient_accumulation_steps}-bs{self.per_device_train_batch_size}-{self.data_selection_unit}-{self.last_layers}-{self.zo_dim}_{self.mezo_topk}_{self.mezo_selection}-{self.max_steps}steps-seed{self.seed}"
                                       )
        
        # For multiple layers, separate by commas
        # Only take LoRA B as it is more important        
        if self.last_layers == "qkv_proj":
            list_last_layer = ['q_proj', 'k_proj', 'v_proj']
        elif self.last_layers == "qkvo_proj":
            list_last_layer = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        elif self.last_layers == "qkvo_proj":
            list_last_layer = ['fc1', 'fc2']
        else:
            list_last_layer = [self.last_layers]
            
        self.last_layers = []
        
        for last_layer in list_last_layer:
            if 'fc' in self.last_layers:
                self.last_layers.append(f'{self.last_layer_index}.mlp.{last_layer}')
            else:
                self.last_layers.append(f'{self.last_layer_index}.self_attn.{last_layer}')
                
                


        self.report_to="wandb"
        self.run_name = self.output_dir.split('/')[-1]
        
        self.remove_unused_columns=False # wee need "source"
        super().__post_init__()
