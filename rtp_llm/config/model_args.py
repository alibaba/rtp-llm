"""ModelArgs class for storing model-related arguments from server_args.

This class is a simple container for user-provided model configuration arguments
that are parsed from command-line arguments and environment variables.
"""

from typing import Optional


class ModelArgs:
    """Simple container for model arguments parsed from server_args.
    
    This class contains all user-provided model configuration arguments.
    These arguments are used to populate ModelConfig after the model architecture
    is determined by the model's _create_config method.
    """
    
    def __init__(self):
        """Initialize ModelArgs with default values."""
        # Paths
        self.ckpt_path: str = ""
        self.tokenizer_path: str = ""
        self.ptuning_path: Optional[str] = None
        self.extra_data_path: str = ""
        self.local_extra_data_path: Optional[str] = None
        self.original_checkpoint_path: Optional[str] = None
        
        # Model type and task
        self.model_type: str = ""
        self.task_type: Optional[str] = None
        
        # Data types and computation
        self.act_type: str = ""
        self.use_float32: bool = False
        
        # Sequence length
        self.max_seq_len: Optional[int] = None
        
        # MLA config
        self.mla_ops_type: Optional[str] = None
        
        # Model override args
        self.json_model_override_args: str = "{}"

def apply_model_args_to_config(model_args: ModelArgs, model_config) -> None:
    """Apply ModelArgs to a ModelConfig instance.
    
    This function sets all user-provided arguments to the ModelConfig,
    overwriting any default values set by _create_config.
    
    Args:
        model_args: ModelArgs instance containing user-provided arguments
        model_config: ModelConfig instance to update
    """
    # Set paths
    if model_args.ckpt_path:
        model_config.ckpt_path = model_args.ckpt_path
    if model_args.tokenizer_path:
        model_config.tokenizer_path = model_args.tokenizer_path
    if model_args.ptuning_path is not None:
        model_config.ptuning_path = model_args.ptuning_path
    if model_args.extra_data_path:
        model_config.extra_data_path = model_args.extra_data_path
    if model_args.local_extra_data_path is not None:
        model_config.local_extra_data_path = model_args.local_extra_data_path
    if model_args.original_checkpoint_path is not None:
        model_config.original_checkpoint_path = model_args.original_checkpoint_path
    
    # Set model type and task type
    if model_args.model_type:
        model_config.model_type = model_args.model_type
    if model_args.task_type:
        # Convert string to enum if needed
        from rtp_llm.ops import TaskType
        try:
            model_config.task_type = TaskType[model_args.task_type]
        except KeyError:
            # If enum conversion fails, keep as string (will be handled by C++ binding)
            pass
    
    # Set data types
    if model_args.act_type:
        model_config.act_type = model_args.act_type
    model_config.use_float32 = model_args.use_float32
    
    # Set sequence length
    if model_args.max_seq_len is not None and model_args.max_seq_len > 0:
        model_config.max_seq_len = model_args.max_seq_len

    # Set MLA ops type (C++ binding handles string to enum conversion)
    if model_args.mla_ops_type:
        model_config.mla_ops_type = model_args.mla_ops_type

    # Apply model override args
    if model_args.json_model_override_args and model_args.json_model_override_args != "{}":
        model_config.json_model_override_args = model_args.json_model_override_args
        # Apply override args to model_config
        model_config.apply_override_args(model_args.json_model_override_args)

