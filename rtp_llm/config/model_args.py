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

    __slots__ = [
        "ckpt_path",
        "tokenizer_path",
        "ptuning_path",
        "extra_data_path",
        "local_extra_data_path",
        "model_type",
        "task_type",
        "act_type",
        "max_seq_len",
        "mla_ops_type",
        "json_model_override_args",
        "phy2log_path",
    ]

    def __init__(self):
        """Initialize ModelArgs with default values."""
        # Paths
        self.ckpt_path: str = ""
        self.tokenizer_path: str = ""
        self.ptuning_path: str = ""
        self.extra_data_path: str = ""
        self.local_extra_data_path: str = ""

        # Model type and task
        self.model_type: str = ""
        self.task_type: Optional[str] = None

        # Data types and computation
        self.act_type: Optional[str] = None

        # Sequence length
        self.max_seq_len: Optional[int] = None

        # MLA config
        self.mla_ops_type: Optional[str] = None

        # Model override args
        self.json_model_override_args: str = "{}"

        # EPLB config
        self.phy2log_path: str = ""
