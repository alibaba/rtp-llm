"""
Test script for Qwen3 model.

This script demonstrates how to create a Qwen3 model with config and weight
from a HuggingFace checkpoint path
"""

import os
import sys

os.environ["LOAD_PYTHON_MODEL"] = "1"

# Optional: Add FasterTransformer path if needed
# sys.path.append("/home/wangyin.yx/workspace/FasterTransformer")

import torch
import rtp_llm.models
from rtp_llm.config.task_type import TaskType
from rtp_llm.distribute.worker_info import ParallelInfo
from rtp_llm.model_factory import ModelFactory
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.models_py.model_desc.qwen3 import Qwen3Model
from rtp_llm.ops import (
    DeviceResourceConfig,
    ParallelismConfig,
    PyAttentionInputs,
    PyModelInputs,
)
from rtp_llm.utils.database import CkptDatabase


class SimpleModelWrapper:
    """Simple wrapper to provide config and weight attributes."""
    def __init__(self, config, weight):
        self.config = config
        self.weight = weight


# Configuration parameters
MODEL_TYPE = "qwen_3"
CHECKPOINT_PATH = "Qwen/Qwen3-0.6B"
os.environ["DEVICE_RESERVE_MEMORY_BYTES"] = str(3 * 1024 * 1024 * 1024)

# Create model configuration from checkpoint path
model_cls = ModelFactory.get_model_cls(MODEL_TYPE)
model_config = model_cls._create_config(CHECKPOINT_PATH)

# Create ModelConfig for ModelLoader
from rtp_llm.model_factory import ModelConfig as ModelConfigType
model_config_for_loader = ModelConfigType(
    model_type=MODEL_TYPE,
    ckpt_path=CHECKPOINT_PATH,
    tokenizer_path=CHECKPOINT_PATH,
    max_seq_len=model_config.max_seq_len,
    seq_size_per_block=1,
    gen_num_per_circle=1,
    quantization="",
    act_type="fp16",
)

# Create parallelism info
env_params = {
    "TP_SIZE": "1",
    "TP_RANK": "0",
    "DP_SIZE": "1",
    "DP_RANK": "0",
    "WORLD_SIZE": "1",
    "WORLD_RANK": "0",
    "LOCAL_WORLD_SIZE": "1",
}
parallel_info = ParallelInfo.from_params(env_params)

# Create config using model's create_config method
config = model_cls.create_config(model_config_for_loader, parallel_info)

# Load weights using ModelLoader
weights_info = ModelFactory.get_weight_cls(MODEL_TYPE)(config, tp_size=1, tp_rank=0)
database = CkptDatabase(CHECKPOINT_PATH, None)
model_weights_loader = ModelLoader(
    TaskType.LANGUAGE_MODEL, weights_info, [], torch.float16, database
)
weights = model_weights_loader.load_weights(device="cpu")

# Create model wrapper with config and weight attributes
model = SimpleModelWrapper(config, weights)

# Create required configs for Qwen3Model
parallelism_config = ParallelismConfig()
parallelism_config.tp_size = 1
parallelism_config.tp_rank = 0

device_resource_config = DeviceResourceConfig()

# Get vocab_size from config
vocab_size = config.vocab_size

# Now we can create Qwen3Model using model.config and model.weight
qwen3_model = Qwen3Model(
    model.config,
    parallelism_config,
    device_resource_config,
    model.weight,
    vocab_size=vocab_size,
)

# Create test input
attention_inputs = PyAttentionInputs()
inputs = PyModelInputs(
    input_ids=torch.randint(0, 10, (1, 10)), attention_inputs=attention_inputs
)

# Test forward pass (if model is properly initialized)
# qwen3_model.forward(inputs)
