import re
import torch
import os
import json
import logging
import shutil
import pynvml
import threading
from enum import Enum
from typing import Optional, Union, Dict, Any, List, Set
from maga_transformer import _ft_pickler
from maga_transformer.device import get_current_device
from pathlib import Path

class AtomicCounter:
    def __init__(self, initial: int=0):
        self.value = initial
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            self.value += 1
            return self.value

    def decrement(self):
        with self._lock:
            self.value -= 1
            return self.value

    def decrement_if_gt_0(self):
        with self._lock:
            if self.value > 0:
                self.value -= 1
                return True
            return False

    def get(self):
        with self._lock:
            return self.value

PathLike = Union[str, Path]

def to_torch_dtype(maybe_str_dtype: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(maybe_str_dtype, torch.dtype):
        dtype = maybe_str_dtype
    else:
        try:
            dtype = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32,
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
                "int8": torch.int8
            }[maybe_str_dtype.lower()]
        except KeyError:
            raise ValueError(f"Cannot convert to torch data type, got {maybe_str_dtype}")
    return dtype

def get_config_from_path(ckpt_path: str) -> Optional[Dict[str, Any]]:
    if os.path.isdir(ckpt_path):
        # load from huggingface
        config_json_path = os.path.join(ckpt_path, 'config.json')
        if os.path.isfile(config_json_path):
            with open(config_json_path, "r", encoding="utf-8") as reader:
                text = reader.read()
                config_dict = json.loads(text)
                return config_dict
    return None

def generate_pad_mask(input_lengths: torch.Tensor, memory_length: int, init_step: int=0):
    """ Generate a pad mask tensor.

    # Args.
        input_lengths: (batch_size * beam_width,), input lengths
        memory_length: the length of key/value cache memory.
        init_step: int, initial step.
    # Return
        masked_tokens: BoolTensor, (batch_size * beam_width, memory_length),
            True if init_step + input_length[i] <= j < init_step + max_input_length,
            where i is a batch-beam index and j is a time step modulo by memory_length.
    """
    max_input_length = input_lengths.max()
    input_lengths = input_lengths.unsqueeze(1)
    shift = init_step % memory_length
    step_indices = torch.arange(
        init_step, init_step + memory_length, device=input_lengths.device)
    step_indices = step_indices.roll(shift).unsqueeze(0).tile(input_lengths.shape[0], 1)
    masked_tokens = torch.logical_and(
        step_indices >= input_lengths, step_indices < init_step + max_input_length)
    return masked_tokens

def get_ckpt_file_from_index(ckpt_path: str, model_index_file: str) -> List[str]:
    with open(model_index_file) as reader:
        index_json = json.loads(reader.read())
    ckpt_set: Set[str] = set()
    for _, ckpt_file in index_json['weight_map'].items():
        ckpt_set.add(ckpt_file)
    return [os.path.join(ckpt_path, ckpt_file) for ckpt_file in ckpt_set]


def load_ckpt(ckpt_path: str) -> Dict[str, Any]:
    if os.path.isfile(ckpt_path):
        return torch.load(ckpt_path, map_location='cpu', pickle_module=_ft_pickler)
    elif os.path.isdir(ckpt_path):
        # just support from huggingface
        model_index_file = os.path.join(ckpt_path, 'pytorch_model.bin.index.json')
        if os.path.exists(model_index_file):
            checkpoints = get_ckpt_file_from_index(ckpt_path, model_index_file)
        else:
            checkpoints = sorted(Path(ckpt_path).glob("*.bin"))
        params: Dict[str, torch.Tensor] = {}
        for ckpt in checkpoints:
            params.update(torch.load(ckpt, map_location='cpu', pickle_module=_ft_pickler))
        return params
    else:
        raise NotImplementedError(f"just support pt file or huggingface: ckpt_path:{ckpt_path}")


def copy_gemm_config():
    if 'HIPPO_APP_INST_ROOT' in os.environ:
        inst_root = os.environ['HIPPO_APP_INST_ROOT']
        gemm_config_path = os.path.join(inst_root, 'gemm_config.in')
        if os.path.exists(gemm_config_path):
            logging.info("Found gemm_config, copy to current path")
            shutil.copy(gemm_config_path, '.')
            return
    logging.info("not found gemm_config in HIPPO_APP_INST_ROOT, not copy")

def get_dtype_size(dtype: torch.dtype) -> int:
    return {torch.int8: 1, torch.half: 2, torch.bfloat16: 2, torch.float: 4}[dtype]

def get_mem_info():
    return get_current_device().get_mem_info()

def to_cuda(tensor: torch.Tensor) -> torch.Tensor:
    return torch.ops.fastertransformer.async_copy_to_gpu(tensor)

def to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return torch.ops.fastertransformer.async_copy_to_cpu(tensor)
