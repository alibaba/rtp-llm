"""
Shared utilities for CUDA Graph tests.
This module provides common model building and KV cache initialization functionality.
"""

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.model_factory import ModelFactory
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.ops.compute_ops import KVCache, PyModelInputs, get_scalar_type, init_device
from rtp_llm.tools.api.hf_model_helper import get_model_info_from_hf


@dataclass
class ModelBuildConfig:
    """Configuration for building a model."""

    model_path: str
    max_seq_len: int = 4096
    tokens_per_block: int = 64
    max_total_tokens: int = 4096
    hack_layer_num: int = 1
    device_reserve_memory_bytes: int = -536870912
    act_type: Optional[str] = None
    device: str = "cuda:0"


@dataclass
class ModelBuildResult:
    """Result of building a model."""

    model: GptModelBase
    model_config: object
    compute_dtype: torch.dtype
    kv_cache: Optional[KVCache] = None
    layer_num: int = 0
    block_nums: int = 0
    kv_head_num: int = 0
    size_per_head: int = 0
    tokens_per_block: int = 0
    engine_config: Optional[EngineConfig] = None
    hidden_size: int = 0  # from model_config (engine build path), for CUDA graph etc.


class CudaGraphTestModelBuilder:
    """
    Shared model builder for CUDA Graph tests.
    Encapsulates common model building logic used by both prefill and decode tests.
    """

    def __init__(self, config: ModelBuildConfig):
        self.config = config
        self.py_env_configs: Optional[PyEnvConfigs] = None
        self.gpt_model = None

    def _set_configs(self) -> None:
        """Set configuration structures instead of environment variables."""
        self.py_env_configs = PyEnvConfigs()

        # Get model info from HuggingFace
        model_path, model_type = get_model_info_from_hf(self.config.model_path, None)
        self.py_env_configs.model_args.model_type = model_type
        self.py_env_configs.model_args.ckpt_path = model_path
        self.py_env_configs.model_args.max_seq_len = self.config.max_seq_len
        self.py_env_configs.kv_cache_config.seq_size_per_block = (
            self.config.tokens_per_block
        )
        self.py_env_configs.profiling_debug_logging_config.hack_layer_num = (
            self.config.hack_layer_num
        )

        if self.config.act_type:
            self.py_env_configs.model_args.act_type = self.config.act_type

        if not self.py_env_configs.model_args.tokenizer_path:
            self.py_env_configs.model_args.tokenizer_path = model_path

        # Set ModelSpecificConfig.load_python_model = True
        self.py_env_configs.model_specific_config.load_python_model = True
        # Set DeviceResourceConfig.device_reserve_memory_bytes
        if self.py_env_configs.device_resource_config.device_reserve_memory_bytes == 0:
            self.py_env_configs.device_resource_config.device_reserve_memory_bytes = (
                self.config.device_reserve_memory_bytes
            )

    def build_model(
        self, init_kv_cache: bool = False, is_casual: bool = True
    ) -> ModelBuildResult:
        """
        Build model using ModelFactory.

        Args:
            init_kv_cache: Whether to initialize KV cache (needed for decode tests)

        Returns:
            ModelBuildResult containing the model and related configurations
        """
        self._set_configs()

        # Create EngineConfig from py_env_configs
        engine_config = EngineConfig.create(self.py_env_configs)

        # Create model configs
        model_config = ModelFactory.create_model_config(
            model_args=self.py_env_configs.model_args,
            lora_config=self.py_env_configs.lora_config,
            kv_cache_config=engine_config.kv_cache_config,
            profiling_debug_logging_config=engine_config.profiling_debug_logging_config,
            generate_env_config=self.py_env_configs.generate_env_config,
            embedding_config=self.py_env_configs.embedding_config,
            quantization_config=self.py_env_configs.quantization_config,
            render_config=self.py_env_configs.render_config,
        )
        model_config.attn_config.is_causal = is_casual
        model_config.attn_config.need_rope_kv_cache = is_casual
        # Update engine_config based on model_config
        ModelFactory.update_engine_config_from_model_config(
            engine_config=engine_config,
            model_config=model_config,
        )

        # hidden_size from model_config (same source as engine_config path)
        hidden_size = getattr(model_config, "hidden_size", 0)
        if not hidden_size and hasattr(model_config, "attn_config"):
            attn = model_config.attn_config
            hidden_size = getattr(attn, "head_num", 0) * getattr(
                attn, "size_per_head", 0
            )

        # Create model using ModelFactory
        self.gpt_model = ModelFactory._create_model(
            model_config=model_config,
            engine_config=engine_config,
            vit_config=None,
            merge_lora=False,
        )

        # Load the model
        self.gpt_model.load()
        compute_dtype = self.gpt_model.weight.dtype
        model = self.gpt_model.py_model
        py_model_config = model.config

        # Init device with new API
        init_device(
            parallelism_config=engine_config.parallelism_config,
            model_config=model_config,
            eplb_config=model_config.eplb_config,
            fmha_config=engine_config.fmha_config,
            device_resource_config=engine_config.device_resource_config,
            moe_config=engine_config.moe_config,
            sp_config=engine_config.sp_config,
            misc_config=engine_config.misc_config,
            profiling_debug_logging_config=engine_config.profiling_debug_logging_config,
            hw_kernel_config=engine_config.hw_kernel_config,
            concurrency_config=engine_config.concurrency_config,
            ffn_disaggregate_config=engine_config.parallelism_config.ffn_disaggregate_config,
            runtime_config=engine_config.runtime_config,
            model_specific_config=engine_config.model_specific_config,
        )

        result = ModelBuildResult(
            model=model,
            model_config=py_model_config,
            compute_dtype=compute_dtype,
            engine_config=engine_config,
            hidden_size=int(hidden_size),
        )

        # Initialize KV cache if requested
        if init_kv_cache:
            self._init_kv_cache(result, py_model_config)
            model.kv_cache = result.kv_cache

        return result

    def _init_kv_cache(self, result: ModelBuildResult, model_config) -> None:
        """Initialize KV cache, similar to auto_model.py"""
        result.kv_cache = KVCache()
        result.layer_num = model_config.num_layers
        result.kv_head_num = model_config.attn_config.kv_head_num
        result.size_per_head = model_config.attn_config.size_per_head
        result.tokens_per_block = model_config.attn_config.tokens_per_block

        result.block_nums = math.ceil(
            self.config.max_total_tokens / result.tokens_per_block
        )
        # since block_id start from 1, so we should add 1 in the corner case
        result.block_nums += 1

        kv_shape = [
            result.layer_num,
            result.block_nums,
            2,
            result.kv_head_num,
            result.tokens_per_block,
            result.size_per_head,
        ]

        torch.manual_seed(42)
        if self.config.device.startswith("cuda"):
            torch.cuda.manual_seed(42)
        kv_cache_total = torch.randn(
            kv_shape, dtype=result.compute_dtype, device=self.config.device
        )
        # KVCache uses single kv_cache_base tensor with shape [..., 2, ...] for k and v
        result.kv_cache.kv_cache_base = kv_cache_total


def profile_normal_forward(
    model: Any,
    inputs: PyModelInputs,
    trace_path: Optional[str] = None,
    profile_dir: Optional[str] = None,
    record_fn_name: str = "forward_normal",
) -> Any:
    """
    Run normal (non-CUDA-graph) model forward under torch.profiler and optionally export chrome trace.

    Args:
        model: Model with .forward(inputs) returning object with .hidden_states.
        inputs: PyModelInputs to run.
        trace_path: Full path for the exported .json trace. If None, uses profile_dir + default name.
        profile_dir: Directory for trace when trace_path is None. Default ".".
        record_fn_name: Name for torch.profiler.record_function.

    Returns:
        outputs from model.forward(inputs). Logs trace path if exported.
    """
    if trace_path is None:
        profile_dir = profile_dir or "."
        trace_path = os.path.join(profile_dir, "normal_forward_profile.json")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        with torch.profiler.record_function(record_fn_name):
            outputs = model.forward(inputs)
        torch.cuda.synchronize()
    try:
        prof.export_chrome_trace(trace_path)
        logging.info(
            "Normal forward profile trace: %s (open in chrome://tracing)",
            trace_path,
        )
    except Exception as e:
        logging.info("Skip profile trace export: %s", e)
    return outputs


def print_py_model_inputs_full(
    m: PyModelInputs, label: str = "py_model_inputs"
) -> None:
    """Print full contents of PyModelInputs (for debugging CUDA graph tests)."""

    def sizes(t: Optional[torch.Tensor]) -> str:
        if t is None or (
            hasattr(t, "numel") and t.numel() == 0 and not hasattr(t, "shape")
        ):
            return "undef"
        if hasattr(t, "shape"):
            return ",".join(str(x) for x in t.shape)
        return "undef"

    def print_tensor_int(
        name: str, t: Optional[torch.Tensor], max_vals: int = 64
    ) -> None:
        if t is None:
            print(f"  attention_inputs.{name}: defined=False sizes=undef")
            return
        defined = True
        sizes_str = sizes(t)
        print(
            f"  attention_inputs.{name}: defined={defined} sizes=[{sizes_str}]", end=""
        )
        if hasattr(t, "numel") and t.numel() > 0:
            cpu = t.cpu() if t.is_cuda else t
            if cpu.dtype in (torch.int32, torch.int64, torch.long):
                n = min(cpu.numel(), max_vals)
                vals = cpu.flatten()[:n].tolist()
                print(f" values({n})= {vals}", end="")
                if cpu.numel() > max_vals:
                    print(" ... (truncated)", end="")
        print()

    print(f"[{label}] full dump:")
    print(
        f"  input_ids: defined={m.input_ids is not None} sizes=[{sizes(m.input_ids)}]"
    )
    if m.input_ids is not None and m.input_ids.numel() > 0:
        ids_cpu = m.input_ids.cpu()
        n = min(ids_cpu.numel(), 2048)
        vals = ids_cpu.flatten()[:n].tolist()
        print(
            f"    values({n})= {vals}"
            + (" ... (truncated)" if ids_cpu.numel() > 2048 else "")
        )
    print(
        f"  input_hiddens: defined={m.input_hiddens is not None} sizes=[{sizes(m.input_hiddens)}]"
    )
    a = m.attention_inputs
    print(
        f"  attention_inputs (scalars): is_prefill={a.is_prefill} is_s_padded={a.is_s_padded} "
        f"is_cuda_graph={getattr(a, 'is_cuda_graph', 'N/A')} "
        f"context_total_kv_length={getattr(a, 'context_total_kv_length', 0)} "
        f"total_tokens={getattr(a, 'total_tokens', 0)}"
    )
    print_tensor_int("input_lengths", getattr(a, "input_lengths", None), 32)
    print_tensor_int("sequence_lengths", getattr(a, "sequence_lengths", None), 32)
    print_tensor_int("prefix_lengths", getattr(a, "prefix_lengths", None), 32)
    print_tensor_int("cu_seqlens", getattr(a, "cu_seqlens", None), 32)
    print_tensor_int("cu_kv_seqlens", getattr(a, "cu_kv_seqlens", None), 32)
    print_tensor_int(
        "decode_cu_seqlens_host", getattr(a, "decode_cu_seqlens_host", None), 32
    )
    print_tensor_int("padding_offset", getattr(a, "padding_offset", None), 256)
    print(
        f"  attention_inputs.kv_cache_block_id_host: defined={a.kv_cache_block_id_host is not None} sizes=[{sizes(a.kv_cache_block_id_host)}]"
    )
    print(
        f"  attention_inputs.kv_cache_block_id_device: defined={a.kv_cache_block_id_device is not None} sizes=[{sizes(a.kv_cache_block_id_device)}]"
    )
    print_tensor_int("prefix_lengths_d", getattr(a, "prefix_lengths_d", None), 32)
    print_tensor_int(
        "sequence_lengths_plus_1_d", getattr(a, "sequence_lengths_plus_1_d", None), 32
    )
    print_tensor_int("input_lengths_d", getattr(a, "input_lengths_d", None), 32)
    print_tensor_int("decode_cu_seqlens_d", getattr(a, "decode_cu_seqlens_d", None), 32)
    dtype_obj = getattr(a, "dtype", None)
    if dtype_obj is not None:
        try:
            dtype_str = str(get_scalar_type(dtype_obj))
        except Exception:
            name_attr = getattr(dtype_obj, "name", None)
            dtype_str = name_attr() if callable(name_attr) else str(dtype_obj)
    else:
        dtype_str = "None"
    print(f"  attention_inputs.dtype: {dtype_str}")
    cs = getattr(a, "cache_store_inputs", None)
    pf = getattr(a, "prefill_cuda_graph_copy_params", None)
    print(
        f"  attention_inputs.cache_store_inputs: {'set' if cs is not None else 'nullopt'}"
    )
    print(
        f"  attention_inputs.prefill_cuda_graph_copy_params: {'set' if pf is not None else 'nullopt'}"
    )
