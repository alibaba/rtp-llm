from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Mapping, Sequence

import torch

from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.model_loader.model_weight_info import ModelWeights

# Assuming these imports are from your project and accessible
from rtp_llm.model_loader.weight_module import WeightModule

from .tipc import CudaIpcHelper, SharedMemIpcMeta, SharedMemoryIPCHelper

# Dictionary for renaming specific layer weight names from an external format
# (e.g., 'verl') to the internal 'rtp-llm' format.
RENAME_DICTIONARY = {
    # verl
    "embed_tokens.weight": "embedding",
    "norm.weight": "final_layernorm.gamma",
    "norm.bias": "final_layernorm.beta",
    "lm_head.weight": "lm_head",
    "input_layernorm.weight": "pre_layernorm_weights.gamma",
    "post_attention_layernorm.weight": "post_layernorm_weights.gamma",
    "self_attn.qkv_proj.weight": "self_attention_weights.query_weight.kernel",
    "self_attn.qkv_proj.bias": "self_attention_weights.query_weight.bias",
    "self_attn.o_proj.weight": "self_attention_weights.attention_output_weight.kernel",
    "mlp.gate_proj.weight": "ffn_weights.intermediate_weight.kernel",
    "mlp.up_proj.weight": "ffn_weights.intermediate_weight3.kernel",
    "mlp.down_proj.weight": "ffn_weights.intermediate_weight2.kernel",
    # roll - megatron
    "mbedding.word_embeddings.weight": "embedding",
    "self_attention.linear_proj.weight": "self_attention_weights.attention_output_weight.kernel",
    "self_attention.linear_proj.bias": "self_attention_weights.attention_output_weight.bias",
    "self_attention.linear_qkv.weight": "self_attention_weights.query_weight.kernel",
    "self_attention.linear_qkv.bias": "self_attention_weights.query_weight.bias",
    "mlp.linear_fc1.layer_norm_weight": "post_layernorm_weights.gamma",
    # ???
    "mlp.linear_fc1.weight": "",
}


def rename_function(layer_name: str) -> str:
    """
    Transforms a layer weight name from an external format (e.g., 'verl')
    into the format required by 'rtp-llm'.
    The input format is expected to be like 'model.layers.1.self_attn_qkv_proj.bias'.
    Args:
        layer_name: The layer weight name string from an external source.
    Returns:
        The transformed layer weight name in 'rtp-llm's internal format.
        For example, 'model.layers.1.self_attn_qkv_proj.bias' might become
        'self_attention_weights.query_weight.bias' if it matches a pattern
        and is in the RENAME_DICTIONARY.
    Error Handling:
        This function does not explicitly raise errors but performs string manipulations
        and dictionary lookups. If an unexpected `layer_name` format is provided,
        it might return a string that is not correctly transformed or recognized
        by downstream components.
    """
    # Remove the "model." prefix
    if layer_name.startswith("model."):
        name: str = layer_name[len("model.") :]
    elif layer_name.startswith("decoder."):
        name: str = layer_name[len("decoder.") :]
    else:
        name: str = layer_name
    if "layers" in layer_name:
        # Remove "layers." prefix
        name = name[len("layers.") :]
        # Remove the layer number and the dot following it (e.g., "1." from "1.self_attn...")
        # This assumes the format "layers.<number>.<rest_of_name>"
        first_dot_after_layers = name.find(".")
        if first_dot_after_layers != -1:
            name = name[first_dot_after_layers + 1 :]
        if name in RENAME_DICTIONARY:
            return RENAME_DICTIONARY[name]
        return name
    else:
        if name in RENAME_DICTIONARY:
            return RENAME_DICTIONARY[name]
        return name


class WeightManager:
    """
    Manages model weight updates, including renaming weights from an external
    source and handling inter-process communication (IPC) for tensor transfer.
    It ensures that incoming tensors are correctly processed and sharded/replicated
    as per the rtp-llm model's internal structure (e.g., for Tensor Parallelism (TP)
    or Pipeline Parallelism (PP)).
    """

    def __init__(
        self,
        device,
        weight: ModelWeights,
        model_weights_loader: ModelLoader,
        non_owned_global_weights: Sequence[str] = (),
    ) -> None:
        """
        Initializes the WeightManager with a model's weights, device information, and weight loader.
        """
        self._s_helper = SharedMemoryIPCHelper()

        # Use the explicit device/weights/loader passed in by the caller (e.g. BaseModel),
        # instead of relying on any global "engine" object.
        if isinstance(device, torch.device):
            self._device = device
        else:
            self._device = torch.device(device)

        self._weights: ModelWeights = weight
        self._weights_loader: ModelLoader = model_weights_loader
        self._weight_module = self._weights_loader._model_weights_info
        self._non_owned_global_weights = frozenset(non_owned_global_weights)
        self._working_stream: torch.cuda.Stream = torch.cuda.Stream(
            device=self._device,
        )
        # TODO: Consider the actual need for this lock. If updates are always
        # serialized via the server's request handling, a per-update lock might
        # be redundant or require finer-grained locking within _weights.update_...
        self._lock = threading.Lock()
        self._rl_update = None
        self._rl_source_shapes = None
        if os.environ.get("RTP_LLM_RL_MODE") == "1":
            config = self._weights_loader.model_config
            if config.model_type not in {"qwen_3", "qwen35_moe_text", "qwen36_moe_text", "qwen38_moe_text"} or config.enable_fp32_lm_head:
                raise ValueError("RL mode requires a supported text model with a BF16 LM head")
            self._rl_required_names = self._weights_loader.get_rl_hf_source_names()
            directory = os.environ.get("RTP_WEIGHT_MANIFEST_DIR")
            if directory:
                import hashlib
                import json
                from pathlib import Path
                load = self._weights_loader.get_load_config()
                tp_rank = int(load.tp_rank)
                dp_rank = int(getattr(load, "dp_rank", 0))
                world_rank = dp_rank * int(load.tp_size) + tp_rank
                record = {
                    "schema": "rtp-required-hf-world-v1" if config.model_type == "qwen38_moe_text" else "qwen36-rtp-required-hf-v1",
                    "model_type": config.model_type,
                    "pid": os.getpid(),
                    "device": str(self._device),
                    "gpu_uuid": str(getattr(torch.cuda.get_device_properties(self._device), "uuid", "")),
                    "world_rank": world_rank,
                    "world_size": int(load.tp_size) * int(getattr(load, "dp_size", 1)),
                    "tp_rank": tp_rank, "tp_size": int(load.tp_size),
                    "dp_rank": dp_rank, "dp_size": int(getattr(load, "dp_size", 1)),
                    "ep_rank": int(getattr(load, "ep_rank", 0)),
                    "ep_size": int(getattr(load, "ep_size", 1)),
                    "pp_size": 1,
                    "num_nodes": int(load.num_nodes),
                    "local_world_size": int(load.tp_size) * int(getattr(load, "dp_size", 1)) // int(load.num_nodes),
                    "incarnation_id": os.environ.get("RTP_LLM_RL_RUN_ID", ""),
                    "ffn_tp_size": int(getattr(load, "ffn_tp_size", load.tp_size)),
                    "names": list(self._rl_required_names),
                    "names_sha256": hashlib.sha256("\n".join(self._rl_required_names).encode()).hexdigest(),
                }
                folder = Path(directory)
                folder.mkdir(parents=True, exist_ok=True)
                output = folder / f"rank-{world_rank:05d}.json"
                temporary = output.with_name(output.name + f".writing-{os.getpid()}")
                temporary.write_text(json.dumps(record, indent=2) + "\n")
                temporary.replace(output)

    def rl_begin(self, max_buffer_bytes: int) -> None:
        """Called only by the native RL control RPC after TP quiescence."""
        import json
        from pathlib import Path

        from safetensors import safe_open

        from rtp_llm.model_loader.rl_weight_update import RlWeightUpdate

        if self._rl_update is not None:
            raise RuntimeError("another RL weight transaction is active")
        config = self._weights_loader.model_config
        if config.model_type not in {"qwen_3", "qwen35_moe_text", "qwen36_moe_text", "qwen38_moe_text"} or config.enable_fp32_lm_head:
            raise ValueError(
                "RL updates require a supported text model with a BF16 LM head"
            )
        if (
            self._non_owned_global_weights
            or self._weights_loader.get_load_config().compute_dtype != torch.bfloat16
        ):
            raise ValueError("RL updates require owned BF16 weights")
        if max_buffer_bytes <= 0:
            raise ValueError("max_buffer_bytes must be positive")
        bootstrap_shapes = getattr(self._weights_loader, "bootstrap_source_shapes", None)
        if self._rl_source_shapes is None and bootstrap_shapes is not None:
            required = set(self._weights_loader.get_rl_hf_source_names())
            if set(bootstrap_shapes) != required:
                raise ValueError("Bootstrap shape manifest differs from the native HF source set")
            self._rl_source_shapes = dict(bootstrap_shapes)
        if self._rl_source_shapes is None:
            checkpoint = Path(config.ckpt_path)
            index = json.loads(
                (checkpoint / "model.safetensors.index.json").read_text()
            )["weight_map"]
            required = set(self._weights_loader.get_rl_hf_source_names())
            missing_sources = required - set(index)
            if missing_sources:
                raise ValueError(f"RL source keys missing from checkpoint index: {sorted(missing_sources)}")
            shapes = {}
            for filename in sorted({index[name] for name in required}):
                with safe_open(checkpoint / filename, framework="pt", device="cpu") as handle:
                    for name in handle.keys():
                        if name in required and index.get(name) == filename:
                            shapes[name] = tuple(handle.get_slice(name).get_shape())
            if set(shapes) != required:
                raise ValueError("checkpoint HF shape manifest is incomplete for required text weights")
            self._rl_source_shapes = shapes
        local_experts = None
        if os.environ.get("RTP_LLM_RL_WORLD32_LOCAL_EXPERTS") == "1":
            from rtp_llm.utils.qwen38_expert_shards import ExpertShardLayout

            if (
                config.model_type != "qwen38_moe_text"
                or os.environ.get("RTP_LLM_RL_WORLD32") != "1"
                or os.environ.get("RTP_LLM_RL_WORLD72") == "1"
            ):
                raise ValueError("Local expert refit is restricted to the opted-in Qwen3.8 EP32 world")
            local_experts = ExpertShardLayout.from_load_config(self._weights_loader.get_load_config())
        if os.environ.get("RTP_RL_Q38_EXPERT_CUDA_GRAPH") == "1":
            from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.qwen38_decode_expert_graph import (
                begin_refit,
            )

            begin_refit(self._device)
        self._rl_update = RlWeightUpdate(
            self._weights_loader,
            self._weights,
            self._device,
            self._rl_source_shapes,
            max_buffer_bytes,
            borrow_singleton_sources=os.environ.get("RTP_RL_BORROW_SINGLETON_HF_SOURCES", "0") == "1",
            fixed_expert_map=(
                config.model_type == "qwen38_moe_text"
                # The no-padding EP32 layout allocates no redundant routing
                # maps; only the padded EP72 transaction has static targets.
                and os.environ.get("RTP_LLM_RL_WORLD72", "0") == "1"
                and os.environ.get("RTP_FIXED_PHYSICAL_EP_LAYOUT", "") == "balanced_v1"
            ),
            source_dtypes=getattr(self._weights_loader, "bootstrap_source_dtypes", None),
            local_experts=local_experts,
        )

    def rl_push(self, name: str, metadata: bytes, timeout: float) -> None:
        from rtp_llm.utils.cuda_ipc import CudaIpcV2

        if self._rl_update is None:
            raise RuntimeError("no active RL weight transaction")
        if self._rl_update.local_experts is not None:
            if name not in self._rl_update.source_shapes:
                raise ValueError(f"unknown HF parameter: {name}")
            metadata = self._rl_update.local_experts.unpack_ipc(name, metadata)
        tensor = CudaIpcV2.import_tensor(metadata)
        self._rl_update.push(name, tensor, timeout=timeout)

    def rl_finish(self, timeout: float) -> dict:
        if self._rl_update is None:
            raise RuntimeError("no active RL weight transaction")
        result = self._rl_update.finish(timeout=timeout)
        self._rl_update = None
        if os.environ.get("RTP_RL_Q38_EXPERT_CUDA_GRAPH") == "1":
            from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.qwen38_decode_expert_graph import (
                finish_refit,
            )

            finish_refit(self._device, result)
        return result

    def extract_layer_number(self, s: str) -> int | None:
        """
        Extracts the layer number (an integer) from a string that follows
        the pattern 'layers.<number>'.
        Args:
            s: The input string, e.g., 'model.layers.2.mlp.gate_proj.weight'.
        Returns:
            The extracted layer number as an integer if found; otherwise, returns `None`.
        Error Handling:
            Returns `None` if the pattern 'layers.<number>' is not found,
            or if the captured group cannot be converted to an integer.
        """
        match = re.search(r"layers\.(\d+)", s)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        else:
            return None

    def update(self, req: dict[str, str]) -> None:
        """
        Receives an Inter-Process Communication (IPC) tensor description and
        updates the corresponding model weights.
        For models with Tensor Parallelism (TP) or Pipeline Parallelism (PP),
        this function expects the transmitted tensor to be a complete, unsharded tensor.
        It then handles the internal sharding or replication according to the
        rtp-llm's specific model parallelism configuration.
        Args:
            req: A dictionary containing the IPC request details. Expected keys are:
                 - "desc": A string describing the tensor's IPC metadata
                           (e.g., `CuIpcTensorMeta` or `SharedMemIpcMeta` encoded string).
                 - "name": A string representing the original name of the weight
                           (e.g., 'model.layers.1.self_attn_qkv_proj.bias').
                 - "method": A string indicating the IPC method used ("cuda_ipc" or "shm").
        Returns:
            None. The method updates internal model weights directly.
        Error Handling:
            - `KeyError`: If "desc", "name", or "method" fields are missing from `req`.
            - `ValueError`: If the "method" is invalid (not "cuda_ipc" or "shm"),
                            or if a layer weight name is invalid and its ID cannot be extracted.
            - `NotImplementedError`: If "cuda_ipc" method is attempted (currently disallowed).
            - `Exception`: If the tensor cannot be built from the IPC metadata (e.g., invalid descriptor).
                          This is a general catch-all for unexpected failures in `_t_helper.build_from_meta`.
        """
        if "desc" not in req:
            raise KeyError(
                "Update request is missing the 'desc' field. "
                "It must contain IPC tensor metadata."
            )
        if "name" not in req:
            raise KeyError(
                "Update request is missing the 'name' field. "
                "It must specify the weight name to update."
            )
        if "method" not in req:
            raise KeyError(
                "Update request is missing the 'method' field. "
                "It must specify the IPC method (e.g., 'cuda_ipc' or 'shm')."
            )
        method: str = req["method"]
        desc: str = req["desc"]
        name: str = req["name"]
        stored_name: str = name

        if method not in {"cuda_ipc", "shm"}:
            raise ValueError(
                f"Invalid IPC method '{method}' provided. Only 'cuda_ipc' and 'shm' are allowed."
            )
        tensor: torch.Tensor | None = None

        if method == "cuda_ipc":
            helper = CudaIpcHelper()
            tensor = helper.build_from_meta(bytes.fromhex(desc))
        else:  # method == "shm"
            sm_meta: SharedMemIpcMeta = SharedMemIpcMeta.decode(desc)
            tensor = self._s_helper.build_from_meta(sm_meta)

        if tensor is None:
            logging.error(
                f"Fail to build tensor from ipc description {desc}, method: {method}"
            )
            # This should ideally not be reached if build_from_meta consistently returns a tensor or raises an error.
            raise Exception(
                f"Failed to build tensor from IPC description '{desc}' using method '{method}'. Tensor is None."
            )

        logging.info(
            f"update weight request: {name}, shape: {tensor.shape}, device: {tensor.device}, dtype: {tensor.dtype}"
        )
        with torch.cuda.stream(self._working_stream):
            config = self._weights_loader.get_load_config()
            if "layers" in name:
                # This is a layer-specific weight
                layer_id: int | None = self.extract_layer_number(name)
                if layer_id is None:
                    raise ValueError(
                        f"Invalid layer weight name format: '{name}'. "
                        "Could not extract layer number. Expected format like 'model.layers.<id>...'"
                    )
                name: str = rename_function(name)
                fail: bool = True

                for receptor in self._weight_module.layer_weights[layer_id]:
                    if receptor.name == name or (
                        "ffn_weights" in name and receptor.name == "__ffn_weights__"
                    ):
                        assert isinstance(receptor, WeightModule)

                        # split tensor into shards
                        shard = receptor.update(
                            tensor=tensor,
                            device=self._device,
                            load_config=config,
                            module_name=name,
                        )
                        if isinstance(shard, dict):
                            shard = next(iter(shard.values()))

                        # update tensor weight
                        self._weights.update_layer_weight(
                            layer_id=layer_id, name=name, data=shard
                        )
                        fail = False

                if fail:
                    raise KeyError(
                        f"{stored_name} not found. wanted name list is {[w.name for w in self._weight_module.layer_weights[layer_id]]}"
                    )

            else:
                # weight is global weight

                name: str = rename_function(name)
                if name in self._non_owned_global_weights:
                    raise PermissionError(
                        f"global weight {name!r} is a non-owning alias; update its owner instead"
                    )
                fail: bool = True
                for weight in self._weight_module.weights:
                    if weight.name == name:
                        shard: dict = weight.update(
                            tensor,
                            self._device,
                            load_config=self._weights_loader.get_load_config(),
                        )
                        if isinstance(shard, dict):
                            shard = next(iter(shard.values()))
                        self._weights.update_global_weight(name=name, data=shard)
                        fail = False

                if fail:
                    raise KeyError(
                        f"{stored_name} not found. wanted name list is {[w.name for w in self._weight_module.weights]}"
                    )

            self._working_stream.synchronize()
