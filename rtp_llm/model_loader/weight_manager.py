from __future__ import annotations

import logging
import re
import threading
from typing import Any, Mapping

import torch

from rtp_llm.async_decoder_engine.base_engine import BaseEngine
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.model_loader.model_weight_info import ModelWeights

# Assuming these imports are from your project and accessible
from rtp_llm.model_loader.weight_module import WeightModule

from .tipc import CudaIpcHelper, SharedMemIpcMeta, SharedMemoryIPCHelper


class WeightManager:
    """
    Manages model weight updates, including renaming weights from an external
    source and handling inter-process communication (IPC) for tensor transfer.
    It ensures that incoming tensors are correctly processed and sharded/replicated
    as per the rtp-llm model's internal structure (e.g., for Tensor Parallelism (TP)
    or Pipeline Parallelism (PP)).
    """

    def __init__(self, engine: BaseEngine) -> None:
        """
        Initializes the WeightManager with an BaseEngine instance.
        Args:
            model: An instance of `BaseEngine` containing the model's structure,
                   device information, and weight loaders.
        Error Handling:
            This constructor does not explicitly raise errors, but relies on the
            correct initialization of the `BaseEngine` and its internal components.
            Issues with `model` object structure could lead to `AttributeError`.
        """
        self._engine: BaseEngine = engine
        self._s_helper = SharedMemoryIPCHelper()
        self._device: torch.device = engine.model.device
        self._weights: ModelWeights = engine.model.weight
        self._weights_loader: ModelLoader = engine.model.model_weights_loader
        self._weight_module = self._weights_loader._model_weights_info
        self._working_stream: torch.cuda.Stream = torch.cuda.Stream(
            device=self._device,
        )
        # TODO: Consider the actual need for this lock. If updates are always
        # serialized via the server's request handling, a per-update lock might
        # be redundant or require finer-grained locking within _weights.update_...
        self._lock = threading.Lock()

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

    def mount(self, name: str, tensor: torch.Tensor) -> None:

        logging.info(
            f"update weight request: {name}, shape: {tensor.shape}, device: {tensor.device}, dtype: {tensor.dtype}"
        )
        with torch.cuda.stream(self._working_stream):
            tensor = tensor.to(self._device, non_blocking=True)
            config = self._weights_loader.get_load_config()

            if "layers" in name:
                # This is a layer-specific weight
                layer_id: int | None = self.extract_layer_number(name)
                if layer_id is None:
                    raise ValueError(
                        f"Invalid layer weight name format: '{name}'. "
                        "Could not extract layer number. Expected format like 'model.layers.<id>...'"
                    )
                if layer_id > len(self._weight_module.layer_weights):
                    raise IndexError("layer index out of range.")

                fail: bool = True
                for receptor in self._weight_module.layer_weights[layer_id]:
                    if name.startswith(f"model.layers.{layer_id}.{receptor.name}"):
                        # 这里需要使用 start with 判断, 因为可以出现 ffn_weights, moe_weights 这样的组合权重
                        # 这些权重在 rtp 里面的名字是 model.layers.0.__ffn_weights__
                        # 但输入的权重是 model.layers.0.__ffn_weights__.intermediate_weights
                        # 这些权重需要被对应的 receptor 处理

                        assert isinstance(receptor, WeightModule)
                        # split tensor into shards
                        _config = config.copy()
                        _config.use_stack_weight = True

                        shard = receptor.update(
                            tensor=tensor,
                            device=self._device,
                            load_config=_config,
                            module_name=name,
                        )
                        if isinstance(shard, dict):
                            shard = next(iter(shard.values()))

                        if "__ffn_weights__" == receptor.name:
                            # 这里需要按照一定规则更换输入的权重名字
                            name = name.replace("__ffn_weights__", "ffn_weights")
                            name = name[name.find("ffn_weights") :]
                            self._weights.update_layer_weight(
                                layer_id=layer_id,
                                name=name,
                                data=shard,
                                is_master=(config.dp_rank == 0 and config.tp_rank == 0),
                            )
                        elif "__moe_weights__" == receptor.name:
                            # 这里需要按照一定规则更换输入的权重名字
                            name = name.replace(
                                "__moe_weights__", "partial_moe_weights"
                            )
                            name = name[name.find("partial_moe_weights") :]
                            self._weights.update_layer_weight(
                                layer_id=layer_id,
                                name=name,
                                data=shard,
                                is_master=(config.dp_rank == 0 and config.tp_rank == 0),
                            )
                        else:
                            # update tensor weight
                            self._weights.update_layer_weight(
                                layer_id=layer_id,
                                name=receptor.name,
                                data=shard,
                                is_master=(config.dp_rank == 0 and config.tp_rank == 0),
                            )
                        fail = False

                if fail:
                    raise KeyError(
                        f"{name} not found. wanted name list is {[f'model.layers.{layer_id}.{w.name}' for w in self._weight_module.layer_weights[layer_id]]}"
                    )

            else:
                # weight is global weight
                fail: bool = True
                for weight in self._weight_module.weights:
                    if f"model.{weight.name}" == name:
                        shard: dict = weight.update(
                            tensor,
                            self._device,
                            load_config=self._weights_loader.get_load_config(),
                        )
                        if isinstance(shard, dict):
                            shard = next(iter(shard.values()))
                        self._weights.update_global_weight(
                            name=weight.name,
                            data=shard,
                            is_master=(config.dp_rank == 0 and config.tp_rank == 0),
                        )
                        fail = False

                if fail:
                    raise KeyError(
                        f"{name} not found. wanted name list is {[f'model.{w.name}' for w in self._weight_module.weights]}"
                    )

            logging.info(f"RtpLLM Finish Weights Update: {name}.")
            self._working_stream.synchronize()

    def update(self, req: Mapping[str, Any]) -> None:
        """
        Receives an Inter-Process Communication (IPC) tensor description and
        updates the corresponding model weights.
        For models with Tensor Parallelism (TP) or Pipeline Parallelism (PP),
        this function expects the transmitted tensor to be a complete, unsharded tensor.
        It then handles the internal sharding or replication according to the
        rtp-llm's specific model parallelism configuration.
        Args:
            req: A dictionary containing the IPC request details. Expected keys are:
                 - "desc": A list of string describing the tensor's IPC metadatas
                           (e.g., `CuIpcTensorMeta` or `SharedMemIpcMeta` encoded string).
                 - "method": A string indicating the IPC method used ("cuda_ipc" or "shm").
        Returns:
            None. The method updates internal model weights directly.
        Error Handling:
            - `KeyError`: If "desc", or "method" fields are missing from `req`.
            - `ValueError`: If the "method" is invalid (not "cuda_ipc" or "shm"),
                            or if a layer weight name is invalid and its ID cannot be extracted.
            - `Exception`: If the tensor cannot be built from the IPC metadata (e.g., invalid descriptor).
                          This is a general catch-all for unexpected failures in `_t_helper.build_from_meta`.
        """
        # --- Validate Request Fields ---
        if "desc" not in req:
            raise KeyError(
                "Update request is missing the 'desc' field. "
                "It must contain IPC tensor metadata."
            )
        if "method" not in req:
            raise KeyError(
                "Update request is missing the 'method' field. "
                "It must specify the IPC method (e.g., 'cuda_ipc' or 'shm')."
            )
        method: str = str(req["method"])
        if method not in {"cuda_ipc", "shm"}:
            raise ValueError(
                f"Invalid IPC method '{method}' provided. Only 'cuda_ipc' and 'shm' are allowed."
            )

        if method == "shm":
            desc: list[str] = req["desc"]
            if desc is None:
                raise KeyError("Empty tensor meta array.")
            if not isinstance(desc, list):
                raise TypeError("Unexpected desc type.")
            if len(desc) == 0:
                raise ValueError("Empty tensor meta array.")

            for raw in desc:
                meta: SharedMemIpcMeta = SharedMemIpcMeta.decode(raw)
                tensor = self._s_helper.build_from_meta(meta)

                logging.info(
                    f"Ipc received tensor: {meta.name}, {tensor.shape}, {tensor.dtype}"
                )

                self.mount(meta.name, tensor)
        else:
            raise NotImplementedError("cuda ipc is not implemented.")
