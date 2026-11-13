# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from vLLM bb233626caa31602728f7ee4625f3d2a4d1a3ad5.
# Source files and explicit RTP adaptations are recorded in vision_sources.json.
"""Runtime boundary for the source-ported, replicated Qwen3.5 ViT layers.

The RTP ViT worker owns a full encoder (the vLLM disable_tp=True mode).
There is no connection to the LLM's process group. vLLM's engine configuration,
platform-specific GEMMs and batch-invariant kernels are not imported here.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Protocol

import torch
from torch import nn

logger = logging.getLogger(__name__)
op_registry: dict[str, type] = {}
op_registry_oot: dict[str, type] = {}


def get_tensor_model_parallel_rank() -> int:
    # Rank within this replicated ViT worker, not the LLM TP rank.
    return 0


def get_tensor_model_parallel_world_size() -> int:
    return 1


def tensor_model_parallel_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError(
        "This ViT port uses disable_tp=True; no ViT TP group is configured"
    )


def tensor_model_parallel_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError(
        "This ViT port uses disable_tp=True; no ViT TP group is configured"
    )


class QuantizationConfig(Protocol):
    """The linear method factory interface; quantized ViT kernels are not ported."""

    def get_quant_method(
        self, layer: nn.Module, prefix: str
    ) -> QuantizeMethodBase | None: ...


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> Sequence[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # NOTE: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class PluggableLayer(nn.Module):
    """
    Base class for pluggable layers.

    A PluggableLayer is a *module-composing* abstraction: it may instantiate other
    ``torch.nn.Module`` objects as sub-layers, and its functionality depends on
    these sub-layers following a generalized invocation sequence. Also, it is stateful
    and may hold parameters or buffers.

    Unlike :class:`CustomOp`, PluggableLayer does NOT provide per-platform
    ``forward_*`` dispatch. Instead, it supports out-of-tree (OOT) replacement
    of the entire layer class at instantiation time, allowing customized
    initialization and submodule composition.
    """

    def __new__(cls, *args, **kwargs):
        try:
            layer_class_name = cls.__name__
        except AttributeError:
            raise TypeError(
                f"Cannot instantiate '{cls.__name__}': its 'name' attribute "
                f"was not set, possibly because it was not decorated with "
                f"@PluggableLayer.register, or it's the PluggableLayer itself."
            ) from None

        if layer_class_name not in op_registry_oot:
            layer_cls_to_instantiate = cls
        else:
            layer_cls_to_instantiate = op_registry_oot[layer_class_name]
            logger.debug(
                "Instantiating pluggable layer: %s using %s",
                layer_class_name,
                str(layer_cls_to_instantiate),
            )
        return super().__new__(layer_cls_to_instantiate)

    # Decorator to register pluggable layers.
    @classmethod
    def register(cls, name: str):
        def decorator(op_cls):
            assert name not in op_registry, f"Duplicate op name: {name}"
            op_cls.name = name
            op_registry[name] = op_cls
            return op_cls

        return decorator

    # Decorator to register out-of-tree(oot) pluggable layers.
    # For OOT pluggable layers:
    #   if in-tree layer class is registered with an oot_custom_layer,
    #   the oot_custom_layer will be used instead.
    @classmethod
    def register_oot(cls, _decorated_layer_cls=None, name: str | None = None):
        def decorator(layer_cls):
            reg_name = name if name is not None else cls.__name__
            assert reg_name not in op_registry_oot, f"Duplicate layer name: {reg_name}"
            layer_cls.name = reg_name
            op_registry_oot[reg_name] = layer_cls
            return layer_cls

        if _decorated_layer_cls is None:
            # Called with parentheses: @PluggableLayer.register_oot()
            # or @PluggableLayer.register_oot(name="...")
            return decorator
        elif isinstance(_decorated_layer_cls, type):  # Check if it's a class
            # Called without parentheses: @PluggableLayer.register_oot
            return decorator(_decorated_layer_cls)
        else:
            raise TypeError("Decorator can only be applied to classes.")


class QuantizeMethodBase(ABC):
    """Base class for different quantized methods."""

    uses_meta_device: bool = False
    """
    Whether this method creates weights on meta device for online quantization.
    When True, weights are created on meta device and quantized layer-wise
    in process_weights_after_loading, reducing peak memory during loading.
    """

    @abstractmethod
    def create_weights(
        self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs
    ):
        """Create weights for a layer.

        The weights will be set as attributes of the layer."""
        raise NotImplementedError

    @abstractmethod
    def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        """Apply the weights in layer to the input tensor.

        Expects create_weights to have been called before on the layer."""
        raise NotImplementedError

    # Not required functions
    def embedding(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        """Gather embeddings in the layer based on indices in the input tensor.

        Expects create_weights to have been called before on the layer."""
        raise NotImplementedError

    # Not required functions
    def tie_weights(self, layer: torch.nn.Module, embed_tokens: torch.nn.Module):
        """Tie ``layer``'s weight to ``embed_tokens``' weight.

        The default shares the weight tensor, which is the standard behavior for
        tied word embeddings and matches what ``ParallelLMHead.tie_weights`` did
        directly before quantization methods became responsible for it.
        Quantization methods that need special weight handling (e.g. repacked
        weights) override this.

        Expects create_weights to have been called before on the layer."""
        layer.weight = embed_tokens.weight
        return layer

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Process the weight after loading.

        This can be used for example, to transpose weights for computation.
        """
        return


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: dict[str, Any] | None,
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(weight, key), f"Overwriting existing tensor attribute: {key}"

        setattr(weight, key, value)


def default_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    return torch.nn.functional.linear(x, weight, bias)


def dispatch_unquantized_gemm():
    # Exact vLLM default CUDA GEMM. CPU also uses this PyTorch fallback in RTP;
    # vLLM CPU weight packing and ROCm-specialized GEMMs are outside this port.
    return default_unquantized_gemm
