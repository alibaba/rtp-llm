import copy
from dataclasses import replace
from typing import Optional

import torch

from rtp_llm.models_py.quant_methods.base import (
    FusedMoEMethodBase,
    QuantizeMethodBase,
    register_moe_quant_method,
    register_quant_method,
)


@register_quant_method("none")
class UnquantizedLinearMethod(QuantizeMethodBase):
    @staticmethod
    def _runtime_signature(layer):
        return (
            layer.weight.device,
            layer.weight.dtype,
            layer.weight.data_ptr(),
            layer.weight.shape,
            layer.weight.stride(),
        )

    def validate_runtime_device(self, device: torch.device) -> None:
        device = torch.device(device)
        if device.type == "cuda" and torch.version.hip is not None:
            from rtp_llm.models_py.quant_methods.dense_layout import (
                rocm_dense_executor_type,
            )

            # Resolve and initialize the real backend before publication or graph
            # capture, never hide import/ABI failure behind F.linear fallback.
            with torch.cuda.device(device):
                rocm_dense_executor_type(False).init_hipblas()

    def runtime_layout_is_ready(self, layer) -> bool:
        return not (layer.weight.is_cuda and torch.version.hip is not None) or (
            getattr(layer, "_dense_runtime_signature", None)
            == self._runtime_signature(layer)
        )

    def encode_lora_delta(self, layer, name, delta):
        if name == "weight" and getattr(layer, "_dense_swizzled", False):
            from rtp_llm.models_py.quant_methods.dense_layout import encode_dense_weight

            return encode_dense_weight(delta)
        return delta

    def prepare_weight_updates(self, layer, weights):
        from rtp_llm.models_py.quant_methods.dense_layout import decode_dense_weight
        from rtp_llm.models_py.runtime_weight_update import prepare_leaf_weight_updates

        conversion = layer
        if getattr(layer, "_dense_swizzled", False):
            conversion = copy.copy(layer)
            conversion._parameters = dict(layer._parameters)
            conversion._parameters["weight"] = torch.nn.Parameter(
                decode_dense_weight(layer.weight.detach().to(device="cpu", copy=True)),
                requires_grad=False,
            )
        updates = prepare_leaf_weight_updates(
            conversion,
            weights,
            reset_fields=(
                "_loaded_parameter_names",
                "_loaded_weight_shards",
                "_loaded_parameter_shards",
                "_fused_loaded_parameters",
                "_merged_per_tensor_scales",
            ),
        )
        return [
            replace(
                update,
                target=layer._parameters[update.name],
                source=self.encode_lora_delta(layer, update.name, update.source),
            )
            for update in updates
        ]

    def create_weights(
        self,
        layer,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> None:
        backing = torch.empty(input_size, output_size, dtype=params_dtype)
        layer.register_parameter(
            "weight",
            torch.nn.Parameter(
                backing.T,
                requires_grad=False,
            ),
        )

    def apply(
        self, layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        executor = getattr(layer, "_dense_runtime_executor", None)
        if executor is not None:
            if self._runtime_signature(layer) != getattr(
                layer, "_dense_runtime_signature", None
            ):
                raise RuntimeError(
                    "Dense runtime storage changed; finalize weights again before inference"
                )
            shape = x.shape[:-1]
            return executor(x.reshape(-1, x.shape[-1])).reshape(
                *shape, layer.weight.shape[0]
            )
        if layer.weight.is_cuda and torch.version.hip is not None:
            raise RuntimeError("ROCm dense weights must be finalized before inference")
        if getattr(layer, "_dense_swizzled", False):
            raise RuntimeError(
                "Swizzled dense parameters are not CPU inference weights"
            )
        if bias is not None and getattr(layer, "bias_as_matrix", False):
            bias = bias.view(1, -1)
        return torch.nn.functional.linear(x, layer.weight, bias)

    def process_weights_after_loading(self, layer) -> None:
        if getattr(layer, "_dense_swizzled", False) and layer.weight.dtype != getattr(
            layer, "_dense_layout_dtype", layer.weight.dtype
        ):
            raise RuntimeError(
                "Changing dtype of packed dense weights requires checkpoint reload"
            )
        expected_stride = (
            (layer.weight.shape[1], 1)
            if getattr(layer, "_dense_swizzled", False)
            else (1, layer.weight.shape[0])
        )
        if layer.weight.ndim != 2 or layer.weight.stride() != expected_stride:
            raise RuntimeError(
                f"Unquantized weight has unexpected stride {layer.weight.stride()}; "
                f"expected {expected_stride}"
            )
        if layer.weight.device.type != "cuda" or torch.version.hip is None:
            return
        from rtp_llm.models_py.quant_methods.dense_layout import (
            can_swizzle_dense,
            encode_dense_weight,
            rocm_dense_executor_type,
        )

        self.validate_runtime_device(layer.weight.device)
        hw_config = getattr(self.quant_config, "hw_kernel_config", None)
        swizzled = bool(getattr(hw_config, "use_swizzleA", False))
        # The current runtime also keeps unaligned BA projections raw. The
        # module's TP-local shape is authoritative, not a legacy weight name.
        swizzled = swizzled and can_swizzle_dense(layer.weight)
        if getattr(layer, "_dense_swizzled", False) and not swizzled:
            raise RuntimeError("Cannot reinterpret an existing swizzled weight as raw")
        if swizzled and not getattr(layer, "_dense_swizzled", False):
            with torch.no_grad():
                # One-time load finalization owns the layout allocation. Match
                # OldLoader's exact [K,N] transpose stride for hipBLASLt, not
                # merely its logical values. Keep the Parameter object; online
                # update contracts are captured after this finalization.
                layer.weight.data = encode_dense_weight(layer.weight)
            layer._dense_swizzled = True
            layer._dense_layout_dtype = layer.weight.dtype
        executor = rocm_dense_executor_type(swizzled)(
            layer.weight.T,
            bias=layer.matmul_bias,
            quant_config=None,
        )
        # Executors hold non-owning views. Registering them as child modules
        # would duplicate parameters and break leaf-owned online transactions.
        layer.__dict__["_dense_runtime_executor"] = executor
        layer._dense_runtime_device = layer.weight.device
        layer._dense_runtime_signature = self._runtime_signature(layer)


@register_moe_quant_method("none")
class UnquantizedFusedMoEMethod(FusedMoEMethodBase):
    def create_weights(
        self,
        layer,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **kwargs,
    ) -> None:
        layer.register_parameter(
            "w13",
            torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    2 * intermediate_size,
                    hidden_size,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "w2",
            torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    hidden_size,
                    intermediate_size,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            ),
        )

    def process_weights_after_loading(self, layer) -> None:
        return None

    def dispatch_weight(self, layer, local_id, projection, parameter_name, tensor):
        if parameter_name == "weight" and not tensor.is_floating_point():
            raise TypeError(
                f"Unquantized MoE {projection}.weight must be floating point, "
                f"got {tensor.dtype}"
            )
        return False
