import logging
from typing import Any, Optional

from torch import Tensor, nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.modules import AttnImplFactory
from rtp_llm.ops import DeviceResourceConfig
from rtp_llm.ops.compute_ops import (
    KVCache,
    PyModelInitResources,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W


class GptModelBase(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config,
        weight: ModelWeights,
        max_generate_batch_size: int,
        fmha_config=None,  # Optional FMHAConfig
        py_hw_kernel_config=None,  # Optional HWKernelConfig
        device_resource_config: Optional[
            DeviceResourceConfig
        ] = None,  # Optional DeviceResourceConfig
    ) -> None:
        super().__init__()
        self.config = config
        self.parallelism_config = parallelism_config
        self.weight = weight
        self.fmha_config = fmha_config
        self.py_hw_kernel_config = py_hw_kernel_config
        self.micro_batch_size: int = (
            1
            if device_resource_config
            and device_resource_config.enable_layer_micro_batch == 0
            else 2
        )
        self.layer_num: int = config.num_layers
        self.vocab_size: int = config.vocab_size

        self.kv_cache: Optional[KVCache] = None
        self.device_type: DeviceType = get_device_type()

        ## (batch_size -> fmha_params)
        self.params_dict: dict[int, Any] = {}

        ## Dynamic decode backend dispatch result: capture bs bucket -> backend class name.
        ## Filled by select_decode_backend (at capture time, called per bs by the engine)
        ## and looked up by prepare_fmha_impl; empty or a miss falls back to fixed priority
        ## (zero behavior change).
        self.backend_plan: dict[int, str] = {}

    def initialize(self, init_resource: PyModelInitResources) -> bool:
        self.kv_cache = init_resource.kv_cache
        if self.kv_cache is not None:
            num_layers = len(self.kv_cache.kv_cache_base_by_layer)
            layer0_shape = (
                self.kv_cache.kv_cache_base_by_layer[0].shape
                if num_layers > 0
                and self.kv_cache.kv_cache_base_by_layer[0] is not None
                else None
            )
            num_scale_layers = len(self.kv_cache.kv_scale_base_by_layer)
            logging.info(
                f"GptModelBase initialized with "
                f"num_kv_layers={num_layers}, "
                f"layer0_kv_cache_shape={layer0_shape}, "
                f"num_scale_layers={num_scale_layers}, "
            )
        return True

    ## for cuda graph attn kernel params' fill
    def fill_params(
        self,
        sequence_lengths: Tensor,
        input_lengths: Tensor,
        kv_cache_block_id_host: Tensor,
        replay_batch_size: int,
        capture_batch_size: int,
        seq_size_per_block: int,
    ):
        assert capture_batch_size in self.params_dict
        params_ptr = self.params_dict[capture_batch_size]
        assert params_ptr is not None
        params_ptr.fillParams(
            sequence_lengths,
            input_lengths,
            kv_cache_block_id_host,
            replay_batch_size,
            seq_size_per_block,
        )

    def select_decode_backend(
        self, inputs: PyModelInputs, is_cuda_graph: bool = True
    ) -> None:
        """Called per bs by the engine at capture time: benchmark on-device and pick a
        decode backend, writing the winner into self.backend_plan[bs].

        If any guard fails or selection fails, no plan is written (prepare_fmha_impl then
        falls back to fixed priority, zero behavior change). Under TP all ranks must call
        in lockstep (the C++ side drives this per bs in the same order on every rank);
        internally only rank0 benchmarks and the winner is broadcast. Active only when
        enable_dynamic_decode_backend is on and the kv cache is ready.
        """
        if not getattr(
            self.py_hw_kernel_config, "enable_dynamic_decode_backend", False
        ):
            return
        # CUDA-only feature: the selectable backends are FlashInfer-based decode impls
        # that only exist on CUDA. On ROCm/PPU/CPU the on-device bench would drive the
        # platform's paged-attention (e.g. aiter pa on ROCm) through the CUDA-graph
        # capture path with synthetic bench inputs, which faults. Skip here so those
        # platforms keep their fixed-priority decode backend (no behavior change).
        if self.device_type != DeviceType.Cuda:
            return
        if self.kv_cache is None:
            return
        try:
            from rtp_llm.models_py.modules.factory.attention.dispatch import (
                backend_selector,
            )

            bs = int(inputs.attention_inputs.input_lengths.size(0))
            winner = backend_selector.run_backend_selection(self, inputs)
            if winner:
                self.backend_plan[bs] = winner
        except (
            Exception
        ) as e:  # noqa: BLE001 -- on failure fall back to fixed priority, never block startup
            logging.warning(
                f"select_decode_backend failed, fallback to fixed priority: {e}"
            )

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> Any:
        # Dynamic dispatch lookup: only on the cuda graph capture path, and only when the
        # plan has an entry for this bs, instantiate the chosen backend; on a miss or
        # instantiation failure fall through to fixed priority below, never assert.
        # Guard: backend_plan only contains DECODE backends; skip when is_prefill
        # (target-model verify/score has is_prefill=True with num_tokens_per_bs > 1).
        if (
            is_cuda_graph
            and self.backend_plan
            and not inputs.attention_inputs.is_prefill
        ):
            bs = int(inputs.attention_inputs.input_lengths.size(0))
            name = self.backend_plan.get(bs)
            if name:
                from rtp_llm.models_py.modules.factory.attention.dispatch import (
                    backend_selector,
                )

                inst = backend_selector.instantiate_decode_impl(
                    self, inputs.attention_inputs, name, is_cuda_graph
                )
                if inst is not None:
                    return inst

        fmha_impl = AttnImplFactory.get_fmha_impl(
            self.config,
            self.parallelism_config,
            self.weight,
            inputs.attention_inputs,
            self.fmha_config,
            is_cuda_graph,
        )
        return fmha_impl

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        raise NotImplementedError("forward method must be implemented in subclass")
