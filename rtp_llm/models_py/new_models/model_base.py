from typing import Any, Optional

from torch import Tensor

from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.ops import DeviceResourceConfig
from rtp_llm.ops.compute_ops import (
    KVCache,
    PyModelInitResources,
    PyModelInputs,
    PyModelOutputs,
)


def required_config_value(config: Any, *names: str) -> Any:
    for name in names:
        if isinstance(config, dict):
            value = config.get(name)
        else:
            value = getattr(config, name, None)
        if value is not None:
            return value
    raise ValueError(f"Model config requires one of {names}")


class NewLoaderModelBase(RtpModule):
    """Runtime base for models whose parameters are owned by the PyModel."""

    def __init__(
        self,
        config: Any,
        parallelism_config,
        fmha_config=None,
        device_resource_config: Optional[DeviceResourceConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.parallelism_config = parallelism_config
        self.fmha_config = fmha_config
        self.weight = None
        self.micro_batch_size = (
            1
            if device_resource_config
            and device_resource_config.enable_layer_micro_batch == 0
            else 2
        )
        self.layer_num = required_config_value(
            config, "num_layers", "num_hidden_layers"
        )
        self.vocab_size = required_config_value(config, "vocab_size")
        self.kv_cache: Optional[KVCache] = None
        self.params_dict: dict[int, Any] = {}

    def initialize(self, init_resource: PyModelInitResources) -> bool:
        self.kv_cache = init_resource.kv_cache
        return True

    def fill_params(
        self,
        sequence_lengths: Tensor,
        input_lengths: Tensor,
        kv_cache_block_id_host: Tensor,
        replay_batch_size: int,
        capture_batch_size: int,
        seq_size_per_block: int,
    ) -> None:
        if capture_batch_size not in self.params_dict:
            raise ValueError(f"No captured FMHA params for batch {capture_batch_size}")
        params = self.params_dict[capture_batch_size]
        if params is None:
            raise RuntimeError("Captured FMHA params cannot be None")
        params.fillParams(
            sequence_lengths,
            input_lengths,
            kv_cache_block_id_host,
            replay_batch_size,
            seq_size_per_block,
        )

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> Any:
        from rtp_llm.models_py.modules import AttnImplFactory

        return AttnImplFactory.get_fmha_impl(
            self.config,
            self.parallelism_config,
            self.weight,
            inputs.attention_inputs,
            self.fmha_config,
            is_cuda_graph,
        )

    def runtime_weight_view(self) -> dict[str, Tensor]:
        raise NotImplementedError

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        raise NotImplementedError
