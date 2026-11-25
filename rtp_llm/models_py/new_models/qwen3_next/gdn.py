"""Direct checkpoint parameters bound to main's Qwen GDN execution path."""

import torch

from rtp_llm.models_py.model_desc.qwen3_next import (
    Qwen3NextGatedDeltaNetDecode,
    Qwen3NextGatedDeltaNetPrefill,
    Qwen3NextGdnExecution,
    Qwen3NextGdnStateParameters,
    RmsNormGated,
)
from rtp_llm.models_py.new_models.qwen3_next.gdn_parameters import (
    Qwen3NextGdnParameters,
)


class Qwen3NextGatedDeltaNet(Qwen3NextGdnExecution, Qwen3NextGdnParameters):
    def __init__(self, *, parallelism_config, norm_activation="silu", **kwargs):
        super().__init__(**kwargs)
        self.parallelism_config = parallelism_config
        self.norm_activation = norm_activation

    def prepare_resident_projection(self):
        self._prepare_input_projection()

    def process_weights_after_loading(self):
        self._prepare_input_projection()
        # Scratch may stage on CPU; take kernel views only after migration.
        state = Qwen3NextGdnStateParameters(
            self.conv1d_w.detach(), self.dt_bias.detach(), self.a_log.detach()
        )
        self.prefill_gdn = Qwen3NextGatedDeltaNetPrefill(
            self.linear_attn_config, self.parallelism_config, state
        )
        self.decode_gdn = Qwen3NextGatedDeltaNetDecode(
            self.linear_attn_config, self.parallelism_config, state
        )
        self.norm = RmsNormGated(
            self.norm_w.detach(),
            eps=self.rms_norm_eps,
            group_size=self.head_v_dim,
            activation=self.norm_activation,
        )

    def _input_project(self, hidden_states):
        return Qwen3NextGdnParameters._input_project(self, hidden_states)
