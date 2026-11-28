"""V4.1 DSpARK projection and sampler parameters, not a V4 alias.

The backbone owns the three independent SWA caches and delayed mHC runtime.
This owner keeps the release checkpoint's 32x32 MXFP8 projection and renamed
Markov embedding/head. It never constructs legacy ModelWeights.
"""

import torch

from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.deepseek_v4_dspark.auxiliary import (
    make_draft_main_norm,
)
from rtp_llm.models_py.new_models.deepseek_v41_dspark.markov_head import (
    V41DSparkMarkovHead,
)
from rtp_llm.models_py.weight_mapper import WeightsMapper


class V41DSparkAuxiliaryWeights(RtpModule):
    def __init__(
        self,
        hidden_size,
        target_layer_ids,
        vocab_size,
        markov_rank,
        *,
        norm_eps=1e-6,
        prefill_commit_only=False,
        head_stage=2,
        device="cpu",
    ):
        super().__init__()
        target_layer_ids = tuple(target_layer_ids)
        if not target_layer_ids or any(
            type(index) is not int or index < 0 for index in target_layer_ids
        ):
            raise ValueError("DSpARK target layers must be nonnegative integers")
        if type(prefill_commit_only) is not bool:
            raise TypeError("prefill_commit_only must be bool")
        for name, value in (
            ("hidden_size", hidden_size),
            ("vocab_size", vocab_size),
            ("markov_rank", markov_rank),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(head_stage) is not int or head_stage < 0:
            raise ValueError("head_stage must be a nonnegative integer")

        from rtp_llm.models_py.new_models.deepseek_v41.linear_owner import (
            Dsv41MXFP8Linear,
        )

        self.target_layer_ids = target_layer_ids
        self.prefill_commit_only = prefill_commit_only
        self.main_proj = Dsv41MXFP8Linear(
            hidden_size * len(target_layer_ids), hidden_size
        )
        self.main_norm = make_draft_main_norm(
            hidden_size, norm_eps, torch.bfloat16, device
        )
        self.markov_head = (
            None
            if prefill_commit_only
            else V41DSparkMarkovHead(
                vocab_size, vocab_size, markov_rank, head_stage=head_stage
            )
        )
        names = {
            "mtp.0.main_proj.weight": "main_proj.weight",
            "mtp.0.main_proj.scale": "main_proj.scale",
            "mtp.0.main_norm.weight": "main_norm.weight",
        }
        if self.markov_head is not None:
            names.update(
                {
                    name: "markov_head." + mapped
                    for name, mapped in self.markov_head._checkpoint_names.items()
                }
            )
        self.WEIGHTS_MAPPER = WeightsMapper(exact_mapping=names)
        self._checkpoint_names = names

    def checkpoint_weight_name_filter(self):
        return self._checkpoint_names.__contains__

    def load_weights(self, weights):
        weights = weights.items() if isinstance(weights, dict) else weights
        super().load_weights(self.WEIGHTS_MAPPER.apply(weights))

    def combine_hidden_states(self, features):
        return self.main_norm(self.main_proj(features))

    def runtime_weight_view(self):
        return (
            {} if self.markov_head is None else self.markov_head.runtime_weight_view()
        )
