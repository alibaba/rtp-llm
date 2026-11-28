"""Parameter owners for the existing V4 DSpARK commit projection and sampler.

The V4 backbone owns the three SWA blocks and final mHC/norm stack. This module
owns only the extra DSpARK tensors; it does not introduce another proposer.
"""

import torch

from rtp_llm.models_py.layers.norm import RMSNorm
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm as CudaRMSNorm
from rtp_llm.models_py.new_models.deepseek_v4_dspark.markov_head import DSparkMarkovHead
from rtp_llm.models_py.runtime_weight_update import prepare_leaf_weight_updates
from rtp_llm.models_py.weight_mapper import WeightsMapper


class _CudaDraftMainNorm(CudaRMSNorm, RtpModule):
    """Use the serving norm's forward with a checkpoint-owned, updatable weight."""

    supports_independent_weight_updates = True

    def prepare_weight_updates(self, weights):
        return prepare_leaf_weight_updates(self, weights)


def make_draft_main_norm(dim, eps, dtype, device):
    # Destination chooses the executor; the Parameter itself is created in
    # the loader's current staging context and filled by checkpoint dispatch.
    if torch.device(device).type == "cuda" and torch.version.cuda is not None:
        weight = torch.nn.Parameter(torch.empty(dim, dtype=dtype), requires_grad=False)
        return _CudaDraftMainNorm(weight, eps)
    return RMSNorm(dim, eps=eps, params_dtype=dtype)


class DSparkAuxiliaryWeights(RtpModule):
    def __init__(
        self,
        hidden_size,
        target_layer_ids,
        target_vocab_size,
        draft_vocab_size,
        markov_rank,
        *,
        norm_eps=1e-6,
        params_dtype=torch.bfloat16,
        prefill_commit_only=False,
        padded_draft_vocab_size=None,
        head_stage=2,
        device="cpu",
    ):
        super().__init__()
        target_layer_ids = tuple(target_layer_ids)
        if not target_layer_ids or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in target_layer_ids
        ):
            raise ValueError(
                "DSpARK target_layer_ids must be nonempty non-negative indices"
            )
        if not isinstance(prefill_commit_only, bool):
            raise TypeError("prefill_commit_only must be bool")
        for name, value in (
            ("target_vocab_size", target_vocab_size),
            ("draft_vocab_size", draft_vocab_size),
            ("markov_rank", markov_rank),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        from rtp_llm.models_py.modules.dsv4.fp8.linear_owner import Dsv4FP8Linear

        self.target_layer_ids = tuple(target_layer_ids)
        self.prefill_commit_only = prefill_commit_only
        self.main_proj = Dsv4FP8Linear(
            hidden_size * len(self.target_layer_ids),
            hidden_size,
            tp_size=1,
            tp_rank=0,
            partition="replicated",
        )
        self.main_norm = make_draft_main_norm(
            hidden_size, norm_eps, params_dtype, device
        )
        self.markov_head = (
            None
            if prefill_commit_only
            else DSparkMarkovHead(
                target_vocab_size,
                draft_vocab_size,
                markov_rank,
                padded_draft_vocab_size=padded_draft_vocab_size,
                head_stage=head_stage,
            )
        )
        self.WEIGHTS_MAPPER = WeightsMapper(
            prefix_mapping={
                "mtp.0.main_proj.": "main_proj.",
                "mtp.0.main_norm.": "main_norm.",
                f"mtp.{head_stage}.markov_head.": "markov_head.",
            }
        )
        self._checkpoint_names = {
            "mtp.0.main_proj.weight",
            "mtp.0.main_proj.scale",
            "mtp.0.main_norm.weight",
        }
        if not prefill_commit_only:
            self._checkpoint_names.update(
                {
                    f"mtp.{head_stage}.markov_head.markov_w1.weight",
                    f"mtp.{head_stage}.markov_head.markov_w2.weight",
                }
            )

    def checkpoint_weight_name_filter(self):
        return self._checkpoint_names.__contains__

    def load_standalone_checkpoint(self, weights):
        # In a complete draft the root mapper handles the checkpoint names.
        # Keep this container transparent to the loader's leaf read planner.
        weights = weights.items() if isinstance(weights, dict) else weights
        self.load_weights(self.WEIGHTS_MAPPER.apply(weights))

    def combine_hidden_states(self, features):
        return self.main_norm(self.main_proj(features))

    def runtime_weight_view(self):
        return (
            {} if self.markov_head is None else self.markov_head.runtime_weight_view()
        )
