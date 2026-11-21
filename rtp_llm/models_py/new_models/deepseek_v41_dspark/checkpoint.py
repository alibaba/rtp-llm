"""V4.1 draft-specific ownership, shared by staging and the main runtime."""

import torch

from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.deepseek_v4.load_geometry import make_lm_head
from rtp_llm.models_py.new_models.deepseek_v41.globals import OwnedV41Matrix
from rtp_llm.models_py.new_models.deepseek_v41_dspark.auxiliary import (
    V41DSparkAuxiliaryWeights,
)


class V41DraftGlobalsMixin:
    def create_global_modules(self, args, head_dtype):
        super().create_global_modules(args, head_dtype)
        if self.commit_only:
            # Both globals are mandatory target loans on PREFILL. Keep shard
            # shape and dtype for binding validation, but allocate no storage.
            with torch.device("meta"):
                self.embed = OwnedV41Matrix(
                    args.vocab_size,
                    args.dim,
                    shard_dim=1,
                    tp_size=args.embedding_tp_size,
                    tp_rank=args.embedding_tp_rank,
                    dtype=torch.bfloat16,
                )
                self.head = make_lm_head(args, head_dtype)

    def checkpoint_weight_name_filter(self):
        base = super().checkpoint_weight_name_filter()
        obsolete = {
            self._checkpoint_global_prefix + name
            for name in (
                "embed.weight",
                "head.weight",
                "hc_head_fn",
                "hc_head_base",
                "hc_head_scale",
                # The released last draft stage includes a training-only
                # confidence predictor. Inference samples with the Markov head.
                # Do not filter any other key under confidence_head.
                "confidence_head.proj.weight",
            )
        }
        return lambda name: name in ("embed.weight", "head.weight") or (
            name not in obsolete and base(name)
        )


class _DraftMapper:
    def __init__(self, model):
        self.backbone, self.auxiliary = model.v4, model.auxiliary

    def map_name(self, name):
        if self.auxiliary.checkpoint_weight_name_filter()(name):
            return "auxiliary." + self.auxiliary.WEIGHTS_MAPPER.map_name(name)
        return "v4." + self.backbone.WEIGHTS_MAPPER.map_name(name)

    def apply(self, weights):
        for name, value in weights:
            yield self.map_name(name), value


class V41DraftParameterMixin:
    runtime_parameter_paths = {
        "embedding": "v4.embed.weight",
        "lm_head": "v4.head.weight",
    }

    def _install_draft_auxiliary(
        self, hidden_size, target_ids, vocab_size, rank, head_stage, *, device="cpu"
    ):
        self.auxiliary = V41DSparkAuxiliaryWeights(
            hidden_size,
            target_ids,
            vocab_size,
            rank,
            norm_eps=self._v4_args.norm_eps,
            prefill_commit_only=self._commit_only_prefill,
            head_stage=head_stage,
            device=device,
        )
        self.WEIGHTS_MAPPER = _DraftMapper(self)

    @property
    def main_proj(self):
        return self.auxiliary.main_proj if hasattr(self, "auxiliary") else None

    @main_proj.setter
    def main_proj(self, value):
        if value is not None:
            raise RuntimeError(
                "Direct V4.1 DSpARK projection cannot be replaced by legacy weights"
            )

    @property
    def main_norm(self):
        return self.auxiliary.main_norm if hasattr(self, "auxiliary") else None

    @main_norm.setter
    def main_norm(self, value):
        if value is not None:
            raise RuntimeError(
                "Direct V4.1 DSpARK norm cannot be replaced by legacy weights"
            )

    def _load_owned_extra_weights(self):
        # Already-owned modules are exposed by properties, not registered twice
        # under runtime aliases. In particular this is NOT target Engram init.
        if self.main_proj is None or self.main_norm is None:
            raise RuntimeError("V4.1 DSpARK auxiliary owner is not initialized")

    def _load_extra_weights(self, weights):
        if weights is not None:
            raise RuntimeError(
                "Direct V4.1 DSpARK must not consume legacy ModelWeights"
            )

    def prepare_borrowed_parameters(self, aliases):
        expected = {"v4.embed.weight", "v4.head.weight"}
        if set(aliases) - expected:
            raise ValueError("V4.1 DSpARK may borrow only target embedding/head")
        if self._commit_only_prefill and set(aliases) != expected:
            raise ValueError("V4.1 DSpARK commit-only requires both target globals")

    def checkpoint_weight_name_filter(self):
        backbone = self.v4.checkpoint_weight_name_filter()
        auxiliary = self.auxiliary.checkpoint_weight_name_filter()
        return lambda name: backbone(name) or auxiliary(name)

    def load_weights(self, weights):
        if self._commit_only_prefill and set(
            getattr(self, "_rtp_borrowed_parameter_bindings", {})
        ) != {"v4.embed.weight", "v4.head.weight"}:
            raise ValueError(
                "V4.1 DSpARK commit-only requires target globals before dispatch"
            )
        items = weights.items() if isinstance(weights, dict) else weights
        RtpModule.load_weights(self, self.WEIGHTS_MAPPER.apply(items))

    def runtime_weight_view(self):
        result = {"embedding": self.v4.embed.weight, "lm_head": self.v4.head.weight}
        if self.v4.norm is not None:
            result["final_layernorm.gamma"] = self.v4.norm.weight
        result.update(self.auxiliary.runtime_weight_view())
        return result
