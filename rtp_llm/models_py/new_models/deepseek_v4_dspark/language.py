"""Direct V4 DSpARK ownership composed with the existing proposer runtime.

No legacy ModelWeights object is constructed or consumed here.
"""

import torch

from rtp_llm.models_py.layers.embedding import VocabParallelEmbedding
from rtp_llm.models_py.model_desc.deepseek_v4_dspark_model import DeepSeekV4DSparkModel
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.deepseek_v4.backbone import OwnedV4Transformer
from rtp_llm.models_py.new_models.deepseek_v4.language import OwnedV4ModelMixin
from rtp_llm.models_py.new_models.deepseek_v4.load_geometry import make_lm_head
from rtp_llm.models_py.new_models.deepseek_v4_dspark.auxiliary import (
    DSparkAuxiliaryWeights,
)
from rtp_llm.models_py.new_models.deepseek_v4_dspark.layout import (
    DSparkV4CheckpointLayout,
)


class DSparkV4Backbone(OwnedV4Transformer):
    def create_global_modules(self, args, params_dtype, head_dtype):
        super().create_global_modules(args, params_dtype, head_dtype)
        if self.commit_only:
            # These placeholders must be rebound to the loaded target before
            # any checkpoint dispatch/device allocation. No proposal globals,
            # final norm/mHC, or Markov parameters exist in this role.
            # These leaves are *always* loans on PREFILL. Bind replaces the
            # metadata-only parameters before dispatch. Never reserve a second
            # pair of full-vocabulary matrices, even transiently on the host.
            with torch.device("meta"):
                self.embed = VocabParallelEmbedding(
                    args.vocab_size, args.dim, params_dtype=params_dtype
                )
                self.head = make_lm_head(args, head_dtype)

    def checkpoint_weight_name_filter(self):
        base = super().checkpoint_weight_name_filter()
        return lambda name: name in ("embed.weight", "head.weight") or base(name)


class _DraftMapper:
    def __init__(self, model):
        self.backbone = model.v4
        self.auxiliary = model.auxiliary
        self.layout = model._draft_layout

    def map_name(self, name):
        return self.layout.mapped(
            name,
            auxiliary_mapper=self.auxiliary.WEIGHTS_MAPPER,
            backbone_mapper=self.backbone.WEIGHTS_MAPPER,
        )

    def apply(self, items):
        for name, tensor in items:
            yield self.map_name(name), tensor


class DeepSeekV4DSparkForCausalLM(OwnedV4ModelMixin, DeepSeekV4DSparkModel, RtpModule):
    preserved_ft_parameter_loans = {
        "v4.embed.weight": "global.embedding",
        "v4.head.weight": "global.lm_head",
    }

    @property
    def eplb_attention_only(self):
        return self._commit_only_prefill

    def __init__(self, model_config, load_config):
        role = getattr(load_config.parallelism_config, "role_type", None)
        self._commit_only_prefill = str(role).upper().rsplit(".", 1)[-1] == "PREFILL"
        super().__init__(model_config, load_config)
        self._draft_layout = DSparkV4CheckpointLayout(self._commit_only_prefill)
        args = self._v4_args
        self.auxiliary = DSparkAuxiliaryWeights(
            args.dim,
            self._dspark_target_layer_ids,
            args.vocab_size,
            args.vocab_size,
            self._dspark_markov_rank,
            norm_eps=args.norm_eps,
            prefill_commit_only=self._commit_only_prefill,
            device=load_config.device,
        )
        self.WEIGHTS_MAPPER = _DraftMapper(self)

    def create_backbone(self, args, load_config):
        args.commit_only = self._commit_only_prefill
        args.n_mtp_layers = 0
        args.n_hash_layers = 0
        return DSparkV4Backbone(
            args,
            params_dtype=load_config.compute_dtype,
            head_dtype=(
                torch.float32
                if getattr(self.config, "enable_fp32_lm_head", False)
                else load_config.compute_dtype
            ),
            layer_prefix="mtp",
            global_prefix="mtp.2.",
        )

    @property
    def main_proj(self):
        return self.auxiliary.main_proj if hasattr(self, "auxiliary") else None

    @main_proj.setter
    def main_proj(self, value):
        if value is not None:
            raise RuntimeError(
                "Owned DSpARK projection cannot be replaced by legacy weights"
            )

    @property
    def main_norm(self):
        return self.auxiliary.main_norm if hasattr(self, "auxiliary") else None

    @main_norm.setter
    def main_norm(self, value):
        if value is not None:
            raise RuntimeError("Owned DSpARK norm cannot be replaced by legacy weights")

    def _load_extra_weights(self, weights):
        if weights is not None:
            raise RuntimeError("Owned DSpARK must not consume legacy ModelWeights")

    def prepare_borrowed_parameters(self, aliases):
        expected = {"v4.embed.weight", "v4.head.weight"}
        if set(aliases) - expected:
            raise ValueError("V4 DSpARK may only borrow target embedding/head")
        if self._commit_only_prefill and set(aliases) != expected:
            raise ValueError("V4 DSpARK commit-only requires both target globals")
        if (
            getattr(self.config, "normalize_lm_head_weight", False)
            or getattr(self.config, "logit_scale", 1.0) != 1.0
        ):
            raise ValueError(
                "Borrowed V4 DSpARK globals cannot request a second head transform"
            )

    def validate_runtime_device(self, device):
        # Called after target loans are bound and before selecting/staging FST
        # files. In particular a missing loan must not allocate a full head.
        super().validate_runtime_device(device)
        self._require_commit_loans()

    def _require_commit_loans(self):
        if self._commit_only_prefill and not getattr(
            self, "_rtp_borrowed_parameter_bindings", None
        ):
            raise ValueError(
                "V4 DSpARK commit-only requires loaded target globals before dispatch"
            )

    def load_weights(self, weights):
        self._require_commit_loans()
        super().load_weights(weights)

    def checkpoint_weight_name_filter(self):
        return self._draft_layout.includes

    def runtime_weight_view(self):
        result = {"embedding": self.v4.embed.weight, "lm_head": self.v4.head.weight}
        if self.v4.norm is not None:
            result["final_layernorm.gamma"] = self.v4.norm.weight
        result.update(self.auxiliary.runtime_weight_view())
        return result
