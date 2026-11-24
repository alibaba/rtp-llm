"""Published draft body + already loaded target-vocabulary parameters."""

from typing import NamedTuple

import torch
from torch import nn

from rtp_llm.models_py.new_models.qwen3_moe_eagle3.body import Eagle3DraftBody
from rtp_llm.models_py.weight_mapper import WeightsMapper


class Eagle3Proposal(NamedTuple):
    logits: torch.Tensor
    recurrent_hidden: torch.Tensor


def target_vocabulary_loans(target):
    # NormalEngine excludes speculative decoding with output vocabulary pruning.
    # Padding can make a compact head look full-sized; shape checks are insufficient.
    if getattr(target, "enable_output_vocab_pruning", False) or len(
        getattr(target, "output_vocab_ids", ())
    ):
        raise ValueError("EAGLE3 requires a full target LM head; output vocabulary pruning is incompatible")
    return ("embedding", "lm_head")


class Eagle3FullVocabularyDraft(Eagle3DraftBody):
    """Full-width release draft with target-owned embedding and vocabulary head.

    The official draft does not contain either vocabulary tensor. Allocate only
    meta placeholders; NewLoader binds both to existing target Parameters
    *before* reading draft weights. In particular, never initialize transient
    full-vocabulary host tables, nor overwrite target weights from a composite
    checkpoint. This class is not a speculative engine registration.
    """

    _vocab_paths = frozenset({"embed_tokens.weight", "lm_head.weight"})
    WEIGHTS_MAPPER = WeightsMapper(
        exact_mapping={"model.embed_tokens.weight": "embed_tokens.weight"}
    )

    @staticmethod
    def capture_ids(config):
        if config.get("architectures") != ["LlamaForCausalLMEagle3"]:
            raise ValueError(
                "expected NVIDIA LlamaForCausalLMEagle3 draft configuration"
            )
        if config.get("draft_vocab_size") not in (None, config["vocab_size"]):
            raise ValueError(
                "reduced-vocabulary EAGLE3 requires a distinct draft mapping"
            )
        eagle = config.get("eagle_config", {})
        ids = eagle.get("eagle_aux_hidden_state_layer_ids")
        if (
            not isinstance(ids, (list, tuple))
            or len(ids) != 3
            or any(type(i) is not int or i < 0 for i in ids)
            or len(set(ids)) != 3
        ):
            raise ValueError(
                "expected three distinct nonnegative target capture layers"
            )
        if (
            config.get("num_hidden_layers") != 1
            or config.get("attention_bias", False)
            or config.get("mlp_bias", False)
            or config.get("hidden_act") != "silu"
            or not eagle.get("use_aux_hidden_state")
            or not eagle.get("use_input_layernorm_in_first_layer")
            or not eagle.get("use_last_layernorm")
            or eagle.get("use_mtp_layernorm", False)
            or config.get("norm_output")
            or config.get("fc_norm")
            or config.get("use_aux_norm")
            or eagle.get("norm_before_fc")
            or config.get("norm_before_fc")
            or config.get("target_hidden_size", config["hidden_size"])
            != config["hidden_size"]
        ):
            raise ValueError("incompatible EAGLE3 block/normalization contract")
        return tuple(ids)

    def __init__(self, config, dtype):
        ids = self.capture_ids(config)
        super().__init__(
            config["hidden_size"],
            config["intermediate_size"],
            config["num_attention_heads"],
            config["num_key_value_heads"],
            config["head_dim"],
            config["rms_norm_eps"],
            dtype,
        )
        self.capture_layer_ids = tuple(ids)
        hidden, vocab = self.hidden_size, config["vocab_size"]
        self.embed_tokens = nn.Embedding(vocab, hidden, device="meta", dtype=dtype)
        self.lm_head = nn.Linear(hidden, vocab, bias=False, device="meta", dtype=dtype)
        self.requires_grad_(False)

    def prepare_borrowed_parameters(self, aliases):
        if set(aliases) != self._vocab_paths:
            raise ValueError("EAGLE3 draft requires both target vocabulary parameters")

    def validate_runtime_device(self, device):
        self.prepare_borrowed_parameters(
            getattr(self, "_rtp_borrowed_parameter_bindings", {})
        )

    def merge_target_captures(self, captured):
        """Concatenate target layer outputs in release-config order, not ID order."""
        if set(captured) != set(self.capture_layer_ids):
            raise ValueError(
                "captured layers do not match configured EAGLE3 target layers"
            )
        parts = [captured[i] for i in self.capture_layer_ids]
        if any(
            x.ndim != 2
            or x.shape != parts[0].shape
            or x.shape[-1] != self.hidden_size
            or x.dtype != parts[0].dtype
            or x.device != parts[0].device
            for x in parts
        ):
            raise ValueError(
                "EAGLE3 capture tensors must align in rows, width, dtype and device"
            )
        return torch.cat(parts, -1)

    def propose(self, token_ids, target_or_draft_hidden, fmha, kv_cache=None):
        if token_ids.ndim != 1:
            raise ValueError("expected row-aligned token IDs")
        body = self.forward(
            self.embed_tokens(token_ids), target_or_draft_hidden, fmha, kv_cache
        )
        return Eagle3Proposal(self.lm_head(body.logits_input), body.recurrent_hidden)
