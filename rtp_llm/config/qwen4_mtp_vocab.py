"""Qwen4 draft/target vocabulary-storage compatibility shared by both loaders."""

from enum import Enum


class MTPVocabPolicy(Enum):
    LOAD_SOURCE = "load_source"
    BORROW = "borrow"
    BORROW_SCALED_FP8 = "borrow_scaled_fp8"


def mtp_vocab_policy(target_config, draft_model_config):
    if getattr(draft_model_config, "_qwen4_mtp_use_dedicated_embeddings", True):
        raise ValueError(
            "qwen4_exp MTP cannot alias a checkpoint that requests dedicated "
            "draft embeddings"
        )

    if bool(getattr(target_config, "enable_output_vocab_pruning", False)) or bool(
        getattr(draft_model_config, "enable_output_vocab_pruning", False)
    ):
        raise ValueError(
            "qwen4_exp MTP cannot alias lm_head when output-vocab pruning is "
            "enabled on the target or draft"
        )
    compatible_fields = (
        "vocab_size",
        "hidden_size",
        "data_type",
        "enable_fp32_lm_head",
        "_qwen4_full_fp8_nvfp4",
    )
    mismatches = [
        name
        for name in compatible_fields
        if getattr(target_config, name, False)
        != getattr(draft_model_config, name, False)
    ]
    if mismatches:
        details = ", ".join(
            f"{name}={getattr(target_config, name)!r}/"
            f"{getattr(draft_model_config, name)!r}"
            for name in mismatches
        )
        raise ValueError(
            "qwen4_exp MTP cannot alias semantically incompatible target "
            f"weights: {details}"
        )
    target_storage = getattr(target_config, "_qwen4_dense_storage", "fp8")
    draft_storage = getattr(draft_model_config, "_qwen4_dense_storage", "fp8")
    mode = getattr(draft_model_config, "_qwen4_mtp_dense_storage_mode", "inherit")
    preserve_source = mode in ("source_bf16", "source_bf16_shared_vocab")
    if target_storage != draft_storage and not (
        preserve_source and target_storage == "fp8" and draft_storage == "source_bf16"
    ):
        raise ValueError("qwen4_exp MTP requires matching dense storage modes")
    target_vocab = getattr(target_config, "_qwen4_vocab_storage", target_storage)
    draft_vocab = getattr(draft_model_config, "_qwen4_vocab_storage", draft_storage)
    if target_vocab != draft_vocab:
        if (
            mode == "source_bf16"
            and target_vocab == "fp8"
            and draft_vocab == "source_bf16"
        ):
            # Returning no aliases keeps the target owner untouched. The
            # existing loader reads the original BF16 vocabulary from the
            # target checkpoint keys declared above, without FP8 scales.
            return MTPVocabPolicy.LOAD_SOURCE
        raise ValueError("qwen4_exp MTP vocabulary storage is incompatible")
    if (
        getattr(target_config, "_qwen4_full_fp8_nvfp4", False) is True
        and target_vocab != "source_bf16"
    ):
        return MTPVocabPolicy.BORROW_SCALED_FP8
    return MTPVocabPolicy.BORROW
