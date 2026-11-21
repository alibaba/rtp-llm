"""Target-owned Qwen4 draft vocabulary loans, including coupled FP8 scales."""

from dataclasses import dataclass

import torch
from torch import nn

from rtp_llm.config.qwen4_mtp_vocab import MTPVocabPolicy, mtp_vocab_policy
from rtp_llm.models_py.module_base import collect_loaded_tensor_ids
from rtp_llm.utils.fp8_constants import FP8_STORAGE_SCALE_SUFFIX


EMBED_SCALE = "embedding" + FP8_STORAGE_SCALE_SUFFIX
HEAD_SCALE = "lm_head.weight_only_quant_scale"


def mtp_checkpoint_filter(policy):
    """Apply *before* opening tensor payloads in the shared target checkpoint."""
    if not isinstance(policy, MTPVocabPolicy):
        raise TypeError("expected Qwen4 MTP vocabulary policy")

    def owns(name):
        return name.startswith("mtp.") or (
            policy is MTPVocabPolicy.LOAD_SOURCE
            and name in ("model.language_model.embed_tokens.weight", "lm_head.weight")
        )

    return owns


@dataclass(frozen=True)
class VocabLeaf:
    """Explicit paths exported by the target, not internal W.* runtime roles."""

    weight: str
    scale: str | None = None


@dataclass(frozen=True)
class _Loan:
    weight: nn.Parameter
    scale: torch.Tensor | None
    weight_ptr: int
    scale_ptr: int | None
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device


class BorrowedVocab(nn.Module):
    """Read-only view: the live parameter and buffer stay registered only on target."""

    def __init__(self, endpoint):
        super().__init__()
        self.endpoint = endpoint

    @property
    def weight(self):
        return self.endpoint.weight

    @property
    def storage_scale(self):
        return self.endpoint.scale

    def _apply(self, fn, recurse=True):
        # There are no owned tensors to allocate, migrate or mark checkpoint-loaded.
        return self


@dataclass(frozen=True)
class Qwen4MTPVocabLoan:
    # Plain (not nn.Module): a draft .to()/FST preparation must never traverse
    # the live target, discard an FP8 scale, or initialize target parameters.
    target: nn.Module
    embedding: _Loan
    head: _Loan
    embedding_path: VocabLeaf
    head_path: VocabLeaf

    @classmethod
    def capture(
        cls,
        target,
        target_config,
        draft_config,
        *,
        embedding,
        head,
        embedding_shape,
        head_shape,
        device,
    ):
        policy = mtp_vocab_policy(target_config, draft_config)
        if policy is MTPVocabPolicy.LOAD_SOURCE:
            return None  # Keep draft source BF16; no reads or writes of target.
        scaled = policy is MTPVocabPolicy.BORROW_SCALED_FP8
        loaded = collect_loaded_tensor_ids(target)
        expected_device = torch.device(device)
        if expected_device.type == "cuda" and expected_device.index is None:
            expected_device = torch.device("cuda", torch.cuda.current_device())

        def bind(path, shape):
            if not isinstance(path, VocabLeaf):
                raise TypeError("expected a target-owned vocabulary leaf")
            weight = target.get_parameter(path.weight)
            if id(weight) not in loaded or weight.requires_grad:
                raise ValueError(
                    "draft cannot borrow uninitialized or trainable target vocabulary"
                )
            if weight.device != expected_device or tuple(weight.shape) != tuple(shape):
                raise ValueError(
                    "target vocabulary rank-local shape or execution device differs"
                )
            if scaled:
                if weight.dtype != torch.float8_e4m3fn or path.scale is None:
                    raise ValueError(
                        "FP8 draft vocabulary requires paired target E4M3 weight and scale"
                    )
                scale = _read_scale(target, path.scale)
                if (
                    scale.device != weight.device
                    or scale.dtype != torch.float32
                    or scale.shape != (1,)
                ):
                    raise ValueError(
                        "target vocabulary scale must be local float32 [1]"
                    )
            else:
                if (
                    weight.dtype not in (torch.float32, torch.bfloat16, torch.float16)
                    or path.scale is not None
                ):
                    raise ValueError(
                        "unquantized draft vocabulary cannot alias a packed/scaled target"
                    )
                scale = None
            return _Loan(
                weight,
                scale,
                weight.data_ptr(),
                scale.data_ptr() if scale is not None else None,
                weight.shape,
                weight.dtype,
                weight.device,
            )

        return cls(
            target,
            bind(embedding, embedding_shape),
            bind(head, head_shape),
            embedding,
            head,
        )

    def validate_binding(self):
        """Run before publishing the draft, after parameter migration/finalization."""
        for path, bound in (
            (self.embedding_path, self.embedding),
            (self.head_path, self.head),
        ):
            weight = self.target.get_parameter(path.weight)
            if (
                weight is not bound.weight
                or weight.data_ptr() != bound.weight_ptr
                or weight.shape != bound.shape
                or weight.dtype != bound.dtype
                or weight.device != bound.device
                or weight.requires_grad
            ):
                raise RuntimeError(
                    "target vocabulary storage changed during draft load"
                )
            if path.scale is not None:
                scale = _read_scale(self.target, path.scale)
                if (
                    scale is not bound.scale
                    or scale.data_ptr() != bound.scale_ptr
                    or scale.shape != (1,)
                    or scale.dtype != torch.float32
                    or scale.device != bound.device
                ):
                    raise RuntimeError(
                        "target vocabulary scale changed during draft load"
                    )


def _read_scale(owner, path):
    leaf_path, _, name = path.rpartition(".")
    leaf = owner.get_submodule(leaf_path)
    # A scale created after load may be a registered buffer, not a source
    # checkpoint parameter. Arbitrary tensor attributes are not an ownership ABI.
    if name in leaf._parameters:
        return leaf.get_parameter(name)
    return leaf.get_buffer(name)
