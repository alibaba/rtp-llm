"""Pinned routed-only SiTU ABI; Decode keeps its BF16-shared backend."""
from functools import lru_cache
import hashlib
import importlib
import importlib.util
import json
import logging
from pathlib import Path


_COMMIT = "a6bbb8000161c0dc3a85a0300a905f76898a7913"
_EPOCH_PATCH = "3685360daa68961c64db6ecce52564fde3cff9c40feafb82095096c64de0e9ef"
_ISOLATED_HELPER = "f0718533722ab2172f3c993aee924175b95480b40901b118fd2fc175217e9e41"
_ISOLATION_FLAGS = {"-fvisibility=hidden", "-fno-gnu-unique"}


def _validate_package(package, torch_version):
    root = Path(package.__file__).resolve().parent
    info = json.loads((root / "BUILD_INFO.json").read_text())
    if (info.get("commit") != _COMMIT
            or info.get("patch_sha256") != _EPOCH_PATCH
            or info.get("isolation_helper_sha256") != _ISOLATED_HELPER
            or set(info.get("isolation_compile_flags", ())) != _ISOLATION_FLAGS
            or info.get("math_source_changed") is not False
            or info.get("torch") != torch_version):
        raise RuntimeError("Unverified routed-only SiTU package or runtime ABI")
    binaries = list(root.glob("_C*.so"))
    if len(binaries) != 1 or hashlib.sha256(binaries[0].read_bytes()).hexdigest() != info.get("binary_sha256"):
        raise RuntimeError("Routed-only SiTU binary does not match its build receipt")
    return info


class _BufferABI:
    def __init__(self, source):
        self.source = source

    def __getattr__(self, name):
        return getattr(self.source, name)

    def get_token_alignment_for_mega_moe_k3(self):
        return self.source.get_token_alignment_for_mega_moe()

    def get_symm_buffer_size_for_mega_moe_k3(self, *args):
        # The old ABI has a ninth num_shared_experts argument. This adapter
        # exclusively serves routed experts; shared BF16 experts stay separate.
        return self.source.get_symm_buffer_size_for_mega_moe(*args, 0)


class _PinnedSiTUBackend:
    def __init__(self, source):
        self.source = source
        self._C = _BufferABI(source._C)
        self.transform_weights_for_mega_moe_k3 = source.transform_weights_for_mega_moe

    def __getattr__(self, name):
        return getattr(self.source, name)

    def get_symm_buffer_for_mega_moe_k3(
        self, *, group, num_experts, num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden, mma_type="fp8xfp4", activation="situ",
    ):
        if activation != "situ" or mma_type != "fp8xfp4":
            raise ValueError("Pinned Prefill backend requires routed MXFP4 SiTU")
        return self.source.get_symm_buffer_for_mega_moe(
            group=group, num_experts=num_experts,
            num_max_tokens_per_rank=num_max_tokens_per_rank, num_topk=num_topk,
            hidden=hidden, intermediate_hidden=intermediate_hidden,
            num_shared_experts=0, mma_type=mma_type, activation=activation,
        )

    def fp8_fp4_mega_moe_k3(
        self, y, l1_weights, l2_weights, sym_buffer, *,
        cumulative_local_expert_recv_stats=None, recipe=(1, 1, 32),
        activation="situ", activation_clamp=None, activation_beta=None,
        activation_linear_beta=None, fast_math=True,
    ):
        if activation != "situ":
            raise ValueError("Pinned Prefill backend only supports SiTU")
        # Names alone are insufficient: this mapping is allowed only for the
        # source/build fingerprint checked above. Its old alpha controls gate
        # tanh; beta controls up tanh, and zero is the optional-up sentinel.
        return self.source.fp8_fp4_mega_moe(
            y=y, l1_weights=l1_weights, l2_weights=l2_weights,
            sym_buffer=sym_buffer,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
            recipe=recipe, activation=activation, activation_clamp=activation_clamp,
            fast_math=fast_math, activation_alpha=activation_beta,
            activation_beta=0.0 if activation_linear_beta is None else activation_linear_beta,
        )


@lru_cache(maxsize=1)
def get_prefill_situ_backend():
    """Prefer a verified local kernel package without replacing deep_gemm.

    An absent optional package retains the standard backend. A present but
    unverified package is an error, rather than an unnoticed precision change.
    Build/import paths and the actual selected backend are part of run evidence.
    """
    if importlib.util.find_spec("k3_native_deep_gemm") is None:
        return None
    import torch

    source = importlib.import_module("k3_native_deep_gemm")
    info = _validate_package(source, torch.__version__)
    logging.info("[MegaMoE] routed-only Prefill backend=%s commit=%s binary=%s",
                 source.__file__, info["commit"], info["binary_sha256"])
    return _PinnedSiTUBackend(source)
