"""Model-owned TP2 LSA reduction with the shared expert epilogue."""

import ctypes
import hashlib
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed.collective_utils import exchange_group_state

# Captured graph nodes contain native context addresses. Retain them if model
# close occurs on only one rank or a graph runner still references the model.
_PROCESS_LIFETIME_CONTEXTS = []


def load_library(path, expected_sha256):
    path = Path(path)
    if (
        not expected_sha256
        or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256
    ):
        raise ValueError("Qwen3.8 LSA library SHA256 mismatch")
    library = ctypes.CDLL(str(path.resolve()))
    library.qwen38_lsa_error.restype = ctypes.c_char_p
    library.qwen38_lsa_abi.restype = ctypes.c_int
    if library.qwen38_lsa_abi() != 2:
        raise ValueError("Qwen3.8 LSA requires ABI 2. Rebuild reduce_gate.cu.")
    library.qwen38_lsa_uid_size.restype = ctypes.c_int
    library.qwen38_lsa_unique_id.argtypes = [ctypes.c_void_p]
    library.qwen38_lsa_create.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.qwen38_lsa_apply.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    library.qwen38_lsa_destroy.argtypes = [ctypes.c_void_p]
    library.qwen38_lsa_destroy.restype = None
    return library


class LsaMoeReduce:
    """One symmetric window shared by serialized forwards of a single model."""

    def __init__(self, library, group, device):
        if dist.get_world_size(group) != 2:
            raise ValueError("Qwen3.8 LSA reduction requires a TP2 group")
        self.library = library
        self.device = torch.device(device)
        self.closed = False
        self.handle = ctypes.c_void_p()
        uid = ctypes.create_string_buffer(library.qwen38_lsa_uid_size())
        state = {"uid": None, "error": None}
        if dist.get_rank(group) == 0:
            try:
                self._check(library.qwen38_lsa_unique_id(uid))
                state["uid"] = bytes(uid).hex()
            except Exception as error:
                state["error"] = str(error)
        peers = exchange_group_state(group, state, "qwen38-lsa-uid")
        if any(peer["error"] for peer in peers):
            raise RuntimeError(str(peers))
        uid.raw = bytes.fromhex(peers[0]["uid"])
        with torch.cuda.device(self.device):
            self._check(
                library.qwen38_lsa_create(
                    dist.get_rank(group),
                    self.device.index,
                    uid,
                    ctypes.byref(self.handle),
                )
            )

    def _check(self, status):
        if status:
            raise RuntimeError(self.library.qwen38_lsa_error().decode())

    def supports(self, value):
        return (
            not self.closed
            and value.is_cuda
            and value.device == self.device
            and value.dtype == torch.bfloat16
            and value.ndim == 2
            and 1 <= value.shape[0] <= 32
            and value.shape[1] == 2560
            and value.is_contiguous()
            and value.data_ptr() % 16 == 0
        )

    def __call__(self, partial, shared, gate):
        if not (
            self.supports(partial)
            and self.supports(shared)
            and partial.shape == shared.shape
            and gate.device == self.device
            and gate.dtype == torch.bfloat16
            and gate.is_contiguous()
            and gate.shape == (partial.shape[0], 1)
        ):
            raise ValueError("LSA epilogue requires BF16 [M,2560] and gate [M,1]")
        output = torch.empty_like(partial)
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            self._check(
                self.library.qwen38_lsa_apply(
                    self.handle,
                    partial.data_ptr(),
                    shared.data_ptr(),
                    gate.data_ptr(),
                    output.data_ptr(),
                    partial.shape[0],
                    stream.cuda_stream,
                )
            )
        return output

    def close(self):
        if self.closed:
            return
        self.closed = True
        _PROCESS_LIFETIME_CONTEXTS.append(self)


def prepare_model_reduce(model, config, parallelism):
    """Agree on the mode and layout before constructing a device communicator."""
    from rtp_llm.models_py.distributed import collective_torch
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
        PureTpRouterNvfp4Cutlass,
    )

    role = str(parallelism.role_type).rsplit(".", 1)[-1]
    if getattr(config, "is_mtp", False) or role != "DECODE":
        return None
    mode = os.getenv("RTP_QWEN4_MOE_REDUCE_GATE_FUSION", "0")
    if parallelism.tp_size != 2:
        if mode != "0":
            raise ValueError("Qwen3.8 LSA shared merge requires TP2")
        return None
    if not dist.is_initialized():
        if mode == "0":
            return None
        raise RuntimeError("Initialize distributed execution before LSA setup")
    group = collective_torch._get_group(collective_torch.Group.TP)
    modes = exchange_group_state(group, mode, "qwen38-lsa-mode")
    if all(value == "0" for value in modes):
        return None
    if modes != ["1", "1"]:
        raise ValueError("RTP_QWEN4_MOE_REDUCE_GATE_FUSION must agree on TP ranks")
    state = {"error": None, "host": socket.gethostname(), "layers": len(model.layers)}
    library = None
    try:
        if not (
            parallelism.ep_size == 2
            and parallelism.dp_size == parallelism.pp_size == 1
            and parallelism.local_world_size == 2
            and parallelism.get_ffn_tp_size() == 1
            and config.hidden_size == 2560
            and getattr(config, "_qwen4_dense_storage", None) == "source_bf16"
            and not config.eplb_config.enable_eplb()
        ):
            raise ValueError(
                "LSA shared merge requires local EP2, BF16 Dense and H2560"
            )
        for layer in model.layers:
            mlp = layer.mlp
            if not (
                type(mlp.fused_moe.router) is PureTpRouterNvfp4Cutlass
                and mlp.shared_expert is not None
                and mlp.shared_expert_gate is not None
                and not mlp.use_ep_shared_allreduce
                and not mlp.use_unified_tp_allreduce
                and getattr(mlp, "_shared_stream", None) is None
                and mlp.shared_expert_gate.weight.is_cuda
                and mlp.shared_expert_gate.weight.dtype == torch.bfloat16
            ):
                raise ValueError(
                    "Unsupported shared-expert or routed collective layout"
                )
        library = load_library(
            os.environ["RTP_QWEN4_MOE_LSA_LIBRARY"],
            os.environ["RTP_QWEN4_MOE_LSA_LIBRARY_SHA256"],
        )
    except Exception as error:
        state["error"] = str(error)
    peers = exchange_group_state(group, state, "qwen38-lsa-layout")
    if (
        any(peer["error"] for peer in peers)
        or len({peer["host"] for peer in peers}) != 1
        or len({peer["layers"] for peer in peers}) != 1
    ):
        raise RuntimeError(f"Qwen3.8 LSA startup agreement failed: {peers}")
    owner = LsaMoeReduce(
        library, group, torch.device("cuda", torch.cuda.current_device())
    )
    for layer in model.layers:
        layer.mlp._reduce_shared_gate = owner
    return owner
