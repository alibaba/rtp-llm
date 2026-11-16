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
_MAX_ROWS = 32


def _aligned_buffer_like(value):
    output = torch.empty_like(value)
    if output.data_ptr() % 16 == 0:
        return output
    # A contiguous view can start at an unaligned BF16 offset. Construct an
    # aligned slice explicitly if the allocator also returns an unaligned base.
    storage = value.new_empty(value.numel() + 7)
    offset = (-storage.data_ptr() % 16) // value.element_size()
    return storage.narrow(0, offset, value.numel()).view_as(value)


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
    if library.qwen38_lsa_abi() != 3:
        raise ValueError("Qwen3.8 LSA requires ABI 3. Rebuild reduce_gate.cu.")
    library.qwen38_lsa_uid_size.restype = ctypes.c_int
    library.qwen38_lsa_unique_id.argtypes = [ctypes.c_void_p]
    library.qwen38_lsa_create.argtypes = [
        ctypes.c_int,
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

    def __init__(self, library, group, device, hidden_size):
        if dist.get_world_size(group) != 2:
            raise ValueError("Qwen3.8 LSA reduction requires a TP2 group")
        if not (
            type(hidden_size) is int
            and 0 < hidden_size <= (2**31 - 1) // _MAX_ROWS
            and hidden_size % 8 == 0
        ):
            raise ValueError(
                "LSA hidden size must be positive, divisible by 8 and fit its row capacity"
            )
        self.hidden_size = hidden_size
        self.library = library
        self.device = torch.device(device)
        self.closed = False
        self.handle = ctypes.c_void_p()
        uid = ctypes.create_string_buffer(library.qwen38_lsa_uid_size())
        state = {"uid": None, "error": None, "hidden": hidden_size}
        if dist.get_rank(group) == 0:
            try:
                self._check(library.qwen38_lsa_unique_id(uid))
                state["uid"] = bytes(uid).hex()
            except Exception as error:
                state["error"] = str(error)
        peers = exchange_group_state(group, state, "qwen38-lsa-uid")
        if (
            any(peer["error"] for peer in peers)
            or len({peer["hidden"] for peer in peers}) != 1
        ):
            raise RuntimeError(str(peers))
        uid.raw = bytes.fromhex(peers[0]["uid"])
        with torch.cuda.device(self.device):
            self._check(
                library.qwen38_lsa_create(
                    dist.get_rank(group),
                    self.device.index,
                    hidden_size,
                    uid,
                    ctypes.byref(self.handle),
                )
            )

    def _check(self, status):
        if status:
            raise RuntimeError(self.library.qwen38_lsa_error().decode())

    def supports(self, value):
        # TP consumes the same logical batch shape on both ranks. Local tensor
        # addresses and strides must never select a different collective.
        return (
            value.ndim == 2
            and 1 <= value.shape[0] <= _MAX_ROWS
            and value.shape[1] == self.hidden_size
        )

    def __call__(self, partial, shared, gate):
        if not (
            not self.closed
            and self.supports(partial)
            and partial.shape == shared.shape
            and gate.shape == (partial.shape[0], 1)
            and all(
                value.device == self.device and value.dtype == torch.bfloat16
                for value in (partial, shared, gate)
            )
        ):
            raise ValueError("LSA epilogue requires local BF16 [M,H] and gate [M,1]")
        partial = partial.contiguous()  # Native code stages it with a byte copy.
        shared = shared.contiguous()
        gate = gate.contiguous()
        if shared.data_ptr() % 16:
            aligned = _aligned_buffer_like(shared)
            aligned.copy_(shared)
            shared = aligned
        output = _aligned_buffer_like(partial)
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
    from rtp_llm.models_py.distributed import collective_torch
    from rtp_llm.models_py.distributed.all_reduce import get_moe_reduce_gate_mode

    group = collective_torch._get_group(collective_torch.Group.TP)
    agreed_mode = get_moe_reduce_gate_mode(group)
    if agreed_mode == "0" or (agreed_mode is None and mode == "0"):
        return None
    if agreed_mode != "1":
        raise RuntimeError("Initialize the TP AllReduce policy before LSA setup")

    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
        PureTpRouterNvfp4Cutlass,
    )

    state = {
        "error": None,
        "host": socket.gethostname(),
        "layers": len(model.layers),
        "hidden": config.hidden_size,
    }
    library = None
    try:
        if not (
            parallelism.ep_size == 2
            and parallelism.dp_size == parallelism.pp_size == 1
            and parallelism.local_world_size == 2
            and parallelism.get_ffn_tp_size() == 1
            and type(config.hidden_size) is int
            and 0 < config.hidden_size <= (2**31 - 1) // _MAX_ROWS
            and config.hidden_size % 8 == 0
            and getattr(config, "_qwen4_dense_storage", None) == "source_bf16"
            and not config.eplb_config.enable_eplb()
        ):
            raise ValueError(
                "LSA shared merge requires local EP2, BF16 Dense and a positive hidden size divisible by 8"
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
                and tuple(mlp.shared_expert_gate.weight.shape)
                == (1, config.hidden_size)
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
        or len({peer["hidden"] for peer in peers}) != 1
    ):
        raise RuntimeError(f"Qwen3.8 LSA startup agreement failed: {peers}")
    owner = LsaMoeReduce(
        library,
        group,
        torch.device("cuda", torch.cuda.current_device()),
        config.hidden_size,
    )
    for layer in model.layers:
        layer.mlp._reduce_shared_gate = owner
    return owner
