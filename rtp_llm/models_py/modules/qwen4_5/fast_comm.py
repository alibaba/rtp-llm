"""NVLink5 Fast AllReduce wrapper for Qwen 4.5-Flash (TP=4 / EP=4).

Provides ``Qwen4_5FastAllReduce`` with three ordered communication backends:
  1. ``flashinfer``: ``flashinfer.comm.allreduce_fusion`` (TRT-LLM / MNNVL
     Lamport workspace, always called with ``launch_with_pdl=True`` and
     ``trigger_completion_at_end=True`` for FlashInfer 0.6.9 CUDA Graph replay
     safety). Workspace is eagerly initialized pre-capture and reused across
     ``hidden_dim`` in ``{2560, 5120}`` (up to ``max_tokens * max(hidden_dims)``).
  2. ``symm_mem``: ``torch.distributed._symmetric_memory`` (``one_shot_all_reduce``
     / ``two_shot_all_reduce_``) with eager ``symm_mem.empty`` + ``rendezvous``.
  3. ``nccl``: Clean ``torch.distributed.all_reduce`` fallback (used automatically
     on unsupported shapes/dtypes, multi-node when IPC unavailable, or large
     prefill batches exceeding the fast workspace token capacity).

All collective backend choices are agreed unanimously across all ranks in the
process group at init time so ranks never diverge between fast and NCCL paths.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _hc_combine_add_kernel(
    streams_ptr,
    h_post_ptr,
    reduced_ptr,
    out_ptr,
    M,
    H,
    stride_sm: tl.int64,
    stride_sk: tl.int64,
    stride_hm: tl.int64,
    stride_rm: tl.int64,
    stride_om: tl.int64,
    stride_ok: tl.int64,
    K_STREAMS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    m = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1)
    if m >= M:
        return
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    r = tl.load(reduced_ptr + m * stride_rm + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    s_base = streams_ptr + m * stride_sm
    h_base = h_post_ptr + m * stride_hm
    o_base = out_ptr + m * stride_om
    for k in tl.static_range(K_STREAMS):
        hp = tl.load(h_base + k).to(tl.float32)
        sv = tl.load(s_base + k * stride_sk + offs_h, mask=mask_h, other=0.0).to(tl.float32)
        ov = (sv + hp * r).to(out_ptr.dtype.element_ty)
        tl.store(o_base + k * stride_ok + offs_h, ov, mask=mask_h)


@triton.jit
def _all_gather_transpose_lastdim_kernel(
    src_ptr,
    dst_ptr,
    M,
    D,
    WS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Copy ``[WS, M, D]`` contiguous gathered ranks to ``[M, WS * D]`` in one pass."""
    m = tl.program_id(0).to(tl.int64)
    r = tl.program_id(1).to(tl.int64)
    db = tl.program_id(2)
    if m >= M or r >= WS:
        return
    offs_d = db * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    vals = tl.load(src_ptr + (r * M + m) * D + offs_d, mask=mask_d)
    tl.store(dst_ptr + (m * WS + r) * D + offs_d, vals, mask=mask_d)

_SUPPORTED_WORLD_SIZES = (2, 4, 8)
_DEFAULT_HIDDEN_DIMS = (5120, 2560)
_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_SYMM_MAX_BYTES = 32 * 1024 * 1024  # 32 MiB (~3276 tokens at H=5120 bf16)

_COMM_CACHE_LOCK = threading.Lock()
_COMM_CACHE: Dict[Tuple[Any, ...], "Qwen4_5FastAllReduce"] = {}


def _read_boot_id() -> str:
    try:
        with open("/proc/sys/kernel/random/boot_id", "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return socket.gethostname()


def _detect_single_node(group: Any) -> bool:
    """Collectively check whether every rank in ``group`` shares one host."""
    world_size = dist.get_world_size(group)
    if world_size <= 1:
        return True
    local_id = _read_boot_id()
    all_ids = [None] * world_size
    dist.all_gather_object(all_ids, local_id, group=group)
    return len(set(all_ids)) == 1


def _agree_bool(local_ok: bool, group: Any, device: torch.device) -> bool:
    """Unanimous collective agreement across ``group`` before fast-path setup."""
    flag_dev = device if device.type == "cuda" else torch.device("cpu")
    try:
        if dist.get_backend(group) == "gloo":
            flag_dev = torch.device("cpu")
    except Exception:
        pass
    flag = torch.tensor(1 if local_ok else 0, dtype=torch.int32, device=flag_dev)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item() == 1)


def _group_identity(group: Any) -> Any:
    if group is None:
        return None
    name = getattr(group, "group_name", None)
    if name is not None:
        return name
    return id(group)


class Qwen4_5FastAllReduce:
    """CUDA-Graph-safe low-latency AllReduce for Qwen 4.5-Flash (H=2560 & H=5120)."""

    def __init__(
        self,
        group: Any,
        device: Optional[torch.device] = None,
        *,
        backend: Optional[str] = None,
        max_tokens: Optional[int] = None,
        hidden_dims: Sequence[int] = _DEFAULT_HIDDEN_DIMS,
        dtype: torch.dtype = torch.bfloat16,
        eager_init: bool = True,
        single_node: Optional[bool] = None,
        flashinfer_backend: Optional[str] = None,
        use_oneshot: Optional[bool] = None,
        fp32_acc: bool = False,
    ) -> None:
        self.group = group
        self.dtype = dtype
        self.hidden_dims = tuple(
            sorted({int(h) for h in hidden_dims if int(h) > 0}, reverse=True)
        )
        if not self.hidden_dims:
            self.hidden_dims = _DEFAULT_HIDDEN_DIMS
        self.max_hidden_dim = max(self.hidden_dims)

        env_max_tokens = os.environ.get("QWEN4_5_FAST_AR_MAX_TOKENS")
        self.max_tokens = int(
            max_tokens
            if max_tokens is not None
            else (env_max_tokens or _DEFAULT_MAX_TOKENS)
        )
        self.max_elements = self.max_tokens * self.max_hidden_dim

        req_backend = (
            backend
            if backend is not None
            else os.environ.get("QWEN4_5_FAST_AR_BACKEND", "auto")
        ).lower()
        if req_backend not in ("auto", "flashinfer", "symm_mem", "nccl"):
            raise ValueError(
                f"Unsupported fast_comm backend={req_backend!r}; "
                "expected one of ('auto', 'flashinfer', 'symm_mem', 'nccl')"
            )
        self.requested_backend = req_backend

        self._fi_backend_pref = (
            flashinfer_backend
            if flashinfer_backend is not None
            else os.environ.get("QWEN4_5_FAST_AR_FLASHINFER_BACKEND", "trtllm")
        ).lower()
        self._use_oneshot = use_oneshot
        self._env_oneshot: Optional[bool] = self._parse_oneshot_env()
        twoshot_thresh_raw = os.environ.get("QWEN4_5_FAST_AR_TWOSHOT_MIN_TOKENS", "0").strip()
        self._twoshot_min_tokens = int(twoshot_thresh_raw) if twoshot_thresh_raw else 0
        self._fp32_acc = bool(fp32_acc)

        self._flashinfer_comm: Any = None
        self._fi_workspace: Any = None
        self._fi_pattern_allreduce: int = 0
        self._fi_pattern_ar_res_norm: int = 1
        self._symm_mem_mod: Any = None
        self._symm_buffer: Optional[torch.Tensor] = None
        self._symm_group_name: Optional[str] = None
        self._symm_use_oneshot: bool = True
        self._gemma_gamma_cache: Dict[int, Tuple[torch.Tensor, int, torch.Tensor]] = {}
        self._destroyed: bool = False
        self._is_gloo: bool = False

        if group is None or not dist.is_initialized():
            self.world_size = 1
            self.rank = 0
            self.device = device or (
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self.active_backend = "noop"
            self.single_node = True
            return

        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        if device is None:
            self.device = (
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        elif isinstance(device, int):
            self.device = torch.device("cuda", device)
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        if self.world_size <= 1 or self.rank < 0:
            self.active_backend = "noop"
            self.single_node = True
            return

        try:
            self._is_gloo = dist.get_backend(group) == "gloo"
        except Exception:
            self._is_gloo = False
        if self._is_gloo or self.device.type != "cuda" or self.requested_backend == "nccl":
            self.active_backend = "gloo" if self._is_gloo else "nccl"
            self.single_node = bool(single_node) if single_node is not None else True
            return

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Qwen4_5FastAllReduce must be initialized eagerly before CUDA Graph capture."
            )

        self.single_node = (
            _detect_single_node(group) if single_node is None else bool(single_node)
        )
        self.active_backend = "nccl"

        if eager_init:
            self.init_workspace()

    def init_workspace(self) -> str:
        """Eagerly allocate collective workspaces prior to CUDA Graph capture."""
        if self.world_size <= 1 or self._is_gloo or self.device.type != "cuda" or self._destroyed:
            return self.active_backend
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Cannot initialize fast_comm workspace during CUDA Graph capture."
            )
        if self.active_backend in ("flashinfer", "symm_mem"):
            return self.active_backend

        if self.requested_backend in ("auto", "flashinfer"):
            if self._try_init_flashinfer():
                self.active_backend = "flashinfer"
                return self.active_backend

        if self.requested_backend in ("auto", "symm_mem"):
            if self._try_init_symm_mem():
                self.active_backend = "symm_mem"
                return self.active_backend

        self.active_backend = "nccl"
        return self.active_backend

    def _try_init_flashinfer(self) -> bool:
        if not self.single_node or self.world_size not in _SUPPORTED_WORLD_SIZES:
            return False

        flashinfer_comm = None
        torch_dist_backend_cls = None
        local_import_ok = False
        try:
            import flashinfer.comm as flashinfer_comm
            from flashinfer.comm.mnnvl import TorchDistBackend as torch_dist_backend_cls

            if (
                hasattr(flashinfer_comm, "allreduce_fusion")
                and hasattr(flashinfer_comm, "create_allreduce_fusion_workspace")
                and hasattr(flashinfer_comm, "AllReduceFusionPattern")
            ):
                local_import_ok = True
        except Exception as exc:
            logger.info("FlashInfer comm unavailable (%s); falling back", exc)

        if not _agree_bool(local_import_ok, self.group, self.device):
            return False

        assert flashinfer_comm is not None and torch_dist_backend_cls is not None
        workspace = None
        local_ws_ok = False
        local_err: Optional[Exception] = None
        try:
            torch.cuda.set_device(self.device)
            workspace = flashinfer_comm.create_allreduce_fusion_workspace(
                backend=self._fi_backend_pref,
                world_size=self.world_size,
                rank=self.rank,
                max_token_num=self.max_tokens,
                hidden_dim=self.max_hidden_dim,
                dtype=self.dtype,
                comm_backend=torch_dist_backend_cls(group=self.group),
            )
            local_ws_ok = workspace is not None
        except Exception as exc:
            local_err = exc

        if not _agree_bool(local_ws_ok, self.group, self.device):
            if workspace is not None:
                try:
                    workspace.destroy()
                except Exception:
                    pass
            logger.warning(
                "FlashInfer workspace creation failed on at least one rank%s",
                f": {local_err}" if local_err is not None else "",
            )
            return False

        self._flashinfer_comm = flashinfer_comm
        self._fi_workspace = workspace
        self._fi_pattern_allreduce = int(
            flashinfer_comm.AllReduceFusionPattern.kAllReduce
        )
        self._fi_pattern_ar_res_norm = int(
            flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm
        )
        logger.info(
            "Initialized Qwen4_5FastAllReduce(flashinfer/%s): rank=%d/%d max_tokens=%d max_hidden=%d",
            getattr(workspace, "backend", self._fi_backend_pref),
            self.rank,
            self.world_size,
            self.max_tokens,
            self.max_hidden_dim,
        )
        return True

    def _try_init_symm_mem(self) -> bool:
        if not self.single_node or self.world_size not in _SUPPORTED_WORLD_SIZES:
            return False

        symm_mem = None
        group_name = getattr(self.group, "group_name", None)
        local_ready = False
        try:
            import torch.distributed._symmetric_memory as symm_mem

            cap_major, _ = torch.cuda.get_device_capability(self.device)
            has_ops = hasattr(torch.ops.symm_mem, "two_shot_all_reduce_") or hasattr(
                torch.ops.symm_mem, "one_shot_all_reduce"
            )
            if cap_major in (9, 10) and group_name is not None and has_ops:
                local_ready = True
        except Exception as exc:
            logger.info("torch._symmetric_memory unavailable (%s)", exc)

        if not _agree_bool(local_ready, self.group, self.device):
            return False

        assert symm_mem is not None and group_name is not None
        buf = None
        local_buf_ok = False
        local_err: Optional[Exception] = None
        try:
            torch.cuda.set_device(self.device)
            elem_size = torch.tensor([], dtype=self.dtype).element_size()
            alloc_elems = max(
                self.max_elements,
                min(_DEFAULT_SYMM_MAX_BYTES // elem_size, self.max_elements * 2),
            )
            buf = symm_mem.empty(alloc_elems, dtype=self.dtype, device=self.device)
            symm_mem.rendezvous(buf, group_name)
            local_buf_ok = True
        except Exception as exc:
            local_err = exc

        if not _agree_bool(local_buf_ok, self.group, self.device):
            logger.warning(
                "torch._symmetric_memory rendezvous failed on at least one rank%s",
                f": {local_err}" if local_err is not None else "",
            )
            return False

        self._symm_mem_mod = symm_mem
        self._symm_buffer = buf
        self._symm_group_name = str(group_name)
        self._symm_use_oneshot = hasattr(torch.ops.symm_mem, "one_shot_all_reduce")
        logger.info(
            "Initialized Qwen4_5FastAllReduce(symm_mem): rank=%d/%d elems=%d",
            self.rank,
            self.world_size,
            buf.numel(),
        )
        return True

    def should_use_flashinfer(self, tensor: torch.Tensor) -> bool:
        if self._fi_workspace is None or self._flashinfer_comm is None:
            return False
        if (
            not tensor.is_cuda
            or tensor.dtype != self.dtype
            or tensor.ndim < 2
            or not tensor.is_contiguous()
        ):
            return False
        H = int(tensor.shape[-1])
        M = int(tensor.numel() // H) if H > 0 else 0
        if M <= 0 or H <= 0 or (H % 8) != 0:
            return False
        if M * H > self.max_elements:
            return False
        if hasattr(self._fi_workspace, "is_buffer_size_sufficient"):
            return bool(
                self._fi_workspace.is_buffer_size_sufficient(
                    self.world_size, M, H, self.dtype
                )
            )
        return True

    def should_use_symm_mem(self, tensor: torch.Tensor) -> bool:
        if self._symm_buffer is None or self._symm_group_name is None:
            return False
        if (
            not tensor.is_cuda
            or tensor.dtype != self.dtype
            or tensor.ndim < 2
            or not tensor.is_contiguous()
        ):
            return False
        n = tensor.numel()
        return 0 < n <= self._symm_buffer.numel()

    @staticmethod
    def _parse_oneshot_env() -> Optional[bool]:
        raw = os.environ.get("QWEN4_5_FAST_AR_ONESHOT", "auto").strip().lower()
        if raw in ("1", "true", "oneshot", "yes"):
            return True
        if raw in ("0", "false", "twoshot", "no"):
            return False
        return None

    def _resolve_flashinfer_oneshot(self, token_num: int) -> Optional[bool]:
        use_oneshot = self._use_oneshot if self._use_oneshot is not None else self._env_oneshot
        if use_oneshot is None and self._twoshot_min_tokens > 0 and token_num >= self._twoshot_min_tokens:
            use_oneshot = False
        # FlashInfer twoshot kernel asserts token_num > world_size.
        if use_oneshot is False and token_num <= self.world_size:
            return True
        return use_oneshot

    def _run_flashinfer(self, x_2d: torch.Tensor, out_2d: torch.Tensor) -> torch.Tensor:
        M, _ = x_2d.shape
        use_oneshot = self._resolve_flashinfer_oneshot(M)
        self._flashinfer_comm.allreduce_fusion(
            input=x_2d,
            workspace=self._fi_workspace,
            pattern=self._fi_pattern_allreduce,
            launch_with_pdl=True,
            trigger_completion_at_end=True,
            output=out_2d,
            use_oneshot=use_oneshot,
            fp32_acc=self._fp32_acc,
        )
        return out_2d

    def _run_symm_mem(self, x_2d: torch.Tensor, out_2d: torch.Tensor) -> torch.Tensor:
        assert self._symm_buffer is not None and self._symm_group_name is not None
        numel = x_2d.numel()
        buf_slice = self._symm_buffer[:numel]
        buf_slice.copy_(x_2d.view(-1))
        if self._symm_use_oneshot and self._use_oneshot is not False:
            res = torch.ops.symm_mem.one_shot_all_reduce(
                buf_slice, "sum", self._symm_group_name
            )
            out_2d.view(-1).copy_(res)
        else:
            torch.ops.symm_mem.two_shot_all_reduce_(
                buf_slice, "sum", self._symm_group_name
            )
            out_2d.view(-1).copy_(buf_slice)
        return out_2d

    def all_reduce(
        self,
        x: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        inplace: bool = False,
    ) -> torch.Tensor:
        """All-reduce ``x`` across ``self.group`` (supports 2D ``[M, H]`` or 3D ``[B, T, H]``).

        Returns a tensor with the same shape and dtype as ``x``. Safe inside CUDA
        Graph capture & replay when ``x.numel() <= self.max_elements``.
        """
        if self.world_size <= 1 or self.group is None or x.numel() == 0:
            if out is not None and out.data_ptr() != x.data_ptr():
                out.copy_(x)
                return out
            return x

        orig_shape = x.shape
        H = int(orig_shape[-1])
        x_c = x if x.is_contiguous() else x.contiguous()
        x_2d = x_c.reshape(-1, H)

        if out is not None:
            if out.shape != orig_shape or out.dtype != x.dtype or out.device != x.device:
                raise ValueError(
                    f"out shape/dtype/device {out.shape}/{out.dtype}/{out.device} "
                    f"must match x {orig_shape}/{x.dtype}/{x.device}"
                )
            out_is_alias_or_noncontig = (
                (not out.is_contiguous()) or (out.data_ptr() == x_2d.data_ptr())
            )
        else:
            out_is_alias_or_noncontig = False

        if self.active_backend == "flashinfer" and self.should_use_flashinfer(x_2d):
            if out is not None and not out_is_alias_or_noncontig:
                out_2d = out.reshape(x_2d.shape)
            else:
                out_2d = torch.empty_like(x_2d)
            self._run_flashinfer(x_2d, out_2d)
            if out is not None and out_is_alias_or_noncontig:
                out.copy_(out_2d.reshape(orig_shape))
                return out
            if inplace:
                x.copy_(out_2d.reshape(orig_shape))
                return x
            return out_2d.reshape(orig_shape)

        if self.active_backend == "symm_mem" and self.should_use_symm_mem(x_2d):
            if out is not None and not out_is_alias_or_noncontig:
                out_2d = out.reshape(x_2d.shape)
            else:
                out_2d = torch.empty_like(x_2d)
            self._run_symm_mem(x_2d, out_2d)
            if out is not None and out_is_alias_or_noncontig:
                out.copy_(out_2d.reshape(orig_shape))
                return out
            if inplace:
                x.copy_(out_2d.reshape(orig_shape))
                return x
            return out_2d.reshape(orig_shape)

        # Clean NCCL / Gloo fallback (used for prefill > max_tokens or when fast backends unavailable)
        if self._is_gloo and x_2d.is_cuda:
            cpu_buf = x_2d.detach().cpu().contiguous()
            dist.all_reduce(cpu_buf, op=dist.ReduceOp.SUM, group=self.group)
            reduced_dev = cpu_buf.to(device=x_2d.device, dtype=x_2d.dtype).reshape(orig_shape)
            if out is not None:
                out.copy_(reduced_dev)
                return out
            if inplace:
                x.copy_(reduced_dev)
                return x
            return reduced_dev

        if out is not None:
            if out.data_ptr() != x_2d.data_ptr():
                out.copy_(x)
            if out.is_contiguous():
                dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.group)
                return out
            tmp = out.contiguous()
            dist.all_reduce(tmp, op=dist.ReduceOp.SUM, group=self.group)
            out.copy_(tmp)
            return out

        if inplace and x.is_contiguous():
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.group)
            return x

        res = x_2d.clone()
        dist.all_reduce(res, op=dist.ReduceOp.SUM, group=self.group)
        if inplace:
            x.copy_(res.reshape(orig_shape))
            return x
        return res.reshape(orig_shape)

    def all_reduce_(self, x: torch.Tensor) -> torch.Tensor:
        """In-place convenience wrapper around :meth:`all_reduce`."""
        return self.all_reduce(x, inplace=True)

    def __call__(
        self,
        x: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        inplace: bool = False,
    ) -> torch.Tensor:
        return self.all_reduce(x, out=out, inplace=inplace)

    def all_reduce_residual_rmsnorm(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """All-reduce ``x``, add ``residual``, and apply standard RMSNorm.

        Returns ``(norm_out, residual_out)``. Uses FlashInfer
        ``kARResidualRMSNorm`` when active and eligible; otherwise cleanly falls
        back to :meth:`all_reduce` + FP32 RMSNorm.
        """
        orig_shape = x.shape
        H = int(orig_shape[-1])
        x_c = x if x.is_contiguous() else x.contiguous()
        res_c = residual if residual.is_contiguous() else residual.contiguous()
        x_2d = x_c.reshape(-1, H)
        res_2d = res_c.reshape(-1, H)

        if (
            self.active_backend == "flashinfer"
            and self.should_use_flashinfer(x_2d)
            and res_2d.shape == x_2d.shape
            and res_2d.dtype == self.dtype
            and weight.is_cuda
            and weight.dtype == self.dtype
            and weight.shape == (H,)
        ):
            res_out = torch.empty_like(x_2d)
            norm_out = torch.empty_like(x_2d)
            use_oneshot = self._resolve_flashinfer_oneshot(x_2d.shape[0])
            self._flashinfer_comm.allreduce_fusion(
                input=x_2d,
                workspace=self._fi_workspace,
                pattern=self._fi_pattern_ar_res_norm,
                launch_with_pdl=True,
                trigger_completion_at_end=True,
                residual_out=res_out,
                norm_out=norm_out,
                residual_in=res_2d,
                rms_gamma=weight.contiguous(),
                rms_eps=float(eps),
                use_oneshot=use_oneshot,
                fp32_acc=self._fp32_acc,
            )
            return norm_out.reshape(orig_shape), res_out.reshape(orig_shape)

        reduced = self.all_reduce(x_c)
        res_out = reduced + res_c
        res_f32 = res_out.float()
        normed = (
            res_f32
            * torch.rsqrt(res_f32.pow(2).mean(dim=-1, keepdim=True) + float(eps))
            * weight.float()
        ).to(x.dtype)
        return normed, res_out

    def _resolve_gemma_gamma(
        self, weight: torch.Tensor, gamma_plus_one: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if gamma_plus_one is not None:
            return gamma_plus_one
        key = id(weight)
        ver = -1 if weight.is_inference() else int(weight._version)
        cached = self._gemma_gamma_cache.get(key)
        if cached is not None and cached[0] is weight and cached[1] == ver:
            return cached[2]
        gamma = (weight.float() + 1.0).to(self.dtype).contiguous()
        if not torch.cuda.is_current_stream_capturing():
            self._gemma_gamma_cache[key] = (weight, ver, gamma)
        return gamma

    def all_reduce_residual_gemma_rmsnorm(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        *,
        gamma_plus_one: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """All-reduce ``x``, add ``residual``, and apply GemmaRMSNorm (``1 + weight``).

        Uses FlashInfer fused ``kARResidualRMSNorm`` with cached
        ``gamma = (1 + weight)`` in BF16/FP16 when active and eligible; falls back
        to :meth:`all_reduce` + FP32 GemmaRMSNorm otherwise.
        """
        orig_shape = x.shape
        H = int(orig_shape[-1])
        x_c = x if x.is_contiguous() else x.contiguous()
        res_c = residual if residual.is_contiguous() else residual.contiguous()
        x_2d = x_c.reshape(-1, H)
        res_2d = res_c.reshape(-1, H)

        if (
            self.active_backend == "flashinfer"
            and self.should_use_flashinfer(x_2d)
            and res_2d.shape == x_2d.shape
            and res_2d.dtype == self.dtype
            and weight.is_cuda
            and weight.shape == (H,)
        ):
            gamma = self._resolve_gemma_gamma(weight, gamma_plus_one)
            if gamma.dtype == self.dtype and gamma.shape == (H,):
                res_out = torch.empty_like(x_2d)
                norm_out = torch.empty_like(x_2d)
                use_oneshot = self._resolve_flashinfer_oneshot(x_2d.shape[0])
                self._flashinfer_comm.allreduce_fusion(
                    input=x_2d,
                    workspace=self._fi_workspace,
                    pattern=self._fi_pattern_ar_res_norm,
                    launch_with_pdl=True,
                    trigger_completion_at_end=True,
                    residual_out=res_out,
                    norm_out=norm_out,
                    residual_in=res_2d,
                    rms_gamma=gamma,
                    rms_eps=float(eps),
                    use_oneshot=use_oneshot,
                    fp32_acc=self._fp32_acc,
                )
                return norm_out.reshape(orig_shape), res_out.reshape(orig_shape)

        reduced = self.all_reduce(x_c)
        res_out = reduced + res_c
        res_f32 = res_out.float()
        gamma_f32 = (
            gamma_plus_one.float()
            if gamma_plus_one is not None
            else (weight.float() + 1.0)
        )
        normed = (
            res_f32
            * torch.rsqrt(res_f32.pow(2).mean(dim=-1, keepdim=True) + float(eps))
            * gamma_f32
        ).to(x.dtype)
        return normed, res_out

    def all_reduce_hc_combine(
        self,
        branch_out: torch.Tensor,
        streams: torch.Tensor,
        h_post: torch.Tensor,
    ) -> torch.Tensor:
        """All-reduce ``branch_out`` and combine into mHC-lite ``streams``.

        Equivalent to ``streams + h_post[..., None] * all_reduce(branch_out)``.
        Uses a single-pass Triton kernel on CUDA (eliminates 3 FP32 intermediates).
        """
        reduced = self.all_reduce(branch_out)
        if (
            streams.is_cuda
            and streams.ndim >= 3
            and h_post.is_cuda
            and reduced.is_cuda
            and streams.shape[-2] <= 16
        ):
            K_streams = int(streams.shape[-2])
            H = int(streams.shape[-1])
            M = int(streams.numel() // (K_streams * H))
            if M == 0 or H == 0:
                return torch.empty_like(streams)
            s_3d = streams.reshape(M, K_streams, H)
            if s_3d.stride(-1) != 1:
                s_3d = s_3d.contiguous()
            hp_2d = h_post.reshape(M, K_streams)
            if hp_2d.stride(-1) != 1:
                hp_2d = hp_2d.contiguous()
            r_2d = reduced.reshape(M, H)
            if r_2d.stride(-1) != 1:
                r_2d = r_2d.contiguous()
            out_3d = torch.empty((M, K_streams, H), dtype=streams.dtype, device=streams.device)
            BLOCK_H = 512 if H >= 512 else triton.next_power_of_2(max(H, 64))
            _hc_combine_add_kernel[(M, triton.cdiv(H, BLOCK_H))](
                s_3d,
                hp_2d,
                r_2d,
                out_3d,
                M,
                H,
                s_3d.stride(0),
                s_3d.stride(1),
                hp_2d.stride(0),
                r_2d.stride(0),
                out_3d.stride(0),
                out_3d.stride(1),
                K_STREAMS=K_streams,
                BLOCK_H=BLOCK_H,
                num_warps=4,
            )
            return out_3d.reshape(streams.shape)
        if reduced.dim() == 2 and streams.dim() == 3:
            reduced = reduced.unsqueeze(1)
        out_f32 = streams.float() + h_post.float().unsqueeze(-1) * reduced.float()
        return out_f32.to(streams.dtype)

    def all_gather(
        self,
        x: torch.Tensor,
        *,
        dim: int = -1,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fast CUDA-Graph-safe AllGather across ``self.group`` along ``dim``.

        For ``dim == -1`` (e.g. ``lm_head`` vocab-sharded logits ``[M, V_local]``),
        runs a single ``dist.all_gather_into_tensor`` into ``[WS, M, D]`` followed
        by a single-pass Triton transpose kernel into ``[M, WS * D]`` (faster and
        lower allocation overhead than ``dist.all_gather(list) + torch.cat``).
        """
        ndim = x.ndim
        norm_dim = dim if dim >= 0 else (ndim + dim)
        if norm_dim < 0 or norm_dim >= ndim:
            raise ValueError(f"Invalid all_gather dim={dim} for tensor shape {tuple(x.shape)}")
        if self.world_size <= 1 or self.group is None or x.numel() == 0:
            if self.world_size <= 1 or self.group is None:
                if out is not None and out.data_ptr() != x.data_ptr():
                    out.copy_(x)
                    return out
                return x
            out_shape = list(x.shape)
            out_shape[norm_dim] *= self.world_size
            return out if out is not None else torch.empty(out_shape, dtype=x.dtype, device=x.device)

        x_c = x if x.is_contiguous() else x.contiguous()
        out_shape = list(x_c.shape)
        out_shape[norm_dim] *= self.world_size

        if self._is_gloo:
            cpu_in = x_c.detach().cpu() if x_c.is_cuda else x_c
            cpu_chunks = [torch.empty_like(cpu_in) for _ in range(self.world_size)]
            dist.all_gather(cpu_chunks, cpu_in, group=self.group)
            res_cpu = torch.cat(cpu_chunks, dim=norm_dim)
            res = res_cpu.to(device=x_c.device, dtype=x_c.dtype) if x_c.is_cuda else res_cpu
            if out is not None:
                out.copy_(res)
                return out
            return res

        if norm_dim == 0:
            if out is not None and out.is_contiguous() and tuple(out.shape) == tuple(out_shape):
                dist.all_gather_into_tensor(out.view(self.world_size, *x_c.shape), x_c, group=self.group)
                return out
            buf = torch.empty(out_shape, dtype=x_c.dtype, device=x_c.device)
            dist.all_gather_into_tensor(buf.view(self.world_size, *x_c.shape), x_c, group=self.group)
            if out is not None:
                out.copy_(buf)
                return out
            return buf

        if norm_dim == ndim - 1 and x_c.is_cuda:
            D = int(x_c.shape[-1])
            M = int(x_c.numel() // D)
            gathered = torch.empty((self.world_size, M, D), dtype=x_c.dtype, device=x_c.device)
            dist.all_gather_into_tensor(gathered, x_c.reshape(M, D), group=self.group)
            use_out = out is not None and out.is_contiguous() and tuple(out.shape) == tuple(out_shape)
            dst = out.reshape(M, self.world_size * D) if use_out else torch.empty(
                (M, self.world_size * D), dtype=x_c.dtype, device=x_c.device
            )
            BLOCK_D = min(1024, triton.next_power_of_2(max(D, 32)))
            _all_gather_transpose_lastdim_kernel[(M, self.world_size, triton.cdiv(D, BLOCK_D))](
                gathered,
                dst,
                M,
                D,
                WS=self.world_size,
                BLOCK_D=BLOCK_D,
                num_warps=4,
            )
            res = dst.reshape(out_shape)
            if out is not None and not use_out:
                out.copy_(res)
                return out
            return res

        gathered = torch.empty((self.world_size, *x_c.shape), dtype=x_c.dtype, device=x_c.device)
        dist.all_gather_into_tensor(gathered, x_c, group=self.group)
        perm = list(range(1, norm_dim + 1)) + [0] + list(range(norm_dim + 1, ndim + 1))
        res = gathered.permute(*perm).contiguous().reshape(out_shape)
        if out is not None:
            out.copy_(res)
            return out
        return res

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        if self._fi_workspace is not None:
            try:
                self._fi_workspace.destroy()
            except Exception:
                pass
            self._fi_workspace = None
        self._symm_buffer = None
        self._gemma_gamma_cache.clear()
        self.active_backend = "nccl" if self.world_size > 1 else "noop"


def get_qwen4_5_fast_all_reduce(
    group: Any,
    device: Optional[torch.device] = None,
    *,
    backend: Optional[str] = None,
    max_tokens: Optional[int] = None,
    hidden_dims: Sequence[int] = _DEFAULT_HIDDEN_DIMS,
    dtype: torch.dtype = torch.bfloat16,
    single_node: Optional[bool] = None,
    flashinfer_backend: Optional[str] = None,
    use_oneshot: Optional[bool] = None,
    fp32_acc: bool = False,
) -> Qwen4_5FastAllReduce:
    """Return a cached, workspace-shared :class:`Qwen4_5FastAllReduce` for ``group``."""
    if group is None or not dist.is_initialized() or dist.get_world_size(group) <= 1:
        return Qwen4_5FastAllReduce(
            group,
            device=device,
            backend=backend,
            max_tokens=max_tokens,
            hidden_dims=hidden_dims,
            dtype=dtype,
            single_node=single_node,
            flashinfer_backend=flashinfer_backend,
            use_oneshot=use_oneshot,
            fp32_acc=fp32_acc,
        )

    if isinstance(device, int):
        dev_obj = torch.device("cuda", device)
    elif isinstance(device, str):
        dev_obj = torch.device(device)
    elif device is not None:
        dev_obj = device
    elif torch.cuda.is_available():
        dev_obj = torch.device("cuda", torch.cuda.current_device())
    else:
        dev_obj = torch.device("cpu")

    req_backend = (
        backend
        if backend is not None
        else os.environ.get("QWEN4_5_FAST_AR_BACKEND", "auto")
    ).lower()
    env_max_tokens = os.environ.get("QWEN4_5_FAST_AR_MAX_TOKENS")
    eff_max_tokens = int(
        max_tokens if max_tokens is not None else (env_max_tokens or _DEFAULT_MAX_TOKENS)
    )
    fi_pref = (
        flashinfer_backend
        if flashinfer_backend is not None
        else os.environ.get("QWEN4_5_FAST_AR_FLASHINFER_BACKEND", "trtllm")
    ).lower()
    max_hidden = max([int(h) for h in hidden_dims if int(h) > 0] or [5120])
    key = (
        _group_identity(group),
        dist.get_rank(group),
        dev_obj.type,
        dev_obj.index,
        req_backend,
        eff_max_tokens,
        max_hidden,
        dtype,
        fi_pref,
        use_oneshot,
        bool(fp32_acc),
    )
    with _COMM_CACHE_LOCK:
        cached = _COMM_CACHE.get(key)
        if cached is not None and not cached._destroyed:
            return cached
        inst = Qwen4_5FastAllReduce(
            group,
            device=dev_obj,
            backend=req_backend,
            max_tokens=eff_max_tokens,
            hidden_dims=hidden_dims,
            dtype=dtype,
            eager_init=True,
            single_node=single_node,
            flashinfer_backend=fi_pref,
            use_oneshot=use_oneshot,
            fp32_acc=fp32_acc,
        )
        _COMM_CACHE[key] = inst
        return inst


def destroy_qwen4_5_fast_all_reduce() -> None:
    """Destroy all cached :class:`Qwen4_5FastAllReduce` workspaces."""
    with _COMM_CACHE_LOCK:
        for inst in _COMM_CACHE.values():
            inst.destroy()
        _COMM_CACHE.clear()


__all__ = [
    "Qwen4_5FastAllReduce",
    "destroy_qwen4_5_fast_all_reduce",
    "get_qwen4_5_fast_all_reduce",
]
