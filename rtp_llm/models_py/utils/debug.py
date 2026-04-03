import os
import logging

_logger = logging.getLogger(__name__)

# === Precision Debug Tensor Dump Utility ===
# Controlled by environment variables:
#   RTP_DUMP_TENSOR=1              — enable tensor dumping
#   RTP_DUMP_TENSOR_DIR=/path      — directory to save .pt files
#   RTP_DUMP_TENSOR_LAYER=0,1,2    — comma-separated layer indices (empty = all)
#   RTP_DUMP_TENSOR_STEPS=0,1      — comma-separated step indices (default: 0 = prefill only)
#   RTP_DUMP_TENSOR_RANK=0         — only dump from this TP rank (default: 0)
#
# Step numbering: step 0 = first forward (prefill), step 1 = first decode, etc.
# Counter increments at the END of each forward (dump_tensor_step_end).

_DUMP_ENABLED: bool = os.environ.get("RTP_DUMP_TENSOR", "0") == "1"
_DUMP_DIR: str = os.environ.get("RTP_DUMP_TENSOR_DIR", "/tmp/rtp_dump_tensors")
_DUMP_LAYERS: set = (
    set(int(x) for x in os.environ.get("RTP_DUMP_TENSOR_LAYER", "").split(",") if x.strip())
    if os.environ.get("RTP_DUMP_TENSOR_LAYER", "")
    else set()
)
_DUMP_STEPS: set = (
    set(int(x) for x in os.environ.get("RTP_DUMP_TENSOR_STEPS", "0").split(",") if x.strip())
    if os.environ.get("RTP_DUMP_TENSOR_STEPS", "0")
    else {0}
)
_DUMP_RANK: int = int(os.environ.get("RTP_DUMP_TENSOR_RANK", "0"))
_dump_step_counter: int = 0
_rank_checked: bool = False
_rank_ok: bool = True


def dump_tensor_enabled() -> bool:
    return _DUMP_ENABLED


def dump_tensor_step_begin():
    pass


def dump_tensor_step_end():
    global _dump_step_counter
    if not _DUMP_ENABLED:
        return
    _dump_step_counter += 1


def _should_dump_layer(layer_idx: int) -> bool:
    if not _DUMP_LAYERS:
        return True
    return layer_idx in _DUMP_LAYERS


def _should_dump_step() -> bool:
    return _dump_step_counter in _DUMP_STEPS


def _check_rank() -> bool:
    global _rank_checked, _rank_ok
    if _rank_checked:
        return _rank_ok
    _rank_checked = True
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            _rank_ok = dist.get_rank() == _DUMP_RANK
        else:
            _rank_ok = True
    except Exception:
        _rank_ok = True
    if _rank_ok:
        _logger.info(f"[DUMP] Tensor dump active on rank {_DUMP_RANK}")
    return _rank_ok


def dump_tensor(
    tensor: 'torch.Tensor',
    name: str,
    layer_idx: int = -1,
    save_file: bool = True,
):
    if not _DUMP_ENABLED:
        return
    if not _check_rank():
        return
    if not _should_dump_step():
        return
    if layer_idx >= 0 and not _should_dump_layer(layer_idx):
        return

    import torch

    t = tensor.detach().float()
    stats = (
        f"[DUMP] step={_dump_step_counter} {name}: "
        f"shape={list(tensor.shape)}, dtype={tensor.dtype}, "
        f"mean={t.mean().item():.6f}, std={t.std().item():.6f}, "
        f"min={t.min().item():.6f}, max={t.max().item():.6f}, "
        f"abs_mean={t.abs().mean().item():.6f}, "
        f"has_nan={tensor.isnan().any().item()}, "
        f"has_inf={tensor.isinf().any().item()}"
    )
    _logger.info(stats)
    print(stats, flush=True)

    if save_file:
        os.makedirs(_DUMP_DIR, exist_ok=True)
        safe_name = name.replace("/", "_").replace(" ", "_")
        file_path = os.path.join(_DUMP_DIR, f"step{_dump_step_counter}_{safe_name}.pt")
        torch.save(tensor.detach().cpu(), file_path)


def set_trace_on_tty():
    """
    启动一个连接到当前终端的 PDB 会话。
    在 Unix-like 系统上工作。
    """
    try:
        import pdb

        tty_r = open("/dev/tty", "r")
        tty_w = open("/dev/tty", "w")
        pdb.Pdb(stdin=tty_r, stdout=tty_w).set_trace()
    except OSError as e:
        print(f"Warning: Could not open /dev/tty: {e}. Skipping pdb.")
        import traceback

        traceback.print_exc()


def remote_debug_breakpoint(host="localhost", port=4444):
    """
    启动一个远程 PDB 会话，监听指定的主机和端口。
    使用 telnet 连接到该主机和端口以进行调试。
    """
    import debugpy

    debugpy.listen((host, port))
    print("Waiting for debugger attach...")
    debugpy.wait_for_client()
    debugpy.breakpoint()


import torch


def cudagraph_debug_kernel(
    data: torch.Tensor | None,
    info_id: int = 1,
    m: int = 0,
    n: int = 0,
    start_row: int = 0,
    start_col: int = 0,
    row_len: int = 0,
    name: str = "cudagraph_debug_kernel",
):
    if data is None:
        return
    print(f"{name} shape is {data.shape}")
    if data.dim() == 1:
        data = data.unsqueeze(0)
    data = data.contiguous().to(torch.float32)
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    row_len = data.size(1) if row_len == 0 else row_len
    n = data.size(1) if (n == 0 or n > data.size(1)) else n
    m = data.size(0) if (m == 0 or m > data.size(0)) else m
    rtp_llm_ops.debug_kernel(
        data=data,
        start_row=start_row,
        start_col=start_col,
        m=m,
        n=n,
        row_len=row_len,  # 每行的长度
        info_id=info_id,
    )
