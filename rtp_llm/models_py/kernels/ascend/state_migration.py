"""Paged state 的按行条件迁移（precopy 优化）。

朴素实现是无条件全量 ``index_copy_``（正确性优先）：每步每层拷贝 batch 个整页
（recurrent ssm_state 32x128x128 fp32 = 2MB/页），而绝大多数行是自拷贝 no-op
——只有恰跨 ``seq_size_per_block`` 边界的序列才真正需要迁移。带宽与 ACL Graph
捕获期驻留（30 层 x batch x 2MB 的 index_select 中间分配）都被浪费。

本模块用 triton kernel 在 kernel 内部跳过 ``src == dst`` 的行（vllm-ascend
``precopy.py`` 同思路）：只有真正跨界的行才读写整页。

图捕获安全：grid 只依赖 batch（bucket 内静态）与 constexpr 的 TILES；kernel
内部对 device 索引做条件跳过——无 nonzero / 布尔掩码索引（那是动态 shape，
捕获会失败或把捕获时刻的 shape 烤死，replay 时需要迁移的行集合已变）。

调用方约定（_cross_page_copy_device / _seed_first_write_pages_device）：
"无需迁移"的行已被 ``torch.where(needs, read, write)`` 做成 ``src == dst``，
因此 kernel 只需一个相等判断即可覆盖自拷贝 / 哨兵（-1 或保留页 0）/ 越界
三种 no-op 情形。

回退路径（CPU tensor / 无 triton / 环境变量 ``RTP_LLM_GDN_PRECOPY=0``）：
等价的无条件 ``index_copy_`` 实现（哨兵行 clamp 到保留页 0 后自拷贝）。
"""

from __future__ import annotations

import os
from typing import Optional

import torch

# 每个内层拷贝步的元素数
_BLOCK = 4096
# 每个 tile 的目标字节数：大行（recurrent 2MB）切多 tile 并行，小行（conv ~48KB）
# 单 tile；上限 32 防止 grid 爆炸
_TILE_BYTES = 256 * 1024
_MAX_TILES = 32

_kernel_cache: Optional[object] = None
_precopy_enabled_cache: Optional[bool] = None


def _precopy_enabled() -> bool:
    """triton precopy 开关（默认开；RTP_LLM_GDN_PRECOPY=0 回退到全量拷贝路径）。"""
    global _precopy_enabled_cache
    if _precopy_enabled_cache is None:
        _precopy_enabled_cache = (
            os.environ.get("RTP_LLM_GDN_PRECOPY", "1").lower() not in ("0", "false")
        )
    return _precopy_enabled_cache


def _get_migrate_kernel():
    """惰性构造 triton kernel（CPU 环境不 import triton）。"""
    global _kernel_cache
    if _kernel_cache is not None:
        return _kernel_cache
    try:
        import triton
        import triton.language as tl
    except ImportError:  # pragma: no cover - depends on the NPU image
        _kernel_cache = False
        return False

    # @triton.jit 的 kernel 体在编译期从模块全局解析自由变量（看不到工厂
    # 函数的局部作用域），必须把 tl 注入模块全局后再定义 kernel。
    globals()["tl"] = tl

    @triton.jit
    def _migrate_rows_kernel(
        state_ptr,      # state 池的 typed 指针（按元素拷贝，dtype 无关）
        src_ptr,        # int32 [rows]：源物理页
        dst_ptr,        # int32 [rows]：目标物理页
        page_stride,    # 每页 dim0 stride（元素数，pool 视图可为 padded）
        row_elems,      # 单页内层元素数（内层连续）
        BLOCK: tl.constexpr,
        TILES: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        src = tl.load(src_ptr + row).to(tl.int64)
        dst = tl.load(dst_ptr + row).to(tl.int64)
        # no-op 行：自拷贝（调用方 where 约定）/ 哨兵。dst < 0 仅与 src == dst
        # 同时出现（needs 要求 write != pad），故无需单独判断。
        if src == dst or src < 0:
            return
        # 大行切 tile 并行；tile 偏移与页偏移都用 int64（真实 pool 的
        # per-page 元素偏移可达 1e8 级，避免 32 位溢出）
        work = tl.cdiv(row_elems, TILES)
        start = tile.to(tl.int64) * work
        end = tl.minimum(start + work, row_elems.to(tl.int64))
        src_row = state_ptr + src * page_stride
        dst_row = state_ptr + dst * page_stride
        offs = tl.arange(0, BLOCK)
        for off in range(start, end, BLOCK):
            mask = off + offs < end
            tl.store(
                dst_row + off + offs,
                tl.load(src_row + off + offs, mask=mask),
                mask=mask,
            )

    _kernel_cache = _migrate_rows_kernel
    return _kernel_cache


def migrate_state_rows(
    state: torch.Tensor,
    src_pages: torch.Tensor,
    dst_pages: torch.Tensor,
) -> None:
    """把 ``state`` 的 ``src_pages[i]`` 页整行拷到 ``dst_pages[i]`` 页（原地）。

    * ``src_pages[i] == dst_pages[i]``（或 src 为负哨兵）的行是 no-op——覆盖
      自拷贝 / 哨兵 / 未分配三种"无需迁移"情形（调用方约定）；
    * ``state`` 形如 ``(pages, ...)``，要求 **dim0 之后内层连续**（生产视图均
      满足：conv 的 FLA 视图与 ssm 的 fp32 视图），dim0 stride 任意（hybrid
      pool 上的 padded 视图 OK）；
    * ``src_pages`` / ``dst_pages`` 形如 ``(batch,)``，int32/int64 tensor
      （图捕获安全：grid 只依赖 batch 与静态 tile 数，索引内容 kernel 内读取）。
    """
    rows = src_pages.shape[0]
    if rows == 0 or state.shape[0] == 0:
        return
    pages = state.shape[0]
    row_elems = state.numel() // pages
    if row_elems == 0:
        return
    page_stride = state.stride(0)

    if _precopy_enabled() and state.device.type == "npu":
        kernel = _get_migrate_kernel()
        if kernel is not False:
            row_bytes = row_elems * state.element_size()
            tiles = max(1, min(_MAX_TILES, row_bytes // _TILE_BYTES))
            kernel[(rows, tiles)](
                state,
                src_pages.to(torch.int32),
                dst_pages.to(torch.int32),
                page_stride,
                row_elems,
                BLOCK=_BLOCK,
                TILES=tiles,
            )
            return

    # 回退：无条件 index_copy_ 全量拷贝（CPU / 无 triton / RTP_LLM_GDN_PRECOPY=0）。
    # 哨兵行 clamp 到保留页 0 后自拷贝，语义与 kernel 等价（调用方约定下
    # src<0 必然 src==dst）。
    src = src_pages.to(torch.int64).clamp(min=0)
    dst = dst_pages.to(torch.int64).clamp(min=0)
    state.index_copy_(0, dst, state.index_select(0, src))
