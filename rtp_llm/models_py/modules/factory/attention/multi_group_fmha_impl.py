import contextlib
from typing import Dict, Generator, Optional

import torch

from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs


@contextlib.contextmanager
def modify_attn_inputs_by_gid(
    attn_inputs: PyAttentionInputs, gid: int
) -> Generator[PyAttentionInputs, None, None]:
    """Temporarily swap kv_cache_block_id_host/device in attn_inputs to the tensors of group gid.

    The context manager restores the original values on exit, preventing state pollution
    when constructing multiple per-group fmha_impl instances.
    If the by_group list is empty or gid is out of range, yield without modification.
    """
    by_group_host = attn_inputs.kv_cache_block_id_host_by_group
    by_group_device = attn_inputs.kv_cache_block_id_device_by_group

    orig_host = attn_inputs.kv_cache_block_id_host
    orig_device = attn_inputs.kv_cache_block_id_device
    attn_inputs.kv_cache_block_id_host = by_group_host[gid]
    attn_inputs.kv_cache_block_id_device = by_group_device[gid]
    try:
        yield
    finally:
        attn_inputs.kv_cache_block_id_host = orig_host
        attn_inputs.kv_cache_block_id_device = orig_device


class MultiGroupFMHAImpl(FMHAImplBase):
    """Transparent proxy for hybrid-attention models with multiple full-attention groups.

    Holds an independent fmha_impl per full-attention group and switches the active
    group via activate_group().

    CUDA Graph compatibility:
    - activate_group() is a pure Python operation and executes normally during graph
      replay (it is not captured by CUDA Graph);
    - GPU kernels inside forward() use each group's own fmha_params GPU buffer (captured);
    - prepare_cuda_graph() updates every group's GPU buffer in-place before replay.

    When activate_group(gid) is called for a non-full-attention layer (e.g. SSM / linear
    attention), gid will not be present in _impls and the call is silently ignored,
    keeping the current active group unchanged.
    """

    def __init__(self, impls: Dict[int, FMHAImplBase]) -> None:
        self._impls: Dict[int, FMHAImplBase] = impls
        self._active_gid: int = next(iter(impls))

    def activate_group(self, gid: int) -> None:
        """Switch the active group. Silently ignored if gid is not a full-attention group."""
        if gid in self._impls:
            self._active_gid = gid

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        return self._impls[self._active_gid].forward(qkv, kv_cache)

    @property
    def fmha_params(self):
        return self._impls[self._active_gid].fmha_params

    def support_cuda_graph(self) -> bool:
        return all(impl.support_cuda_graph() for impl in self._impls.values())

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs) -> None:
        """Called before CUDA Graph replay: fill each group's GPU buffer with its block_ids."""
        for gid, impl in self._impls.items():
            with modify_attn_inputs_by_gid(attn_inputs, gid):
                impl.prepare_cuda_graph(attn_inputs)

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        return True
