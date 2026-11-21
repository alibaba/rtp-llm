"""Read-only capture of the actual restored allocation before model writes."""

import os
from pathlib import Path

import torch
import torch.distributed as dist

from rtp_llm.models_py.modules.dsv4.kv_cache_utils import as_attention_inputs_by_tag
from rtp_llm.utils.hot_hook_runtime import HookContext


def capture_local_restore(ctx: HookContext, scope: str = "target") -> None:
    """Record rank-local restored rows without retaining production block IDs."""
    if os.environ.get("T07_CACHE_PROBE_ROLE") != "prefill":
        return
    resume = int(ctx.kwargs["attn_inputs"].prefix_lengths.reshape(-1)[0])
    if resume <= 0:
        return
    cache = ctx.args[1]
    from rtp_llm.test.smoke.deepseek_v4_tail_reuse_probe import _end_position

    end = _end_position(ctx)
    inputs = as_attention_inputs_by_tag(ctx.kwargs.get("attention_inputs"), cache)
    rows = {}
    for layer in range(int(cache.layer_count)):
        for group in cache.get_layer_cache_groups(layer):
            tagged = inputs.get(str(group.tag))
            if tagged is None:
                continue
            table = tagged.kv_cache_kernel_block_id_device
            ids = torch.unique(table.reshape(-1).long(), sorted=True)
            ids = ids[(ids > 0) & (ids < group.kv_cache_base.shape[0])]
            row = {
                "ids": ids.cpu(),
                "runtime_table": table.detach().cpu().clone(),
                "values": group.kv_cache_base.index_select(0, ids).detach().cpu(),
            }
            if group.kv_scale_base is not None and group.kv_scale_base.numel():
                row["scales"] = group.kv_scale_base.index_select(0, ids).detach().cpu()
            rows[(layer, str(group.tag))] = row
    rank = dist.get_rank() if dist.is_initialized() else 0
    directory = Path(os.environ["T07_CACHE_PROBE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(rows, directory / f"restored_{scope}_rank{rank}_g{resume}_n{end}.pt")
