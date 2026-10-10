"""Test-only planar-cache adapter to the maintained TileLang sparse provider.

The held-input gate requires a finite valid key in every first SWA tile. General
masked rows and runtime provider selection need separate qualification.
"""

import torch

from rtp_llm.models_py.modules.dsv4 import tilelang_kernels


def workspace_decode(query, sinks, caches, plans):
    if not tilelang_kernels.tilelang_available():
        raise RuntimeError("The maintained TileLang sparse provider is unavailable")
    if (
        not query.is_cuda
        or torch.cuda.get_device_capability(query.device) != (12, 0)
        or query.ndim != 3
        or query.shape[0] < 1
        or query.shape[1:] != (64, 512)
        or query.dtype != torch.bfloat16
        or not query.is_contiguous()
        or sinks.shape != (64,)
        or sinks.dtype != torch.float32
        or sinks.device != query.device
        or not sinks.is_contiguous()
        or set(caches) != set(plans)
        or set(caches) not in ({"swa"}, {"swa", "global"})
    ):
        raise ValueError("Unsupported held SM120 TileLang decode geometry")
    rows, segments, slots = query.shape[0], [], []
    for name in ("swa", "global"):
        if name not in caches:
            continue
        cache, plan = caches[name], plans[name]
        width = 128 if name == "swa" else 512
        tags = ("swa_kv",) if name == "swa" else ("csa_kv", "hca_kv")
        if (
            cache.tag not in tags
            or cache.gen_num_per_cycle != 0
            or cache.raw.device != query.device
            or plan.device != query.device
            or plan.dtype != torch.int32
            or plan.shape != (rows, width)
            or not plan.is_contiguous()
        ):
            raise ValueError("Unsupported held SM120 TileLang cache/plan")
        segments.append(cache.read(plan.reshape(-1)).view(rows, width, 512))
        slots.append(plan)
    workspace = torch.cat(segments, dim=1).contiguous()
    live = torch.cat(slots, dim=1) >= 0
    positions = torch.arange(live.shape[1], dtype=torch.int32, device=query.device)
    indices = torch.where(live, positions[None], -1).unsqueeze(1).contiguous()
    output = torch.cat(
        [
            tilelang_kernels.sparse_attn(
                query[:, start : start + 8].reshape(rows, 1, 8, 512).contiguous(),
                workspace,
                sinks[start : start + 8].contiguous(),
                indices,
                512**-0.5,
            )
            for start in range(0, 64, 8)
        ],
        dim=2,
    ).reshape_as(query)
    return output.contiguous(), workspace, indices
