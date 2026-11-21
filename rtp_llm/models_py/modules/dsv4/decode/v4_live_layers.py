"""Bounded device-only witness for the ordinary V4 ARM DSpARK fixture."""
import os

import torch

TARGET = "//rtp_llm/test/smoke:smoke_v4_flash_0731_pd_cp2ep2_dp2ep2_dspark_cprr_async_xgrammar_json_sm100"
CAP = 16


def allocate(v4, device):
    if (os.environ.get("TEST_TARGET") != TARGET
            or not os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
            or len(v4.layers) != 43
            or tuple(v4.capture_aux_hidden_layer_ids) != (40, 41, 42)):
        return None
    # Shared by eager and all captured batch buckets, not allocated per graph.
    # [embed + 43 layers, bounded verify row, raw mHC, hidden dim] = 22 MiB.
    return (torch.empty((44, CAP, 4, 4096), dtype=torch.bfloat16, device=device),
            torch.full((CAP, 2), -1, dtype=torch.int64, device=device))


def begin(state, ids, width):
    # No host/device scalar extraction or Python callbacks at graph replay.
    n = min(ids.numel(), CAP)
    state[1].fill_(-1)
    state[1][:n, 0].copy_(ids[:n])
    state[1][:n, 1].fill_(width)


def put(state, index, h):
    if h.dtype != torch.bfloat16 or h.shape[-2:] != (4, 4096):
        raise ValueError("unexpected V4 verify hidden shape")
    rows = h.flatten(0, 1)[:CAP]
    state[0][index, :rows.size(0)].copy_(rows)


def selected(state, row, width, last_id):
    # Called outside capture, only for the matched and committed #17 row.
    if state is None or not 0 <= row < CAP:
        return None
    if state[1][row].cpu().tolist() != [last_id, width]:
        return None
    return state[0][:, row]
