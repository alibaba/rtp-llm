"""SM120 cases split out of test_prefill_workspace.py.
These cases live in this sibling file, leaving the original test file
untouched."""

"""CPU-only SM120 sibling coverage for ``PrefillWorkspace`` fallback buffers."""
import torch

from rtp_llm.models_py.modules.dsv4.prefill_workspace import PrefillWorkspace


def test_sm120_fallback_buffers_reuse_capacity_and_isolate_roles():
    ws = PrefillWorkspace(
        torch.device("cpu"), q_rows=1, q_dim=1, reserve_cp=False, align_bytes=1
    )

    swa_cache = ws.sm120_packed_cache("swa", num_tokens=65, page_size=64)
    assert tuple(swa_cache.shape) == (2, 64, 584)
    assert (
        ws.sm120_packed_cache("swa", num_tokens=32, page_size=64).data_ptr()
        == swa_cache.data_ptr()
    )

    extra_cache = ws.sm120_packed_cache("extra_2", num_tokens=65, page_size=64)
    assert extra_cache.data_ptr() != swa_cache.data_ptr()

    slots = ws.sm120_linear_slots("swa", 65)
    assert tuple(slots.shape) == (65,)
    assert slots.dtype == torch.int64
    assert ws.sm120_linear_slots("swa", 32).data_ptr() == slots.data_ptr()
    assert ws.sm120_linear_slots("extra_2", 65).data_ptr() != slots.data_ptr()

    grown_cache = ws.sm120_packed_cache("swa", num_tokens=129, page_size=64)
    assert tuple(grown_cache.shape) == (3, 64, 584)
    assert (
        ws.sm120_packed_cache("swa", num_tokens=65, page_size=64).data_ptr()
        == grown_cache.data_ptr()
    )

    grown_slots = ws.sm120_linear_slots("swa", 129)
    assert tuple(grown_slots.shape) == (129,)
    assert ws.sm120_linear_slots("swa", 65).data_ptr() == grown_slots.data_ptr()


if __name__ == "__main__":
    import unittest

    unittest.main()
