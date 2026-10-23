"""Maintained attention validation, including fatal assertions in child contexts."""

import os
import unittest
from unittest.mock import patch

import torch
import torch.multiprocessing as mp

from rtp_llm.models_py.modules.dsv4.fp8.attention import bind_attn_cache
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_attention_owner_sm120 import (
    NativeTaggedCache,
    make_layer,
)


def fixture():
    layer = make_layer(2, 2)
    cache = NativeTaggedCache([layer], pages=8)
    tables = {
        tag: torch.tensor(
            [[1], [2]] if tag in ("swa_kv", "csa_state") else [[1, 0, 8], [2, 3, -1]],
            device="cuda",
            dtype=torch.int32,
        )
        for _, tag in cache.groups
    }
    lengths = torch.ones(2, device="cuda", dtype=torch.int32)
    prefixes = torch.tensor([255, 256], device="cuda", dtype=torch.int32)
    return layer, cache, tables, lengths, prefixes


def invalid_child(_rank, case):
    os.environ["DSV41_ATTN_ASYNC_CHECKS"] = "1"
    if case == "packet":
        from rtp_llm.models_py.modules.dsv4.fp8._v41_compact_cp_plan import (
            CompactCPPlan,
        )
        from rtp_llm.models_py.modules.dsv4.fp8._v41_compact_cp_transport import (
            pack,
            prepare_metadata,
            unpack,
        )
        from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_compact_cp_bytes_sm120 import (
            operands,
        )

        plans = [
            CompactCPPlan("packed", 1, 0, 1, 0, 0, 0, rank, 2, 2) for rank in range(4)
        ]
        metadata = [prepare_metadata(plan, "cuda") for plan in plans]
        packets = []
        for plan, meta in zip(plans, metadata):
            main, index, tail = operands(plan, "cuda")
            packets.append(
                pack(
                    plan,
                    meta,
                    main,
                    index[:, :64].contiguous().view(torch.int8),
                    index[:, 64:].contiguous().view(torch.int32).flatten(),
                    tail,
                )
            )
        received = torch.stack(packets)
        received[2, 32] ^= 1
        unpack(plans, metadata, received)
        torch.cuda.synchronize()
        raise AssertionError("Stale packet identity was accepted")
    layer, cache, tables, lengths, prefixes = fixture()
    if case == "length":
        lengths[0] = 0
    elif case == "prefix":
        prefixes[0] = -1
    elif case == "context":
        prefixes[0] = 512
    elif case == "coverage":
        tables["csa_kv"] = tables["csa_kv"][:, :1].contiguous()
    elif case == "page":
        tables["csa_kv"][1, 1] = 8
    elif case == "alias":
        tables["csa_kv"][1, 1] = 1
    else:
        raise AssertionError(case)
    with bind_attn_cache(layer, cache, tables):
        layer._require_qualified_prefill(lengths, prefixes, cp_on=False)
    torch.cuda.synchronize()
    raise AssertionError("Invalid metadata was accepted")


class AsyncValidation(unittest.TestCase):
    def test_valid_live_columns_and_decode_preserve_cache(self):
        for flag in ("0", "1"):
            with patch.dict(os.environ, {"DSV41_ATTN_ASYNC_CHECKS": flag}):
                layer, cache, tables, lengths, prefixes = fixture()
                saved = {key: raw.clone() for key, raw in cache.storage.items()}
                for decode in (False, True):
                    with bind_attn_cache(layer, cache, tables):
                        layer._require_qualified_prefill(
                            lengths, prefixes, cp_on=False, is_decode=decode
                        )
                    torch.cuda.synchronize()
                for key, raw in cache.storage.items():
                    self.assertTrue(torch.equal(raw, saved[key]))
                with patch.dict(
                    os.environ, {"DSV41_ATTN_ASYNC_CHECKS": str(1 - int(flag))}
                ):
                    with self.assertRaisesRegex(ValueError, "validation mode changed"):
                        layer._check_execution_config()

    def test_invalid_device_metadata_aborts_isolated_context(self):
        for case in (
            "length",
            "prefix",
            "context",
            "coverage",
            "page",
            "alias",
            "packet",
        ):
            with self.subTest(case=case):
                with self.assertRaises(mp.ProcessRaisedException) as failure:
                    mp.spawn(invalid_child, args=(case,), nprocs=1, join=True)
                self.assertIn("device-side assert", str(failure.exception))


if __name__ == "__main__":
    unittest.main()
