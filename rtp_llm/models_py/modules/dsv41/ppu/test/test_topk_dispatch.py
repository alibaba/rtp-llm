"""Dispatch preserves each TopK provider's scores, IDs and failure boundary."""

import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import (
    graph_indexer,
)
from rtp_llm.models_py.modules.dsv41.ppu import (
    local_prefill_hierarchical_topk as hierarchy,
)
from rtp_llm.models_py.modules.dsv41.ppu import local_prefill_sg_topk as sg
from rtp_llm.models_py.modules.dsv41.ppu import prefill_token_topk as local
from rtp_llm.models_py.modules.dsv41.ppu import (
    topk,
)
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_token_topk import (
    _BoundsDouble,
    _finish_double,
    _native_double,
)


class TopKDispatchContract(unittest.TestCase):
    def test_explicit_contract_keeps_dtype_arguments_and_raw_result(self):
        for contract, module, name, options in (
            ("local", local, "try_select_tokens", {"disposable": True}),
            ("sg_dense", sg, "select_dense", {"block_topk": 2048}),
            ("sg_sparse", sg, "select_sparse", {"blocks": object()}),
            (
                "graph",
                graph_indexer,
                "_select_dense_topk",
                {"valid_mask": object(), "topk": 512},
            ),
        ):
            for dtype in (torch.float32, torch.bfloat16):
                scores = torch.tensor(
                    [[torch.nan, torch.inf, -torch.inf, -0.0]], dtype=dtype
                )
                before = scores.view(torch.uint8).clone()
                lengths, marker = object(), object()
                with patch.object(module, name, return_value=marker) as call:
                    self.assertIs(
                        topk.select_tokens(
                            scores, lengths, contract=contract, **options
                        ),
                        marker,
                    )
                self.assertIs(call.call_args.args[0], scores)
                self.assertIs(call.call_args.args[1], lengths)
                self.assertEqual(call.call_args.kwargs, options)
                self.assertTrue(torch.equal(scores.view(torch.uint8), before))

    def test_decline_and_error_never_retry_a_different_contract(self):
        with patch.object(sg, "select_dense", return_value=None), patch.object(
            local,
            "try_select_tokens",
            side_effect=AssertionError("cross-contract retry"),
        ):
            self.assertIsNone(topk.select_tokens(None, None, contract="sg_dense"))
        for error in (RuntimeError("native launch"), ImportError("native import")):
            with patch.object(sg, "select_sparse", side_effect=error) as call:
                with self.assertRaises(type(error)):
                    topk.select_tokens(None, None, contract="sg_sparse", blocks=None)
                call.assert_called_once()
        with self.assertRaisesRegex(ValueError, "unsupported TopK contract"):
            topk.select_tokens(None, None, contract="guess_from_dtype")

    def test_verify_specialization_needs_proof_despite_retired_flag(self):
        generic, bf16 = object(), object()
        ops = NS(dsv4_persistent_topk=generic, dsv4_persistent_topk_bf16_scores=bf16)
        for flag in (None, "0", "1"):
            with patch.dict(os.environ), patch(
                "rtp_llm.ops.compute_ops.rtp_llm_ops", ops
            ), patch.object(topk._quant, "is_ppu", return_value=True):
                os.environ.pop("RTP_V41_PPU_VERIFY_BF16_TOPK", None)
                if flag is not None:
                    os.environ["RTP_V41_PPU_VERIFY_BF16_TOPK"] = flag
                self.assertEqual(
                    topk.native_topk_op(
                        32, torch.device("cuda"), 512, bf16_provenance=True
                    ),
                    (bf16, True),
                )
                self.assertEqual(
                    topk.native_topk_op(32, torch.device("cuda"), 512), (generic, False)
                )

    @torch.no_grad()
    def test_local_hierarchy_automatic_but_outer_gate_and_bounds_remain(self):
        for flag in ("0", "1"):
            scores = torch.arange(1024, dtype=torch.float32).to(torch.bfloat16)[None]
            lengths = torch.tensor([1024], dtype=torch.int32)
            with patch.dict(
                os.environ,
                {local.ENV: "1", "RTP_V41_PPU_LOCAL_PREFILL_HIERARCHICAL_TOPK": flag},
            ), patch.object(
                local, "_device_supported", return_value=True
            ), patch.object(
                local,
                "_native_ops",
                return_value=(Mock(side_effect=AssertionError("wrong provider")), None),
            ), patch.object(
                hierarchy, "is_supported", return_value=True
            ), patch.object(
                hierarchy, "select", side_effect=_native_double
            ) as selected, patch(
                "rtp_llm.models_py.kernels.ppu.attention.ppu_bf16_radix_topk.canonicalize_topk_indices",
                side_effect=lambda out: out,
            ), patch.object(
                local._topk, "_prefill_topk_bounds_kernel", _BoundsDouble()
            ), patch.object(
                local._topk, "finish_tokens", side_effect=_finish_double
            ):
                result = topk.select_tokens(
                    scores, lengths, contract="local", disposable=True
                )
            self.assertEqual(result.shape, (1, 512))
            selected.assert_called_once()
        with patch.dict(os.environ, {local.ENV: "0"}), patch.object(
            local, "_native_ops", side_effect=AssertionError("disabled")
        ):
            self.assertIsNone(
                topk.select_tokens(scores, lengths, contract="local", disposable=True)
            )
        raw = Mock(wraps=_native_double)
        with patch.dict(os.environ, {local.ENV: "1"}), patch.object(
            local, "_device_supported", return_value=True
        ), patch.object(
            local, "_native_ops", return_value=(raw, None)
        ) as factory, patch.object(
            hierarchy,
            "is_supported",
            side_effect=AssertionError("bounds must use radix"),
        ), patch.object(
            local._topk, "_prefill_topk_bounds_kernel", _BoundsDouble()
        ), patch.object(
            local._topk, "finish_tokens", side_effect=_finish_double
        ):
            topk.select_tokens(
                scores,
                lengths,
                contract="local",
                disposable=True,
                bounds=(torch.tensor([7], dtype=torch.int32), lengths),
            )
        factory.assert_called_once_with(True)
        raw.assert_called_once()

    def test_shared_alias_check_catches_distinct_storage_wrappers(self):
        allocation = bytearray(4096)
        left = torch.frombuffer(allocation, dtype=torch.uint8, count=1024)
        overlapping = torch.frombuffer(
            allocation, dtype=torch.uint8, count=1024, offset=512
        )
        separate = torch.frombuffer(
            allocation, dtype=torch.uint8, count=1024, offset=2048
        )
        self.assertTrue(topk.overlaps(left, overlapping))
        self.assertFalse(topk.overlaps(left, separate))
        self.assertTrue(topk.raw_tensor(left))
        self.assertFalse(topk.raw_tensor(torch._neg_view(left.int())))


if __name__ == "__main__":
    unittest.main()
