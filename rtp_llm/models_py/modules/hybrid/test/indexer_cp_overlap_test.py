"""CPU protocol tests for per-call Indexer auxiliary K ownership."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.hybrid.indexer import Indexer


class IndexerCPOverlapTest(unittest.TestCase):
    def test_early_key_precedes_query_projection(self):
        for prequantized in (False, True):
            calls = []
            x, q = torch.zeros(2, 2), torch.zeros(2, 4)
            xs, qs = object(), object()

            def projection(name, expected_scale):
                def run(t, **kwargs):
                    calls.append(name)
                    self.assertEqual(
                        kwargs, {"input_scales": expected_scale} if prequantized else {}
                    )
                    return t

                return run

            def transform(name, value):
                calls.append(name)
                return value

            op = NS(
                fused_rope_hadamard_k=lambda k, p: transform("k_rope", k),
                fused_rope_quant_q=lambda q, p: transform("q_quant", (q, None)),
            )
            obj = NS(
                wk=projection("k_projection", xs),
                wq_b=projection("q_projection", qs),
                k_norm=lambda k: transform("k_norm", k),
                indexer_op=op,
                index_n_heads=2,
                index_head_dim=2,
            )
            cp = NS(
                full_rope_pos_ids=None,
                sharded_slot_mapping=None,
                kv_restore_unpad_indices=None,
            )
            Indexer._fused_forward_prefill_cp_early_k(
                obj,
                q,
                x,
                None,
                None,
                cp,
                x if prequantized else None,
                xs if prequantized else None,
                q if prequantized else None,
                qs if prequantized else None,
                cache_key=lambda *args: calls.append("cache_submit"),
            )
            self.assertEqual(
                calls,
                [
                    "k_projection",
                    "k_norm",
                    "k_rope",
                    "cache_submit",
                    "q_projection",
                    "q_quant",
                ],
            )

    def run_protocol(self, fused=True, fail=None):
        calls = []
        compute = NS(
            wait_event=lambda e: calls.append("join_ready"),
            wait_stream=lambda s: calls.append("failure_join"),
        )
        side = NS(wait_stream=lambda s: calls.append("producer_wait"))
        ready = NS(record=lambda s: calls.append("record_ready"))
        output = [
            NS(record_stream=lambda s: calls.append("record_use")) for _ in range(2)
        ]

        def step(name):
            calls.append(name)
            if fail == name:
                raise ValueError(name)

        def cache_key(*args):
            step("cache_write")

        def prepare(*args, **kwargs):
            step("assemble")
            return output

        def fused_call(*args, cache_key):
            step("fused_qk")
            cache_key(NS(), NS(), NS(), NS())
            return NS(), NS()

        def quant(query, key, cache, slots, restore, cache_key):
            cache_key(key, cache, slots, restore)
            step("quant_q")
            return NS(), NS()

        def weights(*args):
            step("weights")
            return NS()

        def topk(*args, prepared_k):
            self.assertIs(prepared_k, output)
            step("topk")
            return "result"

        obj = NS(
            indexer_op=NS(
                quant_k_cp_only=cache_key, prepare_cp_k=prepare, quant_q_k_cp=quant
            ),
            _prefill_cp_fused_quant_enabled=lambda: fused,
            _fused_forward_prefill_cp_early_k=fused_call,
            _get_q_k_bf16=lambda *args: (NS(), NS()),
            _get_logits_head_gate=weights,
            _compute_topk=topk,
        )
        cp = NS(
            full_rope_pos_ids=NS(),
            total_kv_len=9,
            cu_kv_seqlens_global=NS(),
            cp_size=4,
            indexer_cp_plan=NS(),
            indexer_cp_local_cu=NS(),
            sharded_slot_mapping=NS(),
            kv_restore_unpad_indices=NS(),
        )
        with patch.object(
            torch.cuda, "current_stream", return_value=compute
        ), patch.object(
            torch.cuda, "stream", side_effect=lambda s: nullcontext()
        ), patch.object(
            torch.cuda, "Event", return_value=ready
        ):
            if fail:
                with self.assertRaisesRegex(ValueError, fail):
                    Indexer._forward_cp_overlap(
                        obj,
                        side,
                        NS(device="cpu"),
                        NS(),
                        NS(),
                        NS(),
                        NS(),
                        cp,
                        None,
                        None,
                        None,
                        None,
                    )
            else:
                self.assertEqual(
                    Indexer._forward_cp_overlap(
                        obj,
                        side,
                        NS(device="cpu"),
                        NS(),
                        NS(),
                        NS(),
                        NS(),
                        cp,
                        None,
                        None,
                        None,
                        None,
                    ),
                    "result",
                )
        # No state can leak from this invocation into another runner/step.
        self.assertNotIn("prepared_k", vars(obj))
        self.assertNotIn("ready", vars(obj))
        return calls

    def test_fused_submits_both_gathers_before_independent_gate(self):
        self.assertEqual(
            self.run_protocol(),
            [
                "fused_qk",
                "producer_wait",
                "cache_write",
                "assemble",
                "record_ready",
                "weights",
                "join_ready",
                "record_use",
                "record_use",
                "topk",
            ],
        )

    def test_unfused_query_quantization_stays_after_async_k_submission(self):
        calls = self.run_protocol(False)
        self.assertLess(calls.index("record_ready"), calls.index("quant_q"))
        self.assertLess(calls.index("weights"), calls.index("join_ready"))

    def test_key_submission_failure_joins_stream(self):
        self.assertEqual(self.run_protocol(fail="cache_write")[-1], "failure_join")

    def test_full_k_assembly_failure_joins_stream(self):
        self.assertEqual(self.run_protocol(fail="assemble")[-1], "failure_join")

    def test_gate_failure_joins_outstanding_k_before_unwind(self):
        self.assertEqual(self.run_protocol(fail="weights")[-1], "failure_join")

    def test_topk_failure_keeps_cross_stream_allocator_tracking(self):
        calls = self.run_protocol(fail="topk")
        self.assertEqual(calls.count("record_use"), 2)
        self.assertEqual(calls[-1], "failure_join")

    def test_consecutive_calls_have_no_retained_per_call_state(self):
        self.assertEqual(self.run_protocol(), self.run_protocol())


if __name__ == "__main__":
    unittest.main()
