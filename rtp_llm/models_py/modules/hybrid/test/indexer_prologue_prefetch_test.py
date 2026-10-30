"""Prefetch ownership, exception cleanup and MLA call-site protocol."""

import importlib
import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.device.device_type import DeviceType
from rtp_llm.models_py.modules.hybrid.indexer import Indexer
from rtp_llm.models_py.modules.hybrid.mla_attention import MlaAttention


class IndexerProloguePrefetchTest(unittest.TestCase):
    def protocol(self, early=True, fail=None, consume=True, captured=False):
        calls = []

        def step(name):
            calls.append(name)
            if fail == name:
                raise RuntimeError(name)

        compute = NS(
            wait_stream=lambda s: step("cleanup_join"),
            wait_event=lambda e: step("consumer_join"),
        )
        side = NS(wait_stream=lambda s: step("producer_join"))
        ready = NS(record=lambda s: step("ready"))
        key = NS(record_stream=lambda s: step("key_lifetime"))
        buffers = [
            NS(record_stream=lambda s: step("consumer_lifetime")) for _ in range(2)
        ]

        def projection(*args, **kwargs):
            step("key_projection")
            return key

        def prepare(*args, **kwargs):
            step("history")
            return buffers

        obj = NS(
            wk=projection,
            k_norm=lambda x: x,
            wq_b=lambda *a, **k: torch.zeros(2, 128),
            index_n_heads=1,
            index_head_dim=128,
            _is_sparse_prefill_cp=lambda a: True,
            _prefill_cp_fused_quant_enabled=lambda: True,
            _get_logits_head_gate=lambda *a: step("gate"),
            _compute_topk=lambda *a, **kw: step("topk"),
            indexer_op=NS(
                index_head_dim=128,
                block_size=128,
                prepare_cp_history=prepare,
                finish_cp_history=lambda *a: step("patch"),
                fused_rope_hadamard_k=lambda *a: key,
                quant_k_cp_only=lambda *a: step("cache_write"),
                prepare_cp_k=prepare,
                fused_rope_quant_q=lambda q, p: (q, torch.ones(2, 1)),
            ),
        )
        cp = NS(
            kv_cache_sharded=True,
            cp_size=4,
            full_rope_pos_ids=object(),
            indexer_cp_plan=NS(early_history=object() if early else None),
            sharded_slot_mapping=object(),
            kv_restore_unpad_indices=object(),
            total_kv_len=16,
            cu_kv_seqlens_global=object(),
            indexer_cp_local_cu=object(),
        )
        hidden = NS(is_cuda=True, device="cuda")
        cache = object()
        attn = object()
        seen = []
        with patch.dict(
            Indexer.prefetch_cp_k.__wrapped__.__globals__,
            {
                "_DEVICE_TYPE": Indexer.prefetch_cp_k.__wrapped__.__globals__[
                    "DeviceType"
                ].Cuda
            },
        ), patch.dict(
            "os.environ", {"RTP_LLM_CP_INDEXER_PROLOGUE_PREFETCH": "1"}
        ), patch.object(
            torch.cuda, "current_stream", return_value=compute
        ), patch.object(
            torch.cuda, "stream", side_effect=lambda s: nullcontext()
        ), patch.object(
            torch.cuda, "Event", return_value=ready
        ), patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=captured
        ), patch.object(
            importlib.import_module("rtp_llm.models_py.distributed.pynccl_cp"),
            "indexer_overlap_stream",
            return_value=side,
        ):

            def run():
                with Indexer.prefetch_cp_k(obj, hidden, cache, attn, cp) as state:
                    seen.append(state)
                    if captured:
                        self.assertIsNone(state)
                        return
                    step("mla_projections")
                    if consume:
                        Indexer._consume_prefetched_cp_k(
                            obj, state, hidden, None, cache, None, attn, cp, None, None
                        )
                        self.assertIsNone(state.prepared)
                        with self.assertRaises(ValueError):
                            Indexer._consume_prefetched_cp_k(
                                obj,
                                state,
                                hidden,
                                None,
                                cache,
                                None,
                                attn,
                                cp,
                                None,
                                None,
                            )

            if fail:
                with self.assertRaisesRegex(RuntimeError, fail):
                    run()
            else:
                run()
        self.assertNotIn("prepared", vars(obj))
        for state in seen:
            if state is not None:
                self.assertIsNone(state.prepared)
        return calls

    def test_prefetch_is_ready_before_mla_submission_and_consumer_joins(self):
        for early in (False, True):
            calls = self.protocol(early=early)
            self.assertLess(calls.index("ready"), calls.index("mla_projections"))
            self.assertLess(calls.index("consumer_join"), calls.index("topk"))
            self.assertEqual(calls.count("consumer_lifetime"), 2)
            self.assertNotIn("cleanup_join", calls)

    def test_exception_and_unconsumed_cleanup(self):
        for name in ("history", "key_projection", "patch", "mla_projections", "gate"):
            with self.subTest(name=name):
                self.assertEqual(self.protocol(fail=name)[-1], "cleanup_join")
        self.assertEqual(self.protocol(consume=False)[-1], "cleanup_join")
        # Once the consumer has enqueued its join, TopK failure needs no second join.
        calls = self.protocol(fail="topk")
        self.assertIn("consumer_join", calls)

    def test_reject_foreign_prefetch_before_any_projection(self):
        for field in ("owner", "cache", "params", "consumed"):
            owner, cache, cp = object(), object(), object()
            state = NS(owner=owner, cache=cache, params=cp, consumed=False)
            setattr(state, field, True if field == "consumed" else object())
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "another invocation"
            ):
                Indexer._consume_prefetched_cp_k(
                    owner, state, None, None, cache, None, None, cp, None, None
                )

    def test_disabled_prefetch_does_not_touch_cuda_or_cache(self):
        with patch.dict(
            "os.environ", {"RTP_LLM_CP_INDEXER_PROLOGUE_PREFETCH": "0"}
        ), patch.object(
            torch.cuda, "current_stream", side_effect=AssertionError("CUDA accessed")
        ):
            with Indexer.prefetch_cp_k(object(), None, None, None, None) as state:
                self.assertIsNone(state)

    def test_capture_fallback(self):
        self.assertEqual(self.protocol(captured=True), [])

    def test_mla_wrapper_skips_reuse_and_dense_layers(self):
        for reuse in (False, True):
            for force in (False, True):
                for sparse in (False, True):
                    calls = []

                    @contextmanager
                    def prepare(*args):
                        calls.append("prepare")
                        try:
                            yield "state"
                        finally:
                            calls.append("release")

                    def body(*args, prepared_cp_k):
                        calls.append("body")
                        self.assertEqual(
                            prepared_cp_k,
                            "state" if sparse and not reuse and not force else None,
                        )
                        return "output"

                    obj = NS(
                        indexer=NS(prefetch_cp_k=prepare),
                        reuse_topk_indices=reuse,
                        _forward_impl=body,
                    )
                    fmha = NS(
                        is_sparse=lambda: sparse, attn_inputs=None, cp_params=None
                    )
                    self.assertEqual(
                        MlaAttention.forward(
                            obj, None, fmha, force_reuse_topk_indices=force
                        ),
                        "output",
                    )
                    self.assertEqual(
                        calls,
                        (
                            ["prepare", "body", "release"]
                            if sparse and not reuse and not force
                            else ["body"]
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
