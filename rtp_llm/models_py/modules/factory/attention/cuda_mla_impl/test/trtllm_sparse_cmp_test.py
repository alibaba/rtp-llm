"""Real CMP bridge / TRT adapter GPU integration, not a full CMP-model test.

Imports the production Glm5Cmp.sparse_mla and real TRT op. Only unrelated model,
projection, and MoE construction is bypassed with object.__new__. The side-stream
test simulates the producer/event-join contract of CMP; it does NOT execute the
RTP-kernel CMP prologue, the Indexer, TP collectives, or the complete model.

Run only after reserving a GPU, for example CUDA_VISIBLE_DEVICES=4. The fixture's
cuda:0 then denotes that one visible GPU. This test contains no performance or
full-model precision claim.
"""

import importlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from trtllm_sparse_decode_test import PackedFixture, load_backend_class


class TrtllmSparseCmpBridgeGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) not in (
            (10, 0),
            (10, 3),
        ):
            raise unittest.SkipTest(
                "CMP/TRT bridge test requires an available SM100/103"
            )
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        cls.Op = load_backend_class()
        cls.backend_module = importlib.import_module(cls.Op.__module__)
        cls.Outer = cls.backend_module.TrtllmSparseMlaImpl
        cls.cmp_module = importlib.import_module(
            "rtp_llm.models_py.modules.hybrid.glm5_cmp"
        )
        cls.factory_module = importlib.import_module(
            "rtp_llm.models_py.modules.factory.attention.attn_factory"
        )
        cls.ops_module = importlib.import_module("rtp_llm.ops")

    def tearDown(self):
        # Test-result assertions and explicit test barriers are outside the
        # production bridge being exercised; this synchronizes only cuda:0.
        torch.cuda.synchronize()
        super().tearDown()

    def make_bridge(self, fixture, op, layer_id=7, four_dim=False):
        cmp = object.__new__(self.cmp_module.Glm5Cmp)
        cmp.layer_idx = layer_id
        outer = object.__new__(self.Outer)
        outer.weights = [{}]
        outer.fmha_params = fixture.params
        outer.fmha_impl = op
        cache = fixture.cache.unsqueeze(2) if four_dim else fixture.cache
        layer_cache = SimpleNamespace(kv_cache_base=cache)

        def bridge():
            return cmp.sparse_mla(fixture.q, fixture.topk, outer, layer_cache)

        return bridge

    def assert_result(self, fixture, actual, direct):
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(tuple(actual.shape), (fixture.rows, fixture.heads, 512))
        self.assertTrue(bool(torch.isfinite(actual).all()))
        # The bridge adds no numerical operation. Direct and bridged calls use
        # the same op, shape, metadata and buffers.
        torch.testing.assert_close(actual, direct, atol=0, rtol=0)
        # This reference includes the compatibility adapter's extra FP8
        # quantization. It does not waive the strict FlashMLA replacement gate.
        torch.testing.assert_close(actual, fixture.reference(), atol=0.0008, rtol=0.02)

    def test_real_bridge_dispatches_layer_and_preserves_original_cache(self):
        # H64 is the existing CMP contract. H8/H16 standalone support does not
        # imply the complete CMP model supports attention TP8/TP4.
        fixture = PackedFixture(batch=2, heads=64, queries=1)
        fixture.set_counts([65, 2048])
        op = fixture.new_op()
        original_cache = fixture.cache.clone()
        direct = fixture.forward(op).clone()
        for four_dim in (False, True):
            with self.subTest(four_dim=four_dim):
                bridge = self.make_bridge(fixture, op, layer_id=7, four_dim=four_dim)
                with patch.object(op, "forward", wraps=op.forward) as forwarded:
                    actual = bridge().clone()
                self.assertEqual(forwarded.call_count, 1)
                args, kwargs = forwarded.call_args
                self.assertIs(args[0], fixture.q)
                self.assertIs(args[2], fixture.topk)
                self.assertEqual(args[1].data_ptr(), fixture.cache.data_ptr())
                self.assertEqual(kwargs, {"layer_id": 7})
                self.assert_result(fixture, actual, direct)
                self.assertTrue(torch.equal(fixture.cache, original_cache))

    def test_real_bridge_cuda_graph_observes_multi_query_updates(self):
        fixture = PackedFixture(batch=2, heads=64, queries=6)
        fixture.set_counts([65] * fixture.rows)
        op = fixture.new_op(cuda_graph=True)
        bridge = self.make_bridge(fixture, op, layer_id=0)
        for _ in range(3):
            bridge()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = bridge()
        graph.replay()
        before = output.clone()
        fixture.q.mul_(1.75)
        fixture.set_counts([0 if row % 3 == 0 else 2048 for row in range(fixture.rows)])
        original_cache = fixture.cache.clone()
        graph.replay()
        actual = output.clone()
        direct = fixture.forward(op).clone()
        self.assert_result(fixture, actual, direct)
        self.assertFalse(torch.equal(before, actual), "Replay ignored changed inputs")
        self.assertTrue(torch.equal(fixture.cache, original_cache))

    def test_side_stream_producer_join_and_bridge_are_captured_together(self):
        fixture = PackedFixture(batch=2, heads=64, queries=1)
        fixture.set_counts([65, 65])
        op = fixture.new_op(cuda_graph=True)
        bridge = self.make_bridge(fixture, op, layer_id=3)
        producer = torch.cuda.Stream(device=fixture.device)
        caller_ready = torch.cuda.Event()
        producer_done = torch.cuda.Event()
        source_q = fixture.q.clone()
        source_cache = fixture.cache.clone()
        source_topk = fixture.topk.clone()

        def joined_bridge():
            # Exactly the stream ownership boundary being simulated: producers
            # finish Q/KV/TopK before the caller dispatches sparse attention.
            # No host barrier is inserted between producer and bridge.
            caller = torch.cuda.current_stream()
            caller_ready.record(caller)
            with torch.cuda.stream(producer):
                producer.wait_event(caller_ready)
                fixture.q.copy_(source_q)
                fixture.cache.copy_(source_cache)
                fixture.topk.copy_(source_topk)
                producer_done.record(producer)
            caller.wait_event(producer_done)
            return bridge()

        for _ in range(3):
            joined_bridge()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = joined_bridge()
        graph.replay()
        before = output.clone()
        # Modify source tensors in place, not graph-captured destination
        # tensors. Correct output requires the captured side-stream copies and
        # its join dependency to execute on every replay.
        source_q.mul_(-2.0)
        source_cache[:, :, 528:].zero_()
        source_topk.fill_(-1)
        source_topk[1].copy_(
            torch.arange(2048, device=fixture.device, dtype=torch.int32)
        )
        graph.replay()
        actual = output.clone()
        direct = fixture.forward(op).clone()
        self.assert_result(fixture, actual, direct)
        self.assertFalse(
            torch.equal(before, actual), "Captured producer was not replayed"
        )
        self.assertTrue(torch.equal(fixture.q, source_q))
        self.assertTrue(torch.equal(fixture.topk, source_topk))
        # Only the simulated producer changes persistent cache; attention must
        # leave its output bytes untouched afterwards.
        self.assertTrue(torch.equal(fixture.cache, source_cache))
        torch.testing.assert_close(
            actual[0], torch.zeros_like(actual[0]), atol=0, rtol=0
        )

    def test_explicit_selector_uses_real_backend_class_for_all_decode_roles(self):
        factory = self.factory_module._get_glm5_trtllm_impl
        config = SimpleNamespace(
            is_sparse=True,
            use_mla=True,
            kv_cache_dtype=self.backend_module.KvCacheDataType.FP8,
            head_num=64,
            kv_lora_rank=512,
            nope_head_dim=192,
            rope_head_dim=64,
            kernel_tokens_per_block=64,
            indexer_topk=2048,
        )
        model = SimpleNamespace(model_type="glm_5", quant_config=None, max_seq_len=8192)
        weights = SimpleNamespace(weights=[{}], get_global_weight=lambda _: None)
        parallelism = self.ops_module.ParallelismConfig()
        with (
            patch.dict(os.environ, {"GLM5_SPARSE_DECODE_BACKEND": "trtllm_gen"}),
            # Dispatch and new-subclass construction are real. Skip only the
            # parent constructor's model-weight/C++ metadata initialization.
            patch.object(
                self.backend_module.SparseMlaImpl, "__init__", return_value=None
            ) as initialized,
        ):
            for model_type in ("glm_5", "glm_5_mtp"):
                model.model_type = model_type
                for role in ("decode", "verify", "extend"):
                    with self.subTest(model=model_type, role=role):
                        inputs = SimpleNamespace(
                            is_prefill=role != "decode",
                            is_target_verify=role == "verify",
                            is_draft_extend=role == "extend",
                        )
                        actual = factory(
                            model, config, parallelism, weights, inputs, None, True
                        )
                        self.assertIsInstance(actual, self.Outer)
                        self.assertIs(
                            initialized.call_args.kwargs["fmha_impl"], self.Op
                        )
            prefill = SimpleNamespace(
                is_prefill=True, is_target_verify=False, is_draft_extend=False
            )
            self.assertIsNone(
                factory(model, config, parallelism, weights, prefill, None, True)
            )
            self.assertEqual(initialized.call_count, 6)

    def make_selector_inputs(self, parallelism, role, model_type="glm_5"):
        # Use the real C++ parallelism binding to derive per-rank head count.
        config = SimpleNamespace(
            is_sparse=True,
            use_mla=True,
            kv_cache_dtype=self.backend_module.KvCacheDataType.FP8,
            head_num=64 // parallelism.get_attn_tp_size(),
            kv_lora_rank=512,
            nope_head_dim=192,
            rope_head_dim=64,
            kernel_tokens_per_block=64,
            indexer_topk=2048,
        )
        model = SimpleNamespace(
            model_type=model_type, quant_config=None, max_seq_len=8192
        )
        weights = SimpleNamespace(weights=[{}], get_global_weight=lambda _: None)
        inputs = SimpleNamespace(
            is_prefill=role != "decode",
            is_target_verify=role == "verify",
            is_draft_extend=role == "extend",
        )
        return model, config, parallelism, weights, inputs, None

    def test_real_sharded_prefill_cp_rejected_without_active_cp(self):
        # PREFILL_CP is normally D-side P topology metadata. However, raw TP>1
        # plus kv_cache_sharded really constructs CPSlotMapper independently
        # of is_enabled(); the ordinary 656 reader cannot consume that layout.
        parallelism = self.ops_module.ParallelismConfig()
        parallelism.tp_size = 8
        parallelism.role_type = self.ops_module.RoleType.DECODE
        cp = parallelism.prefill_cp_config
        cp.method = self.ops_module.CPRotateMethod.PREFILL_CP
        cp.kv_cache_sharded = True
        cp.prefill_cp_size = 8
        self.assertFalse(cp.is_enabled())
        with (
            patch.dict(os.environ, {"GLM5_SPARSE_DECODE_BACKEND": "trtllm_gen"}),
            patch.object(
                self.backend_module.SparseMlaImpl, "__init__", return_value=None
            ) as initialized,
        ):
            with self.assertRaises(ValueError):
                self.factory_module._get_glm5_trtllm_impl(
                    *self.make_selector_inputs(parallelism, "decode"), True
                )
            initialized.assert_not_called()

    def test_real_cp_configuration_matrix_guards_selection(self):
        # 6 methods x 3 TP sizes x 2 cache layouts x 4 attention roles x
        # graph/eager x main/MTP models = 576 real-config selection cases.
        # Only model-weight/metadata allocation in the existing parent
        # constructor is mocked; new backend class and config methods are real.
        methods = self.ops_module.CPRotateMethod
        cp_methods = (
            methods.DISABLED,
            methods.ALL_GATHER,
            methods.ALL_GATHER_WITH_OVERLAP,
            methods.ALLTOALL,
            methods.PREFILL_CP,
            methods.UNKNOWN,
        )
        active_methods = (
            methods.ALL_GATHER,
            methods.ALL_GATHER_WITH_OVERLAP,
            methods.ALLTOALL,
        )
        checked = 0
        branches = {"prefill_unchanged": 0, "rejected": 0, "selected": 0}
        with (
            patch.dict(os.environ, {"GLM5_SPARSE_DECODE_BACKEND": "trtllm_gen"}),
            patch.object(
                self.backend_module.SparseMlaImpl, "__init__", return_value=None
            ) as initialized,
        ):
            for method in cp_methods:
                for tp_size in (1, 2, 8):
                    for sharded in (False, True):
                        parallelism = self.ops_module.ParallelismConfig()
                        parallelism.tp_size = tp_size
                        parallelism.role_type = self.ops_module.RoleType.DECODE
                        cp = parallelism.prefill_cp_config
                        cp.method = method
                        cp.kv_cache_sharded = sharded
                        # A remote P width alone must not reject local TP1 D.
                        cp.prefill_cp_size = 8
                        active_cp = method in active_methods
                        self.assertEqual(cp.is_enabled(), active_cp)
                        unsafe = active_cp or (sharded and tp_size > 1)
                        for role in ("decode", "verify", "extend", "prefill"):
                            for is_cuda_graph in (False, True):
                                for model_type in ("glm_5", "glm_5_mtp"):
                                    with self.subTest(
                                        method=str(method),
                                        tp_size=tp_size,
                                        sharded=sharded,
                                        role=role,
                                        graph=is_cuda_graph,
                                        model=model_type,
                                    ):
                                        initialized.reset_mock()
                                        args = self.make_selector_inputs(
                                            parallelism, role, model_type
                                        )
                                        factory = (
                                            self.factory_module._get_glm5_trtllm_impl
                                        )
                                        if role == "prefill":
                                            self.assertIsNone(
                                                factory(*args, is_cuda_graph)
                                            )
                                            initialized.assert_not_called()
                                            branches["prefill_unchanged"] += 1
                                        elif unsafe:
                                            with self.assertRaises(ValueError):
                                                factory(*args, is_cuda_graph)
                                            initialized.assert_not_called()
                                            branches["rejected"] += 1
                                        else:
                                            selected = factory(*args, is_cuda_graph)
                                            self.assertIsInstance(selected, self.Outer)
                                            initialized.assert_called_once()
                                            self.assertIs(
                                                initialized.call_args.kwargs[
                                                    "parallelism_config"
                                                ],
                                                parallelism,
                                            )
                                            self.assertIs(
                                                initialized.call_args.kwargs[
                                                    "fmha_impl"
                                                ],
                                                self.Op,
                                            )
                                            branches["selected"] += 1
                                        checked += 1
        self.assertEqual(checked, 576)
        self.assertEqual(
            branches,
            {"prefill_unchanged": 144, "rejected": 288, "selected": 144},
        )
        print(f"CP_SELECTOR_REAL_CONFIG_MATRIX: total={checked}, branches={branches}")


if __name__ == "__main__":
    unittest.main()
