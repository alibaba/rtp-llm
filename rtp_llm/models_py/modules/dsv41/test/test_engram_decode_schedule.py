"""Variable decode buckets, phase boundaries and actual native lookup graphs.

Small pinned tables and lightweight attention markers test correctness and
ownership, not full-model latency. CUDA execution requires a reserved GPU.
"""

import ast
import gc
import os
import unittest
import uuid
import weakref
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv41 import _engram_decode_prefetch as prefetch
from rtp_llm.models_py.modules.dsv41 import _engram_decode_schedule as schedule
from rtp_llm.models_py.modules.dsv41 import _engram_quantized_triton as direct

ENV = {"DSV41_DECODE_ENGRAM_PREFETCH": "1", "DSV41_ENGRAM_DEVICE_SCALES": "1"}


class BoundaryCore:
    _supports_engram_lookup_schedule = True

    def decode(self, marker):
        hook = getattr(self, "_engram_lookup_schedule", None)
        if hook is not None:
            hook.before_core()
        marker()
        if hook is not None:
            hook.after_core()


class BoundaryAttention:
    _supports_engram_lookup_schedule = True
    tp_size = 1

    def __init__(self):
        self._mega_attention = BoundaryCore()

    def forward_decode(self, marker=lambda: None, *, native=True):
        hook = getattr(self, "_engram_lookup_schedule", None)
        try:
            if hook is not None:
                hook.before_attention()
                hook.before_projection(native)
            marker()
            if native:
                self._mega_attention.decode(marker)
            marker()
            if hook is not None:
                hook.after_attention()
        except BaseException:
            if hook is not None:
                hook.abort()
            raise


class Model:
    def __init__(self):
        self.layers = [
            SimpleNamespace(engram=None, engram_hashes=None, attn=BoundaryAttention())
            for _ in range(15)
        ]


def cpu_model(tokens):
    model = Model()
    device = torch.device("cuda:0")
    for index, target in enumerate((1, 14)):
        lease = object()
        embedding = SimpleNamespace(
            _quantized_lookup_supported=True,
            _uva=(object(), object()),
            _pinned=object(),
            _device_scale_lease=lease,
            _num_sms=148,
            _device_scale_resources=lambda lease=lease: lease,
        )
        module = SimpleNamespace(
            embed_tokens=embedding,
            _direct_fp8_lookup=True,
            q_weight=SimpleNamespace(is_cuda=True, device=device),
            layout=SimpleNamespace(n_hash_cols=24, head_dim=256, layer_ids=(1, 14)),
            layer_hash_index=index,
            _lookup_work=None,
        )
        module._decode_scheduled_stream = SimpleNamespace(
            target=target,
            ready=True,
            grid=120 if target == 1 else 80,
            matches=lambda e, expected=embedding: e is expected,
            stream=Mock(),
            done=Mock(),
        )
        model.layers[target].engram = module
        model.layers[target].engram_hashes = SimpleNamespace(
            shape=(tokens, 24),
            dtype=torch.int64,
            is_cuda=True,
            device=device,
            is_contiguous=lambda: True,
            record_stream=Mock(),
        )
    return model


class DecodeScheduleCPU(unittest.TestCase):
    def mock_cuda(self, stack):
        main = Mock(name="main")
        stack.enter_context(patch.dict(os.environ, ENV))
        stack.enter_context(
            patch.object(torch.cuda, "current_stream", return_value=main)
        )
        stack.enter_context(
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False)
        )
        stack.enter_context(
            patch.object(torch.cuda, "stream", side_effect=lambda *_: nullcontext())
        )
        outputs = SimpleNamespace(pair=lambda: (Mock(), Mock()))
        stack.enter_context(
            patch.object(direct, "allocate_outputs", return_value=outputs)
        )
        whole = stack.enter_context(patch.object(direct, "launch_epilogue"))
        slices = stack.enter_context(patch.object(schedule.sliced, "launch"))
        return main, whole, slices

    def test_partition_covers_all_rows_once_and_keeps_small_work_early(self):
        for tokens in (*range(1, 2049), 32768, 32769):
            pieces = schedule.partition_tokens(tokens)
            cursor = 0
            for start, count in pieces:
                self.assertEqual(start, cursor)
                cursor += count
            self.assertEqual(cursor, tokens)
            counts = [count for _, count in pieces]
            self.assertEqual(counts, sorted(counts, reverse=True))
            self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertEqual(
            schedule.partition_tokens(384), tuple((i * 48, 48) for i in range(8))
        )
        for invalid in (0, -1, True, 384.0):
            with self.assertRaises(ValueError):
                schedule.partition_tokens(invalid)

    def test_positive_buckets_launch_exactly_one_whole_and_nonempty_pieces(self):
        for batch, query in (
            (1, 1),
            (1, 6),
            (7, 1),
            (7, 6),
            (63, 6),
            (64, 6),
            (65, 6),
            (128, 6),
            (385, 1),
            (32768, 1),
        ):
            tokens = batch * query
            with self.subTest(batch=batch, query=query), ExitStack() as stack:
                main, whole, slices = self.mock_cuda(stack)
                model = cpu_model(tokens)
                plan = prefetch.DecodeEngramPrefetch(
                    model, SimpleNamespace(batch_size=batch, q_len_per_req=query)
                )
                try:
                    for index, layer in enumerate(model.layers):
                        plan.before_layer(index)
                        layer.attn.forward_decode()
                        for work in plan.work:
                            self.assertEqual(work.joined.get(main, 0), work.epoch)
                    self.assertEqual(whole.call_count, 1)
                    self.assertEqual(whole.call_args.args[-1], 120)
                    expected = [
                        (start, count)
                        for start, count in schedule.partition_tokens(tokens)
                        if count
                    ]
                    self.assertEqual(
                        [call.args[4:6] for call in slices.call_args_list], expected
                    )
                    self.assertTrue(
                        all(call.kwargs["grid"] == 80 for call in slices.call_args_list)
                    )
                    for work in plan.work:
                        self.assertEqual(work.next_token, tokens)
                finally:
                    plan.close()
                self.assertTrue(
                    all(
                        not hasattr(layer.attn, "_engram_lookup_schedule")
                        for layer in model.layers
                    )
                )
                self.assertTrue(
                    all(model.layers[i].engram._lookup_work is None for i in (1, 14))
                )

    def test_empty_and_chunked_consumer_buckets_remain_serial(self):
        for tokens in (0, 32769):
            with ExitStack() as stack:
                _, whole, slices = self.mock_cuda(stack)
                model = cpu_model(tokens)
                plan = prefetch.DecodeEngramPrefetch(
                    model, SimpleNamespace(batch_size=tokens, q_len_per_req=1)
                )
                for index in range(15):
                    plan.before_layer(index)
                self.assertEqual(plan.work, [])
                plan.close()
                whole.assert_not_called()
                slices.assert_not_called()

    def test_cancel_after_partial_lookup_drains_and_restores_serial_consumer(self):
        with ExitStack() as stack:
            main, _, slices = self.mock_cuda(stack)
            model = cpu_model(385)
            plan = prefetch.DecodeEngramPrefetch(
                model, SimpleNamespace(batch_size=77, q_len_per_req=5)
            )
            try:
                plan.before_layer(10)
                model.layers[10].attn.forward_decode()
                work = model.layers[14].engram._lookup_work
                model.layers[11].attn.forward_decode(native=False)
                for index in (12, 13):
                    model.layers[index].attn.forward_decode()
                self.assertIsNone(model.layers[14].engram._lookup_work)
                self.assertEqual(work.joined[main], work.epoch)
                self.assertEqual(slices.call_count, 2)
            finally:
                plan.close()

    def test_enqueue_failure_and_failed_join_restore_hooks(self):
        for fail_join in (False, True):
            with ExitStack() as stack:
                main, _, slices = self.mock_cuda(stack)
                model = cpu_model(9)
                plan = prefetch.DecodeEngramPrefetch(
                    model, SimpleNamespace(batch_size=9, q_len_per_req=1)
                )
                plan.before_layer(10)
                slices.side_effect = RuntimeError("enqueue")
                with self.assertRaisesRegex(RuntimeError, "enqueue"):
                    model.layers[10].attn.forward_decode()
                if fail_join:
                    plan.work[0].joined.clear()
                    main.wait_stream.side_effect = RuntimeError("join")
                    with self.assertRaisesRegex(RuntimeError, "join"):
                        plan.close()
                else:
                    plan.close()
                for layer in model.layers:
                    self.assertFalse(hasattr(layer.attn, "_engram_lookup_schedule"))

    def test_old_two_layer_lookahead_is_not_a_fallback(self):
        with ExitStack() as stack:
            _, whole, slices = self.mock_cuda(stack)
            model = cpu_model(384)
            model.layers[0].attn.tp_size = 2
            model.layers[10].attn._mega_attention = None
            plan = prefetch.DecodeEngramPrefetch(
                model, SimpleNamespace(batch_size=64, q_len_per_req=6)
            )
            for index in range(15):
                plan.before_layer(index)
            self.assertEqual(plan.work, [])
            plan.close()
            whole.assert_not_called()
            slices.assert_not_called()

    def test_cold_capture_has_no_stream_creation_and_reentrant_plan_rolls_back(self):
        with ExitStack() as stack:
            self.mock_cuda(stack)
            model = cpu_model(7)
            module = model.layers[1].engram
            state = module._decode_scheduled_stream
            del module._decode_scheduled_stream
            with patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=True
            ), patch.object(schedule, "_ScheduleStream") as create:
                self.assertIsNone(prefetch.prepare(module))
                create.assert_not_called()
            module._decode_scheduled_stream = state
            metadata = SimpleNamespace(batch_size=7, q_len_per_req=1)
            first = prefetch.DecodeEngramPrefetch(model, metadata)
            try:
                hook = model.layers[10].attn._engram_lookup_schedule
                with self.assertRaisesRegex(RuntimeError, "not reentrant"):
                    prefetch.DecodeEngramPrefetch(model, metadata)
                self.assertIs(model.layers[10].attn._engram_lookup_schedule, hook)
            finally:
                first.close()

    def test_actual_native_core_places_hooks_around_backend_call(self):
        path = Path(__file__).resolve().parents[1] / "fp8/_v41_mega_attention.py"
        tree = ast.parse(path.read_text())
        method = next(
            node
            for cls in tree.body
            if isinstance(cls, ast.ClassDef) and cls.name == "MegaAttention"
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "decode"
        )
        namespace = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ("_HEAD_DIM", "_ROPE_DIM")
        }
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        for enabled in (False, True):
            for fail in (False, True):
                events = []

                def backend(*args):
                    events.append("core")
                    if fail:
                        raise RuntimeError("backend")
                    return 1, 2, None

                owner = SimpleNamespace(
                    pad_q=lambda q: q,
                    backend=SimpleNamespace(
                        fused_norm_rope_attn_rope_cast_decode=backend
                    ),
                    softmax_scale=1.0,
                    sink=None,
                    cos_sin=None,
                    padded_groups=8,
                    _engram_lookup_schedule=(
                        SimpleNamespace(
                            before_core=lambda: events.append("before"),
                            after_core=lambda: events.append("after"),
                        )
                        if enabled
                        else None
                    ),
                )
                args = (
                    owner,
                    torch.zeros(1, 64, 512),
                    torch.zeros(1, 1, 528),
                    torch.zeros(1, 128),
                    torch.zeros(1),
                )
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "backend"):
                        namespace["decode"](*args)
                else:
                    self.assertEqual(namespace["decode"](*args), (1, 2))
                expected = (["before"] if enabled else []) + ["core"]
                if enabled and not fail:
                    expected.append("after")
                self.assertEqual(events, expected)

    def test_actual_attention_wrapper_orders_and_drains_on_failure(self):
        path = Path(__file__).resolve().parents[1] / "fp8/attention_v41.py"
        tree = ast.parse(path.read_text())
        method = next(
            n
            for cls in tree.body
            if isinstance(cls, ast.ClassDef)
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "forward_decode"
        )

        class Base:
            def forward_decode(self, *args, **kwargs):
                if self.fail:
                    raise RuntimeError("attention")
                return 42

        definition = ast.ClassDef(
            name="Wrapper",
            bases=[ast.Name(id="Base", ctx=ast.Load())],
            keywords=[],
            body=[method],
            decorator_list=[],
        )
        namespace = {"Base": Base}
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[definition], type_ignores=[])
                ),
                str(path),
                "exec",
            ),
            namespace,
        )
        for fail in (False, True):
            owner = namespace["Wrapper"]()
            owner.fail, owner._engram_lookup_schedule = fail, Mock()
            if fail:
                with self.assertRaisesRegex(RuntimeError, "attention"):
                    owner.forward_decode(None, None)
                owner._engram_lookup_schedule.abort.assert_called_once()
                owner._engram_lookup_schedule.after_attention.assert_not_called()
            else:
                self.assertEqual(owner.forward_decode(None, None), 42)
                owner._engram_lookup_schedule.after_attention.assert_called_once()
            owner._engram_lookup_schedule.before_attention.assert_called_once()


def oracle(weight, scales, ids):
    valid = (ids >= 0) & (ids < weight.shape[0])
    safe = ids.clamp(0, weight.shape[0] - 1).reshape(-1)
    q = weight.view(torch.uint8).index_select(0, safe).reshape(*ids.shape, 256)
    sf = scales.index_select(0, safe).reshape(*ids.shape, 8)
    q[~valid], sf[~valid] = 0, 127
    groups = sf.reshape(ids.shape[0], 48, 4).to(torch.int64)
    packed = sum(groups[..., i] << (8 * i) for i in range(4)).to(torch.int32)
    storage = torch.full(
        (48, (ids.shape[0] + 3) // 4 * 4), 0x7F7F7F7F, dtype=torch.int32
    )
    storage[:, : ids.shape[0]] = packed.T
    return q.reshape(ids.shape[0], 6144), storage


class DecodeScheduleCUDA(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
            (10, 0),
            (10, 3),
        ):
            self.skipTest("requires reserved SM100/SM103 GPU")
        from rtp_llm.models_py.modules.dsv41.engram import (
            HostEngramEmbedding,
            _PinnedSharedTable,
        )

        self.device = torch.device("cuda", torch.cuda.current_device())
        context = patch.dict(os.environ, ENV)
        context.start()
        self.addCleanup(context.stop)
        self.addCleanup(torch.cuda.synchronize)
        self.model, self.tables = Model(), {}
        self.marker = torch.zeros(1, device=self.device)
        for index, target in enumerate((1, 14)):
            weight = (
                (
                    torch.arange(257)[:, None] * 17
                    + torch.arange(256)[None, :] * 29
                    + target
                )
                % 256
            ).to(torch.uint8)
            scales = (
                (torch.arange(257)[:, None] * 13 + torch.arange(8)[None, :] * 31) % 256
            ).to(torch.uint8)
            self.tables[target] = weight, scales
            pinned = _PinnedSharedTable(
                weight.view(torch.float8_e4m3fn),
                scales,
                "schedule-" + uuid.uuid4().hex,
                self.device,
            )
            group = (
                torch.distributed.group.WORLD
                if torch.distributed.is_initialized()
                else None
            )
            embedding = HostEngramEmbedding(
                pinned.weight,
                pinned.scales,
                pinned=pinned,
                scale_group=group,
                scale_identity=f"schedule-{self._testMethodName}-L{target}",
            )
            if group is not None and torch.distributed.get_world_size(group) > 1:
                self.assertEqual(
                    embedding._scale_shard_count,
                    torch.distributed.get_world_size(group),
                )
            self.model.layers[target].engram = SimpleNamespace(
                embed_tokens=embedding,
                _direct_fp8_lookup=True,
                q_weight=torch.empty(1, device=self.device),
                layout=SimpleNamespace(n_hash_cols=24, head_dim=256, layer_ids=(1, 14)),
                layer_hash_index=index,
                _lookup_work=None,
            )

    def ids(self, tokens, generation, target):
        ids = (
            torch.arange(tokens * 24).reshape(tokens, 24) * 37
            + generation * 19
            + target
        ) % 257
        ids[::3, ::4] = -1
        ids[1::7, 1::5] = 257
        return ids

    def run_model(
        self, inputs, metadata, *, enabled=True, fallback_layer=None, abandon=False
    ):
        for target in (1, 14):
            self.model.layers[target].engram_hashes = inputs[target]
        with patch.dict(
            os.environ, {"DSV41_DECODE_ENGRAM_PREFETCH": "1" if enabled else "0"}
        ):
            plan = prefetch.DecodeEngramPrefetch(self.model, metadata)
        outputs, active = {}, (
            set(plan._schedule.targets) if plan._schedule is not None else set()
        )
        try:
            for index, layer in enumerate(self.model.layers):
                plan.before_layer(index)
                if abandon and index == 10:
                    layer.attn._engram_lookup_schedule.before_attention()
                    layer.attn._engram_lookup_schedule.before_projection(True)
                    break
                if layer.engram is not None:
                    module, ids = layer.engram, inputs[index]
                    work = module._lookup_work
                    if work is None:
                        outputs[index] = module.embed_tokens.lookup_quantized(
                            ids, self.device
                        )
                    else:
                        outputs[index] = work.consume(ids, ids)
                        module._lookup_work = None
                layer.attn.forward_decode(
                    lambda: self.marker.add_(1), native=index != fallback_layer
                )
            return outputs, active
        finally:
            plan.close()

    def assert_pair(self, pair, ids, target):
        q, sf = pair
        expected, storage = oracle(*self.tables[target], ids)
        self.assertTrue(torch.equal(q.cpu().view(torch.uint8), expected))
        self.assertTrue(
            torch.equal(direct._as_outputs(pair).scale_storage.cpu(), storage)
        )

    @torch.inference_mode()
    def test_multiple_batch_and_query_graphs_changed_ids_padding_and_storage(self):
        records = []
        try:
            for batch, query in (
                (1, 1),
                (1, 6),
                (2, 1),
                (7, 1),
                (8, 1),
                (9, 1),
                (7, 6),
                (37, 1),
                (63, 6),
                (64, 6),
                (65, 6),
                (128, 6),
                (385, 1),
                (256, 6),
            ):
                tokens = batch * query
                inputs = {t: self.ids(tokens, 0, t).to(self.device) for t in (1, 14)}
                metadata = SimpleNamespace(batch_size=batch, q_len_per_req=query)
                _, active = self.run_model(inputs, metadata)
                self.assertEqual(active, {1, 14})
                torch.cuda.synchronize(self.device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs, active = self.run_model(inputs, metadata)
                records.append((graph, metadata, inputs, outputs, tokens))
            pointers = [
                pair[0].data_ptr()
                for _, _, _, outputs, _ in records
                for pair in outputs.values()
            ]
            self.assertEqual(len(pointers), len(set(pointers)))
            for generation in (1, 2, 3):
                for graph, metadata, inputs, outputs, tokens in reversed(records):
                    for target in (1, 14):
                        inputs[target].copy_(self.ids(tokens, generation, target))
                        outputs[target][0].view(torch.uint8).fill_(0xA5)
                        direct._as_outputs(outputs[target]).scale_storage.fill_(-1)
                    metadata._engram_decode_tables.validate()
                    graph.replay()
                    torch.cuda.synchronize(self.device)
                    for target in (1, 14):
                        self.assert_pair(
                            outputs[target],
                            self.ids(tokens, generation, target),
                            target,
                        )
        finally:
            torch.cuda.synchronize(self.device)
            for graph, *_ in records:
                graph.reset()

    @torch.inference_mode()
    def test_runtime_core_fallback_abandonment_and_cold_capture(self):
        tokens = 37
        inputs = {t: self.ids(tokens, 0, t).to(self.device) for t in (1, 14)}
        for module in (self.model.layers[t].engram for t in (1, 14)):
            module.embed_tokens.lookup_quantized(inputs[1], self.device, prepare=True)
        torch.cuda.synchronize(self.device)
        metadata = SimpleNamespace(batch_size=tokens, q_len_per_req=1)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                outputs, active = self.run_model(inputs, metadata)
            self.assertEqual(active, set())
            graph.replay()
            torch.cuda.synchronize(self.device)
            for target in (1, 14):
                self.assert_pair(outputs[target], inputs[target].cpu(), target)
        finally:
            graph.reset()
        for fallback in (10, 11, 13):
            outputs, active = self.run_model(
                inputs,
                SimpleNamespace(batch_size=tokens, q_len_per_req=1),
                fallback_layer=fallback,
            )
            self.assertEqual(active, {1, 14})
            torch.cuda.synchronize(self.device)
            for target in (1, 14):
                self.assert_pair(outputs[target], inputs[target].cpu(), target)
        self.run_model(
            inputs, SimpleNamespace(batch_size=tokens, q_len_per_req=1), abandon=True
        )
        torch.cuda.synchronize(self.device)
        self.assertTrue(
            all(self.model.layers[t].engram._lookup_work is None for t in (1, 14))
        )

        abandoned_meta = SimpleNamespace(batch_size=tokens, q_len_per_req=1)
        abandoned = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(abandoned):
                outputs, _ = self.run_model(inputs, abandoned_meta, abandon=True)
            abandoned.replay()
            torch.cuda.synchronize(self.device)
            self.assert_pair(outputs[1], inputs[1].cpu(), 1)
        finally:
            abandoned.reset()

    @torch.inference_mode()
    def test_replaced_table_requires_new_metadata_and_keeps_old_graph_alive(self):
        from rtp_llm.models_py.modules.dsv41.engram import (
            HostEngramEmbedding,
            _PinnedSharedTable,
        )

        inputs = {t: self.ids(9, 0, t).to(self.device) for t in (1, 14)}
        metadata = SimpleNamespace(batch_size=3, q_len_per_req=3)
        self.run_model(inputs, metadata)
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                outputs, active = self.run_model(inputs, metadata)
            self.assertEqual(active, {1, 14})
            module = self.model.layers[14].engram
            old = weakref.ref(module.embed_tokens._pinned)
            weight, scales = self.tables[14]
            pinned = _PinnedSharedTable(
                (weight ^ 1).view(torch.float8_e4m3fn),
                scales,
                "schedule-reload-" + uuid.uuid4().hex,
                self.device,
            )
            module.embed_tokens = HostEngramEmbedding(
                pinned.weight, pinned.scales, pinned=pinned
            )
            gc.collect()
            self.assertIsNotNone(old())
            with self.assertRaisesRegex(RuntimeError, "recreate decode CUDA graphs"):
                metadata._engram_decode_tables.validate()
            graph.replay()
            torch.cuda.synchronize(self.device)
            self.assert_pair(outputs[14], inputs[14].cpu(), 14)
            new_metadata = SimpleNamespace(batch_size=9, q_len_per_req=1)
            new_outputs, active = self.run_model(inputs, new_metadata)
            self.assertEqual(active, {1, 14})
            torch.cuda.synchronize(self.device)
            expected, _ = oracle(weight ^ 1, scales, inputs[14].cpu())
            self.assertTrue(
                torch.equal(new_outputs[14][0].cpu().view(torch.uint8), expected)
            )
        finally:
            torch.cuda.synchronize(self.device)
            graph.reset()
        del metadata, graph
        gc.collect()
        self.assertIsNone(old())


if __name__ == "__main__":
    unittest.main()
