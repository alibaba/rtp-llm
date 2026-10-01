"""The native PageRR fixture through the C++ Graph runner and serial models."""
from types import SimpleNamespace
from contextlib import nullcontext
import faulthandler
import os
import json
import copy
from pathlib import Path

import torch

from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class PageRRTestModel(GptModelBase):
    """Two real MLA layers with fixture-produced Q/KV and independent cache pools.

    Input IDs select mutable fixture payloads, like an embedding lookup. The
    production factory, attention, metadata, cache writer and V projection all
    execute unchanged. This is an attention lifecycle test, not a full K3 model.
    """

    def __init__(self, fixture, parallelism, fmha_config, max_batch, queries, max_seq_len,
                 cache_capacity_tokens=131076):
        self.queries = queries
        self.query_heads = fixture.q.shape[1]
        self.local_heads = fixture.config.head_num
        self.nope = fixture.config.nope_head_dim
        self.dim = self.nope + fixture.config.rope_head_dim
        self.hidden_size = 2 * self.local_heads * 128
        self.graph_instances = {}
        self.layer_map = [1, fixture.group_id, 1, fixture.group_id]
        first = fixture.weights[fixture.layer_id]
        weights = [{}, first, {}, {W.mla_kc: first[W.mla_kc] * 0.5,
                                    W.mla_vc: first[W.mla_vc] * -2.0}]
        attention_config = fixture.config
        device = fixture.q.device
        angles = (torch.arange(max_seq_len + queries + 16, device=device)[:, None]
                  * (torch.arange(32, device=device)[None, :] + 1) / 97)
        cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1)
        config = SimpleNamespace(
            num_layers=4, vocab_size=max_batch * queries, max_seq_len=max_seq_len,
            quant_config=None, getAttentionConfigs=lambda _: attention_config,
        )
        # The real factory receives the same weight-owner interface as the
        # existing native test, including the global RoPE cache.
        weight = SimpleNamespace(weights=weights,
                                 get_global_weight=lambda _: cos_sin)
        super().__init__(config, parallelism, weight, max_batch, fmha_config)
        self.cos_sin = cos_sin
        payload_width = self.query_heads * self.dim + 512 + 64
        self.payload = torch.zeros((max_batch * queries, payload_width),
                                   dtype=torch.bfloat16, device=device)
        owner_page = fixture.config.tokens_per_block
        # Physical pool size follows the fixture's largest live working set,
        # independently of the model capacity used for Graph descriptors.
        local_capacity = ((cache_capacity_tokens + owner_page * parallelism.tp_size - 1)
                          // (owner_page * parallelism.tp_size) * owner_page)
        pages = max_batch * ((local_capacity + fixture.config.kernel_tokens_per_block - 1)
                             // fixture.config.kernel_tokens_per_block) + 8
        self.caches = [torch.zeros((pages, fixture.config.kernel_tokens_per_block, 576),
                                  dtype=fixture.dtype, device=device) for _ in range(2)]

    def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
        impl = super().prepare_fmha_impl(inputs, is_cuda_graph)
        if is_cuda_graph:
            # The C++ runner retains these same objects. Keeping the last one
            # per bucket lets this test inspect fixed metadata/workspace pointers.
            key = (inputs.attention_inputs.context_total_kv_length
                   if inputs.attention_inputs.is_mtp_draft_update
                   else impl.fmha_params.local_causal_lens.shape[0])
            self.graph_instances[key] = impl
        return impl

    def forward(self, inputs, fmha_impl=None):
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        payload = self.payload.index_select(0, inputs.input_ids.long())
        second_payload = payload.clone()
        q_end = self.query_heads * self.dim
        q = payload[:, :q_end].reshape(-1, self.query_heads, self.dim).contiguous()
        ckv = payload[:, q_end:q_end + 512].contiguous()
        k_pe = payload[:, q_end + 512:]
        assert k_pe.stride(0) == payload.shape[1] and k_pe.storage_offset() > 0
        first = fmha_impl.forward(q, ckv, k_pe,
                                  SimpleNamespace(kv_cache_base=self.caches[0]), 1)
        second_q = second_payload[:, :q_end].reshape(-1, self.query_heads, self.dim).contiguous()
        second_q[..., :self.nope].mul_(-2.0)
        second_q[..., self.nope:].neg_()
        second_k_pe = second_payload[:, q_end + 512:]
        second_k_pe.neg_()
        second = fmha_impl.forward(second_q, -ckv, second_k_pe,
                                   SimpleNamespace(kv_cache_base=self.caches[1]), 3)
        # KC/2 with -2*Q_nope and negative Q_rope/KV keeps logits unchanged.
        # Negative V and -2*VC make layer two's reference exactly 2*layer one.
        return PyModelOutputs(torch.cat((first.flatten(1), second.flatten(1)), dim=1)
                              .to(inputs.input_hiddens.dtype))

    def load(self, fixture, fake=False):
        rows = fixture.q.shape[0]
        payload = torch.cat((fixture.q.flatten(1), fixture.ckv, fixture.k_pe), dim=1)
        self.payload[:rows].copy_(payload)
        count = fixture.cache.kv_cache_base.shape[0]
        assert count <= self.caches[0].shape[0]
        if not fake:
            self.caches[0][:count].copy_(fixture.cache.kv_cache_base)
            self.caches[1][:count].copy_((-fixture.cache.kv_cache_base.float()).to(fixture.dtype))
        inputs = PyModelInputs()
        inputs.input_ids = torch.arange(rows, device=fixture.q.device, dtype=torch.int32)
        inputs.input_hiddens = torch.zeros((rows, self.hidden_size), device=fixture.q.device,
                                          dtype=torch.bfloat16)
        attention = fixture.inputs
        groups = attention.kv_cache_kernel_block_id_device_by_group
        if fake:
            # StreamCacheResource::fakeInitKVBlock uses reserved physical page
            # zero. A fake step must not replace any live request's cache data.
            groups = [torch.zeros_like(table) for table in groups]
            attention.kv_cache_kernel_block_id_device_by_group = groups
            attention.is_fake_stream = True
        attention.kv_cache_kernel_block_id_device = groups[fixture.group_id]
        attention.kv_cache_kernel_block_id_host_by_group = [t.cpu().pin_memory() for t in groups]
        attention.kv_cache_kernel_block_id_host = attention.kv_cache_kernel_block_id_host_by_group[fixture.group_id]
        attention.kv_cache_block_id_device = attention.kv_cache_kernel_block_id_device
        attention.kv_cache_block_id_host = attention.kv_cache_kernel_block_id_host
        layer_map = torch.tensor(self.layer_map, dtype=torch.int32).pin_memory()
        attention.kv_cache_layer_to_group = layer_map
        attention.kv_cache_layer_to_group_host = layer_map
        attention.sequence_lengths_host = attention.sequence_lengths.cpu().pin_memory()
        batch = len(fixture.prefixes)
        attention.cu_seqlens_host = (torch.arange(batch + 1, dtype=torch.int32)
                                    * self.queries).pin_memory()
        attention.cu_seqlens = attention.cu_seqlens_host.cuda()
        attention.decode_cu_seqlens_d = attention.cu_seqlens
        attention.cu_kv_seqlens = torch.cat((torch.zeros(1, dtype=torch.int32, device=fixture.q.device),
            (attention.prefix_lengths + self.queries).cumsum(0, dtype=torch.int32)))
        if self.queries == 1:
            # Ordinary Decode publishes sequence_lengths; its native graph has
            # no prefix_lengths destination (reserved for prefill/verify).
            attention.prefix_lengths = torch.empty(0, dtype=torch.int32, device=fixture.q.device)
            attention.prefix_lengths_host = torch.empty(0, dtype=torch.int32)
        attention.padding_offset = torch.zeros(rows, dtype=torch.int32, device=fixture.q.device)
        attention.logical_request_count = attention.physical_request_count = batch
        attention.logical_token_count = attention.physical_token_count = rows
        inputs.attention_inputs = attention
        return inputs

    def check(self, fixture, output, expected=None):
        expected = (fixture.expected if expected is None else expected).flatten(1)
        expected = torch.cat((expected, expected * 2), dim=1).to(output.dtype)
        torch.testing.assert_close(output, expected,
                                   atol=2e-3 if fixture.dtype == torch.float8_e4m3fn else 1e-3,
                                   rtol=0.015)


def _check_split_buffer_growth(rank, size, parallelism, fmha_config, fixture_fn,
                               model_max_seq_len, fp8, probability_reference):
    """Keep both modes' native graphs alive across growth of their FP32 arenas."""
    from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
    from rtp_llm.ops import DecodeCPMLAFusionMode

    def fixture(queries, generation, prefixes):
        return fixture_fn(parallelism.tp_rank, parallelism.tp_size, queries, fp8,
                          generation + 100 * parallelism.dp_rank, device_index=rank,
                          q_replicated=parallelism.decode_cp_q_replicated,
                          prefix_lengths=prefixes, page=1024, kernel_page=128,
                          independent_requests=True)

    graphs = []
    try:
        for mode in (DecodeCPMLAFusionMode.FUSED, DecodeCPMLAFusionMode.UNFUSED):
            config = copy.deepcopy(fmha_config)
            config.decode_cp_mla_fusion_mode = mode
            initial = fixture(1, 0, [127] * 16)
            model = PageRRTestModel(initial, parallelism, config, max_batch=16,
                                   queries=1, max_seq_len=model_max_seq_len,
                                   cache_capacity_tokens=8193)
            runner = CudaGraphRunner()
            graphs.append((runner, model))
            runner.init_decode(
                model, hidden_size=model.hidden_size, max_seq_len=model_max_seq_len,
                tokens_per_block=1024, kernel_tokens_per_block=128,
                decode_capture_batch_sizes=[16], num_tokens_per_bs=1,
                is_target_verify=False, max_context_batch_size=16,
                kv_cache_layer_to_group=model.layer_map, kv_cache_group_num=3,
                sequence_parallel_size=parallelism.tp_size, sp_steps=0,
                model_data_type=torch.bfloat16,
            )
            backend = model.graph_instances[16].fmha_impl.fia2a_backend
            old_buffers = backend.buffers
            assert backend.splits > 1 and backend.mode == mode
            assert old_buffers.output.dtype == torch.float32

            # On SM103/H96, B16/Q1 allocates 64 rows (S4); B1/Q4 needs
            # 96 rows (S24). Q changes the query geometry, not the workspace key.
            larger = fixture(4, 1, [8192])
            other = PageRRTestModel(larger, parallelism, config, max_batch=1,
                                   queries=4, max_seq_len=model_max_seq_len,
                                   cache_capacity_tokens=8196)
            inputs = other.load(larger)
            impl = other.prepare_fmha_impl(inputs, False)
            grown = impl.fmha_impl.fia2a_backend
            output = other.forward(inputs, impl).hidden_states
            torch.cuda.synchronize()
            other.check(larger, output, probability_reference(larger, grown))
            assert grown.workspace is backend.workspace
            assert grown.splits > 1 and grown.mode == mode
            assert grown.buffers.output.dtype == torch.float32
            assert grown.buffers.capacity > old_buffers.capacity
            assert grown.buffers is not old_buffers and backend.buffers is old_buffers
            print(f"DCP SPLIT_GROWTH rank={rank} dtype={fp8} mode={mode.name} "
                  f"capacity={old_buffers.capacity}->{grown.buffers.capacity} "
                  f"S={backend.splits}->{grown.splits}", flush=True)
            del other, impl, output
            if backend.a2a_buffers is not None:
                old_a2a = backend.a2a_buffers
                # Grow packed-source capacity independently of the local
                # split arena, then replay the original graph below.
                grow_batch = old_a2a.capacity // 4 + 1
                larger = fixture(4, 2, [8192] * grow_batch)
                other = PageRRTestModel(larger, parallelism, config, max_batch=grow_batch,
                                       queries=4, max_seq_len=model_max_seq_len,
                                       cache_capacity_tokens=8196)
                inputs = other.load(larger)
                impl = other.prepare_fmha_impl(inputs, False)
                grown = impl.fmha_impl.fia2a_backend
                output = other.forward(inputs, impl).hidden_states
                torch.cuda.synchronize()
                other.check(larger, output, probability_reference(larger, grown))
                assert grown.a2a_buffers.capacity > old_a2a.capacity
                assert backend.a2a_buffers is old_a2a
                print(f"DCP A2A_GROWTH rank={rank} dtype={fp8} "
                      f"capacity={old_a2a.capacity}->{grown.a2a_buffers.capacity}", flush=True)
                del other, impl, output

        # Both graphs remain captured while each overwrites its own current
        # outputs. Replay their retained allocations with changed live contents.
        for generation in (2, 3):
            for runner, model in graphs:
                values = [0, 1, 1023, 1024, 4095, 8192]
                current = fixture(1, generation, [values[i % len(values)] for i in range(16)])
                inputs = model.load(current)
                assert runner.canRun(inputs) and runner.getCurrentRealGraphSize() == 16
                output = runner.forward(inputs).hidden_states
                torch.cuda.synchronize()
                backend = model.graph_instances[16].fmha_impl.fia2a_backend
                model.check(current, output, probability_reference(current, backend))
        print(f"DCP SPLIT_GROWTH_INTERLEAVE PASS rank={rank} dtype={fp8}", flush=True)
    finally:
        for runner, _ in graphs:
            runner.reset()


@torch.inference_mode()
def run_lifecycle(rank, size, parallelism, fmha_config, fixture_fn, assert_result,
                  probability_reference):
    from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner

    from rtp_llm.ops import DecodeCPMLABackend, DecodeCPMLAFusionMode

    backend_name = fmha_config.decode_cp_mla_backend
    draft = os.environ.get("DCP_TEST_DRAFT_GRAPH_ONLY") == "1"
    model_max_seq_len = int(os.environ.get("DCP_TEST_MODEL_MAX_SEQ_LEN", "1048576"))
    growth_seen = False
    faulthandler.enable(all_threads=True)
    faulthandler.dump_traceback_later(60, repeat=True)
    if backend_name == DecodeCPMLABackend.FIA2A and not draft:
        parallelism.decode_cp_q_replicated = False
        for fp8 in (False, True):
            _check_split_buffer_growth(rank, size, parallelism, fmha_config,
                                      fixture_fn, model_max_seq_len, fp8, probability_reference)
    for q_replicated in (False, True):
        parallelism.decode_cp_q_replicated = q_replicated
        for fp8 in ((False,) if draft else (False, True)):
            for queries in (3, 4):
                print(f"DCP LIFECYCLE START rank={rank} dtype={fp8} qrep={q_replicated} Q={queries}", flush=True)
                def fixture(batch, generation, prefixes):
                    return fixture_fn(
                        parallelism.tp_rank, parallelism.tp_size, queries, fp8,
                        generation + 100 * parallelism.dp_rank, draft=draft, device_index=rank,
                        q_replicated=q_replicated, prefix_lengths=prefixes,
                        page=1024, kernel_page=128, independent_requests=True,
                    )

                first = fixture(1, 0, [127])
                model = PageRRTestModel(first, parallelism, fmha_config,
                                        max_batch=32, queries=queries, max_seq_len=model_max_seq_len)
                runner = CudaGraphRunner()
                try:
                    print(f"DCP LIFECYCLE CAPTURE_START rank={rank}", flush=True)
                    common = dict(
                        hidden_size=model.hidden_size, max_seq_len=model_max_seq_len,
                        tokens_per_block=1024, kernel_tokens_per_block=128,
                        num_tokens_per_bs=queries, max_context_batch_size=16,
                        kv_cache_layer_to_group=model.layer_map, kv_cache_group_num=3,
                        sequence_parallel_size=parallelism.tp_size,
                        sp_steps=queries - 1,
                    )
                    if draft:
                        runner.init_draft_prefill(model, **common)
                    else:
                        runner.init_decode(model, decode_capture_batch_sizes=[1, 4, 8, 16],
                                           is_target_verify=True, model_data_type=torch.bfloat16,
                                           **common)
                    torch.cuda.synchronize()
                    print(f"DCP LIFECYCLE CAPTURE_DONE rank={rank}", flush=True)
                    captured = {key: impl for key, impl in model.graph_instances.items() if key > 0}
                    if draft:
                        assert sorted(captured) == [b * queries for b in range(1, 17)]
                        assert all(impl.fmha_params.local_causal_lens.shape == (16, queries)
                                   for impl in captured.values())
                    pointers = {
                        batch: tuple(t.data_ptr() for t in (
                            impl.fmha_params.positions_d, impl.fmha_params.local_causal_lens,
                            (impl.fmha_params.query_block_tables
                             if impl.fmha_params.query_block_tables is not None
                             else impl.fmha_params.block_tables),
                        ) if t is not None) for batch, impl in captured.items()
                    }
                    workspace = (next(iter(captured.values())).fmha_impl.fia2a_backend.workspace
                                 if backend_name == DecodeCPMLABackend.FIA2A else None)
                    largest_capture = max(captured.values(),
                                          key=lambda impl: impl.fmha_params.local_causal_lens.numel())
                    original_buffers = (largest_capture.fmha_impl.fia2a_backend.buffers
                                        if workspace is not None else None)
                    if workspace is not None:
                        for impl in captured.values():
                            backend = impl.fmha_impl.fia2a_backend
                            requested = fmha_config.decode_cp_mla_fusion_mode
                            assert backend.requested_mode == requested
                            if requested == DecodeCPMLAFusionMode.AUTO:
                                assert backend.fused == (backend.splits == 1)
                            else:
                                assert backend.mode == requested
                    captured_buffers = {
                        key: impl.fmha_impl.fia2a_backend.buffers
                        for key, impl in captured.items()
                    } if workspace is not None else {}

                    def check_layers(model, f, impl, output):
                        expected = probability_reference(f, impl.fmha_impl.fia2a_backend)
                        model.check(f, output, expected)
                        rows, batch = f.q.shape[0], len(f.prefixes)
                        meta = impl.fmha_params
                        held = impl.attn_inputs
                        physical_batch = meta.local_causal_lens.shape[0]
                        assert held.logical_request_count == batch
                        assert held.logical_token_count == rows
                        assert held.physical_request_count == physical_batch
                        assert held.physical_token_count == physical_batch * queries
                        if impl.is_cuda_graph:
                            assert meta.block_tables.shape[1] == (
                                (model_max_seq_len + 1023) // 1024 + queries - 1
                            ) * 8
                        tail_slots = meta.slot_mapping[rows:]
                        assert torch.all((tail_slots == -1) | ((tail_slots >= 0) & (tail_slots < 128)))
                        assert torch.all(meta.block_tables[batch:] == 0)
                        live_meta = SimpleNamespace(
                            positions_d=meta.positions_d[:rows],
                            local_causal_lens=meta.local_causal_lens[:batch],
                            slot_mapping=meta.slot_mapping[:rows],
                        )
                        width = model.local_heads * 128
                        for layer in (0, 1):
                            view = SimpleNamespace(**vars(f))
                            view.cache = SimpleNamespace(kv_cache_base=model.caches[layer])
                            if layer:
                                view.canonical = (-f.canonical.float()).to(f.dtype)
                                view.expected = f.expected * 2
                            actual = output[:, layer * width:(layer + 1) * width].reshape(
                                rows, model.local_heads, 128
                            ).bfloat16()
                            assert_result(view, SimpleNamespace(fmha_params=live_meta), actual,
                                          parallelism.tp_rank, parallelism.tp_size,
                                          expected=expected * 2 if layer else expected)

                    # Replay several physical buckets. The C++ runner generates
                    # rectangular Q3 padding and owns metadata copies/event ordering.
                    for generation, batch in enumerate((1, 5, 16)):
                        values = [1023, 1024, 1025, 4095, 4096]
                        if backend_name == DecodeCPMLABackend.FIA2A:
                            values[:2] = [0, 1]
                        f = fixture(batch, generation, [values[i % len(values)] for i in range(batch)])
                        inputs = model.load(f)
                        assert runner.canRun(inputs)
                        bucket = batch * queries if draft else runner.getCurrentRealGraphSize()
                        if not draft:
                            assert bucket >= batch and bucket * queries % parallelism.tp_size == 0
                        prepare_stream = torch.cuda.Stream()
                        prepare_stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(prepare_stream):
                            runner.prepareAttentionInputs(inputs)
                            ready = torch.cuda.Event()
                            ready.record()
                        # AsyncRunner::sync owns this stream dependency in RTP.
                        torch.cuda.current_stream().wait_event(ready)
                        trace_dir = os.environ.get("DCP_TEST_GRAPH_TRACE_DIR")
                        collect = bool(trace_dir and fp8 and q_replicated and queries == 4
                                       and not draft and batch in (1, 16))
                        with (torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CPU,
                                        torch.profiler.ProfilerActivity.CUDA],
                            record_shapes=True,
                        ) if collect else nullcontext()) as profile:
                            output = runner.forward(inputs).hidden_states
                            torch.cuda.synchronize()
                        if collect:
                            directory = Path(trace_dir)
                            directory.mkdir(parents=True, exist_ok=True)
                            stage = "small" if batch == 1 else "large"
                            trace_path = directory / f"mla_{stage}_owner{parallelism.dp_rank}_wr{rank}_probe.json"
                            profile.export_chrome_trace(str(trace_path))
                            events = json.loads(trace_path.read_text())["traceEvents"]
                            assert any(event.get("name", "").startswith("cudaGraphLaunch")
                                       for event in events), trace_path
                            if workspace is not None:
                                backend = captured[bucket].fmha_impl.fia2a_backend
                                kernels = [event["name"] for event in events
                                           if event.get("cat") == "kernel"]
                                pull = backend.a2a_buffers is not None
                                assert sum("PeerBarrier" in name for name in kernels) == (
                                    2 * len(model.caches) if backend.fused or pull else 0
                                ), trace_path
                                assert any("nccl" in name.lower() for name in kernels) == (
                                    not backend.fused and not pull
                                ), trace_path
                                assert sum("_a2a_pull" in name for name in kernels) == (
                                    len(model.caches) if pull else 0
                                ), trace_path
                        impl = captured[bucket]
                        check_layers(model, f, impl, output)
                        assert pointers[bucket] == tuple(t.data_ptr() for t in (
                                impl.fmha_params.positions_d, impl.fmha_params.local_causal_lens,
                                (impl.fmha_params.query_block_tables
                                 if impl.fmha_params.query_block_tables is not None
                                 else impl.fmha_params.block_tables),
                        ) if t is not None)

                    # Real -> fake -> real keeps collective participation and
                    # may write reserved page 0, but cannot touch live pages.
                    real_pages = torch.unique(f.inputs.kv_cache_kernel_block_id_device_by_group[f.group_id])
                    real_pages = real_pages[real_pages > 0].long()
                    before_fake = [cache[real_pages].clone() for cache in model.caches]
                    fake = fixture(1, 8, [0])
                    fake_inputs = model.load(fake, fake=True)
                    assert runner.canRun(fake_inputs)
                    fake_bucket = queries if draft else runner.getCurrentRealGraphSize()
                    runner.forward(fake_inputs)
                    torch.cuda.synchronize()
                    fake_slots = captured[fake_bucket].fmha_params.slot_mapping
                    assert torch.all((fake_slots == -1) | ((fake_slots >= 0) & (fake_slots < 128)))
                    for cache, previous in zip(model.caches, before_fake):
                        torch.testing.assert_close(cache[real_pages].float(), previous.float(), atol=0, rtol=0)

                    # Eager reconstruction shares the TP-group arena. An older
                    # graph retains its exact buffers after later growth.
                    grown_buffers = None
                    for generation in (3, 4):
                        f = fixture(32, generation, [4095 + i % 3 for i in range(32)])
                        inputs = model.load(f)
                        impl = model.prepare_fmha_impl(inputs, False)
                        output = model.forward(inputs, impl).hidden_states
                        torch.cuda.synchronize()
                        check_layers(model, f, impl, output)
                        if workspace is not None:
                            assert impl.fmha_impl.fia2a_backend.workspace is workspace
                            current_buffers = impl.fmha_impl.fia2a_backend.buffers
                            if grown_buffers is None:
                                grown_buffers = current_buffers
                                if 32 * queries > original_buffers.capacity:
                                    assert grown_buffers is not original_buffers
                                    growth_seen = True
                            else:
                                assert current_buffers is grown_buffers

                    # B16 uses the exact S1 allocation that B32 grew above;
                    # B5 alone only exercises a different split-buffer key.
                    for batch in (5, 16):
                        values = [128, 129, 130, 4095, 4096]
                        f = fixture(batch, 5, [values[i % len(values)] for i in range(batch)])
                        inputs = model.load(f)
                        assert runner.canRun(inputs)
                        bucket = batch * queries if draft else runner.getCurrentRealGraphSize()
                        output = runner.forward(inputs).hidden_states
                        torch.cuda.synchronize()
                        check_layers(model, f, captured[bucket], output)
                    if original_buffers is not None:
                        for key, impl in captured.items():
                            candidate = impl.fmha_impl.fia2a_backend
                            assert candidate.buffers is captured_buffers[key]
                        other_fixture = fixture(16, 6, [127] * 16)
                        other = PageRRTestModel(other_fixture, parallelism, fmha_config,
                                               max_batch=32, queries=queries, max_seq_len=model_max_seq_len)
                        other_inputs = other.load(other_fixture)
                        other_impl = other.prepare_fmha_impl(other_inputs)
                        other_output = other.forward(other_inputs, other_impl).hidden_states
                        torch.cuda.synchronize()
                        other.check(other_fixture, other_output,
                                    probability_reference(other_fixture, other_impl.fmha_impl.fia2a_backend))
                        # No model-level arena identity is required. Serial
                        # models may share resources; both must still execute.
                        inputs = model.load(f)
                        assert runner.canRun(inputs)
                        output = runner.forward(inputs).hidden_states
                        torch.cuda.synchronize()
                        check_layers(model, f, captured[bucket], output)
                        del other, other_impl, other_output
                    if parallelism.tp_rank == 0:
                        print(f"DCP LIFECYCLE PASS backend={backend_name.name} ranks={size} "
                              f"dtype={'FP8' if fp8 else 'BF16'} qrep={q_replicated} Q={queries} draft={draft} "
                              f"buckets={sorted(captured)} layers=2 eager_growth=32 "
                              f"owner={parallelism.dp_rank} TP={parallelism.tp_size}", flush=True)
                finally:
                    # Graph-owned NCCL callbacks must be released even when
                    # a comparator or replay rejects an input.
                    runner.reset()
                del runner, model
                torch.cuda.synchronize()
    if backend_name == DecodeCPMLABackend.FIA2A:
        assert growth_seen, "lifecycle run did not exercise arena growth with an older graph alive"
    faulthandler.cancel_dump_traceback_later()
