"""Real SM90 peer-scale BF16 lookup, outer-model ownership and changed graphs.

The caller owns three GPU leases. WORLD rendezvous always precedes the final
noncontiguous subgroup: Torch 2.11's rendezvous history is process-wide.
Direct FP8 lookup remains disabled on SM90. The reference is the original full
local GPU-scale HostEngramEmbedding path, including scale-byte 0/255 semantics.
"""

import ast
import gc
import os
import tempfile
import unittest
import weakref
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from rtp_llm.models_py.modules.dsv41 import _engram_decode_prefetch as ownership
from rtp_llm.models_py.modules.dsv41 import _engram_scale_shards as provider
from rtp_llm.models_py.modules.dsv41.engram import (
    Engram,
    EngramLayout,
    HostEngramEmbedding,
    NgramHashState,
    _PinnedSharedTable,
    create_engram_prefetch,
)
from rtp_llm.models_py.modules.dsv41.test.test_engram_scale_shards import (
    forbid_forward_collectives,
)


def _same_bits(actual, expected):
    if actual.shape != expected.shape or actual.dtype != torch.bfloat16:
        raise AssertionError("BF16 output geometry changed")
    if not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
        raise AssertionError("BF16 bytes differ from full local GPU scales")


def _tables(group, device, rows, label, finite=False):
    rank, peers = dist.get_rank(group), dist.get_world_size(group)
    identity = f"sm90-scale-ut-{dist.get_process_group_ranks(group)}-{label}-{rows}"
    row = torch.arange(rows, dtype=torch.int64)[:, None]
    if finite:
        weight = (((row * 17 + torch.arange(256)[None, :]) % 33 - 16) / 16).to(
            torch.float8_e4m3fn
        )
        scales = ((row + torch.arange(8)[None, :]) % 6 + 124).to(torch.uint8)
    else:
        weight = (
            ((row * 17 + torch.arange(256)[None, :] * 29) % 256)
            .to(torch.uint8)
            .view(torch.float8_e4m3fn)
        )
        scales = ((row * 13 + torch.arange(8)[None, :] * 31) % 256).to(torch.uint8)
    pinned = _PinnedSharedTable(weight, scales, identity, device)
    full = HostEngramEmbedding(pinned.weight, pinned.scales, pinned=pinned)
    sharded = HostEngramEmbedding(
        pinned.weight,
        pinned.scales,
        pinned=pinned,
        scale_group=group,
        scale_identity=identity,
    )
    descriptor = sharded._uva[1]
    assert isinstance(descriptor, provider.ShardedScaleDescriptor)
    assert descriptor.group_size == peers and descriptor.rank == rank
    assert descriptor.rows_per_shard == (rows + peers - 1) // peers
    assert descriptor.local_scales.shape == (descriptor.rows_per_shard, 8)
    assert sharded._legacy_scale_uva is None
    assert not sharded._quantized_lookup_supported
    again = provider.create_sharded_scales(pinned.scales, group, device, identity)
    assert again is descriptor
    dist.barrier(group=group)  # All full copies and owner shards exist first.
    if rank == 0:
        pinned.scales.fill_(91)
    dist.barrier(group=group)
    return pinned, full, sharded


def _ids(rows, tokens, generation, peers, dtype):
    ids = (torch.arange(tokens * 24).reshape(tokens, 24) * 37 + generation * 53) % rows
    capacity = (rows + peers - 1) // peers
    if tokens:
        for owner in range(peers):
            begin = owner * capacity
            if begin < rows:
                ids[0, owner] = (
                    begin if generation % 2 else min(rows, begin + capacity) - 1
                )
        large = 2**62 if dtype == torch.int64 else 2**31 - 1
        ids[:, -4:] = torch.tensor([-1, rows, large, -large])
    return ids.to(dtype)


def _host_graphs(group, device, label):
    records = []
    peers = dist.get_world_size(group)
    for rows in (257, 1):  # includes empty owner shards when N < P
        pinned, full, sharded = _tables(group, device, rows, label)
        model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.engram = SimpleNamespace(embed_tokens=sharded)
        layer.engram_hashes = None
        model.layers = [layer]
        graphs = []
        try:
            for dtype in (torch.int32, torch.int64):
                for tokens in (0, 1, 16, 96, 128, 257):
                    backing = torch.empty(
                        (tokens * 2 + 2, 50), dtype=dtype, device=device
                    )
                    ids = backing[1 : 1 + tokens * 2 : 2, 1:49:2]
                    ids.copy_(_ids(rows, tokens, 0, peers, dtype))
                    for prefetch in (False, True):
                        out = torch.empty(
                            (tokens, 24, 256), dtype=torch.bfloat16, device=device
                        )
                        sharded(ids, device, prefetch=prefetch, out=out)
                        metadata = SimpleNamespace(
                            batch_size=max(tokens, 1), q_len_per_req=1
                        )
                        graph = torch.cuda.CUDAGraph()
                        with forbid_forward_collectives(), torch.cuda.graph(graph):
                            plan = ownership.DecodeEngramPrefetch(model, metadata)
                            try:
                                sharded(ids, device, prefetch=prefetch, out=out)
                            finally:
                                plan.close()
                        address = out.data_ptr()
                        for generation in (1, 2, 3):
                            ids.copy_(_ids(rows, tokens, generation, peers, dtype))
                            expected = full(ids, device, prefetch=prefetch)
                            out.view(torch.uint8).fill_(0xA5)
                            with forbid_forward_collectives():
                                graph.replay()
                            _same_bits(out, expected)
                            assert out.data_ptr() == address
                        metadata._engram_decode_tables.validate()
                        graphs.append((graph, metadata))
                        records.append(
                            dict(
                                table_rows=rows,
                                tokens=tokens,
                                dtype=str(dtype),
                                stride=list(ids.stride()),
                                prefetch=prefetch,
                                changed_replays=3,
                                status="PASS",
                            )
                        )
            torch.cuda.synchronize(device)
            dist.barrier(group=group)
        finally:
            for graph, metadata in graphs:
                graph.reset()
                del metadata._engram_decode_tables
            torch.cuda.synchronize(device)
            # Registration may only retire after every rank has drained readers.
            dist.barrier(group=group)
    return records


def _real_decode_loop():
    """Use the production inner owner-reuse/finally code and real injection.

    Attention/MoE are outside this scale-source test. Only their layer-math
    callback is replaced; the actual Block._inject_engram executes WKV and gate.
    """
    from rtp_llm.models_py.modules.dsv41.block import Block

    source = Path(__file__).resolve().parents[1] / "decode" / "forward.py"
    function = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "forward_layers"
    )
    namespace = dict(
        _rt=SimpleNamespace(ENABLED=False),
        _profiler=SimpleNamespace(
            make_layer_forward_range=lambda: lambda _: nullcontext()
        ),
        prepare_model_gate_routing_metadata=lambda *args: None,
        forward_decode_layer=lambda layer, hidden, *args, **kwargs: Block._inject_engram(
            layer, hidden
        ),
        expert_load_for_layer=lambda *args: None,
    )
    future = ast.parse("from __future__ import annotations").body[0]
    exec(compile(ast.Module([future, function], []), str(source), "exec"), namespace)
    return namespace["forward_layers"]


def _model_fixture(group, device, label):
    from rtp_llm.models_py.model_desc.deepseek_v41_model import DeepSeekV41Model
    from rtp_llm.models_py.modules.dsv41.block import Block
    from rtp_llm.models_py.modules.dsv41.engram_distributed import EngramModelAttachment
    from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear

    config = dict(
        engram_layer_ids=[1, 14],
        engram_num_embeddings=[32768, 32768],
        engram_max_ngram_size=4,
        engram_n_heads=8,
        engram_head_dim=256,
        engram_compressed_vocab_size=997,
        engram_pad_token_id=2,
        engram_vocab_size=997,
        image_token_id=991,
    )
    layout = EngramLayout(config)
    state = NgramHashState(layout, token_map=list(range(997)), device=device)
    state.prepare_device(device)
    tables = [
        _tables(group, device, 32768, label + f"-layer{i}", finite=True)
        for i in (1, 14)
    ]
    torch.manual_seed(20261123)
    projection = V41MXFP8Linear(
        (torch.randn(25600, 6144, device=device) * 0.01).to(torch.float8_e4m3fn),
        torch.ones(800, 192, device=device).to(torch.float8_e8m0fnu),
    )
    q = torch.randn(4, 5120, device=device, dtype=torch.bfloat16) * 0.01
    k = torch.randn_like(q) * 0.01

    def make(sharded):
        model = DeepSeekV41Model.__new__(DeepSeekV41Model)
        torch.nn.Module.__init__(model)
        model.v4 = torch.nn.Module()
        model.v4.embed = torch.nn.Embedding(1, 1, device=device, dtype=torch.bfloat16)
        layers = []
        modules = []
        for i in range(15):
            layer = Block.__new__(Block)
            torch.nn.Module.__init__(layer)
            layer.layer_id, layer.engram = i, None
            layer.engram_hashes = layer.engram_token_mask = layer.engram_prefetch = None
            if i in (1, 14):
                index = (1, 14).index(i)
                layer.engram = Engram(
                    layout,
                    index,
                    tables[index][2 if sharded else 1],
                    projection,
                    q,
                    k,
                    1e-20,
                )
                assert not layer.engram._direct_fp8_lookup
                modules.append((i, index, layer.engram))
            layers.append(layer)
        model.v4.layers = torch.nn.ModuleList(layers)
        model.v4.capture_aux_hidden_layer_ids = ()
        model.v4._mtp_hidden_buffer = None
        model.v4._hc_head_reduce = lambda hidden: hidden.flatten(-2)
        model.v4.norm = torch.nn.Identity()
        model._engram_attachment = EngramModelAttachment(
            phase="READY",
            external=False,
            hash_state=state,
            layers=tuple(modules),
            layer_indices=((1, 0), (14, 1)),
        )
        model.kv_cache = object()
        model._v4_args = SimpleNamespace(v41_config=config)
        model._engram_forward_active = False
        model._is_decode_role = True
        model._image_plan = None
        model._engram_prefetch = None
        model._engram_prefetch_max_rows = 0
        if sharded:
            model._engram_prefetch, model._engram_prefetch_max_rows = (
                create_engram_prefetch(
                    device, ((i, module.embed_tokens) for i, _, module in modules)
                )
            )
            assert model._engram_prefetch is not None
        model._engram_verify_prefetch_enabled = model._engram_prefetch is not None
        model._engram_prefetch_has_device_scales = bool(
            model._engram_prefetch is not None
            and any(
                m.embed_tokens._device_scale_lease is not None for _, _, m in modules
            )
        )
        assert model._engram_prefetch_has_device_scales == sharded
        return model

    return make(False), make(True), tables


def _model_graphs(group, device, label):
    from rtp_llm.models_py.model_desc.deepseek_v41_base_model import DeepSeekV4Model

    reference, candidate, tables = _model_fixture(group, device, label)
    inner = _real_decode_loop()
    actual_owner = ownership.DecodeEngramPrefetch
    captured = []

    def consumer(model, inputs, fmha_impl=None):
        outer = getattr(model.v4, "_engram_lookup_prefetch", None)
        if model._engram_prefetch_has_device_scales:
            assert isinstance(outer, actual_owner)
            if torch.cuda.is_current_stream_capturing():
                assert (
                    outer._capture_permit is not None and outer._capture_permit.active
                )
                fmha_impl.metadata._engram_decode_tables.validate()
        # Clone the shared input because the production Engram consumer may use
        # its documented in-place output path.
        return inner(
            model.v4,
            None,
            inputs.input_ids,
            fmha_impl.metadata,
            None,
            prepare_hidden_fn=lambda **kwargs: inputs.hidden.clone().view(
                fmha_impl.metadata.batch_size, fmha_impl.metadata.q_len_per_req, 4, 5120
            ),
        )

    cases = []
    try:
        with patch.object(DeepSeekV4Model, "forward", new=consumer):
            for verify, span in ((False, 1), (True, 4), (True, 6), (True, 8)):
                for batch in (1, 8, 16):
                    tokens = batch * span
                    windows = torch.empty((tokens, 4), device=device, dtype=torch.int32)
                    hidden = torch.empty(
                        (tokens, 4, 5120), device=device, dtype=torch.bfloat16
                    )
                    inputs = SimpleNamespace(
                        attention_inputs=SimpleNamespace(
                            is_prefill=verify, is_target_verify=verify
                        ),
                        engram_token_windows=windows,
                        input_ids=torch.zeros(tokens, dtype=torch.int32, device=device),
                        hidden=hidden,
                        multimodal_inputs=SimpleNamespace(multimodal_features=None),
                    )
                    meta = SimpleNamespace(batch_size=batch, q_len_per_req=span)
                    fmha = SimpleNamespace(metadata=meta)
                    reference_fmha = SimpleNamespace(
                        metadata=SimpleNamespace(batch_size=batch, q_len_per_req=span)
                    )
                    windows.zero_()
                    hidden.fill_(0.25)
                    for _ in range(2):
                        candidate(inputs, fmha)
                        reference(inputs, reference_fmha)
                    graph = torch.cuda.CUDAGraph()
                    with patch.object(
                        ownership, "DecodeEngramPrefetch", wraps=actual_owner
                    ) as constructors:
                        with forbid_forward_collectives(), torch.cuda.graph(graph):
                            output = candidate(inputs, fmha)
                        assert (
                            constructors.call_count == 1
                        ), "inner decode created a second owner"
                    del constructors  # Mock call history must not retain the model.
                    assert candidate.v4._engram_lookup_prefetch is None
                    assert candidate._engram_prefetch._origin is None
                    assert not candidate._engram_forward_active
                    assert all(
                        layer.engram_hashes is None for layer in candidate.v4.layers
                    )
                    meta._engram_decode_tables.validate()
                    address = output.data_ptr()
                    for generation in (1, 2, 3):
                        live = (
                            torch.arange(tokens * 4).reshape(tokens, 4) * 19
                            + generation * 37
                        ) % 997
                        live[0, 0] = -1 if generation == 2 else 17 + generation
                        if tokens > 1:
                            live[-1, 0] = 991
                        windows.copy_(live)
                        hidden.copy_(torch.randn_like(hidden) * 0.1)
                        expected = reference(inputs, reference_fmha)
                        output.view(torch.uint8).fill_(0xA5)
                        with forbid_forward_collectives():
                            graph.replay()
                        _same_bits(output, expected)
                        assert output.data_ptr() == address
                    # Reject stale generations without touching any scale bytes.
                    embedding = candidate.v4.layers[1].engram.embed_tokens
                    original = embedding._uva
                    embedding._uva = tuple(list(original))
                    try:
                        with unittest.TestCase().assertRaisesRegex(
                            RuntimeError, "recreate decode CUDA graphs"
                        ):
                            meta._engram_decode_tables.validate()
                    finally:
                        embedding._uva = original
                    meta._engram_decode_tables.validate()
                    captured.append((graph, meta))
                    cases.append(
                        dict(
                            batch=batch,
                            span=span,
                            target_verify=verify,
                            changed_replays=3,
                            real_outer_and_prepare=True,
                            real_inner_owner_reuse=True,
                            real_block_wkv_and_inject=True,
                            status="PASS",
                        )
                    )
            # A representative full-GPU scale replica must exercise the same
            # new verify policy. Reuse B16/span8; preserve the fixture's public
            # (reference, candidate, tables) interface used by component benches.
            full_plan, full_limit = create_engram_prefetch(
                device,
                (
                    (i, module.embed_tokens)
                    for i, _, module in reference._engram_attachment.layers
                ),
            )
            assert full_plan is not None
            reference._engram_prefetch = full_plan
            reference._engram_prefetch_max_rows = full_limit
            full_verify = reference._engram_prefetch is not None
            full_device_scales = full_verify and any(
                module.embed_tokens._device_scale_lease is not None
                for _, _, module in reference._engram_attachment.layers
            )
            assert full_verify and full_device_scales
            reference._engram_verify_prefetch_enabled = full_verify
            reference._engram_prefetch_has_device_scales = full_device_scales
            control_meta = SimpleNamespace(batch_size=batch, q_len_per_req=span)
            control_fmha = SimpleNamespace(metadata=control_meta)
            for _ in range(2):
                reference(inputs, control_fmha)
            control_graph = torch.cuda.CUDAGraph()
            with patch.object(
                ownership, "DecodeEngramPrefetch", wraps=actual_owner
            ) as constructors:
                with forbid_forward_collectives(), torch.cuda.graph(control_graph):
                    control_output = reference(inputs, control_fmha)
                assert constructors.call_count == 1
            del constructors
            for generation in (1, 2, 3):
                windows.copy_(
                    (torch.arange(tokens * 4).reshape(tokens, 4) + generation * 43)
                    % 997
                )
                hidden.copy_(torch.randn_like(hidden) * 0.1)
                # The oracle is the original full-local-scale synchronous path,
                # not the candidate's prefetched result.
                reference._engram_prefetch = None
                reference._engram_verify_prefetch_enabled = False
                reference._engram_prefetch_has_device_scales = False
                try:
                    expected = reference(inputs, reference_fmha)
                finally:
                    reference._engram_prefetch = full_plan
                    reference._engram_verify_prefetch_enabled = full_verify
                    reference._engram_prefetch_has_device_scales = full_device_scales
                control_output.view(torch.uint8).fill_(0xA5)
                with forbid_forward_collectives():
                    control_graph.replay()
                _same_bits(control_output, expected)
            assert full_plan._origin is None
            assert reference.v4._engram_lookup_prefetch is None
            control_meta._engram_decode_tables.validate()
            captured.append((control_graph, control_meta))
            cases.append(
                dict(
                    batch=batch,
                    span=span,
                    target_verify=True,
                    full_gpu_scale_verify_control=True,
                    changed_replays=3,
                    status="PASS",
                )
            )
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        # Metadata, rather than the returned tensor, owns captured generations.
        retained = weakref.ref(candidate.v4.layers[1].engram.embed_tokens)
        del embedding, original, candidate, reference, tables
        gc.collect()
        assert retained() is not None
        for graph, meta in captured:
            graph.reset()
            del meta._engram_decode_tables
        captured.clear()
        torch.cuda.synchronize(device)
        gc.collect()
        assert retained() is None, "retired graph metadata retained the embedding"
        return cases
    finally:
        for graph, meta in captured:
            graph.reset()
            if hasattr(meta, "_engram_decode_tables"):
                del meta._engram_decode_tables
        torch.cuda.synchronize(device)
        dist.barrier(group=group)


@torch.inference_mode()
def run_distributed_cases(device=None):
    """Caller initializes WORLD=3 and locks all visible devices before entry."""
    assert dist.is_initialized() and dist.get_world_size() == 3
    device = torch.device(
        device if device is not None else f"cuda:{torch.cuda.current_device()}"
    )
    assert torch.cuda.get_device_capability(device) == (9, 0)
    with patch.dict(
        os.environ, {"DSV41_ENGRAM_DEVICE_SCALES": "1", "DSV41_ENGRAM_PREFETCH": "1"}
    ):
        group = provider.select_scale_group(device, "sm90-bf16-world-then-subgroup")
        assert group is dist.group.WORLD
        results = [
            dict(
                group="world3",
                host=_host_graphs(group, device, "world"),
                model=_model_graphs(group, device, "world"),
            )
        ]
        # Last rendezvous domain in this process. Rank1 must not subsequently
        # enter a WORLD symmetric rendezvous with its now different history.
        subgroup = dist.new_group(
            ranks=[0, 2], backend="nccl", timeout=timedelta(seconds=120)
        )
        if dist.get_rank() in (0, 2):
            results.append(
                dict(
                    group="subgroup_0_2",
                    host=_host_graphs(subgroup, device, "subgroup"),
                    model=_model_graphs(subgroup, device, "subgroup"),
                )
            )
        dist.barrier()
    return dict(
        status="PASS_SM90_BF16_SCALE_SHARDS_REAL_OUTER_MODEL_GRAPH",
        groups=results,
        direct_fp8_enabled=False,
        scope="tiny tables/real WKV and injection, no served model",
    )


def _worker(rank, store):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method="file://" + store,
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=180),
        device_id=torch.device("cuda", rank),
    )
    try:
        result = run_distributed_cases(torch.device("cuda", rank))
        assert result["status"] == "PASS_SM90_BF16_SCALE_SHARDS_REAL_OUTER_MODEL_GRAPH"
    finally:
        dist.destroy_process_group()


class SM90EngramScaleShardsCUDA(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.device_count() >= 3,
        "requires three peer-accessible SM90 GPUs",
    )
    def test_world_then_noncontiguous_subgroup_bf16_and_real_model_graph(self):
        if any(torch.cuda.get_device_capability(i) != (9, 0) for i in range(3)):
            self.skipTest("SM90 GPUs required")
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory(
            prefix="sm90-engram-scale-shards-"
        ) as directory:
            mp.spawn(
                _worker, args=(str(Path(directory) / "store"),), nprocs=3, join=True
            )


if __name__ == "__main__":
    unittest.main()
