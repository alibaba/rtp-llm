"""Page-RR Decode production pipeline against a replicated mathematical reference.

DCP_TEST_WORLD_SIZE selects 4/8/16 ranks. A smaller-rank pass validates the
same contracts but does not imply that a larger communication topology ran.
"""

import multiprocessing as mp
import os
import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.distributed.collective_torch import (
    destroy_distributed_environment,
    init_distributed_environment,
)
from rtp_llm.models_py.modules.factory.attention.attn_factory import get_mla_impl
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import mla_dcp_comm
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_dcp_comm import MlaDcpCommunicator
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.page_rr_mla_decode import PageRRMlaDecodeImpl
from rtp_llm.ops import KvCacheDataType, NcclCommConfig, ParallelismConfig, RoleType
from rtp_llm.ops.compute_ops import PyAttentionInputs
from rtp_llm.test.utils.port_util import PortManager
from rtp_llm.utils.model_weight import W


def _cycle_prefixes(batch, values):
    return [values[index % len(values)] for index in range(batch)]


# Every case runs metadata preparation, the real distributed attention path,
# a replicated mathematical reference, and mutable CUDA Graph replay.
_CASES = (
    (1, 1, [31]),
    (2, 1, [127, 128]),
    (8, 1, [1023, 1024, 1025, 4095, 4096, 4097, 32767, 32768]),
    (32, 1, _cycle_prefixes(32, [63, 64, 65, 127, 128, 129])),
    (128, 1, _cycle_prefixes(128, [31, 32, 33, 127, 128, 129])),
    (512, 1, _cycle_prefixes(512, [31, 32, 33, 127, 128, 129])),
    (1, 2, [32767]),
    (8, 2, _cycle_prefixes(8, [4095, 4096, 4097, 32767])),
    (256, 2, _cycle_prefixes(256, [4095, 4096])),
    (32, 4, _cycle_prefixes(32, [65535, 65536, 65537, 131071])),
    (128, 4, _cycle_prefixes(128, [1023, 1024, 1025, 4095, 4096, 4097])),
    (8, 8, _cycle_prefixes(8, [131071, 131072, 131073])),
    (64, 8, _cycle_prefixes(64, [4095, 4096, 4097])),
    (16, 16, _cycle_prefixes(16, [262143, 262144, 262145])),
    (32, 16, _cycle_prefixes(32, [1048575, 1048576, 1048577])),
)


def _rotate(value, positions, cos_sin):
    cos, sin = cos_sin[positions].chunk(2, dim=-1)
    while cos.ndim < value.ndim:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    x, y = value.float().chunk(2, dim=-1)
    return torch.cat((x * cos - y * sin, y * cos + x * sin), dim=-1).to(value.dtype)


def _fixture(rank, size, queries, fp8, generation=0, *, draft=False, prefix_lengths=None, page=128, kernel_page=64):
    torch.manual_seed(1701 + generation)
    device = torch.device("cuda", rank)
    heads, latent, rope, nope, value = 96, 512, 64, 128, 128
    # The short request crosses into a 128:1 split during replay; the other
    # spans a whole CP period. This distinguishes LSE weighting from averaging.
    prefixes = list(prefix_lengths) if prefix_lengths is not None else [page - 2 + generation, page * size + 3 + generation]
    batch, local_heads = len(prefixes), heads // size
    unique_prefixes = list(dict.fromkeys(prefixes))
    prefix_to_group = {prefix: group for group, prefix in enumerate(unique_prefixes)}
    prefix_groups = [prefix_to_group[prefix] for prefix in prefixes]
    group_count = len(unique_prefixes)
    group_ids = torch.tensor(prefix_groups, dtype=torch.long, device=device)
    lengths = [p + queries for p in prefixes]
    tokens = batch * queries
    max_len = max(lengths) + 16
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    q_scale, kv_scale = (0.01, 0.02) if fp8 else (1.0, 1.0)
    q_groups = (torch.randn(group_count, queries, heads, nope + rope, device=device) * 0.2).bfloat16()
    ckv_groups = (torch.randn(group_count, queries, latent, device=device) * 0.2).bfloat16()
    k_pe_groups = (torch.randn(group_count, queries, rope, device=device) * 0.2).bfloat16()
    torch.manual_seed(3301)
    kc = (torch.randn(heads, nope, latent, device=device) * 0.02).bfloat16()
    vc = (torch.randn(heads, latent, value, device=device) * 0.02).bfloat16()
    angles = torch.arange(max_len, device=device)[:, None] * (torch.arange(rope // 2, device=device)[None, :] + 1) / 97
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1)
    group_positions = torch.tensor(
        [[prefix + query for query in range(queries)] for prefix in unique_prefixes],
        dtype=torch.int32,
        device=device,
    )
    q_rotated_groups = q_groups.clone()
    q_rotated_groups[..., nope:] = _rotate(
        q_groups[..., nope:].reshape(group_count * queries, heads, rope),
        group_positions.reshape(-1).long(),
        cos_sin,
    ).view(group_count, queries, heads, rope)
    k_rotated_groups = _rotate(
        k_pe_groups.reshape(group_count * queries, rope),
        group_positions.reshape(-1).long(),
        cos_sin,
    ).view(group_count, queries, rope)
    canonical = (torch.randn(group_count, max_len, latent + rope, device=device) * 0.2).bfloat16()
    for group, prefix in enumerate(unique_prefixes):
        canonical[group, prefix:prefix + queries] = torch.cat(
            (ckv_groups[group], k_rotated_groups[group]), dim=-1
        )
    encoded = (canonical.float() / kv_scale).clamp(-448, 448).to(dtype) if fp8 else canonical
    q_all = q_groups[group_ids].reshape(tokens, heads, nope + rope)
    q_rotated = q_rotated_groups[group_ids].reshape(tokens, heads, nope + rope)
    ckv = ckv_groups[group_ids].reshape(tokens, latent)
    k_pe = k_pe_groups[group_ids].reshape(tokens, rope)
    positions = group_positions[group_ids].reshape(tokens)

    # Real kernel tables are padded views and may select a native draft FULL
    # group distinct from the singular view last selected by a KDA layer.
    width = (max(lengths) + page * size - 1) // (page * size) * (page // kernel_page)
    table_storage = torch.full((3, batch, width + 3), -1, dtype=torch.int32, device=device)
    group_id = 2 if draft else 0
    table = table_storage[group_id, :, :width]
    page_count = batch * width
    logical_pages = torch.arange(page_count, device=device).view(batch, width)
    physical_pages = 1 + (
        page_count - 1 - logical_pages + generation
    ) % page_count
    table.copy_(physical_pages.to(torch.int32))
    local_pages = torch.arange(width, device=device)
    pages_per_owner = page // kernel_page
    global_starts = (
        (local_pages // pages_per_owner) * size + rank
    ) * page + (local_pages % pages_per_owner) * kernel_page
    global_positions = global_starts[:, None] + torch.arange(
        kernel_page, device=device
    )
    safe_positions = global_positions.clamp_max(max_len - 1)
    packed_pages = encoded[
        group_ids[:, None, None], safe_positions[None]
    ]
    valid = global_positions[None] < torch.tensor(
        prefixes, device=device
    )[:, None, None]
    packed_pages = torch.where(
        valid[..., None], packed_pages, torch.zeros((), dtype=dtype, device=device)
    )
    raw_cache = torch.zeros(
        (page_count + 1, kernel_page, latent + rope),
        dtype=dtype,
        device=device,
    )
    raw_cache[physical_pages.reshape(-1)] = packed_pages.reshape(
        page_count, kernel_page, latent + rope
    )
    fields = dict(
        is_prefill=queries > 1, is_target_verify=queries > 1 and not draft,
        is_mtp_draft_update=queries > 1 and draft,
        is_cuda_graph=False, total_tokens=tokens,
        input_lengths=torch.full((batch,), queries, dtype=torch.int32, device=device),
        input_lengths_host=torch.full((batch,), queries, dtype=torch.int32),
        prefix_lengths=torch.tensor(prefixes, dtype=torch.int32, device=device),
        prefix_lengths_host=torch.tensor(prefixes, dtype=torch.int32),
        sequence_lengths=torch.tensor(prefixes, dtype=torch.int32, device=device),
        sequence_lengths_plus_1_d=torch.tensor([p + 1 for p in prefixes], dtype=torch.int32, device=device),
        kv_cache_kernel_block_id_device_by_group=[table_storage[g, :, :width] for g in range(3)],
        kv_cache_kernel_block_id_device=table_storage[1, :, :width],
        # Native Graph keeps the CPU map in this field, without a _host mirror.
        kv_cache_layer_to_group=torch.tensor([2] if draft else [1, 0], dtype=torch.int32).pin_memory(),
        cache_store_inputs=None,
    )
    inputs = PyAttentionInputs()
    for name, field in fields.items():
        setattr(inputs, name, field)
    config = SimpleNamespace(
        head_num=local_heads, kv_lora_rank=latent, rope_head_dim=rope, nope_head_dim=nope,
        kernel_tokens_per_block=kernel_page, tokens_per_block=page, softmax_extra_scale=1.0,
        use_mla=True, indexer_topk=2048, is_sparse=False, mla_fp8_compute=fp8, mla_fp8_q_scale=q_scale, mla_fp8_kv_scale=kv_scale,
        kv_cache_dtype=KvCacheDataType.FP8 if fp8 else KvCacheDataType.BASE,
        rope_config=SimpleNamespace(is_neox_style=True),
    )
    head_slice = slice(rank * local_heads, (rank + 1) * local_heads)
    weights = [{W.mla_kc: kc[head_slice], W.mla_vc: vc[head_slice]}]
    if not draft:
        weights.insert(0, {})  # The target starts with LINEAR, not FULL MLA.
    absorbed = torch.bmm(q_rotated[..., :nope].transpose(0, 1), kc).transpose(0, 1)
    absorbed = torch.cat((absorbed, q_rotated[..., nope:]), dim=-1)
    if fp8:
        absorbed = (absorbed.float() / q_scale).clamp(-448, 448).to(dtype)
    representative_rows = torch.tensor(
        [prefixes.index(prefix) for prefix in unique_prefixes],
        dtype=torch.long,
        device=device,
    )
    local_q = absorbed.view(batch, queries, heads, latent + rope)[
        representative_rows, :, head_slice
    ]
    scores = torch.bmm(
        local_q.reshape(group_count, queries * local_heads, latent + rope).float(),
        encoded.float().transpose(1, 2),
    ).view(group_count, queries, local_heads, max_len)
    scores.mul_((nope + rope) ** -0.5 * q_scale * kv_scale)
    causal_lengths = torch.tensor(
        [[prefix + query + 1 for query in range(queries)] for prefix in unique_prefixes],
        dtype=torch.int64,
        device=device,
    )
    causal_mask = torch.arange(max_len, device=device).view(1, 1, 1, -1)
    scores.masked_fill_(causal_mask >= causal_lengths[:, :, None, None], -float("inf"))
    merged = torch.bmm(
        scores.softmax(-1).reshape(group_count, queries * local_heads, max_len),
        encoded[..., :latent].float(),
    ).view(group_count, queries, local_heads, latent)
    merged.mul_(kv_scale)
    merged_by_head = (
        merged.bfloat16()
        .permute(2, 0, 1, 3)
        .reshape(local_heads, group_count * queries, latent)
    )
    expected_groups = torch.bmm(merged_by_head, vc[head_slice]).permute(1, 0, 2)
    expected = expected_groups.view(group_count, queries, local_heads, value)[
        group_ids
    ].reshape(tokens, local_heads, value)
    return SimpleNamespace(
        config=config, inputs=inputs, weights=weights, cos_sin=cos_sin,
        q=q_all[:, head_slice].contiguous(), ckv=ckv, k_pe=k_pe,
        cache=SimpleNamespace(kv_cache_base=raw_cache), expected=expected,
        positions=positions, canonical=encoded, prefix_groups=prefix_groups,
        prefixes=prefixes, dtype=dtype,
        layer_id=0 if draft else 1, group_id=group_id,
    )


def _reference_local_prefix_tokens(position, rank, size):
    # Count owned complete pages and the optional partial page. This avoids
    # scanning a million Python integers for every query of the 1M fixtures.
    pages, tail = divmod(position + 1, 128)
    owned_pages = (pages + size - 1 - rank) // size
    return owned_pages * 128 + (tail if pages % size == rank else 0)


def _assert_result(fixture, impl, output, rank, size):
    atol = 2e-3 if fixture.dtype == torch.float8_e4m3fn else 1e-3
    torch.testing.assert_close(output, fixture.expected, atol=atol, rtol=0.015)
    metadata = impl.fmha_params
    torch.testing.assert_close(metadata.positions_d, fixture.positions, rtol=0, atol=0)
    expected_lengths = [_reference_local_prefix_tokens(p, rank, size) for p in fixture.positions.tolist()]
    assert metadata.local_causal_lens.flatten().tolist() == expected_lengths
    slots = metadata.slot_mapping.tolist()
    for row, position in enumerate(fixture.positions.tolist()):
        if (position // 128) % size != rank:
            assert slots[row] == -1
        else:
            assert slots[row] >= 0
            actual = fixture.cache.kv_cache_base.view(-1, 576)[slots[row]]
            batch = row // (len(slots) // len(fixture.prefixes))
            group = fixture.prefix_groups[batch]
            cache_atol = 0 if fixture.dtype == torch.float8_e4m3fn else 1e-6
            torch.testing.assert_close(
                actual.float(),
                fixture.canonical[group, position].float(),
                rtol=0,
                atol=cache_atol,
            )


def _worker(rank, size, port):
    torch.cuda.set_device(rank)
    parallelism = ParallelismConfig()
    parallelism.world_rank = parallelism.local_rank = parallelism.tp_rank = rank
    parallelism.world_size = parallelism.local_world_size = parallelism.tp_size = size
    parallelism.dp_size = 1
    parallelism.role_type = RoleType.DECODE
    parallelism.decode_cp_kv_cache_sharded = True
    init_distributed_environment(parallelism, NcclCommConfig(nccl_ip="127.0.0.1"), port, timeout=180)
    graph = None
    try:
        for fp8 in (False, True):
            for batch, queries, prefixes in _CASES:
                assert batch * queries <= 512
                fixture = _fixture(rank, size, queries, fp8, draft=not fp8, prefix_lengths=prefixes)
                if queries == 1:
                    # Match the runner's synthetic capture descriptor;
                    # replay below switches to live plus-one lengths.
                    fixture.inputs.sequence_lengths_plus_1_d.zero_()
                impl = get_mla_impl(
                    fixture.config,
                    SimpleNamespace(weights=fixture.weights, get_global_weight=lambda key: fixture.cos_sin),
                    fixture.inputs, parallelism_config=parallelism,
                    is_cuda_graph=True,
                )
                assert isinstance(impl, PageRRMlaDecodeImpl)
                comm = mla_dcp_comm.get_mla_dcp(fixture.config, fixture.q.device, fixture.dtype)
                assert comm.backend == "a2a"
                assert impl.fmha_params.cache_group_id == fixture.group_id, (
                    f"Page-RR selected group {impl.fmha_params.cache_group_id}, expected {fixture.group_id}"
                )
                output = impl.forward(fixture.q.clone(), fixture.ckv, fixture.k_pe.clone(), fixture.cache, fixture.layer_id)
                _assert_result(fixture, impl, output, rank, size)
                q, k_pe = fixture.q.clone(), fixture.k_pe.clone()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        impl.forward(q, fixture.ckv, k_pe, fixture.cache, fixture.layer_id)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = impl.forward(q, fixture.ckv, k_pe, fixture.cache, fixture.layer_id)
                addresses = tuple(x.data_ptr() for x in (impl.fmha_params.positions_d, impl.fmha_params.slot_mapping, impl.fmha_params.local_causal_lens, impl.fmha_params.query_block_tables))
                for generation in (1, 2):
                    live = _fixture(rank, size, queries, fp8, generation, draft=not fp8, prefix_lengths=prefixes)
                    # Weights/RoPE stay fixed; mutate live tensors at
                    # their captured addresses, including physical IDs.
                    q.copy_(live.q)
                    k_pe.copy_(live.k_pe)
                    fixture.ckv.copy_(live.ckv)
                    fixture.cache.kv_cache_base.copy_(live.cache.kv_cache_base)
                    for field in ("prefix_lengths", "sequence_lengths", "sequence_lengths_plus_1_d"):
                        getattr(fixture.inputs, field).copy_(getattr(live.inputs, field))
                    fixture.inputs.kv_cache_kernel_block_id_device_by_group[fixture.group_id].copy_(live.inputs.kv_cache_kernel_block_id_device_by_group[live.group_id])
                    impl.prepare_cuda_graph(fixture.inputs)
                    allocated = torch.cuda.memory_allocated()
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.cuda.memory_allocated() == allocated
                    live.cache = fixture.cache
                    _assert_result(live, impl, output, rank, size)
                    assert addresses == tuple(x.data_ptr() for x in (impl.fmha_params.positions_d, impl.fmha_params.slot_mapping, impl.fmha_params.local_causal_lens, impl.fmha_params.query_block_tables))
                graph.reset()
                graph = None
                torch.cuda.synchronize()
                if rank == 0:
                    print(f"DCP PASS ranks={size} backend=a2a dtype={fixture.dtype} batch={batch} q={queries}", flush=True)

        fixture = _fixture(rank, size, 1, False)
        a2a = MlaDcpCommunicator(96 // size, 576, 512, fixture.dtype, fixture.q.device)
        assert a2a.backend == "a2a"
        try:
            MlaDcpCommunicator(96 // size, 576, 512, torch.float32, fixture.q.device)
        except ValueError as error:
            assert "BF16/E4M3" in str(error)
        else:
            raise AssertionError("unsupported communication dtype must fail during initialization")
        if rank == 0:
            print("DCP PASS fixed A2A backend and lazy communicator contract", flush=True)
    finally:
        # NCCL finalization waits for captured graph callbacks to be released.
        # Destroy the graph while its communicator is still alive.
        if graph is not None:
            torch.cuda.synchronize()
            graph.reset()
        mla_dcp_comm._communicators.clear()
        destroy_distributed_environment()


class PageRRMlaDecodeTest(unittest.TestCase):
    def test_prefix_reference_matches_token_ownership(self):
        for size in (4, 8, 16):
            counts = [0] * size
            for position in range(128 * size * 3 + 19):
                counts[(position // 128) % size] += 1
                for rank in range(size):
                    self.assertEqual(_reference_local_prefix_tokens(position, rank, size), counts[rank])

    def test_production_page_rr_decode(self):
        size = int(os.environ.get("DCP_TEST_WORLD_SIZE", "8"))
        self.assertIn(size, (4, 8, 16))
        self.assertGreaterEqual(torch.cuda.device_count(), size, "DCP test must not pass by skipping missing GPUs")
        mp.set_start_method("spawn", force=True)
        ports, locks = PortManager().get_consecutive_ports(1)
        processes = [mp.Process(target=_worker, args=(rank, size, ports[0]), name=f"dcp-rank-{rank}") for rank in range(size)]
        try:
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=840)
                self.assertFalse(process.is_alive(), f"{process.name} timed out")
                self.assertEqual(process.exitcode, 0, process.name)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                if process.pid is not None:
                    process.join(timeout=10)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=10)
            for lock in locks:
                lock.__exit__(None, None, None)



if __name__ == "__main__":
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    unittest.main()
