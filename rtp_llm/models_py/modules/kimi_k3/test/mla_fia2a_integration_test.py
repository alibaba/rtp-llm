"""Real K3 paged MLA wrapper, including absorption, cache insertion and V-up.

The device-input fixture is the Python module boundary, as in K3 JIT warmup.
It does not exercise C++ PyWrappedModel.buildPyAttentionInputs or a full model.
No attention, projection, cache-write or communication operation is mocked.
"""

import json
import math
import os
import socket
import unittest
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.distributed.collective_torch import (
    destroy_distributed_environment,
    init_distributed_environment,
)
from rtp_llm.models_py.modules.kimi_k3.mla import KimiK3MlaImpl
from rtp_llm.ops import (
    AttentionConfigs,
    FMHAConfig,
    KvCacheDataType,
    NcclCommConfig,
    ParallelismConfig,
    RoleType,
)
from rtp_llm.ops.compute_ops import LayerKVCache
from rtp_llm.utils.model_weight import W

WORLD, HEADS, LOCAL_HEADS = 4, 96, 24
LATENT, SUFFIX, NOPE, VALUE = 512, 64, 128, 128
PHYSICAL_PAGE, KERNEL_PAGE = 1024, 128
BATCH, QUERIES, HISTORY = 2, 4, 6144
SCALE = (NOPE + SUFFIX) ** -0.5


@dataclass
class _DeviceInputs:
    """Tensor/flag inputs; native read-only device mirrors have no Python ctor."""

    input_lengths: torch.Tensor
    prefix_lengths: torch.Tensor
    prefix_lengths_device: torch.Tensor
    sequence_lengths_plus_1_device: torch.Tensor
    kv_cache_kernel_block_id_device: torch.Tensor
    cu_seqlens_device: torch.Tensor
    valid_token_mask: torch.Tensor
    physical_token_count: int = BATCH * QUERIES
    logical_token_count: int = BATCH * QUERIES
    is_prefill: bool = False
    is_target_verify: bool = False
    is_mtp_draft_update: bool = False
    is_cuda_graph: bool = True
    cache_store_inputs: object = None
    cache_store_writer: object = None


def _parallelism(rank, replicated=True):
    config = ParallelismConfig()
    config.world_size = config.local_world_size = config.tp_size = WORLD
    config.world_rank = config.local_rank = config.tp_rank = rank
    config.dp_size = config.ep_size = config.pp_size = 1
    config.dp_rank = config.ep_rank = config.pp_rank = 0
    config.role_type = RoleType.DECODE if replicated else RoleType.PREFILL
    config.prefill_cp_config.kv_cache_sharded = True
    return config


class _Fixture:
    def __init__(self, rank, fp8, replicated, draft):
        self.rank, self.fp8 = rank, fp8
        self.replicated, self.draft = replicated, draft
        self.device = torch.device("cuda", rank)
        self.dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
        self.owned_by_rank = [
            torch.tensor(
                [p for p in range(HISTORY) if (p // PHYSICAL_PAGE) % WORLD == owner],
                dtype=torch.long,
                device=self.device,
            )
            for owner in range(WORLD)
        ]
        self.owned = self.owned_by_rank[rank]
        self.columns = math.ceil(self.owned.numel() / KERNEL_PAGE)
        self.tables = [
            torch.arange(
                1 + bank * BATCH * self.columns,
                1 + (bank + 1) * BATCH * self.columns,
                dtype=torch.int32,
                device=self.device,
            ).reshape(BATCH, self.columns)
            for bank in range(2)
        ]
        self.cache = torch.zeros(
            (1 + 2 * BATCH * self.columns, KERNEL_PAGE, LATENT + SUFFIX),
            dtype=self.dtype,
            device=self.device,
        )
        self.layer_cache = LayerKVCache(self.cache, PHYSICAL_PAGE, 0)
        self.inputs = _DeviceInputs(
            input_lengths=torch.full((BATCH,), QUERIES, dtype=torch.int32),
            prefix_lengths=torch.empty(BATCH, dtype=torch.int32),
            prefix_lengths_device=torch.empty(
                BATCH, dtype=torch.int32, device=self.device
            ),
            sequence_lengths_plus_1_device=torch.empty(
                BATCH, dtype=torch.int32, device=self.device
            ),
            kv_cache_kernel_block_id_device=self.tables[0].clone(),
            cu_seqlens_device=torch.empty(
                BATCH + 1, dtype=torch.int32, device=self.device
            ),
            valid_token_mask=torch.ones(
                BATCH * QUERIES, dtype=torch.bool, device=self.device
            ),
            is_prefill=draft,
            is_target_verify=not draft,
            is_mtp_draft_update=draft,
        )
        # Sparse but head-dependent projections make the independent reference
        # explicit indexing. Production still executes both actual BF16 BMMs.
        heads = torch.arange(HEADS, device=self.device)[:, None]
        feature = torch.arange(LATENT, device=self.device)[None, :]
        self.k_indices = (feature + 3 * heads) % NOPE
        self.k_gain = (0.25 + ((feature // NOPE) % 2) * 0.25).expand(HEADS, -1)
        k_weight = torch.zeros(
            (HEADS, NOPE, LATENT), dtype=torch.bfloat16, device=self.device
        )
        k_weight.scatter_(
            1, self.k_indices[:, None, :], self.k_gain[:, None, :].to(torch.bfloat16)
        )
        local_heads = torch.arange(
            rank * LOCAL_HEADS, (rank + 1) * LOCAL_HEADS, device=self.device
        )[:, None]
        value = torch.arange(VALUE, device=self.device)[None, :]
        self.v_first = (value + 5 * local_heads) % NOPE + (local_heads % 4) * NOPE
        self.v_second = (value + 17 + 7 * local_heads) % NOPE + (
            (local_heads + 1) % 4
        ) * NOPE
        v_weight = torch.zeros(
            (LOCAL_HEADS, LATENT, VALUE), dtype=torch.bfloat16, device=self.device
        )
        v_weight.scatter_(1, self.v_first[:, None, :], 0.5)
        v_weight.scatter_(1, self.v_second[:, None, :], -0.25)
        self.head_slice = (
            slice(None)
            if replicated
            else slice(rank * LOCAL_HEADS, (rank + 1) * LOCAL_HEADS)
        )
        self.weights = [
            {W.mla_kc: k_weight[self.head_slice].contiguous(), W.mla_vc: v_weight}
        ]
        query_heads = HEADS if replicated else LOCAL_HEADS
        self.query = torch.empty(
            (BATCH * QUERIES, query_heads, NOPE + SUFFIX),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.current_kv = torch.empty(
            (BATCH * QUERIES, LATENT), dtype=torch.bfloat16, device=self.device
        )
        self.current_pe = torch.empty(
            (BATCH * QUERIES, SUFFIX), dtype=torch.bfloat16, device=self.device
        )
        self.select_phase(0)

    def select_phase(self, phase):
        self.prefixes = [4094, 3070] if phase == 0 else [1022, 2046]
        counts = [QUERIES, QUERIES]
        if self.draft:
            counts = [3, 2] if phase == 0 else [2, 0]
        self.active = [q < counts[b] for b in range(BATCH) for q in range(QUERIES)]
        self.inputs.input_lengths.copy_(torch.tensor(counts, dtype=torch.int32))
        self.inputs.prefix_lengths.copy_(torch.tensor(self.prefixes, dtype=torch.int32))
        self.inputs.prefix_lengths_device.copy_(self.inputs.prefix_lengths)
        self.inputs.sequence_lengths_plus_1_device.copy_(
            self.inputs.prefix_lengths_device + 1
        )
        self.inputs.kv_cache_kernel_block_id_device.copy_(self.tables[phase])
        self.inputs.cu_seqlens_device.copy_(
            torch.tensor(
                [0, counts[0], sum(counts)], dtype=torch.int32, device=self.device
            )
        )
        self.inputs.valid_token_mask.copy_(
            torch.tensor(self.active, device=self.device)
        )
        row = torch.arange(BATCH * QUERIES, device=self.device)[:, None, None]
        head = torch.arange(HEADS, device=self.device)[None, :, None]
        q_feature = torch.arange(NOPE + SUFFIX, device=self.device)[None, None, :]
        raw_query = (
            ((row * 17 + head * 13 + q_feature * 7 + phase * 19) % 37 - 18).float() / 64
            + ((head % 7).float() - 3) / 32
        ).to(torch.bfloat16)
        self.query.copy_(raw_query[:, self.head_slice])
        # Reference absorption uses explicit lookup, never the production BMM,
        # quantizer, PageRR helper or FIA output.
        selected = (
            raw_query[..., :NOPE]
            .float()
            .gather(2, self.k_indices[None].expand(BATCH * QUERIES, -1, -1))
        )
        self.absorbed = torch.cat(
            ((selected * self.k_gain[None]).to(torch.bfloat16), raw_query[..., NOPE:]),
            dim=-1,
        )
        self.encoded_query = self.absorbed.to(self.dtype)
        position = torch.arange(HISTORY, device=self.device)[None, :, None]
        request = torch.arange(BATCH, device=self.device)[:, None, None]
        feature = torch.arange(LATENT + SUFFIX, device=self.device)[None, None, :]
        history = (
            (
                (
                    (position * 11 + feature * 5 + request * 23 + phase * 29) % 31 - 15
                ).float()
                / 128
                + ((position // PHYSICAL_PAGE) % WORLD).float() / 8
                + request.float() / 64
                + phase / 4
            )
            .to(torch.bfloat16)
            .to(self.dtype)
        )
        current_row = torch.arange(BATCH * QUERIES, device=self.device)[:, None]
        current_feature = torch.arange(LATENT + SUFFIX, device=self.device)[None, :]
        inserts = (
            ((current_row * 7 + current_feature * 13 + phase * 17) % 23 - 11).float()
            / 32
            + 0.75
            + phase / 8
        ).to(torch.bfloat16)
        self.current_kv.copy_(inserts[:, :LATENT])
        self.current_pe.copy_(inserts[:, LATENT:])
        encoded_inserts = inserts.to(self.dtype)
        # Fill the selected bank with old history first. The expected bytes
        # then apply only independently enumerated owner/active insertions.
        first = 1 + phase * BATCH * self.columns
        payload = history.view(torch.uint8).index_select(1, self.owned)
        self.cache[first : first + BATCH * self.columns].view(torch.uint8).copy_(
            payload.reshape_as(
                self.cache[first : first + BATCH * self.columns].view(torch.uint8)
            )
        )
        self.expected_cache = self.cache.view(torch.uint8).clone()
        self.expected_global = history.clone()
        self.expected_slots, self.expected_bounds = [], []
        table = self.tables[phase].cpu().tolist()
        for b, prefix in enumerate(self.prefixes):
            bounds = []
            for q in range(QUERIES):
                row = b * QUERIES + q
                position = prefix + q
                active = self.active[row]
                visible = [
                    p
                    for p in range(position + 1)
                    if (p // PHYSICAL_PAGE) % WORLD == self.rank
                ]
                bounds.append(len(visible) if active else 0)
                slot = -1
                if active:
                    self.expected_global[b, position].view(torch.uint8).copy_(
                        encoded_inserts[row].view(torch.uint8)
                    )
                    if (position // PHYSICAL_PAGE) % WORLD == self.rank:
                        local_index = len(visible) - 1
                        block = table[b][local_index // KERNEL_PAGE]
                        offset = local_index % KERNEL_PAGE
                        slot = block * KERNEL_PAGE + offset
                        self.expected_cache[block, offset].copy_(
                            encoded_inserts[row].view(torch.uint8)
                        )
                self.expected_slots.append(slot)
            self.expected_bounds.append(bounds)

    def reference(self):
        outputs = []
        start, end_head = self.rank * LOCAL_HEADS, (self.rank + 1) * LOCAL_HEADS
        head_index = torch.arange(LOCAL_HEADS, device=self.device)[:, None]
        for b, prefix in enumerate(self.prefixes):
            for q in range(QUERIES):
                row = b * QUERIES + q
                if not self.active[row]:
                    outputs.append(
                        torch.zeros((LOCAL_HEADS, VALUE), device=self.device)
                    )
                    continue
                end = prefix + q + 1
                query = self.encoded_query[row, start:end_head].float()
                partials, lses = [], []
                for owned in self.owned_by_rank:
                    positions = owned[owned < end]
                    if not positions.numel():
                        partials.append(
                            torch.zeros((LOCAL_HEADS, LATENT), device=self.device)
                        )
                        lses.append(
                            torch.full((LOCAL_HEADS,), -torch.inf, device=self.device)
                        )
                        continue
                    kv = self.expected_global[b].float().index_select(0, positions)
                    scores = query @ kv.T * SCALE
                    # Per-owner O is BF16 on the wire, then final merged O is
                    # BF16 before V-up. Keep both explicit rounding points.
                    partials.append(
                        (scores.softmax(-1) @ kv[:, :LATENT]).to(torch.bfloat16).float()
                    )
                    lses.append(scores.logsumexp(-1))
                weights = torch.stack(lses).softmax(0)
                latent = (
                    (torch.stack(partials) * weights[:, :, None])
                    .sum(0)
                    .to(torch.bfloat16)
                    .float()
                )
                projected = (
                    latent[head_index, self.v_first] * 0.5
                    - latent[head_index, self.v_second] * 0.25
                ).to(torch.bfloat16)
                outputs.append(projected.float())
        return torch.stack(outputs)


def _run_case(rank, fp8, replicated, draft, directory):
    fixture = _Fixture(rank, fp8, replicated, draft)
    config = AttentionConfigs()
    config.head_num, config.kv_head_num = LOCAL_HEADS, 1
    config.kv_lora_rank, config.nope_head_dim, config.rope_head_dim = (
        LATENT,
        NOPE,
        SUFFIX,
    )
    config.v_head_dim, config.size_per_head = VALUE, NOPE + SUFFIX
    config.tokens_per_block, config.kernel_tokens_per_block = PHYSICAL_PAGE, KERNEL_PAGE
    config.dtype, config.use_mla, config.mla_fp8_compute = torch.bfloat16, True, fp8
    config.kv_cache_dtype = KvCacheDataType.FP8 if fp8 else KvCacheDataType.BASE
    config.softmax_extra_scale, config.max_seq_len = 1.0, HISTORY
    fmha = FMHAConfig()
    impl = KimiK3MlaImpl(
        config,
        fixture.inputs,
        fixture.weights,
        fmha_config=fmha,
        max_seq_len=HISTORY,
        is_cuda_graph=True,
        parallelism_config=_parallelism(rank, replicated),
    )
    if impl.fp8_compute != fp8 or impl.q_replicated != replicated:
        raise AssertionError("K3 constructor selected an unexpected dtype or Q layout")
    if impl.fia2a_backend.dtype != fixture.dtype:
        raise AssertionError("FIA2A compute dtype must match Q/cache without fallback")
    if impl.requires_host_metadata or not impl.can_capture_cuda_graph_prepare:
        raise AssertionError("K3 paged attention must prepare device metadata in graph")
    communicator = impl.page_rr_communicator
    tail = torch.empty(1, device=fixture.device)

    def step():
        impl.prepare_cuda_graph(fixture.inputs)
        output = impl.forward(
            fixture.query,
            fixture.current_kv,
            fixture.current_pe,
            fixture.layer_cache,
            0,
        )
        # The production wrapper returns before O-projection's following TP
        # collective. A real TP collective with output consumption provides
        # that same peer-buffer reuse ordering contract in this module test.
        tail.copy_(output.float().sum().reshape(1))
        dist.all_reduce(tail, group=communicator.group)
        return output

    def check(output):
        if (
            output.shape != (BATCH * QUERIES, LOCAL_HEADS, VALUE)
            or output.dtype != torch.bfloat16
        ):
            raise AssertionError(
                "K3 V-up output layout/dtype differs from the production contract"
            )
        if not bool(torch.isfinite(output).all()) or not bool(
            torch.isfinite(tail).all()
        ):
            raise AssertionError(
                "K3 MLA or its following collective produced nonfinite values"
            )
        expected = fixture.reference()
        torch.testing.assert_close(output.float(), expected, atol=2e-3, rtol=3e-2)
        if not torch.equal(fixture.cache.view(torch.uint8), fixture.expected_cache):
            raise AssertionError(
                "Owner-only cache insertion or inactive-slot preservation differs bytewise"
            )
        metadata = impl.page_rr_metadata
        if metadata.slot_mapping.cpu().tolist() != fixture.expected_slots:
            raise AssertionError(
                "K3 cache slots differ from independent physical-page ownership"
            )
        if metadata.local_causal_lens.cpu().tolist() != fixture.expected_bounds:
            raise AssertionError(
                "K3 causal lengths differ from independent key enumeration"
            )
        if impl.fia2a_backend.scan_lengths.cpu().tolist() != [
            max(b) for b in fixture.expected_bounds
        ]:
            raise AssertionError("FIA2A did not refresh live maximum query bounds")
        if fp8:
            torch.testing.assert_close(
                impl._absorbed_query_buffer,
                fixture.absorbed[:, fixture.head_slice],
                rtol=0,
                atol=0,
            )
            if not torch.equal(
                impl._query_buffer.view(torch.uint8),
                fixture.encoded_query[:, fixture.head_slice]
                .contiguous()
                .view(torch.uint8),
            ):
                raise AssertionError(
                    "Actual BF16 absorption/FP8 quantization differs from the explicit reference"
                )
        inactive = torch.tensor([not v for v in fixture.active], device=fixture.device)
        if bool(inactive.any()) and not bool((output[inactive] == 0).all()):
            raise AssertionError("Inactive draft rows must stay zero through V-up")
        return float((output.float() - expected).abs().max())

    stream = torch.cuda.Stream(device=fixture.device)
    stream.wait_stream(torch.cuda.current_stream(fixture.device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            eager = step()
    torch.cuda.current_stream(fixture.device).wait_stream(stream)
    torch.cuda.synchronize()
    eager_error = check(eager)
    dist.barrier(group=communicator.group)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = step()
    pointers = tuple(
        t.data_ptr()
        for t in (
            fixture.query,
            fixture.current_kv,
            fixture.current_pe,
            fixture.cache,
            fixture.inputs.prefix_lengths_device,
            fixture.inputs.valid_token_mask,
            impl.page_rr_metadata.local_causal_lens,
            impl.page_rr_metadata.query_block_tables,
            impl.fia2a_backend.scan_lengths,
            impl.fia2a_backend.table,
            output,
        )
    )
    errors = []
    for phase in (0, 1):
        if phase:
            fixture.select_phase(phase)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        errors.append(check(output))
    after = tuple(
        t.data_ptr()
        for t in (
            fixture.query,
            fixture.current_kv,
            fixture.current_pe,
            fixture.cache,
            fixture.inputs.prefix_lengths_device,
            fixture.inputs.valid_token_mask,
            impl.page_rr_metadata.local_causal_lens,
            impl.page_rr_metadata.query_block_tables,
            impl.fia2a_backend.scan_lengths,
            impl.fia2a_backend.table,
            output,
        )
    )
    if pointers != after:
        raise AssertionError("Captured K3 input/metadata/output storage changed")
    name = f"{'fp8' if fp8 else 'bf16'}_{'draft' if draft else 'target'}_qrep{int(replicated)}"
    trace = f"{name}_rank{rank}.trace.json"
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        dist.barrier(group=communicator.group)
        graph.replay()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(directory / trace))
    names = [
        event.name
        for event in profiler.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    cache_symbol = (
        "concat_and_cache_mla_kernel"
        if fp8
        else "mla_decode_q_concat_cache_insert_kernel"
    )
    if not any(cache_symbol in name for name in names):
        raise AssertionError("Trace lacks the actual production K3 cache-write kernel")
    graph.reset()
    dist.barrier(group=communicator.group)
    return {
        "case": name,
        "B": BATCH,
        "Q": QUERIES,
        "H": HEADS,
        "q_replicated": replicated,
        "fp8_compute_and_cache": fp8,
        "phase": "draft_update" if draft else "target_verify",
        "selected_fusion": impl.fia2a_backend.mode,
        "splits": impl.fia2a_backend.splits,
        "compact_tables": impl.page_rr_metadata.query_block_tables.shape[0] == BATCH,
        "eager_max_abs": eager_error,
        "graph_max_abs_by_phase": errors,
        "cache_bytes_checked": True,
        "graph_storage_stable": True,
        "graph_mutated_raw_q_kv_pe_prefixes_table_and_valid_mask": True,
        "input_boundary": "device-tensor fixture; not C++ input construction or full model",
        "trace": trace,
    }


def _worker(rank, port, output_directory):
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    cc = torch.cuda.get_device_capability(rank)
    if ".".join(map(str, cc)) != os.environ["FIA2A_EXPECTED_CC"]:
        raise AssertionError(f"Unexpected GPU compute capability {cc}")
    properties = torch.cuda.get_device_properties(rank)
    init_distributed_environment(
        _parallelism(rank), NcclCommConfig(nccl_ip="127.0.0.1"), port, timeout=300
    )
    try:
        directory = Path(output_directory)
        cases = [
            (True, True, False),
            (True, False, False),
            (False, True, True),
            (False, False, True),
            (True, True, True),
        ]
        with torch.inference_mode():
            results = [
                _run_case(rank, fp8, replicated, draft, directory)
                for fp8, replicated, draft in cases
            ]
        (directory / f"rank{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "hardware": {
                        "name": properties.name,
                        "compute_capability": cc,
                        "sm_count": properties.multi_processor_count,
                        "torch": torch.__version__,
                        "cuda_runtime": torch.version.cuda,
                    },
                    "cases": results,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        destroy_distributed_environment()


class K3MlaFia2aIntegrationTest(unittest.TestCase):
    def test_real_wrapper_tp4_and_mutable_graph(self):
        self.assertTrue(torch.cuda.is_available(), "Requires real CUDA GPUs")
        self.assertEqual(torch.cuda.device_count(), WORLD, "Allocate four locked GPUs")
        directory = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        self.assertIsNotNone(
            directory, "Run via the Bazel target for retained evidence"
        )
        Path(directory).mkdir(parents=True, exist_ok=True)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        mp.spawn(_worker, args=(port, directory), nprocs=WORLD, join=True)


if __name__ == "__main__":
    unittest.main()
