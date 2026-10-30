"""Real TP4 PageRR FIA2A routes and mutable CUDA Graph replay."""

import itertools
import math
import os
import socket
import unittest
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.distributed.collective_torch import (
    destroy_distributed_environment,
    get_mla_dcp,
    init_distributed_environment,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.page_rr_mla_metadata import (
    PageRRMlaMetadata,
)
from rtp_llm.models_py.utils.cutlass import setup_cutlass_import_path
from rtp_llm.ops import AttentionConfigs, NcclCommConfig, ParallelismConfig, RoleType

WORLD, HEADS, LOCAL_HEADS = 4, 96, 24
PHYSICAL_PAGE, KERNEL_PAGE = 1024, 128
LATENT, SUFFIX = 512, 64
MAX_TOKENS = 6144
SCALE = 192**-0.5


@dataclass
class _DecodeInputs:
    """Device input buffers normally supplied by the model's input bindings."""

    input_lengths: torch.Tensor
    physical_token_count: int
    is_target_verify: bool
    is_mtp_draft_update: bool
    is_cuda_graph: bool
    prefix_lengths_device: torch.Tensor
    sequence_lengths_plus_1_device: torch.Tensor
    kv_cache_kernel_block_id_device: torch.Tensor
    cu_seqlens_device: torch.Tensor
    valid_token_mask: torch.Tensor


class _Fixture:
    def __init__(self, rank, batch, queries, dtype, draft_update):
        self.rank, self.batch, self.queries = rank, batch, queries
        self.dtype, self.draft_update = dtype, draft_update
        self.device = torch.device("cuda", rank)
        # Enumerate ownership independently of production PageRR metadata.
        owned = [p for p in range(MAX_TOKENS) if (p // PHYSICAL_PAGE) % WORLD == rank]
        self.owned = torch.tensor(owned, dtype=torch.long, device=self.device)
        self.columns = math.ceil(len(owned) / KERNEL_PAGE)
        self.tables = [
            torch.arange(
                1 + bank * batch * self.columns,
                1 + (bank + 1) * batch * self.columns,
                dtype=torch.int32,
                device=self.device,
            ).view(batch, self.columns)
            for bank in range(2)
        ]
        self.cache = torch.zeros(
            (1 + 2 * batch * self.columns, KERNEL_PAGE, LATENT + SUFFIX),
            dtype=dtype,
            device=self.device,
        )
        self.queries_by_phase, self.global_kv = [], []
        row = torch.arange(batch * queries, device=self.device, dtype=torch.int64)[
            :, None, None
        ]
        head = torch.arange(HEADS, device=self.device, dtype=torch.int64)[None, :, None]
        feature = torch.arange(LATENT + SUFFIX, device=self.device, dtype=torch.int64)[
            None, None, :
        ]
        position = torch.arange(MAX_TOKENS, device=self.device, dtype=torch.int64)[
            None, :, None
        ]
        request = torch.arange(batch, device=self.device, dtype=torch.int64)[
            :, None, None
        ]
        for phase in range(2):
            query = (
                (row * 17 + head * 13 + feature * 7 + phase * 19) % 37 - 18
            ).float() / 64
            query = query + ((head % 7).float() - 3) / 32 + ((row % 3).float() - 1) / 32
            self.queries_by_phase.append(query.to(dtype))
            noise = (
                (position * 11 + feature * 5 + request * 23 + phase * 29) % 31 - 15
            ).float() / 128
            signal = (
                (position // PHYSICAL_PAGE) % WORLD
            ).float() / 8 + request.float() / 64
            logical = (noise + signal + phase / 4).to(dtype)
            self.global_kv.append(logical)
        self.query = self.queries_by_phase[0].clone()
        self.inputs = _DecodeInputs(
            input_lengths=torch.full((batch,), queries, dtype=torch.int32),
            physical_token_count=batch * queries,
            is_target_verify=queries > 1 and not draft_update,
            is_mtp_draft_update=draft_update,
            is_cuda_graph=True,
            prefix_lengths_device=torch.empty(
                batch, dtype=torch.int32, device=self.device
            ),
            sequence_lengths_plus_1_device=torch.empty(
                batch, dtype=torch.int32, device=self.device
            ),
            kv_cache_kernel_block_id_device=self.tables[0].clone(),
            cu_seqlens_device=torch.empty(
                batch + 1, dtype=torch.int32, device=self.device
            ),
            valid_token_mask=torch.ones(
                batch * queries, dtype=torch.bool, device=self.device
            ),
        )
        self.prefixes, self.active = [], []
        self.select_phase(0)

    def select_phase(self, phase):
        self.prefixes = [
            (4099 if phase == 0 else 1025) + request * 17
            for request in range(self.batch)
        ]
        counts = [self.queries] * self.batch
        active = [True] * (self.batch * self.queries)
        if self.draft_update:
            for request in range(self.batch):
                counts[request] = max(1, self.queries - request % 3)
                if phase and request == self.batch - 1:
                    counts[request] = 0
                for query in range(self.queries):
                    # A live earlier query with an inactive final physical
                    # query exercises scan=max(bounds), not bounds[:,-1].
                    active[request * self.queries + query] = (
                        query < counts[request]
                        and query != self.queries - 1
                        and not (phase and request == 0 and query == 1)
                    )
        self.active = active
        # Update cache contents after capture as well as switching table IDs.
        # Byte copies support E4M3 without requiring float8 indexing kernels.
        payload = self.global_kv[phase].view(torch.uint8).index_select(1, self.owned)
        first = 1 + phase * self.batch * self.columns
        target = self.cache[first : first + self.batch * self.columns].view(torch.uint8)
        target.copy_(payload.reshape_as(target))
        self.query.copy_(self.queries_by_phase[phase])
        self.inputs.input_lengths.copy_(torch.tensor(counts, dtype=torch.int32))
        self.inputs.prefix_lengths_device.copy_(
            torch.tensor(self.prefixes, dtype=torch.int32, device=self.device)
        )
        self.inputs.sequence_lengths_plus_1_device.copy_(
            self.inputs.prefix_lengths_device + 1
        )
        self.inputs.kv_cache_kernel_block_id_device.copy_(self.tables[phase])
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        self.inputs.cu_seqlens_device.copy_(
            torch.tensor(offsets, dtype=torch.int32, device=self.device)
        )
        self.inputs.valid_token_mask.copy_(
            torch.tensor(active, dtype=torch.bool, device=self.device)
        )

    def reference(self, phase):
        outputs, local_bounds = [], []
        for request, prefix in enumerate(self.prefixes):
            bound_row = []
            for query in range(self.queries):
                row = request * self.queries + query
                end = prefix + query + 1 if self.active[row] else 0
                bound_row.append(
                    sum((p // PHYSICAL_PAGE) % WORLD == self.rank for p in range(end))
                )
                if not end:
                    outputs.append(
                        torch.zeros((LOCAL_HEADS, LATENT), device=self.device)
                    )
                    continue
                q = self.queries_by_phase[phase][
                    row, self.rank * LOCAL_HEADS : (self.rank + 1) * LOCAL_HEADS
                ].float()
                kv = self.global_kv[phase][request, :end].float()
                if self.dtype == torch.float8_e4m3fn:
                    q = q * 0.5
                    kv = kv * 0.25
                scores = (q @ kv.T) * SCALE
                outputs.append(scores.softmax(-1) @ kv[:, :LATENT])
            local_bounds.append(bound_row)
        return torch.stack(outputs).transpose(0, 1), local_bounds


def _run_case(
    rank,
    communicator,
    *,
    dtype,
    queries,
    replicated,
    batch,
    draft_update,
    expected_transport=None,
):
    from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.page_rr_fia2a_backend import (
        PageRRFIA2AMLABackend,
    )

    fixture = _Fixture(rank, batch, queries, dtype, draft_update)
    compact = replicated and queries > 1
    metadata = PageRRMlaMetadata(PHYSICAL_PAGE, KERNEL_PAGE, WORLD, rank)
    metadata.prepare_decode(fixture.inputs, compact_queries=compact)
    config = AttentionConfigs()
    config.kv_lora_rank = LATENT
    config.rope_head_dim = SUFFIX
    config.kernel_tokens_per_block = KERNEL_PAGE
    backend = PageRRFIA2AMLABackend(config, communicator, dtype, token_major=replicated)
    fp8 = dtype == torch.float8_e4m3fn
    backend.prepare(
        metadata, SCALE * (0.5 * 0.25 if fp8 else 1.0), 0.25 if fp8 else 1.0
    )
    tail = torch.zeros(1, device=fixture.device)

    def step():
        metadata.prepare_decode(
            fixture.inputs, forbid_realloc=True, compact_queries=compact
        )
        if replicated:
            query = fixture.query
        else:
            query = fixture.query[:, rank * LOCAL_HEADS : (rank + 1) * LOCAL_HEADS]
            query = communicator.query_gather(query.transpose(0, 1).contiguous())
        result = backend.forward(query, fixture.cache)
        # This is the model's following TP collective: it prevents a peer
        # from overwriting its source arena before every rank consumed it.
        dist.all_reduce(tail, group=communicator.group)
        return result

    def check(output, phase):
        expected, bounds = fixture.reference(phase)
        if not bool(torch.isfinite(output).all()):
            raise AssertionError("FIA2A output contains nonfinite values")
        torch.testing.assert_close(output.float(), expected, atol=2e-3, rtol=3e-2)
        if metadata.local_causal_lens.cpu().tolist() != bounds:
            raise AssertionError(
                "Prepared bounds disagree with independent owner enumeration"
            )
        if backend.scan_lengths.cpu().tolist() != [max(row) for row in bounds]:
            raise AssertionError(
                "Scan lengths did not follow the live per-query maximum"
            )
        inactive = torch.tensor(
            [not value for value in fixture.active], device=fixture.device
        )
        if bool(inactive.any()) and not bool((output[:, inactive] == 0).all()):
            raise AssertionError(
                "Inactive queries must produce exact zero merged output"
            )

    stream = torch.cuda.Stream(device=fixture.device)
    stream.wait_stream(torch.cuda.current_stream(fixture.device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream(fixture.device).wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier(group=communicator.group)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = step()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    check(output, 0)
    pointers = (
        metadata.local_causal_lens.data_ptr(),
        metadata.query_block_tables.data_ptr(),
        backend.scan_lengths.data_ptr(),
        backend.table.data_ptr(),
        fixture.query.data_ptr(),
    )
    fixture.select_phase(1)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    check(output, 1)
    after = (
        metadata.local_causal_lens.data_ptr(),
        metadata.query_block_tables.data_ptr(),
        backend.scan_lengths.data_ptr(),
        backend.table.data_ptr(),
        fixture.query.data_ptr(),
    )
    if pointers != after:
        raise AssertionError("Graph metadata or input storage changed during replay")
    actual_transport = (
        "producer"
        if backend.fused
        else ("custom" if backend.a2a_buffers is not None else "nccl")
    )
    if expected_transport is not None and actual_transport != expected_transport:
        raise AssertionError(
            f"Expected automatic {expected_transport}, executed {actual_transport}"
        )
    graph.reset()
    dist.barrier(group=communicator.group)


def _worker(rank, port):
    torch.cuda.set_device(rank)
    capability = torch.cuda.get_device_capability(rank)
    expected = os.environ["FIA2A_EXPECTED_CC"]
    if ".".join(map(str, capability)) != expected:
        raise AssertionError(f"Expected CC {expected}, received {capability}")
    setup_cutlass_import_path()
    parallelism = ParallelismConfig()
    parallelism.world_size = parallelism.local_world_size = parallelism.tp_size = WORLD
    parallelism.world_rank = parallelism.local_rank = parallelism.tp_rank = rank
    parallelism.dp_size = parallelism.ep_size = parallelism.pp_size = 1
    parallelism.dp_rank = parallelism.ep_rank = parallelism.pp_rank = 0
    parallelism.role_type = RoleType.DECODE
    parallelism.prefill_cp_config.kv_cache_sharded = True
    init_distributed_environment(
        parallelism, NcclCommConfig(nccl_ip="127.0.0.1"), port, timeout=300
    )
    try:
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            config = AttentionConfigs()
            config.head_num, config.kv_lora_rank, config.rope_head_dim = (
                LOCAL_HEADS,
                LATENT,
                SUFFIX,
            )
            communicator = get_mla_dcp(config, torch.device("cuda", rank), dtype)
            for queries, replicated in itertools.product((1, 4), (True, False)):
                _run_case(
                    rank,
                    communicator,
                    dtype=dtype,
                    queries=queries,
                    replicated=replicated,
                    batch=2,
                    draft_update=queries > 1,
                )
            # The low payload, custom-pull lower boundary and full query work
            # naturally select all three transports without an override.
            for batch, queries, expected_transport in (
                (1, 1, "nccl"),
                (2, 4, "custom"),
                (16, 4, "producer"),
            ):
                _run_case(
                    rank,
                    communicator,
                    dtype=dtype,
                    queries=queries,
                    replicated=True,
                    batch=batch,
                    draft_update=False,
                    expected_transport=expected_transport,
                )
    finally:
        destroy_distributed_environment()


class PageRRFia2aTest(unittest.TestCase):
    def test_tp4_routes_and_mutable_graphs(self):
        self.assertTrue(torch.cuda.is_available(), "FIA2A GPU tests require CUDA")
        self.assertEqual(
            torch.cuda.device_count(), WORLD, "Allocate exactly four locked GPUs"
        )
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        mp.spawn(_worker, args=(port,), nprocs=WORLD, join=True)


if __name__ == "__main__":
    unittest.main()
