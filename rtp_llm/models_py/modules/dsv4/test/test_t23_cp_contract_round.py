"""Independent, CPU-only T23 CP contract round.

The collective is mocked at the ``cp.py`` boundary.  These tests deliberately
do not import the model/provider, PPU bindings, DeepEP, or the PD transport.
They pin the sequence contract that T23 must preserve before a device gate is
attempted.
"""

import contextlib
import importlib.util
import sys
import traceback
import types
from contextlib import contextmanager
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[5]


def _import_cp():
    name = "rtp_llm.models_py.modules.dsv4.cp"
    if name in sys.modules:
        return sys.modules[name]
    collective = types.ModuleType("rtp_llm.models_py.distributed.collective_torch")

    class _Group:
        TP = "TP"

    collective.Group = _Group
    collective.all_gather = lambda local, group=None: local
    for package in (
        "rtp_llm",
        "rtp_llm.models_py",
        "rtp_llm.models_py.distributed",
        "rtp_llm.models_py.modules",
        "rtp_llm.models_py.modules.dsv4",
    ):
        sys.modules.setdefault(package, types.ModuleType(package))
    sys.modules[collective.__name__] = collective
    profiler = types.ModuleType("rtp_llm.models_py.modules.dsv4._profiler")
    profiler.record_function_range = lambda *_a, **_k: contextlib.nullcontext()
    sys.modules[profiler.__name__] = profiler
    spec = importlib.util.spec_from_file_location(
        name, _REPO_ROOT / "rtp_llm/models_py/modules/dsv4/cp.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CP = _import_cp()


def _parametrize(values):
    """Tiny stdlib-only parameterization helper for standalone CPU execution."""

    def decorate(fn):
        def wrapped():
            for value in values:
                fn(value)

        return wrapped

    return decorate


@contextmanager
def _raises(expected, match=None):
    try:
        yield
    except expected as error:
        if match is not None:
            assert match in str(error), (match, str(error))
    else:
        raise AssertionError(f"expected {expected.__name__}")


class _CpInfo:
    def __init__(self, padding, restore, lengths, chunks):
        self.prefill_qkv_padding_mask = padding
        self.prefill_qkv_restore_indice = restore
        self.prefill_actual_input_lengths_cpu = torch.tensor(lengths, dtype=torch.int32)
        self.prefill_cp_chunk_lengths = torch.tensor(chunks, dtype=torch.int32)


def _metadata(chunks, actual_lengths, cp_size):
    """Build independent zig-zag metadata for request-concatenated B>=1 input."""
    total_chunk = sum(chunks)
    restore = torch.empty(cp_size * total_chunk, dtype=torch.int32)
    padding_parts = []
    chunk_offset = 0
    sequence_offset = 0
    for chunk, actual in zip(chunks, actual_lengths):
        assert chunk % 2 == 0 and 0 <= actual <= chunk * cp_size
        padded = chunk * cp_size
        mask = torch.zeros(padded, dtype=torch.int32)
        mask[:actual] = 1
        padding_parts.append(mask)
        pair = chunk // 2
        for rank in range(cp_size):
            local = rank * total_chunk + chunk_offset
            first = torch.arange(
                sequence_offset + rank * pair, sequence_offset + (rank + 1) * pair
            )
            second = torch.arange(
                sequence_offset + padded - (rank + 1) * pair,
                sequence_offset + padded - rank * pair,
            )
            restore[first] = torch.arange(local, local + pair, dtype=torch.int32)
            restore[second] = torch.arange(
                local + pair, local + 2 * pair, dtype=torch.int32
            )
        chunk_offset += chunk
        sequence_offset += padded
    return torch.cat(padding_parts), restore


@_parametrize([2, 4, 8])
def test_cp_positions_restore_batched_prefix_padding(cp_size):
    chunks = [4, 6]
    actual = [chunks[0] * cp_size - 1, chunks[1] * cp_size - 2]
    padding, restore = _metadata(chunks, actual, cp_size)
    info = _CpInfo(padding, restore, actual, chunks)
    contexts = [
        CP.build_cp_context(
            info,
            cp_size=cp_size,
            cp_rank=rank,
            chunk_length=sum(chunks),
            device=torch.device("cpu"),
            position_offset=torch.tensor([11, 29]),
        )
        for rank in range(cp_size)
    ]

    # Every real global token appears exactly once, while padding never enters
    # the restore stream. Prefix offsets are per request, not a scalar.
    real_positions = []
    for ctx in contexts:
        assert ctx.global_positions.numel() == sum(chunks)
        expected_offsets = ctx.prefix_lengths.gather(0, ctx.req_id_per_token.long())
        padded_req_offsets = torch.tensor(
            [0, chunks[0] * cp_size], dtype=torch.long
        ).gather(0, ctx.req_id_per_token.long())
        relative_local = ctx.relative_positions - padded_req_offsets
        assert torch.equal(
            ctx.global_positions[ctx.local_is_real] - relative_local[ctx.local_is_real],
            expected_offsets[ctx.local_is_real],
        )
        # Padding positions are clamped to the last valid token of their own
        # request, then receive that request's prefix offset as well.
        max_real = torch.tensor(actual, dtype=torch.long).sub(1).clamp_min(0)
        expected_padding_local = torch.minimum(
            relative_local,
            max_real.gather(0, ctx.req_id_per_token.long()),
        )
        assert torch.equal(
            ctx.global_positions[~ctx.local_is_real],
            (expected_padding_local + expected_offsets)[~ctx.local_is_real],
        )
        real_positions.append(ctx.relative_positions[ctx.local_is_real])
    union = torch.cat(real_positions).sort().values
    expected = torch.cat(
        [
            torch.arange(0, actual[0]),
            torch.arange(chunks[0] * cp_size, chunks[0] * cp_size + actual[1]),
        ]
    )
    assert torch.equal(union, expected)
    assert all(ctx.prefix_lengths.tolist() == [11, 29] for ctx in contexts)


def test_cp_t0_is_empty_and_does_not_construct_a_fake_row():
    info = _CpInfo(
        torch.empty(0, dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        [0],
        [0],
    )
    ctx = CP.build_cp_context(info, 2, 0, 0, torch.device("cpu"))
    assert ctx.seq_len_full == 0
    assert ctx.chunk_length == 0
    assert ctx.relative_positions.numel() == 0
    assert ctx.global_positions.numel() == 0
    assert ctx.unpad_restore.numel() == 0


@_parametrize([2, 4, 8])
def test_mocked_unsharded_gather_restores_global_request_order(cp_size):
    chunks = [4, 4]
    actual = [7, 5]
    padding, restore = _metadata(chunks, actual, cp_size)
    info = _CpInfo(padding, restore, actual, chunks)
    contexts = [
        CP.build_cp_context(info, cp_size, rank, sum(chunks), torch.device("cpu"))
        for rank in range(cp_size)
    ]
    rank_values = []
    for rank, ctx in enumerate(contexts):
        # Value encodes the rank-local row; restore must recover global values.
        rank_values.append((rank * 10000 + torch.arange(sum(chunks))).unsqueeze(1))
    gathered = torch.cat(rank_values, dim=0)

    CP.all_gather = lambda local, group=None: gathered
    for ctx in contexts:
        out = CP.cp_all_gather_full_varlen(rank_values[ctx.cp_rank], ctx).squeeze(1)
        expected = gathered[ctx.unpad_restore, 0]
        assert torch.equal(out, expected)


_T13_POOL_CONTRACTS = [
    # tag, compression/ring ratio, entries per block, logical entry bytes,
    # dtype, and scalar elements per entry for the mocked block tensor.
    ("csa_kv", 4, 64, 584, torch.uint8, 584),
    ("hca_kv", 128, 2, 584, torch.uint8, 584),
    ("indexer_kv", 4, 64, 68, torch.uint8, 68),
    ("indexer_state", 4, 8, 2048, torch.float32, 512),
    ("csa_state", 4, 8, 8192, torch.float32, 2048),
    ("hca_state", 128, 128, 4096, torch.float32, 1024),
    ("swa_kv", 128, 128, 584, torch.uint8, 584),
]


@_parametrize(_T13_POOL_CONTRACTS)
def test_mocked_sharded_gather_covers_every_t13_pool(pool):
    tag, ratio, entries, entry_bytes, dtype, scalars_per_entry = pool
    assert tag
    assert ratio in (4, 128)
    assert (
        scalars_per_entry * torch.empty((), dtype=dtype).element_size() == entry_bytes
    )
    cp_size = 2
    total_blocks = 3
    local_table = torch.tensor([1, 4], dtype=torch.int32)
    block_shape = (entries, scalars_per_entry)
    local_pool = torch.zeros((8, *block_shape), dtype=dtype)
    for block in range(local_pool.size(0)):
        local_pool[block].fill_(block + 1)
    rank0 = local_pool.index_select(0, local_table.long())
    rank1 = torch.stack(
        [
            torch.full(block_shape, 101, dtype=dtype),
            torch.full(block_shape, 103, dtype=dtype),
        ]
    )
    CP.all_gather = lambda local, group=None: torch.cat([rank0, rank1], dim=0)
    out = CP.cp_gather_request_pool_blocks(
        local_pool,
        local_table,
        cp_size=cp_size,
        cp_rank=0,
        total_logical_blocks=total_blocks,
    )
    assert out.shape == (total_blocks, *block_shape)
    assert out.dtype is dtype
    assert torch.all(out[0] == 2)
    assert torch.all(out[1] == 101)
    assert torch.all(out[2] == 5)


if __name__ == "__main__":
    failures = []
    for name, function in sorted(globals().items()):
        if not name.startswith("test_") or not callable(function):
            continue
        try:
            function()
            print("PASS", name)
        except (
            Exception
        ) as error:  # noqa: BLE001 - test harness must report all failures
            failures.append((name, error))
            print("FAIL", name, repr(error))
            traceback.print_exc()
    if failures:
        print("FAILED", len(failures))
        raise SystemExit(1)
    print("ALL_PASS", len([name for name in globals() if name.startswith("test_")]))
