"""Stage 5b-3 — per-iteration pool reader UTs (CPU only).

Verifies factory dispatch + the per-iteration assemble pipeline:
  dequant owned → all_gather → restore → scatter → workspace.

The Triton dequant kernel is stubbed with an identity passthrough so we
test the dataflow shape, not the kernel itself. The all_gather stub
returns its input (cp_size=1 simulation) — sharded-path dataflow under
real cp_size>1 is exercised end-to-end by Stage 6 PD smoke.
"""

import importlib.util
import sys
import types
from pathlib import Path

import torch

_DEQUANT_CALLS = []
_REPO_ROOT = Path(__file__).resolve().parents[5]


def _stub_modules():
    mod_name = "rtp_llm.models_py.distributed.collective_torch"
    if mod_name not in sys.modules:
        stub = types.ModuleType(mod_name)

        class _Group:
            TP = "TP"

        def _stub_all_gather(local, group=None):
            return local

        stub.Group = _Group
        stub.all_gather = _stub_all_gather
        sys.modules[mod_name] = stub
        for p in (
            "rtp_llm",
            "rtp_llm.models_py",
            "rtp_llm.models_py.distributed",
        ):
            if p not in sys.modules:
                sys.modules[p] = types.ModuleType(p)

    sk_name = "rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton"
    if sk_name not in sys.modules:
        sk = types.ModuleType(sk_name)

        def _stub_dequant(**kwargs):
            _DEQUANT_CALLS.append({k: v for k, v in kwargs.items()})
            # Identity-fill: write zeros (we test scatter / dataflow only).
            kwargs["out"].zero_()

        sk.dequantize_and_gather_k_cache = _stub_dequant
        sys.modules[sk_name] = sk
        for p in (
            "rtp_llm.models_py.modules",
            "rtp_llm.models_py.modules.dsv4",
            "rtp_llm.models_py.modules.dsv4.fp8",
        ):
            if p not in sys.modules:
                sys.modules[p] = types.ModuleType(p)

    cp_name = "rtp_llm.models_py.modules.dsv4.cp"
    if cp_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            cp_name,
            _REPO_ROOT / "rtp_llm/models_py/modules/dsv4/cp.py",
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[cp_name] = mod
        spec.loader.exec_module(mod)

        # Override gather with an identity handle (cp_size=1 simulation).
        class _IdHandle:
            def __init__(self, t):
                self._t = t

            def wait(self):
                return self._t

        def _identity_gather(local, cp_ctx, group=None):
            return _IdHandle(local)

        mod.cp_all_gather_full_async = _identity_gather


def _import_pr():
    _stub_modules()
    name = "_dsv4_pool_reader_for_test_5b3"
    spec = importlib.util.spec_from_file_location(
        name,
        _REPO_ROOT / "rtp_llm/models_py/modules/dsv4/fp8/_pool_reader.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PR = _import_pr()


def _fake_cp_ctx(cp_size: int, cp_rank: int = 0):
    ctx = types.SimpleNamespace()
    ctx.cp_size = cp_size
    ctx.cp_rank = cp_rank
    return ctx


# ---------- Factory ----------
def test_factory_kv_cache_not_sharded_returns_local():
    r = PR.make_compressed_k_pool_reader(
        cp_ctx=_fake_cp_ctx(cp_size=4),
        kv_cache_sharded=False,
    )
    assert isinstance(r, PR.LocalPoolReader)


def test_factory_cp_size_one_returns_local():
    r = PR.make_compressed_k_pool_reader(
        cp_ctx=_fake_cp_ctx(cp_size=1),
        kv_cache_sharded=True,
        per_req_total_kv_lens=torch.tensor([12], dtype=torch.int64),
        block_size=4,
    )
    assert isinstance(r, PR.LocalPoolReader)


def test_factory_no_cp_ctx_returns_local():
    r = PR.make_compressed_k_pool_reader(
        cp_ctx=None,
        kv_cache_sharded=True,
        per_req_total_kv_lens=torch.tensor([12], dtype=torch.int64),
        block_size=4,
    )
    assert isinstance(r, PR.LocalPoolReader)


def test_factory_all_cold_returns_local():
    r = PR.make_compressed_k_pool_reader(
        cp_ctx=_fake_cp_ctx(cp_size=4),
        kv_cache_sharded=True,
        per_req_total_kv_lens=torch.zeros(2, dtype=torch.int64),
        block_size=4,
    )
    assert isinstance(r, PR.LocalPoolReader)


def test_factory_sharded_returns_cp_reader():
    r = PR.make_compressed_k_pool_reader(
        cp_ctx=_fake_cp_ctx(cp_size=4),
        kv_cache_sharded=True,
        per_req_total_kv_lens=torch.tensor([0, 16, 0], dtype=torch.int64),
        block_size=4,
    )
    assert isinstance(r, PR.CPShardedPoolReader)
    assert r.cfg.cp_ctx.cp_size == 4
    # 16 tokens, vb=16 → 1 vb → 4 local tokens per req
    assert r.cfg.total_local_kv == 4


def test_factory_missing_args_raises():
    try:
        PR.make_compressed_k_pool_reader(
            cp_ctx=_fake_cp_ctx(cp_size=4),
            kv_cache_sharded=True,
            per_req_total_kv_lens=None,
            block_size=4,
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# ---------- Helpers ----------
def test_compute_local_seq_lens_basic():
    out = PR._compute_local_seq_lens(torch.tensor([16, 17, 0]), cp_size=2, block_size=4)
    # vb=8. T=16→2vb→8; T=17→3vb→12; T=0→0.
    assert torch.equal(out, torch.tensor([8, 12, 0]))


def test_pack_padded_to_flat():
    padded = torch.zeros((3, 5, 2))
    padded[0, 0] = torch.tensor([1.0, 2.0])
    padded[0, 1] = torch.tensor([3.0, 4.0])
    padded[1, 0] = torch.tensor([5.0, 6.0])
    lens = torch.tensor([2, 1, 0])
    flat = PR._pack_padded_to_flat(padded, lens, total=3, D=2)
    assert flat.shape == (3, 2)
    assert torch.equal(flat[0], torch.tensor([1.0, 2.0]))
    assert torch.equal(flat[2], torch.tensor([5.0, 6.0]))


def test_pack_padded_to_flat_empty():
    padded = torch.zeros((1, 0, 2))
    lens = torch.tensor([0])
    flat = PR._pack_padded_to_flat(padded, lens, total=0, D=2)
    assert flat.shape == (0, 2)


def test_scatter_flat_to_workspace():
    out = torch.full((2, 6, 3), -1.0)
    restored = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    seq_lens = torch.tensor([3, 2])
    PR._scatter_flat_to_workspace(restored, out, seq_lens, offset=1)
    # req0 → out[0, 1:4]; req1 → out[1, 1:3]
    assert torch.equal(out[0, 1:4], restored[0:3])
    assert torch.equal(out[1, 1:3], restored[3:5])
    assert torch.equal(out[0, 0], torch.full((3,), -1.0))
    assert torch.equal(out[0, 4], torch.full((3,), -1.0))


# ---------- End-to-end fill (CPU dataflow) ----------
def test_local_reader_fill_passthrough():
    _DEQUANT_CALLS.clear()
    r = PR.LocalPoolReader()
    out = torch.zeros((1, 4, 2))
    r.fill(
        out=out,
        k_cache=torch.zeros((4, 4, 2)),
        seq_lens=torch.tensor([4]),
        gather_lens=None,
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        block_size=4,
        offset=0,
    )
    assert len(_DEQUANT_CALLS) == 1


def test_cp_sharded_fill_dataflow_cp1_passthrough():
    """cp_size=1 in fill stage: gather is identity → restore is identity →
    dataflow lands intact in workspace."""
    _DEQUANT_CALLS.clear()
    cp_ctx = _fake_cp_ctx(cp_size=1, cp_rank=0)
    per_req = torch.tensor([8], dtype=torch.int64)
    block_size = 4
    restore = PR.__dict__.get("build_kv_allgather_restore_indices")  # not exported
    # Use cp.py builder directly
    cp_mod = sys.modules["rtp_llm.models_py.modules.dsv4.cp"]
    restore_idx = cp_mod.build_kv_allgather_restore_indices(
        per_req, 1, block_size, torch.device("cpu")
    )
    cfg = PR.CPShardConfig(
        cp_ctx=cp_ctx,
        per_req_total_kv_lens=per_req,
        restore_indices=restore_idx,
        block_size=block_size,
        total_local_kv=8,
    )
    reader = PR.CPShardedPoolReader(cfg)
    out = torch.full((1, 8, 2), -1.0)
    reader.fill(
        out=out,
        k_cache=torch.zeros((4, 4, 2)),
        seq_lens=torch.tensor([8]),
        gather_lens=None,
        block_table=torch.zeros((1, 2), dtype=torch.int32),
        block_size=block_size,
        offset=0,
    )
    # stub dequant zeros local_packed → restored is zeros → out becomes zeros.
    assert torch.equal(out[0, 0:8], torch.zeros((8, 2)))
    assert len(_DEQUANT_CALLS) == 1, "exactly one dequant call per fill"


def test_cp_sharded_fill_block_size_mismatch_raises():
    cp_ctx = _fake_cp_ctx(cp_size=2, cp_rank=0)
    per_req = torch.tensor([8], dtype=torch.int64)
    cp_mod = sys.modules["rtp_llm.models_py.modules.dsv4.cp"]
    restore_idx = cp_mod.build_kv_allgather_restore_indices(
        per_req, 2, 4, torch.device("cpu")
    )
    reader = PR.CPShardedPoolReader(
        PR.CPShardConfig(
            cp_ctx=cp_ctx,
            per_req_total_kv_lens=per_req,
            restore_indices=restore_idx,
            block_size=4,
            total_local_kv=4,
        )
    )
    try:
        reader.fill(
            out=torch.zeros((1, 8, 2)),
            k_cache=torch.zeros((4, 4, 2)),
            seq_lens=torch.tensor([8]),
            gather_lens=None,
            block_table=torch.zeros((1, 1), dtype=torch.int32),
            block_size=8,  # mismatch!
            offset=0,
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError on block_size mismatch")


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok  {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if failures else 0)
