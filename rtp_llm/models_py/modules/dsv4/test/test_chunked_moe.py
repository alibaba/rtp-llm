from __future__ import annotations

import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _stub_package(name: str, path: str) -> None:
    module = types.ModuleType(name)
    module.__path__ = [path]
    sys.modules.setdefault(name, module)


_stub_package("rtp_llm", os.path.join(_REPO, "rtp_llm"))
_stub_package("rtp_llm.models_py", os.path.join(_REPO, "rtp_llm", "models_py"))
_stub_package(
    "rtp_llm.models_py.modules",
    os.path.join(_REPO, "rtp_llm", "models_py", "modules"),
)
_stub_package(
    "rtp_llm.models_py.modules.dsv4",
    os.path.join(_REPO, "rtp_llm", "models_py", "modules", "dsv4"),
)
_stub_package(
    "rtp_llm.models_py.modules.dsv4.moe",
    os.path.join(_REPO, "rtp_llm", "models_py", "modules", "dsv4", "moe"),
)
_stub_package(
    "rtp_llm.models_py.modules.dsv4.moe.strategies",
    os.path.join(_REPO, "rtp_llm", "models_py", "modules", "dsv4", "moe", "strategies"),
)

_ops_pkg = types.ModuleType("librtp_compute_ops")
_ops_pkg.__path__ = []
_ops_mod = types.ModuleType("librtp_compute_ops.rtp_llm_ops")
sys.modules.setdefault("librtp_compute_ops", _ops_pkg)
sys.modules.setdefault("librtp_compute_ops.rtp_llm_ops", _ops_mod)

from rtp_llm.models_py.modules.dsv4.chunk_env import dsv4_chunk_tokens_from_env
from rtp_llm.models_py.modules.dsv4.moe.moe_layer import (
    DEFAULT_MOE_CHUNK_TOKENS,
    MoE,
    chunked_moe_enabled,
    cp_padded_tokens_per_rank_bound,
    moe_chunk_tokens_from_env,
    resolve_moe_max_tokens_per_rank,
)
from rtp_llm.models_py.modules.dsv4.moe.strategies.mega import _mega_output_capacity


class _FakeGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_id_chunks: list[torch.Tensor] = []
        self.token_chunks: list[int] = []

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor):
        self.token_chunks.append(x.size(0))
        self.input_id_chunks.append(input_ids.detach().clone())
        weights = torch.ones((x.size(0), 1), dtype=torch.float32, device=x.device)
        indices = input_ids.view(-1, 1).to(torch.long)
        return weights, indices


class _FakeSharedExecutor:
    def __init__(self) -> None:
        self.token_chunks: list[int] = []
        self._out: torch.Tensor | None = None

    def start(self, shared_experts, x: torch.Tensor) -> None:
        self.token_chunks.append(x.size(0))
        self._out = x.float() + 1.0

    def finish(self) -> torch.Tensor:
        assert self._out is not None
        out = self._out
        self._out = None
        return out


class _FakeStrategy(nn.Module):
    name = "mega"

    def __init__(self, cap: int) -> None:
        super().__init__()
        self.cap = cap
        self.token_chunks: list[int] = []

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        self.token_chunks.append(x.size(0))
        if x.size(0) > self.cap:
            raise RuntimeError(f"chunk overflow: {x.size(0)} > {self.cap}")
        return x.float() * 2.0


def _fake_moe(dim: int, cap: int, is_decode_role: bool = False) -> MoE:
    moe = MoE.__new__(MoE)
    nn.Module.__init__(moe)
    moe.layer_id = 0
    moe.dim = dim
    moe.max_tokens_per_rank = cap
    moe._is_decode_role = is_decode_role
    moe.gate = _FakeGate()
    moe.shared_experts = nn.Identity()
    moe._shared_executor = _FakeSharedExecutor()
    moe._strategy = _FakeStrategy(cap)
    return moe


class ChunkedMoETest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = mock.patch.dict(
            os.environ,
            {
                "DSV4_MOE_CHUNK_PREFILL": "1",
                "DSV4_MOE_STRICT_FUSED": "0",
                "DSV4_SHARED_EXPERT_BF16_ADD": "1",
            },
            clear=False,
        )
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()

    def test_env_helpers_default_on(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(chunked_moe_enabled())
            self.assertEqual(DEFAULT_MOE_CHUNK_TOKENS, 16384)
            self.assertEqual(moe_chunk_tokens_from_env(), DEFAULT_MOE_CHUNK_TOKENS)
        with mock.patch.dict(os.environ, {"DSV4_MOE_CHUNK_TOKENS": "4096"}):
            self.assertEqual(moe_chunk_tokens_from_env(), 4096)
        with mock.patch.dict(os.environ, {"DSV4_MOE_CHUNK_TOKENS": "bad"}):
            self.assertEqual(moe_chunk_tokens_from_env(), DEFAULT_MOE_CHUNK_TOKENS)

    def test_global_chunk_tokens_overrides_legacy_chunk_envs(self):
        with mock.patch.dict(
            os.environ,
            {
                "DSV4_CHUNK_TOKENS": "16384",
                "DSV4_MOE_CHUNK_PREFILL": "0",
                "DSV4_MOE_CHUNK_TOKENS": "4096",
                "DSV4_FP8_INDEXER_SCORE_CHUNK_ROWS": "8192",
            },
            clear=True,
        ):
            self.assertTrue(chunked_moe_enabled())
            self.assertEqual(moe_chunk_tokens_from_env(), 16384)
            self.assertEqual(
                dsv4_chunk_tokens_from_env("DSV4_FP8_INDEXER_SCORE_CHUNK_ROWS"),
                16384,
            )
            budget = resolve_moe_max_tokens_per_rank(
                max_seq_len=1048576,
                current_max_tokens_per_rank=1048576,
                cp_size=4,
                max_generate_batch_size=8,
            )
        self.assertEqual(budget, 16384)

    def test_global_chunk_tokens_zero_disables_all_chunking(self):
        with mock.patch.dict(
            os.environ,
            {
                "DSV4_CHUNK_TOKENS": "0",
                "DSV4_MOE_CHUNK_PREFILL": "1",
                "DSV4_MOE_CHUNK_TOKENS": "4096",
                "DSV4_ATTN_OUT_CHUNK_TOKENS": "8192",
            },
            clear=True,
        ):
            self.assertFalse(chunked_moe_enabled())
            self.assertEqual(moe_chunk_tokens_from_env(), 0)
            self.assertEqual(
                dsv4_chunk_tokens_from_env("DSV4_ATTN_OUT_CHUNK_TOKENS"),
                0,
            )
            budget = resolve_moe_max_tokens_per_rank(
                max_seq_len=1048576,
                current_max_tokens_per_rank=65536,
                cp_size=4,
                max_generate_batch_size=8,
            )
        self.assertEqual(budget, 65536)

    def test_token_budget_caps_cp_1m_to_moe_chunk(self):
        with mock.patch.dict(
            os.environ,
            {"DSV4_MOE_CHUNK_PREFILL": "1", "DSV4_MOE_CHUNK_TOKENS": "65536"},
        ):
            budget = resolve_moe_max_tokens_per_rank(
                max_seq_len=1048576,
                current_max_tokens_per_rank=1048576,
                cp_size=4,
                max_generate_batch_size=8,
            )
        self.assertEqual(budget, 65536)

    def test_cp_token_budget_includes_zigzag_padding(self):
        self.assertEqual(cp_padded_tokens_per_rank_bound(200002, 4), 50002)
        with mock.patch.dict(os.environ, {"DSV4_MOE_CHUNK_PREFILL": "0"}):
            budget = resolve_moe_max_tokens_per_rank(
                max_seq_len=200002,
                current_max_tokens_per_rank=200002,
                cp_size=4,
                max_generate_batch_size=8,
            )
        self.assertEqual(budget, 50002)

    def test_token_budget_never_expands_existing_cap(self):
        with mock.patch.dict(
            os.environ,
            {"DSV4_MOE_CHUNK_PREFILL": "1", "DSV4_MOE_CHUNK_TOKENS": "65536"},
        ):
            budget = resolve_moe_max_tokens_per_rank(
                max_seq_len=1048576,
                current_max_tokens_per_rank=8192,
                cp_size=4,
                max_generate_batch_size=8,
            )
        self.assertEqual(budget, 8192)

    def test_token_budget_decode_uses_batch_size(self):
        with mock.patch.dict(
            os.environ,
            {"DSV4_MOE_CHUNK_PREFILL": "1", "DSV4_MOE_CHUNK_TOKENS": "65536"},
        ):
            budget = resolve_moe_max_tokens_per_rank(
                max_seq_len=1048576,
                current_max_tokens_per_rank=1048576,
                cp_size=4,
                max_generate_batch_size=32,
                is_decode_role=True,
            )
        self.assertEqual(budget, 32)

    def test_token_budget_decode_accounts_for_speculative_width(self):
        budget = resolve_moe_max_tokens_per_rank(
            max_seq_len=1048576,
            current_max_tokens_per_rank=1048576,
            cp_size=4,
            max_generate_batch_size=1024,
            is_decode_role=True,
            is_speculative=True,
            gen_num_per_cycle=4,
        )
        self.assertEqual(budget, 5120)

    def test_token_budget_decode_non_speculative_uses_batch_size(self):
        budget = resolve_moe_max_tokens_per_rank(
            max_seq_len=1048576,
            current_max_tokens_per_rank=1048576,
            cp_size=4,
            max_generate_batch_size=1024,
            is_decode_role=True,
            is_speculative=False,
            gen_num_per_cycle=4,
        )
        self.assertEqual(budget, 1024)

    def test_token_budget_ignores_role_type_env(self):
        with mock.patch.dict(
            os.environ,
            {"ROLE_TYPE": "DECODE", "DSV4_MOE_CHUNK_PREFILL": "0"},
        ):
            budget = resolve_moe_max_tokens_per_rank(
                max_seq_len=200002,
                current_max_tokens_per_rank=200002,
                cp_size=4,
                max_generate_batch_size=8,
            )
        self.assertEqual(budget, 50002)

    def test_chunk_splits_flat_tokens_and_preserves_order(self):
        moe = _fake_moe(dim=3, cap=5)
        x = torch.arange(17 * 3, dtype=torch.float32).view(17, 3)
        input_ids = torch.arange(17, dtype=torch.long)

        out = moe(x, input_ids)

        self.assertEqual(moe.gate.token_chunks, [5, 5, 5, 2])
        self.assertEqual(moe._shared_executor.token_chunks, [5, 5, 5, 2])
        self.assertEqual(moe._strategy.token_chunks, [5, 5, 5, 2])
        self.assertTrue(torch.equal(out, x * 3.0 + 1.0))

    def test_input_ids_are_sliced_with_token_chunks(self):
        moe = _fake_moe(dim=2, cap=4)
        x = torch.arange(11 * 2, dtype=torch.float32).view(11, 2)
        input_ids = torch.arange(100, 111, dtype=torch.long)

        moe(x, input_ids)

        chunks = [c.tolist() for c in moe.gate.input_id_chunks]
        self.assertEqual(
            chunks, [[100, 101, 102, 103], [104, 105, 106, 107], [108, 109, 110]]
        )

    def test_chunking_avoids_strategy_token_overflow(self):
        moe = _fake_moe(dim=4, cap=8)
        x = torch.randn(33, 4)
        input_ids = torch.arange(33, dtype=torch.long)

        moe(x, input_ids)

        self.assertLessEqual(max(moe._strategy.token_chunks), 8)

    def test_varlen_multi_request_flat_order(self):
        moe = _fake_moe(dim=1, cap=6)
        request_lengths = [3, 17, 5]
        total = sum(request_lengths)
        x = torch.arange(total, dtype=torch.float32).view(total, 1)
        input_ids = torch.arange(total, dtype=torch.long)

        out = moe(x, input_ids)

        self.assertEqual(out.view(-1).tolist(), (x.view(-1) * 3.0 + 1.0).tolist())
        self.assertEqual(moe.gate.token_chunks, [6, 6, 6, 6, 1])

    def test_small_input_keeps_single_call(self):
        moe = _fake_moe(dim=2, cap=8)
        x = torch.randn(7, 2)
        input_ids = torch.arange(7, dtype=torch.long)

        out = moe(x, input_ids)

        self.assertEqual(moe.gate.token_chunks, [7])
        self.assertTrue(torch.allclose(out, x * 3.0 + 1.0))

    def test_env_can_disable_chunking(self):
        moe = _fake_moe(dim=2, cap=4)
        x = torch.randn(9, 2)
        input_ids = torch.arange(9, dtype=torch.long)
        with mock.patch.dict(os.environ, {"DSV4_MOE_CHUNK_PREFILL": "0"}):
            with self.assertRaisesRegex(RuntimeError, "chunk overflow"):
                moe(x, input_ids)

    def test_decode_asserts_instead_of_chunking(self):
        moe = _fake_moe(dim=2, cap=4, is_decode_role=True)
        x = torch.randn(9, 2)
        input_ids = torch.arange(9, dtype=torch.long)

        with self.assertRaisesRegex(AssertionError, "decode must not use chunked MoE"):
            moe(x, input_ids)

        self.assertEqual(moe.gate.token_chunks, [])

    def test_cuda_graph_capture_asserts_instead_of_chunking(self):
        moe = _fake_moe(dim=2, cap=4)
        x = torch.randn(9, 2)
        input_ids = torch.arange(9, dtype=torch.long)

        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            with mock.patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=True
            ):
                with self.assertRaisesRegex(
                    AssertionError, "CUDA graph capture must not use chunked MoE"
                ):
                    moe(x, input_ids)

        self.assertEqual(moe.gate.token_chunks, [])

    def test_input_ids_must_match_flat_tokens(self):
        moe = _fake_moe(dim=2, cap=4)
        x = torch.randn(5, 2)
        input_ids = torch.arange(4, dtype=torch.long)

        with self.assertRaisesRegex(RuntimeError, "input_ids/token mismatch"):
            moe(x, input_ids)

    def test_mega_output_capacity_uses_aligned_buffer_capacity(self):
        aligned = SimpleNamespace(num_max_tokens_per_rank=50304)
        self.assertEqual(_mega_output_capacity(aligned, 50000), 50304)
        smaller = SimpleNamespace(num_max_tokens_per_rank=49920)
        self.assertEqual(_mega_output_capacity(smaller, 50000), 50000)


if __name__ == "__main__":
    unittest.main()
