"""CPU allocation contracts; transport is mocked, tensor storage is real."""

import ast
import sys
import unittest
from itertools import product
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parent.parent


def load_model():
    # Follow the standalone JIT tests: execute production methods without
    # loading engine DSOs or CUDA operators for a CPU allocation check.
    path = ROOT / "dsa_model_adapter.py"
    tree = ast.parse(path.read_text())
    tree.body = next(node.body for node in tree.body if isinstance(node, ast.ClassDef))
    namespace = {
        "torch": torch,
        "__package__": "rtp_llm.models_py.modules.kimi_k3",
    }
    exec(compile(tree, str(path), "exec"), namespace)
    methods = {name: value for name, value in namespace.items() if callable(value)}
    return type("CapacityModel", (torch.nn.Module,), methods)


class PrefillRuntimeCapacityTest(unittest.TestCase):
    def model(self, tokens, tp=8):
        model = load_model()()
        model.tp_size = tp
        model._max_moe_tokens = tokens
        # Deliberately stale full-context capacities must not size buffers.
        model.config = NS(
            hidden_size=4,
            max_seq_len=1048576,
            moe_prefill_max_tokens_per_rank=2097152,
        )
        model.prefill_query_cp = True
        model._outer_natural_enabled = True
        model._row_exchange = None
        model.layers = []
        model.norm = torch.nn.LayerNorm(4, dtype=torch.bfloat16)
        model.dsa_workspace = NS()
        model.register_buffer("_mtp_prefill_hidden_buffer", None, persistent=False)
        model._mtp_prefill_hidden_valid_rows = 0
        return model

    def initialize_route(self, model, requests):
        def exchange(group, device, rows, hidden, *, chunk_capacity):
            return NS(max_rows=rows, chunk_capacity=min(rows, chunk_capacity))

        modules = {
            "rtp_llm.models_py.distributed.collective_torch": NS(
                Group=NS(TP=0), get_process_group=lambda _: None
            ),
            "rtp_llm.models_py.distributed.zigzag_row_exchange": NS(
                DirectedRowExchange=Mock(side_effect=exchange)
            ),
            "rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear": NS(
                CudaF16Linear=object
            ),
            "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear": NS(
                CudaFp8GEMMLinear=object
            ),
            "rtp_llm.models_py.modules.kimi_k3.natural_dsa": NS(
                route_allocation_budget=lambda _: (0, 0)
            ),
        }
        with patch.dict(sys.modules, modules):
            model._initialize_natural_outer(NS(max_context_batch_size=requests))

    def initialize_hidden(self, model, requests=3):
        model._initialize_mtp_hidden_handoff(
            NS(is_speculative=True, max_context_batch_size=requests)
        )

    def test_route_and_handoff_follow_the_same_physical_capacity(self):
        for tp, tokens in product((1, 8, 16), (256, 1024, 16384)):
            with self.subTest(tp=tp, tokens=tokens):
                model = self.model(tokens, tp)
                self.initialize_route(model, 3)
                self.initialize_hidden(model)
                rows = tokens // tp
                self.assertEqual(model._row_exchange.max_rows, rows)
                self.assertEqual(model.dsa_workspace.route_capacity, rows)
                self.assertEqual(model.dsa_workspace.route_chunk_rows, min(rows, 8192))
                self.assertEqual(model._mtp_prefill_hidden_buffer.shape, (rows, 4))

    def test_reinitialize_preserves_storage_and_published_values(self):
        model = self.model(256)
        self.initialize_hidden(model)
        storage = model._mtp_prefill_hidden_buffer
        rows = storage.shape[0]
        model._query_layout = NS(local_rows=rows)
        hidden = torch.arange(rows * 4, dtype=torch.bfloat16).view(rows, 4)
        expected = hidden.clone()
        published = model._publish_mtp_prefill_hidden(hidden)
        hidden.zero_()
        torch.testing.assert_close(published, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            model.get_mtp_target_hidden_states(-1), expected, rtol=0, atol=0
        )
        self.initialize_hidden(model)
        self.assertIs(storage, model._mtp_prefill_hidden_buffer)
        model._max_moe_tokens *= 2
        with self.assertRaisesRegex(ValueError, "Cannot replace"):
            self.initialize_hidden(model)

    def test_overflow_is_rejected_before_publication(self):
        model = self.model(256)
        self.initialize_hidden(model)
        model._query_layout = NS(local_rows=33)
        with self.assertRaisesRegex(ValueError, "rank-local CP storage"):
            model._publish_mtp_prefill_hidden(torch.empty(33, 4, dtype=torch.bfloat16))
        self.assertEqual(model._mtp_prefill_hidden_valid_rows, 0)

    def test_non_speculative_and_non_cp_do_not_allocate_handoff(self):
        model = self.model(256)
        model._initialize_mtp_hidden_handoff(NS(is_speculative=False))
        self.assertFalse(model.has_mtp_hidden_buffer())
        model.prefill_query_cp = False
        self.initialize_hidden(model)
        self.assertFalse(model.has_mtp_hidden_buffer())


if __name__ == "__main__":
    unittest.main()
