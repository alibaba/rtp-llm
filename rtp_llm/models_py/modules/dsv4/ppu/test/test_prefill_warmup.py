"""Ownership and metadata tests; real PPU numerics are a separate gate."""

import ast
import gc
import logging
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
import unittest
import weakref

import torch


ROOT = Path(__file__).resolve().parents[6]
PATH = ROOT / "rtp_llm/models_py/modules/dsv4/ppu/prefill/warmup.py"
scope = dict(torch=torch, Mapping=Mapping, logging=logging, PyAttentionInputs=SimpleNamespace,
             PyModelOutputs=lambda hidden, aux=None: (hidden, aux))
tree = ast.parse(PATH.read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)],
                        type_ignores=[]), str(PATH), "exec"), scope)


class PrefillWarmupTest(unittest.TestCase):
    def setUp(self):
        self.v4 = SimpleNamespace(
            fp8_kv_cache=True, _prefill_ws_q_rows=5, _prefill_ws_q_dim=2,
            _cp_size=1, _mtp_hidden_buffer=torch.full((5, 4), -1.),
            _mtp_hidden_valid_tokens=1, _mtp_last_hidden_buffer=torch.full((4, 2), -2.),
            _mtp_last_hidden_valid_tokens=2)
        self.cu = torch.tensor([0, 3, 5, 9, 10], dtype=torch.int32)
        self.positions = torch.arange(10)
        self.attn = {}
        for tag, width in (("swa_kv", 2), ("csa_kv", 4)):
            self.attn[tag] = SimpleNamespace(
                is_prefill=True, is_cuda_graph=False, is_s_padded=False,
                is_target_verify=False, context_parallel_info=None,
                cache_store_inputs=None, cache_store_writer=None,
                dtype=torch.bfloat16, input_lengths=torch.diff(self.cu),
                prefix_lengths=torch.tensor([0, 8, 16, 32], dtype=torch.int32),
                sequence_lengths=torch.empty(0, dtype=torch.int32),
                kv_cache_block_id=torch.zeros(4, width, dtype=torch.int32),
                kv_cache_block_id_device=torch.zeros(4, width, dtype=torch.int32),
                kv_cache_kernel_block_id=torch.arange(4 * width).reshape(4, width),
                kv_cache_kernel_block_id_device=torch.arange(4 * width).reshape(4, width),
                sequence_lengths_plus_1_device=torch.empty(0, dtype=torch.int32),
                cu_kv_seqlens_device=torch.tensor([0, 3, 13, 33, 66]),
                headwise_config=None)
        self.parallel = SimpleNamespace(ep_size=1, tp_size=4, world_size=4)

    def groups(self, **kwargs):
        return scope["reserved_page_warmup_groups"](
            self.v4, self.parallel, self.attn, self.cu,
            kwargs.get("tokens", 10), kwargs.get("prepare"), kwargs.get("capture"))

    def test_whole_requests_and_real_or_graph_requests_untouched(self):
        self.assertEqual(self.groups(), [(0, 2, 0, 5), (2, 4, 5, 10)])
        self.assertIsNone(self.groups(tokens=5))
        self.assertIsNone(self.groups(prepare=object()))
        self.assertIsNone(self.groups(capture=SimpleNamespace(enabled=True)))
        for name, value in (("is_cuda_graph", True), ("cache_store_writer", object())):
            a = self.attn["swa_kv"]
            old = getattr(a, name)
            setattr(a, name, value)
            self.assertIsNone(self.groups())
            setattr(a, name, old)
        self.attn["swa_kv"].kv_cache_block_id[0, 0] = 1
        self.assertIsNone(self.groups())

    def test_uneven_groups_keep_complete_request_boundaries(self):
        self.cu = torch.tensor([0, 3, 7, 9, 10], dtype=torch.int32)
        for a in self.attn.values():
            a.input_lengths = torch.diff(self.cu)
        self.assertEqual(self.groups(), [(0, 1, 0, 3), (1, 2, 3, 7), (2, 4, 7, 10)])

    def test_single_request_cannot_be_split(self):
        self.v4._prefill_ws_q_rows = 3
        with self.assertRaisesRegex(ValueError, "whole requests"):
            self.groups()

    def test_workspace_reuse_aux_offsets_and_release_including_failure(self):
        fixed, last = self.v4._mtp_hidden_buffer, self.v4._mtp_last_hidden_buffer
        refs, pointers, calls = [], [], []

        class Workspace:
            def __init__(owner, device, **kwargs):
                owner.buffer = torch.empty(5, 2)
                refs.append(weakref.ref(owner))
                refs.append(weakref.ref(owner.buffer))

        scope["PrefillWorkspace"] = Workspace
        tables = {tag: a.kv_cache_kernel_block_id for tag, a in self.attn.items()}

        def forward(v4, kv, ids, positions, cu, blocks, **kwargs):
            index = len(calls)
            calls.append(ids.tolist())
            pointers.append(kwargs["workspace"].buffer.data_ptr())
            r0, r1, t0, t1 = [(0, 2, 0, 5), (2, 4, 5, 10)][index]
            for tag, a in kwargs["attention_inputs"].items():
                self.assertEqual(a.input_lengths.tolist(), torch.diff(self.cu)[r0:r1].tolist())
                self.assertEqual(a.prefix_lengths.tolist(), self.attn[tag].prefix_lengths[r0:r1].tolist())
                self.assertTrue(torch.equal(a.kv_cache_kernel_block_id_device, tables[tag][r0:r1]))
                self.assertEqual(a.cu_kv_seqlens_device.tolist(),
                                 (self.attn[tag].cu_kv_seqlens_device[r0:r1+1]
                                  - self.attn[tag].cu_kv_seqlens_device[r0]).tolist())
            v4._mtp_hidden_buffer.copy_(ids[:, None].expand(-1, 4))
            v4._mtp_hidden_valid_tokens = len(ids)
            v4._mtp_last_hidden_buffer.fill_(index)
            return ids[:, None].expand(-1, 2).float()

        hidden, aux = scope["forward_warmup_groups"](
            self.v4, None, torch.arange(10), self.positions, self.cu,
            tables, self.attn, self.groups(), forward)
        self.assertEqual(calls, [list(range(5)), list(range(5, 10))])
        self.assertEqual(len(set(pointers)), 1)
        self.assertEqual(aux[:, 0].tolist(), list(range(10)))
        self.assertEqual(hidden[:, 0].tolist(), list(range(10)))
        self.assertIs(self.v4._mtp_hidden_buffer, fixed)
        self.assertIs(self.v4._mtp_last_hidden_buffer, last)
        self.assertEqual(self.v4._mtp_hidden_valid_tokens, 1)
        self.assertEqual(self.v4._mtp_last_hidden_valid_tokens, 2)
        gc.collect()
        self.assertTrue(all(r() is None for r in refs))

        def fail(*args, **kwargs):
            raise RuntimeError("layer failed")

        with self.assertRaisesRegex(RuntimeError, "layer failed"):
            scope["forward_warmup_groups"](
                self.v4, None, torch.arange(10), self.positions, self.cu,
                tables, self.attn, self.groups(), fail)
        gc.collect()
        self.assertIs(self.v4._mtp_hidden_buffer, fixed)
        self.assertIs(self.v4._mtp_last_hidden_buffer, last)
        self.assertTrue(all(r() is None for r in refs))


if __name__ == "__main__":
    unittest.main()
