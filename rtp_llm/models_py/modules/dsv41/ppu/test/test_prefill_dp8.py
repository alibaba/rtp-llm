"""DP8 native Prefill with EP8/ENP8, uneven batches and empty owners."""

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu.cp import ThreadedCPGroup
from rtp_llm.models_py.modules.dsv41.ppu.model import PPUModelCore
from rtp_llm.models_py.modules.dsv41.ppu.native_cache import NativeCacheBridge
from rtp_llm.models_py.modules.dsv41.ppu.native_model import NativeModelRunner
from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
from rtp_llm.models_py.modules.dsv41.ppu.test.test_native_cache import NativePools
from rtp_llm.models_py.modules.dsv41.ppu.weight import (
    V41Geometry,
    load_ppu_weights,
    synthetic_config,
    write_synthetic_checkpoint,
)


class _EPOps:
    def __init__(self, ops, view):
        self.ops, self.view = ops, view
        self.layers = []
        self.invalid_rows = 0

    def __getattr__(self, name):
        return getattr(self.ops, name)

    def moe(self, x, expert_ids, expert_weights, *, layer_id, context=None):
        self.layers.append(layer_id)
        self.invalid_rows += int((expert_ids < 0).all(dim=1).sum())
        return self.view.moe(x, expert_ids, expert_weights, layer_id)


class DPPrefillTest(unittest.TestCase):
    def test_native_dp8_ep8_enp8_independent_requests_and_idle_owners(self):
        config = synthetic_config()
        geo = V41Geometry.from_config(config)
        group = ThreadedCPGroup(8)
        cores, raw_ops = [], {}
        with tempfile.TemporaryDirectory(prefix="v41-dp-prefill-") as root:
            checkpoint = write_synthetic_checkpoint(config, Path(root), seed=27)
            for rank in range(8):
                view = group.rank_view(rank)
                ops = PPUOps(
                    ep_rank=rank,
                    ep_size=8,
                    num_routed_experts=geo.n_routed_experts,
                    num_local_experts=geo.num_local_experts(8),
                    hidden_size=geo.hidden_size,
                    inter_dim=geo.moe_inter_dim,
                    swiglu_limit=geo.swiglu_limit,
                    mhc_config=geo.mhc_config(),
                    sm_scale=geo.sm_scale(),
                    window=geo.window,
                    enpm_group=view,
                )
                bundle = load_ppu_weights(
                    config,
                    checkpoint,
                    ep_rank=rank,
                    ep_size=8,
                    enp_rank=rank,
                    enp_size=8,
                )
                bundle.bind(ops, enpm_group=view)
                raw_ops[rank] = ops
                cores.append(PPUModelCore(config, bundle, _EPOps(ops, view)))
        group.register_ops(raw_ops)

        scenarios = (
            [[3 + rank % 3] for rank in range(8)],
            [[], [], [], [], [7], [], [], []],
            [[3, 5], [], [4], [], [6, 2], [], [], [1]],
            [[] for _ in range(8)],
        )
        # The work/stop vote has its own native-model tests. This test uses
        # real ENP rendezvous and dispatches every token to the eight actual
        # synthetic expert shards, with independent native page pools.
        for lengths_by_rank in scenarios:
            with self.subTest(lengths=lengths_by_rank):
                busy = any(lengths_by_rank)

                def forward(rank):
                    core = cores[rank]
                    lengths = lengths_by_rank[rank]
                    batch = max(1, len(lengths))
                    native = NativePools(config, batch=batch)
                    before = {
                        key: handle.kv_cache_base.clone()
                        for key, handle in native.handles.items()
                    }
                    ids = (
                        torch.cat(
                            [
                                (torch.arange(length) + rank * 7 + row) % geo.vocab_size
                                for row, length in enumerate(lengths)
                            ]
                        )
                        if lengths
                        else torch.zeros(1, dtype=torch.int64)
                    )
                    windows = []
                    cursor = 0
                    for length in lengths:
                        tokens = ids[cursor : cursor + length].tolist()
                        windows.extend(
                            [
                                [
                                    tokens[pos - back] if pos >= back else -1
                                    for back in range(geo.engram_max_ngram_size)
                                ]
                                for pos in range(length)
                            ]
                        )
                        cursor += length
                    for entry in native.inputs.values():
                        entry.is_prefill = True
                        entry.input_lengths = torch.tensor(lengths or [1])
                        entry.prefix_lengths = torch.zeros(batch, dtype=torch.int64)
                    inputs = NS(
                        input_ids=ids,
                        attention_inputs=native.inputs,
                        engram_token_windows=torch.tensor(windows, dtype=torch.int64),
                        is_fake_stream=not lengths,
                        warmup=False,
                    )
                    runner = NativeModelRunner(core, dp_rank=rank, dp_size=8)
                    runner.initialize(NS(kv_cache=native, is_decode_role=False))
                    core.ops.layers.clear()
                    core.ops.invalid_rows = 0
                    with patch.object(runner, "_work_available", return_value=busy):
                        hidden = runner.forward(inputs)
                    self.assertEqual(hidden.shape, (ids.numel(), geo.hidden_size))
                    self.assertTrue(bool(torch.isfinite(hidden).all()))
                    self.assertEqual(
                        core.ops.layers, list(range(geo.num_layers)) if busy else []
                    )
                    if not lengths:
                        self.assertEqual(native.calls, [])
                        self.assertEqual(int(hidden.count_nonzero()), 0)
                        self.assertEqual(
                            core.ops.invalid_rows, geo.num_layers if busy else 0
                        )
                        for key, handle in native.handles.items():
                            self.assertTrue(
                                torch.equal(handle.kv_cache_base, before[key])
                            )
                    else:
                        self.assertEqual(set(native.calls), set(native.handles))
                        cache = NativeCacheBridge(
                            config, native, native.inputs
                        ).import_requests(lengths)
                        self.assertEqual(
                            cache._request_tokens, dict(enumerate(lengths))
                        )
                    return hidden

                with ThreadPoolExecutor(max_workers=8) as executor:
                    list(executor.map(forward, range(8)))


if __name__ == "__main__":
    unittest.main()
