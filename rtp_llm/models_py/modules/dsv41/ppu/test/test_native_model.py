"""Native scheduler inputs, CacheStore publication and decode continuation."""

import tempfile
from concurrent.futures import ThreadPoolExecutor
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu.decode import DecodeStepOrchestrator
from rtp_llm.models_py.modules.dsv41.ppu.cp import ThreadedCPGroup, build_zigzag_plan
from rtp_llm.models_py.modules.dsv41.ppu.model import PPUModelCore
from rtp_llm.models_py.modules.dsv41.ppu.native_model import NativeModelRunner
from rtp_llm.models_py.modules.dsv41.ppu.native_protocol import FRAMEWORK_POOL_TAGS
from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
from rtp_llm.models_py.modules.dsv41.ppu.test.test_native_cache import NativePools
from rtp_llm.models_py.modules.dsv41.ppu.weight import (
    V41Geometry,
    load_ppu_weights,
    synthetic_config,
    write_synthetic_checkpoint,
)


class NativeModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = synthetic_config()
        cls.geo = V41Geometry.from_config(cls.config)
        cls.ops = PPUOps(
            ep_size=1,
            num_routed_experts=cls.geo.n_routed_experts,
            num_local_experts=cls.geo.n_routed_experts,
            hidden_size=cls.geo.hidden_size,
            inter_dim=cls.geo.moe_inter_dim,
            swiglu_limit=cls.geo.swiglu_limit,
            mhc_config=cls.geo.mhc_config(),
            sm_scale=cls.geo.sm_scale(),
            window=cls.geo.window,
        )
        with tempfile.TemporaryDirectory(prefix="v41-native-model-") as root:
            checkpoint = write_synthetic_checkpoint(cls.config, Path(root), seed=13)
            cls.bundle = load_ppu_weights(cls.config, checkpoint)
        cls.bundle.bind(cls.ops)

    def runner(self, *, decode=False):
        core = PPUModelCore(self.config, self.bundle, self.ops, device="cpu")
        runner = NativeModelRunner(core)
        pools = NativePools(self.config, batch=1)
        self.assertTrue(runner.initialize(NS(kv_cache=pools, is_decode_role=decode)))
        return runner, pools

    def inputs(self, pools, tokens, *, prefix=0, prefill=True, fake=False):
        tokens = [int(token) for token in tokens]
        ids = tokens[prefix:]
        windows = torch.tensor(
            [
                [
                    tokens[pos - back] if pos >= back else -1
                    for back in range(self.geo.engram_max_ngram_size)
                ]
                for pos in range(prefix, len(tokens))
            ],
            dtype=torch.int64,
        )
        for entry in pools.inputs.values():
            entry.is_prefill = prefill
            entry.input_lengths = torch.tensor([len(ids)])
            entry.prefix_lengths = torch.tensor([prefix])
            entry.sequence_lengths = torch.tensor([prefix])
        return NS(
            input_ids=torch.tensor(ids),
            attention_inputs=pools.inputs,
            engram_token_windows=windows,
            is_fake_stream=fake,
            warmup=False,
        )

    def test_prefill_publishes_and_decode_uses_scheduler_history(self):
        prefill, native = self.runner()
        prompt = (torch.arange(15) % self.geo.vocab_size).tolist()
        hidden = prefill.forward(self.inputs(native, prompt))
        reference, _ = self.runner()
        result = reference.core.prefill(torch.tensor(prompt), apply_lm_head=True)
        self.assertTrue(torch.equal(hidden, result.hidden))
        self.assertEqual(
            set(native.calls),
            {
                (owner, FRAMEWORK_POOL_TAGS[tag])
                for owner, tag in reference.core.cache.pool_keys()
            },
        )
        decode = NativeModelRunner(reference.core)
        decode.initialize(NS(kv_cache=native, is_decode_role=True))
        uninterrupted = DecodeStepOrchestrator(
            ops=self.ops, bundle=self.bundle, cache=reference.core.cache
        )
        uninterrupted.add_request(0, prompt)
        pending = int(result.logits[-1].argmax())
        tokens = list(prompt)
        with patch.object(
            decode.core, "prefill", side_effect=AssertionError("D ran prefill")
        ):
            for _ in range(4):
                prefix = len(tokens)
                tokens.append(pending)
                actual = decode.forward(
                    self.inputs(native, tokens, prefix=prefix, prefill=False)
                )
                expected = uninterrupted.step({0: pending})
                self.assertTrue(torch.equal(actual, expected.hidden))
                pending = int(expected.logits[0].argmax())
        # Decode commits pages without republishing the P-side writer.
        self.assertEqual(len(native.calls), len(native.handles))

    def test_idle_world_has_no_cache_or_collective_work(self):
        runner, native = self.runner(decode=True)
        inputs = self.inputs(native, [0], fake=True, prefill=False)
        with patch.object(
            runner, "_orchestrator", side_effect=AssertionError("idle work")
        ):
            self.assertTrue(
                torch.equal(
                    runner.forward(inputs),
                    torch.zeros(1, self.geo.hidden_size, dtype=torch.bfloat16),
                )
            )
        self.assertEqual(native.calls, [])

    def test_idle_rank_with_active_peers_only_runs_placeholders(self):
        runner, native = self.runner(decode=True)
        inputs = self.inputs(native, [0], fake=True, prefill=False)
        with patch.object(runner, "_work_available", return_value=True), patch.object(
            runner, "_orchestrator", wraps=runner._orchestrator
        ) as factory:
            actual = runner.forward(inputs)
        self.assertEqual(tuple(actual.shape), (1, self.geo.hidden_size))
        self.assertEqual(factory.call_args.args[0]._request_tokens, {})
        self.assertEqual(native.calls, [])

    def test_peer_stop_vote_quiesces_idle_rank_before_releasing_native_loop(self):
        runner, native = self.runner(decode=True)
        inputs = self.inputs(native, [0], fake=True, prefill=False)
        observed = []
        runner.bind_native_stop(lambda: False, lambda: observed.append(True))

        def peer_stops(value, group=None):
            self.assertEqual(value.tolist(), [0, 0])
            value[1] = 1

        with patch("torch.distributed.is_available", return_value=True), patch(
            "torch.distributed.is_initialized", return_value=True
        ), patch(
            "torch.distributed.all_reduce", side_effect=peer_stops
        ) as vote, patch.object(
            runner, "_orchestrator", side_effect=AssertionError("stopped model work")
        ):
            for _ in range(2):
                self.assertEqual(
                    tuple(runner.forward(inputs).shape), (1, self.geo.hidden_size)
                )
        self.assertEqual(observed, [True])
        self.assertEqual(vote.call_count, 1)
        self.assertEqual(native.calls, [])

    def test_missing_decode_history_is_rejected(self):
        runner, native = self.runner(decode=True)
        inputs = self.inputs(native, [1, 2, 3, 4], prefix=3, prefill=False)
        inputs.engram_token_windows = torch.empty(0)
        with self.assertRaisesRegex(ValueError, "missing scheduler Engram"):
            runner.forward(inputs)

    def test_native_cp_returns_local_rows_for_cpp_output_gather(self):
        # Exercise the native CP input/output boundary independently of the
        # already-tested CP layer math. C++ performs the outer output gather.
        tokens = torch.arange(17)
        plans = [
            build_zigzag_plan(tokens, torch.tensor([0, 17]), cp_size=8, cp_rank=rank)
            for rank in range(8)
        ]
        group = ThreadedCPGroup(8)
        windows = torch.tensor(
            [
                [
                    int(tokens[pos - back]) if pos >= back else -1
                    for back in range(self.geo.engram_max_ngram_size)
                ]
                for pos in range(17)
            ]
        )
        restore = torch.zeros(plans[0].padded_seq_len, dtype=torch.int64)
        restore[:17] = plans[0].unpad_restore
        mask = torch.arange(plans[0].padded_seq_len) < 17

        def forward(rank):
            plan = plans[rank]
            core = NS(
                config=self.config, device="cpu", geometry=self.geo, weights=self.bundle
            )

            def prefill(ids, **kwargs):
                self.assertTrue(torch.equal(ids, plan.local_input_ids))
                self.assertTrue(
                    torch.equal(kwargs["lookback_token_ids"], torch.full((1, 3), -1))
                )
                return NS(
                    hidden=tokens[:, None].expand(17, self.geo.hidden_size).float()
                )

            core.prefill = prefill
            runner = NativeModelRunner(
                core, cp_rank=rank, cp_size=8, group=group.rank_view(rank)
            )
            info = NS(
                prefill_qkv_restore_indice_cpu=restore,
                prefill_qkv_padding_mask_cpu=mask,
                prefill_actual_input_lengths_cpu=torch.tensor([17]),
                prefill_prefix_lengths_cpu=torch.tensor([0]),
            )
            local_windows = windows[plan.local_positions].clone()
            inputs = NS(
                input_ids=plan.local_input_ids,
                engram_token_windows=local_windows,
                is_fake_stream=False,
                warmup=False,
                attention_inputs={
                    "swa_kv": NS(is_prefill=True, context_parallel_info=info)
                },
            )
            return runner.forward(inputs)

        with patch(
            "rtp_llm.models_py.modules.dsv41.ppu.native_model.DistributedCPGroup",
            side_effect=lambda view: view,
        ), ThreadPoolExecutor(max_workers=8) as executor:
            outputs = list(executor.map(forward, range(8)))
        gathered = torch.cat(outputs)
        restored = gathered[plans[0].unpad_restore]
        self.assertTrue(torch.equal(restored[:, 0], tokens.float()))
        for rank, output in enumerate(outputs):
            self.assertTrue(
                torch.equal(output[:, 0], plans[rank].local_positions.float())
            )


if __name__ == "__main__":
    unittest.main()
