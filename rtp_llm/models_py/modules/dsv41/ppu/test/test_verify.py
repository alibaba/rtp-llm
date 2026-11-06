"""Compare deferred verify state to independent ordinary decode continuation."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu.decode import DecodeStepOrchestrator
from rtp_llm.models_py.modules.dsv41.ppu.model import PPUModelCore
from rtp_llm.models_py.modules.dsv41.ppu.native_cache import NativeCacheBridge
from rtp_llm.models_py.modules.dsv41.ppu.native_model import NativeModelRunner
from rtp_llm.models_py.modules.dsv41.ppu.test.test_ced import make_ops
from rtp_llm.models_py.modules.dsv41.ppu.test.test_native_cache import NativePools
from rtp_llm.models_py.modules.dsv41.ppu.verify import VerifyLayout, VerifyTransaction
from rtp_llm.models_py.modules.dsv41.ppu.weight import (
    V41Geometry,
    load_ppu_weights,
    synthetic_config,
    write_synthetic_checkpoint,
)


class RingPools(NativePools):
    """Fixed native pools really wrap onto the same physical block."""

    def __init__(self, config, batch=1):
        super().__init__(config, batch=batch, blocks=24)
        for tag in ("swa_kv", "csa_state"):
            primary = self.inputs.get(tag)
            if primary is not None:
                table = primary.kv_cache_block_id_device.clone()
                for row in range(batch):
                    table[row].fill_(row + 1)
                primary.kv_cache_block_id_device = table
                primary.kv_cache_kernel_block_id_device = table


class VerifyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = synthetic_config()
        cls.geo = V41Geometry.from_config(cls.config)
        cls.ops = make_ops(cls.geo)
        with tempfile.TemporaryDirectory() as root:
            path = write_synthetic_checkpoint(cls.config, Path(root), seed=67)
            cls.bundle = load_ppu_weights(cls.config, path)
        cls.bundle.bind(cls.ops)

    def seeded(self, prefixes):
        native = RingPools(self.config, batch=len(prefixes))
        core = PPUModelCore(self.config, self.bundle, self.ops, device="cpu")
        core.capture_layer_ids = (5, 6, 7)
        tokens = [list(range(length)) for length in prefixes]
        for row, prompt in enumerate(tokens):
            core.prefill(
                torch.tensor(prompt), request_ids=torch.full((len(prompt),), row)
            )
        bridge = NativeCacheBridge(self.config, native, native.inputs)
        bridge.export_requests(core.cache, prefixes)
        runner = NativeModelRunner(core)
        runner.initialize(NS(kv_cache=native, is_decode_role=True))
        return runner, native, tokens

    def inputs(self, native, prompts, lengths):
        ids, windows = [], []
        prefixes = [len(prompt) for prompt in prompts]
        for prompt, count in zip(prompts, lengths):
            history = list(prompt)
            for index in range(count):
                token = 20 + index
                ids.append(token)
                history.append(token)
                windows.append(
                    (
                        [
                            history[-1 - back] if len(history) > back else -1
                            for back in range(self.geo.engram_max_ngram_size)
                        ]
                    )
                )
        for primary in native.inputs.values():
            primary.is_prefill = True
            primary.is_target_verify = True
            primary.input_lengths = torch.tensor(lengths)
            primary.prefix_lengths = torch.tensor(prefixes)
        return NS(
            input_ids=torch.tensor(ids),
            request_id=torch.arange(len(lengths)) + 410,
            attention_inputs=native.inputs,
            engram_token_windows=torch.tensor(windows),
            is_fake_stream=False,
            warmup=False,
        )

    def native_bytes(self, native):
        return {
            key: value.kv_cache_base.clone() for key, value in native.handles.items()
        }

    def assert_bytes(self, native, before):
        for key, value in before.items():
            torch.testing.assert_close(
                native.handles[key].kv_cache_base, value, rtol=0, atol=0
            )

    def oracle(self, native, prompts, counts):
        bridge = NativeCacheBridge(self.config, native, native.inputs)
        prefixes = [len(prompt) for prompt in prompts]
        cache = bridge.import_requests(prefixes)
        step = DecodeStepOrchestrator(
            ops=self.ops, bundle=self.bundle, cache=cache, capture_layer_ids=(5, 6, 7)
        )
        for row, prompt in enumerate(prompts):
            step.add_request(row, prompt)
        hidden = [[] for _ in prompts]
        features = [[] for _ in prompts]
        for column in range(max(counts, default=0)):
            tokens = {
                row: 20 + column for row, count in enumerate(counts) if column < count
            }
            result = step.step(tokens)
            for index, row in enumerate(tokens):
                hidden[row].append(result.hidden[index])
                features[row].append(result.mtp_target_hidden_states[index])
        return cache, hidden, features

    def assert_cache_prefix(self, native, expected, ends):
        bridge = NativeCacheBridge(self.config, native, native.inputs)
        restored = bridge.import_requests(ends)
        for row, end in enumerate(ends):
            for key in expected.pool_keys():
                positions = bridge._positions(key, end)
                if key[1] in ("global_kv_1", "global_kv_2", "indexer_kv"):
                    a = expected._slots_for_blocks(key, row, positions.numel())
                    b = restored._slots_for_blocks(key, row, positions.numel())
                else:
                    a = expected.slots_at_positions(*key, row, positions)
                    b = restored.slots_at_positions(*key, row, positions)
                torch.testing.assert_close(
                    expected._gather(key, a), restored._gather(key, b), rtol=0, atol=0
                )

    def test_dense_and_packed_causal_features_match_ordinary_decode(self):
        for lengths in ([4, 4], [1, 3], [6, 2]):
            with self.subTest(lengths=lengths):
                runner, native, prompts = self.seeded([9, 10])
                inputs = self.inputs(native, prompts, lengths)
                before = self.native_bytes(native)
                expected, hidden, features = self.oracle(native, prompts, lengths)
                actual, aux = runner.forward(inputs, return_aux=True)
                torch.testing.assert_close(
                    actual,
                    torch.stack([h for rows in hidden for h in rows]),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    aux,
                    torch.stack([h for rows in features for h in rows]),
                    rtol=0,
                    atol=0,
                )
                self.assert_bytes(native, before)
                runner.finish_speculative_forward(
                    inputs.request_id, torch.tensor([9, 10]), torch.tensor(lengths)
                )
                self.assert_cache_prefix(
                    native, expected, [9 + lengths[0], 10 + lengths[1]]
                )

    def test_zero_partial_all_acceptance_restore_swa_wrap_odd_pairs_and_indexer(self):
        for accepted in (0, 1, 3, 6):
            with self.subTest(accepted=accepted):
                runner, native, prompts = self.seeded([9])
                inputs = self.inputs(native, prompts, [6])
                before = self.native_bytes(native)
                expected, _, _ = self.oracle(native, prompts, [accepted])
                runner.forward(inputs)
                self.assert_bytes(native, before)
                runner.finish_speculative_forward(
                    inputs.request_id, torch.tensor([9]), torch.tensor([accepted])
                )
                if accepted == 0:
                    self.assert_bytes(native, before)
                self.assert_cache_prefix(native, expected, [9 + accepted])

    def test_cancel_abort_then_page_reuse_does_not_publish_old_round(self):
        runner, native, prompts = self.seeded([9])
        inputs = self.inputs(native, prompts, [4])
        before = self.native_bytes(native)
        runner.forward(inputs)
        runner.abort_speculative_forward()
        self.assert_bytes(native, before)
        inputs.request_id += 1000
        runner.forward(inputs)
        runner.finish_speculative_forward(
            inputs.request_id, torch.tensor([9]), torch.tensor([1])
        )
        expected, _, _ = self.oracle_from_original(prompts, [1])
        self.assert_cache_prefix(native, expected, [10])

    def oracle_from_original(self, prompts, lengths):
        _, native, _ = self.seeded([len(prompt) for prompt in prompts])
        self.inputs(native, prompts, lengths)
        return self.oracle(native, prompts, lengths)

    def test_transaction_rejects_identity_order_prefix_and_double_finalize(self):
        runner, native, prompts = self.seeded([9, 10])
        inputs = self.inputs(native, prompts, [2, 2])
        before = self.native_bytes(native)
        runner.forward(inputs)
        tx = runner._pending_verify
        for ids, prefix in (
            (torch.tensor([411, 410]), [9, 10]),
            (torch.tensor([410, 411]), [10, 10]),
        ):
            with self.assertRaisesRegex(ValueError, "request/round"):
                tx.finish(ids, torch.tensor(prefix), torch.tensor([1, 1]))
            self.assert_bytes(native, before)
        tx.finish(inputs.request_id, torch.tensor([9, 10]), torch.tensor([0, 0]))
        with self.assertRaisesRegex(RuntimeError, "already finalized"):
            tx.finish(inputs.request_id, torch.tensor([9, 10]), torch.tensor([0, 0]))
        self.assert_bytes(native, before)
        runner.abort_speculative_forward()

    def test_layout_requires_real_anchor_and_exact_request_major_rows(self):
        for lengths, prefixes, rows in (([0], [9], 0), ([2], [9], 3), ([2], [], 2)):
            with self.assertRaises(ValueError):
                VerifyLayout.from_inputs(
                    NS(
                        input_lengths=torch.tensor(lengths),
                        prefix_lengths=torch.tensor(prefixes),
                    ),
                    rows,
                )

    def test_publication_exception_restores_every_affected_native_byte(self):
        runner, native, prompts = self.seeded([9])
        inputs = self.inputs(native, prompts, [4])
        before = self.native_bytes(native)
        runner.forward(inputs)
        bridge = runner._pending_verify.bridge
        original = bridge._copy_rows
        writes = 0

        def fail_once(key, row, positions, packed=None):
            nonlocal writes
            if packed is not None:
                writes += 1
                if writes == 3:
                    raise RuntimeError("injected native copy failure")
            return original(key, row, positions, packed)

        with patch.object(bridge, "_copy_rows", side_effect=fail_once):
            with self.assertRaisesRegex(RuntimeError, "injected native copy"):
                runner.finish_speculative_forward(
                    inputs.request_id, torch.tensor([9]), torch.tensor([2])
                )
        self.assert_bytes(native, before)
        with self.assertRaisesRegex(RuntimeError, "no pending"):
            runner.finish_speculative_forward(
                inputs.request_id, torch.tensor([9]), torch.tensor([1])
            )
