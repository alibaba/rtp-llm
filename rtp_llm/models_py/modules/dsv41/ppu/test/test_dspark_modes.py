"""Stdlib mode contracts, or one explicit --device PPU transaction fixture.

The source checks never import Torch. The device arm uses synthetic native
pools and actual PPU cache/row/transaction methods with independent whole-byte
expected state. Neither arm substitutes for the 40+3 native RDMA smoke.
"""

import ast
import importlib.util
import json
import os
import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

PPU = Path(__file__).resolve().parents[1]
ROOT = PPU.parents[3]


def load_light(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MODES = load_light("rtp_llm.models_py.modules.dsv41.ppu.dspark_modes", PPU / "dspark_modes.py")
POLICY = load_light("rtp_llm.models_py.modules.dsv41.ppu.replay_policy", PPU / "replay_policy.py")
PROTOCOL = load_light("n7_native_protocol", PPU / "native_protocol.py")
MODE = MODES.SharedDSparkMode


class SharedDSparkModeContracts(unittest.TestCase):
    def test_all_25_modes_and_capacity(self):
        count = 0
        for gamma in (1, 3, 5):
            choices = [MODE(gamma, c) for c in ("off", "observe")]
            for budget in sorted({0, gamma // 2 if gamma > 1 else 0, gamma}):
                choices += [MODE(gamma, "bounded", verify, budget) for verify in ("dense", "packed")]
            choices.append(MODE(gamma, "bounded", "packed", gamma, "adaptive"))
            for mode in choices:
                mode.validate()
                with self.subTest(mode=asdict(mode)):
                    env = {MODES.PROFILE_ENV: json.dumps(asdict(mode)), "DSV41_CED": "1", "DSV41_SWA_BOUNDED_REPLAY": "1"}
                    policy = POLICY.SharedPrefillPolicy.from_config(env=env)
                    self.assertEqual(policy.max_rollback_tokens, gamma + 1)
                    self.assertGreaterEqual(policy.keep_tokens, policy.window + mode.max_verified_rows)
                    self.assertEqual(MODE.from_config(mode.freeze({}), env={}), mode)
                count += 1
        self.assertEqual(count, 25)

    def test_actual_launch_both_roles_and_protocol_identity(self):
        # Only execute the real import-light launcher functions, with the real
        # runtime config parser. No model/bootstrap import on the local host.
        precision_name = "rtp_llm.models_py.modules.dsv41.ppu.moe_precision"
        load_light(precision_name, PPU / "moe_precision.py")
        runtime = load_light("n7_runtime_config", ROOT / "config/v41_runtime.py")
        tree = ast.parse((ROOT / "tools/v41/launch.py").read_text())
        selected = ast.Module(body=[x for x in tree.body if isinstance(x, ast.FunctionDef)
                                   and x.name in {"preset_env", "bounded_prefill_wait_timeout_ms"}], type_ignores=[])
        ns = {"os": os, "json": json, "DEFAULT_PREFILL_WAIT_TIMEOUT_MS": 1800000,
              "MAX_PREFILL_WAIT_TIMEOUT_MS": 86400000, "env_bool": runtime.env_bool,
              "PPU_STRATEGY": runtime.PPU_STRATEGY, "V41RuntimeConfig": runtime.V41RuntimeConfig}
        exec(compile(selected, "launch.py", "exec"), ns)
        for mode in (MODE(1), MODE(3, "bounded", "dense", 0), MODE(5, "bounded", "packed", 2, "adaptive"),
                     MODE(3, sample_from_anchor=False)):
            configs = []
            for role in ("prefill", "decode"):
                env = ns["preset_env"](role, "/checkpoint", parent={}, kv_cache_sharded=True,
                    ced_mode="bounded", dspark_gamma=mode.gamma, dspark_confidence=mode.confidence,
                    dspark_verify_mode=mode.verify, dspark_verify_budget=mode.budget,
                    dspark_budget_mode=mode.budget_mode, dspark_sample_from_anchor=mode.sample_from_anchor)
                self.assertEqual(MODE.from_config(env=env), mode)
                if role == "prefill":
                    self.assertEqual((env["SP_DSPARK_CONFIDENCE_MODE"], env["SP_DSPARK_VERIFY_MODE"],
                                      env["SP_DSPARK_VERIFY_BUDGET_MODE"], env["SP_DSPARK_VERIFY_BUDGET"]),
                                     ("off", "dense", "fixed", "-1"))
                with patch.dict(os.environ, env, clear=True):
                    config = POLICY.SharedPrefillPolicy.from_config().freeze({})
                    config = POLICY.PrefillPolicy.from_config().freeze(config)
                    config["_ppu_cp_cache"] = {"enabled": True, "size": 8, "rank": 0, "sliced_prefill": role == "prefill"}
                    configs.append(PROTOCOL.native_cache_protocol(config))
            self.assertEqual(configs[0], configs[1])
            self.assertEqual(configs[0]["speculative_state"], mode.protocol())
        with patch.dict(os.environ, {}, clear=True):
            seeds = [PROTOCOL.native_cache_key_hash_seed(mode.freeze({"_ppu_shared_prefill_policy": asdict(POLICY.SharedPrefillPolicy(max_rollback_tokens=4))}))
                     for mode in (MODE(3), MODE(3, "observe"), MODE(3, "bounded", "packed", 1))]
            self.assertEqual(len(set(seeds)), 3)

    def test_ragged_rows_and_unsupported_configs(self):
        self.assertEqual(MODES.target_loop_geometry((2, 5), True), (1, 7))
        self.assertEqual(MODES.target_loop_geometry((4, 4), False), (2, 4))
        for lengths, packed in (((), True), ((2, 0), True), ((2, 5), False)):
            with self.assertRaises(ValueError):
                MODES.target_loop_geometry(lengths, packed)
        for mode in (MODE(2), MODE(3, "observe", "packed", 1), MODE(3, "bounded", "dense", 1, "adaptive"),
                     MODE(3, "bounded", "packed", -1), MODE(3, "observe", sample_from_anchor=False)):
            with self.assertRaises(ValueError):
                mode.validate()
        self.assertEqual(MODE(5, sample_from_anchor=False).validate().query_width, 6)

    def test_actual_registration_consumes_profile_before_factory(self):
        path = ROOT / "models/deepseek_v41.py"
        tree = ast.parse(path.read_text())
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "DeepSeekV41DSpark")
        cls.body = [x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "_create_config"]

        class DeepSeekV41:
            @classmethod
            def _create_config(cls, path):
                return NS(
                    deepseek_v41_config={
                        "checkpoint": path, "num_nextn_predict_layers": 2,
                        "compress_ratios": [4, 0, 0], "dspark_n_routed_experts": 8,
                        "dspark_num_experts_per_tok": 2,
                    },
                    num_layers=1, mm_model_config=NS(), attn_config=NS(),
                    dspark_sample_from_anchor=True,
                )

        scope = {"DeepSeekV41": DeepSeekV41, "DeepSeekV4DSpark": object,
                 "v41_backend_for_config": lambda: "ppu"}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
        mode = MODE(3, sample_from_anchor=False)
        with patch.dict(
            os.environ, {MODES.PROFILE_ENV: json.dumps(asdict(mode))}, clear=True
        ):
            config = scope["DeepSeekV41DSpark"]._create_config("checkpoint")
        self.assertFalse(config.dspark_sample_from_anchor)
        self.assertEqual(config.deepseek_v41_config[MODES.CONFIG_KEY], asdict(mode))


def device_suite():
    import torch
    from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import SharedNativeCache
    from rtp_llm.models_py.modules.dsv41.ppu.prefill_runtime import SharedForwardState
    from rtp_llm.models_py.modules.dsv41.ppu.verify_transaction import SharedVerifyCache, SharedVerifyTransaction
    from rtp_llm.models_py.modules.dsv41.ppu.verify import VerifyLayout
    from rtp_llm.models_py.modules.dsv41.ppu.dspark_attention import unsampled_anchor_metadata

    class TransactionDevice(unittest.TestCase):
        @torch.inference_mode()
        def test_ragged_zero_partial_all_odd_pair_and_swa_wrap(self):
            self.assertTrue(torch.cuda.is_available(), "actual PPU required")
            device = torch.device("cuda", 0)
            indices, sizes = unsampled_anchor_metadata(
                torch.tensor([129, 7, 0], device=device), torch.tensor([True, True, False], device=device),
                width=4, window=128, virtual_span=512,
            )
            self.assertEqual(sizes.tolist(), [132, 11, 0])
            for rid, prefix in enumerate((129, 7, 0)):
                expected = [((rid + 1) * 512 + p) for p in range(max(0, prefix - 128), prefix + 4)] if rid < 2 else []
                for row in indices[rid * 4:(rid + 1) * 4].tolist():
                    self.assertEqual([p for p in row if p >= 0], expected)
            geometry = NS(num_layers=2, kv_source_layer_ids=(1,), window=128,
                          ratio=lambda layer: 2 if layer == 1 else 0,
                          engram_enabled=lambda: True, engram_max_ngram_size=4)
            widths = {"swa_kv": 528, "global_kv_2": 288, "indexer_kv": 68, "csa_state": 1024}
            framework = {"global_kv_2": "csa_kv", "indexer_kv": "indexer_kv", "swa_kv": "swa_kv", "csa_state": "csa_state"}
            keys = ((0, "swa_kv"), (1, "swa_kv"), (1, "global_kv_2"), (1, "indexer_kv"), (1, "csa_state"))
            entries = {"swa_kv": 128, "csa_kv": 128, "indexer_kv": 128, "csa_state": 2}
            spans = {"swa_kv": 128, "csa_kv": 256, "indexer_kv": 256, "csa_state": 128}
            handles, tagged = {}, {}
            table_host = [[1, 2, 3], [4, 5, 6]]
            table = torch.tensor(table_host, device=device, dtype=torch.int32)
            for layer, tag in keys:
                name = framework[tag]
                dtype = torch.float32 if tag == "csa_state" else torch.uint8
                value = torch.full((7, entries[name] * widths[tag] + 16), 173, device=device, dtype=dtype)
                handles[layer, name] = NS(kv_cache_base=value)
                tagged[name] = NS(kv_cache_block_id=table, kv_cache_block_id_device=table,
                                  kv_cache_kernel_block_id=table, kv_cache_kernel_block_id_device=table)
            native = NS(get_layer_cache=lambda layer, tag: handles[layer, tag],
                        get_entries_per_block=lambda tag: entries[tag], get_seq_size_per_block=lambda tag: spans[tag],
                        get_kernel_seq_size_per_block=lambda tag: spans[tag])
            layout = VerifyLayout.from_inputs(NS(input_lengths=torch.tensor([2, 5]), prefix_lengths=torch.tensor([127, 129])), 7)
            ids = torch.arange(10, 17, device=device, dtype=torch.int32)
            windows = []
            for start, length in zip(layout.starts, layout.lengths):
                history = [7, 6, 5]
                for token in ids[start:start + length].tolist():
                    window = [token, *history]
                    windows.append(window)
                    history = window[:-1]
            windows = torch.tensor(windows, device=device, dtype=torch.int32)
            request_ids = torch.tensor([101, 202])
            original = {key: handles[key[0], framework[key[1]]].kv_cache_base.clone() for key in keys}
            observed = []
            for counts in ((0, 0), (1, 2), (2, 5)):
                for key, value in original.items():
                    handles[key[0], framework[key[1]]].kv_cache_base.copy_(value)
                cache = SharedVerifyCache({}, geometry, native, tagged, (129, 134), device=device)
                state = SharedForwardState.__new__(SharedForwardState)
                state.prefill, state.inputs = False, NS(input_ids=ids)
                state.query_prefixes, state.lengths = layout.prefixes, layout.lengths
                pos, requests, cu, valid = state.rows()
                self.assertEqual(pos.tolist(), [127, 128, 129, 130, 131, 132, 133])
                self.assertEqual(requests.tolist(), [0, 0, 1, 1, 1, 1, 1])
                self.assertEqual(cu.tolist(), [0, 2, 7])
                tx = SharedVerifyTransaction(cache, request_ids, layout, windows, ids, call_id=1, max_backtracking=6)
                expected = {key: value.clone() for key, value in original.items()}
                for key in keys:
                    width, tag = widths[key[1]], key[1]
                    if tag == "swa_kv":
                        payload = ((pos[:, None] * 7 + torch.arange(width, device=device) + requests[:, None]) % 151).to(torch.uint8)
                        cache.stage_swa(key[0], pos, requests, payload, layout.prefixes)
                    for rid, (base, length, count) in enumerate(zip(layout.prefixes, layout.lengths, counts)):
                        full = tag in ("global_kv_2", "indexer_kv")
                        positions = list(range(base // 2 * 2, (base + length) // 2 * 2, 2)) if full else list(range(base, base + length))
                        raw = torch.tensor(positions, device=device, dtype=torch.long)
                        payload = ((raw[:, None] * 7 + torch.arange(width, device=device) + rid) % 151).to(cache.pools[key].tensor.dtype)
                        slots = cache.slots_at_positions(*key, rid, raw)
                        if tag != "swa_kv":
                            cache.insert_rows(*key, slots, payload)
                        end = base + count
                        # Independent scalar oracle: no production coordinate,
                        # acceptance-mask, planar-copy or latest-slot helpers.
                        for row, position in enumerate(positions):
                            keep = base // 2 <= position // 2 < end // 2 if full else base <= position < end
                            if tag == "csa_state":
                                keep = keep and position >= max(base, end - 2)
                            if not keep:
                                continue
                            name = framework[tag]
                            block = table_host[rid][position // spans[name]]
                            offset = position % spans[name] // 2 if full else position % entries[name]
                            for column in range(width):
                                destination = offset * 64 + column if tag == "indexer_kv" and column < 64 else (
                                    entries[name] * 64 + offset * 4 + column - 64 if tag == "indexer_kv" else offset * width + column)
                                expected[key][block, destination] = (position * 7 + column + rid) % 151
                self.assertTrue(all(torch.equal(pool.tensor, original[key]) for key, pool in cache.pools.items()))
                # A stale request order never publishes any candidate byte.
                with self.assertRaises(ValueError):
                    tx.finish(request_ids.flip(0), torch.tensor(layout.prefixes), torch.tensor(counts))
                evidence = tx.finish(request_ids, torch.tensor(layout.prefixes), torch.tensor(counts))
                self.assertEqual(evidence["accepted_ends"], [127 + counts[0], 129 + counts[1]])
                for key, pool in cache.pools.items():
                    self.assertTrue(torch.equal(pool.tensor, expected[key]), (counts, key))
                self.assertTrue(tx.closed and not cache.journal)
                observed.append({"counts": counts, "published_rows": evidence["native_rows_published"], "whole_native_bytes": "EXACT"})
            print(json.dumps({"schema": "n7-device-transaction-v1", "ragged_lengths": [2, 5],
                              "loop_view": MODES.target_loop_geometry((2, 5), True), "cases": observed}))
    return unittest.defaultTestLoader.loadTestsFromTestCase(TransactionDevice)


if __name__ == "__main__":
    if "--device" in sys.argv:
        result = unittest.TextTestRunner(verbosity=2).run(device_suite())
        sys.exit(not result.wasSuccessful())
    unittest.main()
