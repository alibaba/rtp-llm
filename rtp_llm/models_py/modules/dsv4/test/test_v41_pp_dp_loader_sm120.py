"""Shared CEP4PP2/DP8EP8 checkpoint slices and a single-device worker control."""

import gc
import json
import os
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from rtp_llm.models_py.model_desc.deepseek_v4_model import Dsv4SharedRuntimeBufferStore
from rtp_llm.models_py.model_desc.deepseek_v41_model import DeepSeekV41Model
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_loader_worker_sm120 import (
    load_configuration,
    make_loader,
    save_checkpoint,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_worker_sm120 import (
    context_tables,
    decode_inputs,
    inputs,
    resources,
    tagged_cache,
)
from rtp_llm.models_py.modules.dsv4.test import v41_checkpoint_test_support as handoff
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_execution_sm120 import (
    raw_snapshot,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_preflight_sm120 import (
    clean_worker,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_pp_loader_sm120 import make_pp_loader
from rtp_llm.models_py.modules.dsv4.test.test_v41_stage_slices_sm120 import wide_fixture
from rtp_llm.utils.model_weight import W


class V41PPDPCheckpointSM120Test(unittest.TestCase):
    @torch.inference_mode()
    def test_shared_named_checkpoint_and_local_continuation(self):
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        root = Path(os.environ["DSV41_TEST_PP_DP_LOADER_ROOT"]).resolve()
        checkpoint = root / "checkpoint"
        checkpoint.mkdir(parents=True)
        torch.manual_seed(41701)
        args, weights = wide_fixture(experts=16)
        weights.global_weights[W.lm_head][127].copy_(
            weights.global_weights[W.lm_head][126]
        )
        save_checkpoint(checkpoint, args, weights)
        oracles = handoff.loader_oracles(args, weights)
        handoff.save_json(root / "loader-oracles.json", oracles)
        del weights
        gc.collect()
        torch.cuda.empty_cache()
        loaded_records = {role: {} for role in oracles}
        for role in oracles:
            for rank in range(8):
                config, kv_config = load_configuration(checkpoint, max_seq_len=512)
                if role == "source_loader":
                    _, _, loader = make_pp_loader(config, kv_config, rank)
                else:
                    _, loader = make_loader(config, kv_config, 8, rank)
                names, _ = loader._generate_weight_info()
                self.assertFalse(any("engram" in name for name in names))
                loaded = loader.load_weights(f"cuda:{torch.cuda.current_device()}")
                signature = handoff.producer.loader_signature(loaded)
                self.assertEqual(signature, oracles[role][str(rank)])
                loaded_records[role][str(rank)] = signature
                handoff.emit(
                    "DSV41_SHARED_CHECKPOINT_LOADER",
                    role=role,
                    rank=rank,
                    entries_exact=len(signature),
                    ep_size=4 if role == "source_loader" else 8,
                    pp_size=2 if role == "source_loader" else 1,
                    local_compute=False,
                )
                del loader, loaded
                gc.collect()
                torch.cuda.empty_cache()
        handoff.save_json(root / "loaded.json", loaded_records)
        choices = {
            f"mhc_{component}_mode": mode
            for component, mode in handoff.producer.MHC_MODES.items()
        }
        os.environ.update(
            {
                "DSV41_ENGRAM_HOST_BYTES": str(2**20),
                "DSV41_ENGRAM_CACHE_DIR": str(root / "tables"),
                "DSV4_MOE_STRATEGY": "auto",
                "MOEDBG": "0",
            }
        )
        (root / "tables").mkdir()
        prompts = [
            [(3 + request * 17 + token) % 127 for token in range(length)]
            for request, length in enumerate((127, 129))
        ]
        results = []
        with handoff.producer.pp_mhc_equations(choices, False):
            for phase in range(2):
                Dsv4SharedRuntimeBufferStore._reset_for_test()
                config, kv_config = load_configuration(checkpoint, max_seq_len=512)
                config.deepseek_v41_config.update(choices)
                parallelism, loader = make_loader(config, kv_config, 1, 0)
                loaded = loader.load_weights(f"cuda:{torch.cuda.current_device()}")
                head = loaded.global_weights[W.lm_head].clone()
                model = DeepSeekV41Model(config, parallelism, loaded, None, 2)
                del loader, loaded
                try:
                    self.assertTrue(model.initialize(resources()))
                    self.assertTrue(
                        model.initialize(resources(tagged_cache(model, pages=8)))
                    )
                    self.assertEqual(len(model.v4.layers), 40)
                    self.assertEqual(len(model.kv_cache.groups), 51)
                    tables = context_tables(model.kv_cache)
                    hidden = model.forward(
                        inputs(
                            model.kv_cache,
                            [token for prompt in prompts for token in prompt],
                            [len(prompt) for prompt in prompts],
                            tables=tables,
                        )
                    ).hidden_states
                    clean_worker(model)
                    self.assertEqual(hidden.shape, (256, 5120))
                    self.assertTrue(torch.isfinite(hidden).all())
                    logits = F.linear(hidden[[126, 255]], head).float()
                    tokens = logits.argmax(-1).tolist()
                    result = {
                        "prefill_hidden": hidden.cpu().clone(),
                        "prefill_logits": logits.cpu().clone(),
                        "prefill_cache": raw_snapshot(model.kv_cache),
                        "steps": [],
                    }
                    starts = [127, 129]
                    history = [prompt[-3:][::-1] for prompt in prompts]
                    for step in range(4):
                        request = decode_inputs(
                            model.kv_cache,
                            tokens,
                            starts,
                            history,
                            prompt_lengths=[127, 129],
                            tables=tables,
                        )
                        windows = request.engram_token_windows.cpu().clone()
                        hidden = model.forward(request).hidden_states
                        clean_worker(model)
                        self.assertEqual(hidden.shape, (2, 5120))
                        self.assertTrue(torch.isfinite(hidden).all())
                        logits = F.linear(hidden, head).float()
                        next_tokens = logits.argmax(-1).tolist()
                        self.assertTrue(all(0 <= token < 127 for token in next_tokens))
                        result["steps"].append(
                            {
                                "step": step,
                                "starts": starts.copy(),
                                "history": [row.copy() for row in history],
                                "input_tokens": tokens,
                                "output_tokens": next_tokens,
                                "windows": windows,
                                "hidden": hidden.cpu().clone(),
                                "logits": logits.cpu().clone(),
                                "cache": raw_snapshot(model.kv_cache),
                            }
                        )
                        history = [
                            [token, *row[:2]] for token, row in zip(tokens, history)
                        ]
                        tokens, starts = next_tokens, [start + 1 for start in starts]
                    result.update(start=starts, history=history, next_tokens=tokens)
                    handoff.producer.save_tensors(root / f"worker{phase}.pt", result)
                    results.append(result)
                finally:
                    model.close()
                    self.assertEqual(model._worker_state, "closed")
                    Dsv4SharedRuntimeBufferStore._reset_for_test()
                del model, head
                gc.collect()
                torch.cuda.empty_cache()
        tensors = handoff.producer.require_exact_tree(*results)
        self.assertTrue(
            all(path.name.endswith(".lock") for path in (root / "tables").iterdir())
        )
        summary = {
            "checkpoint_sha256": {
                path.name: handoff.digest(path)
                for path in checkpoint.iterdir()
                if path.is_file()
            },
            "choices": choices,
            "layers": 40,
            "hidden_size": 5120,
            "experts": 16,
            "loader_profiles": 16,
            "two_cold_workers_exact": True,
            "tensors_exact": tensors,
            "prefill_lengths": [127, 129],
            "decode_rounds_per_worker": 4,
            "compute_ep_size": 1,
            "compute_world_size": 1,
            "standalone_decode_role": False,
            "distributed_handoff": False,
        }
        handoff.save_json(root / "completion.json", summary)
        handoff.emit("DSV41_SHARED_CHECKPOINT_COMPARISON", **summary)


if __name__ == "__main__":
    unittest.main()
