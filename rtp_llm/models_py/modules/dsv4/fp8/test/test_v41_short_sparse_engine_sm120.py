"""Bounded native PREFILL/PDFUSION smoke using maintained numerical choices."""

import gc
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.multiprocessing as mp

from rtp_llm.models_py.modules.dsv4.fp8.test import (
    test_v41_checkpoint_engine_sm120 as engine,
)

CHOICES = {
    "rotary_compute_device": "cuda",
    "sparse_attention_mode": "short_preserved",
}


def run_engine_rank(rank, port, checkpoint, directory, prompts, new_tokens):
    original = engine.DeepSeekV41Model.initialize

    def initialize(model, resource):
        result = original(model, resource)
        assert result and model._materialized
        assert model._rotary_compute_device == "cuda"
        assert model._sparse_attention_mode == "short_preserved"
        assert len(model.v4.layers) == 40
        for layer in model.v4.layers:
            assert layer.attn._sparse_attention_mode == "short_preserved"
            assert layer.attn._rotary_compute_device == "cuda"
            assert layer.attn.freqs_cis.device == torch.device("cuda", rank)
            layer.attn._check_execution_config()
        print(
            "DSV41_SHORT_ENGINE_OWNER "
            + json.dumps(
                dict(
                    rank=rank, layers=40, new_tokens=new_tokens, device=rank, **CHOICES
                )
            ),
            flush=True,
        )
        return result

    with patch.object(engine.DeepSeekV41Model, "initialize", initialize):
        engine.run_rank(
            rank, port, checkpoint, directory, prompts, False, new_tokens, 256
        )


class ShortSparseEngineSM120Test(unittest.TestCase):
    def run_smoke(self, new_tokens):
        self.assertEqual(torch.cuda.device_count(), 8)
        torch.manual_seed(41990)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, weights = engine.fixture(
                experts=16, full_schedule=True, max_seq_len=256
            )
            # Keep greedy image-token ties deterministic in the bounded fixture.
            head = weights.global_weights[engine.W.lm_head]
            head[127].copy_(head[126])
            engine.save_checkpoint(root, args, weights)
            del weights
            gc.collect()
            torch.cuda.empty_cache()
            config_path = root / "config.json"
            config = json.loads(config_path.read_text())
            config["text_config"].update(CHOICES)
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            loaded, _ = engine.load_configuration(root)
            for key, value in CHOICES.items():
                self.assertEqual(loaded.deepseek_v41_config[key], value)
            directory = root / "cache"
            directory.mkdir()
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            mp.spawn(
                run_engine_rank,
                args=(port, str(root), str(directory), [[3, 4, 5], [6, 7]], new_tokens),
                nprocs=8,
                join=True,
            )
            self.assertTrue(
                all(path.name.endswith(".lock") for path in directory.iterdir())
            )
        print(
            "DSV41_SHORT_ENGINE_SMOKE "
            + json.dumps(
                dict(
                    new_tokens=new_tokens,
                    phase="PREFILL" if new_tokens == 1 else "PDFUSION",
                    layers=40,
                    routing="real",
                    profiles=["baseline"],
                    completed=True,
                    **CHOICES
                )
            ),
            flush=True,
        )

    def test_01_prefill(self):
        self.run_smoke(1)

    def test_02_decode_after_prefill(self):
        self.run_smoke(9)


if __name__ == "__main__":
    unittest.main()
