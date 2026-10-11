"""CPU-only tests for the jev_vl subcommand of convert_decision_head.py.

Synthesizes a miniature JEV-27B-VL-shaped checkpoint (base shards + peft
adapter + decision head + calibration), runs the converter, and checks the
LoRA merge math, the 264-row verbalizer head, and the decision block.
"""

import itertools
import json
import string
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file

from rtp_llm.tools.convert_decision_head import convert_jev_vl

HIDDEN = 8
FFN = 16
VOCAB = 400
RANK = 2
SCALE = 2.0  # lora_alpha 4 / r 2


def _codes():
    return (list(string.ascii_uppercase) + [
        a + b for a, b in itertools.product(string.ascii_uppercase, repeat=2)
    ])[:256]


def _labels_and_ids():
    codes = _codes()
    labels = ["false", "true"] + [str(i) for i in range(6)] + codes
    ids = list(range(100, 100 + len(labels)))
    return labels, ids


def _write(path, obj):
    Path(path).write_text(json.dumps(obj))


def _make_ckpt(root: Path):
    labels, ids = _labels_and_ids()
    torch.manual_seed(0)
    base_gate = torch.randn(FFN, HIDDEN)
    base_lm_head = torch.randn(VOCAB, HIDDEN)
    gate_a, gate_b = torch.randn(RANK, HIDDEN), torch.randn(FFN, RANK)
    head_a, head_b = torch.randn(RANK, HIDDEN), torch.randn(VOCAB, RANK)
    bias = torch.randn(24) / 100

    ckpt = root / "jev"
    ckpt.mkdir()
    _write(
        ckpt / "config.json",
        {
            "model_type": "qwen3_5",
            "vision_start_token_id": 248053,
            "vision_end_token_id": 248054,
        },
    )
    _write(
        ckpt / "model.safetensors.index.json",
        {
            "metadata": {"total_size": 0},
            "weight_map": {
                "lm_head.weight": "model-00001-of-00002.safetensors",
                "model.language_model.layers.0.mlp.gate_proj.weight": (
                    "model-00002-of-00002.safetensors"
                ),
            },
        },
    )
    from safetensors.torch import save_file

    save_file(
        {"lm_head.weight": base_lm_head}, str(ckpt / "model-00001-of-00002.safetensors")
    )
    save_file(
        {"model.language_model.layers.0.mlp.gate_proj.weight": base_gate},
        str(ckpt / "model-00002-of-00002.safetensors"),
    )
    _write(ckpt / "calibration.json", {"per_kind": {"noul": 1.5, "score": 1.1, "choice": 1.2}})

    adapter = ckpt / "adapter_vllm"
    adapter.mkdir()
    _write(
        adapter / "adapter_config.json",
        {"r": RANK, "lora_alpha": 4.0, "use_rslora": False},
    )
    _write(
        adapter / "decision_head.json",
        {
            "verbalizer_ids": ids[:24],
            "bias": bias.tolist(),
            "slots": {
                "num_slots": 24,
                "ranges": {"noul": [0, 2], "score": [2, 8], "choice": [8, 24]},
                "verbalizers": labels[:24],
                "template_version": "bare-v1",
            },
        },
    )
    save_file(
        {
            "base_model.model.model.language_model.layers.0.mlp.gate_proj.lora_A.weight": gate_a,
            "base_model.model.model.language_model.layers.0.mlp.gate_proj.lora_B.weight": gate_b,
            "base_model.model.lm_head.lora_A.weight": head_a,
            "base_model.model.lm_head.lora_B.weight": head_b,
        },
        str(adapter / "adapter_model.safetensors"),
    )
    # vocab fallback for label resolution (AutoTokenizer cannot load this dir)
    _write(ckpt / "vocab.json", {lab: i for lab, i in zip(labels, ids)})

    return {
        "ckpt": ckpt,
        "gate": base_gate + SCALE * (gate_b @ gate_a),
        "lm_head": base_lm_head + SCALE * (head_b @ head_a),
        "labels": labels,
        "ids": ids,
        "bias": bias,
    }


class ConvertJevVlTest(unittest.TestCase):
    def test_merge_head_and_decision_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _make_ckpt(root)
            target = root / "target"
            block = convert_jev_vl(str(fixture["ckpt"]), str(target))

            self.assertEqual(block["prompt_format"], "bare")
            self.assertEqual(block["head_type"], "linear")
            self.assertEqual(block["num_labels"], 264)
            self.assertEqual(block["slot_ranges"]["choice"], [8, 264])
            self.assertEqual(block["slot_ranges"]["noul"], [0, 2])
            self.assertEqual(block["slot_ranges"]["score"], [2, 8])
            self.assertEqual(block["temperatures"], {"noul": 1.5, "score": 1.1, "choice": 1.2})
            self.assertEqual(len(block["codes"]), 256)
            self.assertEqual(block["codes"][:4], ["A", "B", "C", "D"])

            # every adapter tensor consumed: shards carry the merged weights
            merged_gate = load_file(
                str(target / "model-00002-of-00002.safetensors")
            )["model.language_model.layers.0.mlp.gate_proj.weight"]
            torch.testing.assert_close(merged_gate, fixture["gate"])
            merged_lm_head = load_file(
                str(target / "model-00001-of-00002.safetensors")
            )["lm_head.weight"]
            torch.testing.assert_close(merged_lm_head, fixture["lm_head"])
            self.assertTrue((target / ".jev_lora_merged").exists())

            # head rows are the merged lm_head rows at the label ids
            head = load_file(str(target / "decision_head.safetensors"))
            expected_rows = merged_lm_head.index_select(
                0, torch.tensor(fixture["ids"], dtype=torch.long)
            )
            self.assertEqual(tuple(head["decision_head.weight"].shape), (264, HIDDEN))
            torch.testing.assert_close(
                head["decision_head.weight"], expected_rows
            )
            expected_bias = torch.cat(
                [fixture["bias"], torch.zeros(264 - 24)]
            )
            torch.testing.assert_close(
                head["decision_head.bias"], expected_bias
            )

            # config + index patched; text-only decision strips the vision ids
            config = json.loads((target / "config.json").read_text())
            self.assertNotIn("vision_start_token_id", config)
            self.assertNotIn("vision_end_token_id", config)
            self.assertEqual(config["decision"]["slot_ranges"], block["slot_ranges"])
            index = json.loads(
                (target / "model.safetensors.index.json").read_text()
            )
            self.assertEqual(
                index["weight_map"]["decision_head.weight"],
                "decision_head.safetensors",
            )

    def test_idempotent_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _make_ckpt(root)
            target = root / "target"
            first = convert_jev_vl(str(fixture["ckpt"]), str(target))
            # second run on the same target: merge marker + identical head skip
            second = convert_jev_vl(str(target), str(target))
            self.assertEqual(first, second)
            merged_gate = load_file(
                str(target / "model-00002-of-00002.safetensors")
            )["model.language_model.layers.0.mlp.gate_proj.weight"]
            torch.testing.assert_close(merged_gate, fixture["gate"])

    def test_all_adapter_tensors_must_be_consumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _make_ckpt(root)
            from safetensors.torch import load_file, save_file

            adapter = fixture["ckpt"] / "adapter_vllm"
            tensors = dict(load_file(str(adapter / "adapter_model.safetensors")))
            # an adapter row targeting a tensor absent from the checkpoint
            tensors["base_model.model.model.visual.block.lora_A.weight"] = torch.randn(
                RANK, HIDDEN
            )
            tensors["base_model.model.model.visual.block.lora_B.weight"] = torch.randn(
                4, RANK
            )
            save_file(tensors, str(adapter / "adapter_model.safetensors"))
            with self.assertRaisesRegex(ValueError, "absent from the checkpoint"):
                convert_jev_vl(str(fixture["ckpt"]), str(root / "target"))


if __name__ == "__main__":
    unittest.main()
