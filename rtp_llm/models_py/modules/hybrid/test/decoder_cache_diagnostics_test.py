import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.model_desc.decoder_cache import decoder_cache_forward


class DecoderFailure(RuntimeError):
    pass


class DecoderCacheDiagnosticsTest(unittest.TestCase):
    def test_real_diagnostics_outputs_follow_prefill_lifecycle(self):
        root = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        for name, enabled, is_prefill, failed in (
            ("success", "1", True, False),
            ("failure", "1", True, True),
            ("disabled", "0", True, False),
            ("decode", "1", False, False),
        ):
            with self.subTest(name=name):
                output_root = root / name
                inputs = SimpleNamespace(
                    is_prefill=is_prefill,
                    input_lengths=torch.tensor([82, 13], dtype=torch.int32),
                    prefix_lengths=torch.tensor([4096, 64], dtype=torch.int32),
                )
                fmha = SimpleNamespace(
                    attn_inputs=inputs,
                    fmha_impl=SimpleNamespace(attn_inputs=inputs),
                )
                model = SimpleNamespace(
                    layered_prefill_cache=None,
                    pinned_mla_groups={},
                    parallelism_config=SimpleNamespace(tp_rank=3),
                    prepare_fmha_impl=lambda _: fmha,
                )
                with patch.dict(
                    os.environ,
                    {
                        "RTP_LLM_PREFILL_CACHE_STATS": enabled,
                        "TEST_UNDECLARED_OUTPUTS_DIR": str(output_root),
                    },
                ):
                    if failed:
                        with self.assertRaises(DecoderFailure):
                            with decoder_cache_forward(model, None):
                                raise DecoderFailure(name)
                    else:
                        with decoder_cache_forward(model, None):
                            pass
                    inputs.is_prefill = False
                    with decoder_cache_forward(model, None):
                        pass
                output_file = output_root / "prefill_forwards_rank3.jsonl"
                if enabled == "1" and is_prefill:
                    rows = [
                        json.loads(line) for line in output_file.read_text().splitlines()
                    ]
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0]["failed"], failed)
                    self.assertEqual(rows[0]["batch"], 2)
                    self.assertEqual(rows[0]["exec_tokens"], 95)
                    self.assertEqual(rows[0]["kv_tokens"], 4255)
                    self.assertEqual(rows[0]["sequence_lengths"], [4178, 77])
                    self.assertEqual(rows[0]["rank"], 3)
                else:
                    self.assertFalse(output_file.exists())


if __name__ == "__main__":
    unittest.main()
