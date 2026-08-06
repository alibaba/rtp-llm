import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv4.moe import topk_dump


class MegaMoeTopkDumpTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.prefill_dir = self.root / "prefill"
        self.decode_dir = self.root / "decode"
        self.query_file = self.root / "current_query"
        self.env = mock.patch.dict(
            os.environ,
            {
                "DSV4_MEGA_MOE_TOPK_DUMP_PREFILL_DIR": str(self.prefill_dir),
                "DSV4_MEGA_MOE_TOPK_DUMP_DECODE_DIR": str(self.decode_dir),
                "DSV4_MEGA_MOE_TOPK_DUMP_QUERY_FILE": str(self.query_file),
                "DSV4_MEGA_MOE_TOPK_DUMP_STEPS": "3",
                "DSV4_MEGA_MOE_TOPK_DUMP_PREFILL_STEPS": "1",
                "RANK": "3",
            },
        )
        self.env.start()
        self.strategy = SimpleNamespace(cfg=SimpleNamespace(layer_id=7))
        self.topk = torch.tensor([[11, 12], [21, 22]], dtype=torch.int32)

    def tearDown(self) -> None:
        self.env.stop()
        self.temp_dir.cleanup()

    def _dump(self, *, decode: bool = True, fake: bool = False) -> None:
        with topk_dump.forward_context(
            is_decode_role=decode,
            is_fake_stream=fake,
            model_name="DeepSeekV4Model",
        ):
            topk_dump.maybe_dump(self.strategy, self.topk, token_count=1)

    def test_requires_active_query_and_real_stream(self) -> None:
        self._dump()
        self.query_file.write_text("trace-a")
        self._dump(fake=True)
        self.assertEqual(list(self.root.rglob("*.pt")), [])

    def test_separates_role_query_step_and_layer(self) -> None:
        self.query_file.write_text("trace-a")
        self._dump()
        self._dump()
        self._dump()
        self._dump()
        self._dump(decode=False)
        self._dump(decode=False)

        self.query_file.write_text("trace-b")
        self._dump()

        paths = sorted(self.root.rglob("*.pt"))
        relative_paths = [path.relative_to(self.root).as_posix() for path in paths]
        self.assertEqual(len(paths), 5)
        self.assertIn(
            "decode/query_trace-a/step_000/DeepSeekV4Model/layer_007/rank_003.pt",
            relative_paths,
        )
        self.assertIn(
            "decode/query_trace-a/step_001/DeepSeekV4Model/layer_007/rank_003.pt",
            relative_paths,
        )
        self.assertIn(
            "decode/query_trace-a/step_002/DeepSeekV4Model/layer_007/rank_003.pt",
            relative_paths,
        )
        self.assertIn(
            "decode/query_trace-b/step_000/DeepSeekV4Model/layer_007/rank_003.pt",
            relative_paths,
        )
        self.assertIn(
            "prefill/query_trace-a/step_000/DeepSeekV4Model/layer_007/all_tokens.pt",
            relative_paths,
        )
        for path in paths:
            torch.testing.assert_close(
                torch.load(path, weights_only=True), self.topk[:1]
            )

    def test_prefill_cp_gathers_all_tokens_and_only_rank_zero_writes(self) -> None:
        self.query_file.write_text("trace-c")
        cp_ctx = SimpleNamespace(cp_size=2, cp_rank=0, chunk_length=2)
        gathered = torch.tensor([[11, 12], [31, 32], [21, 22]], dtype=torch.int32)

        with mock.patch(
            "rtp_llm.models_py.modules.dsv4.cp.cp_all_gather_full",
            return_value=gathered,
        ) as gather:
            with topk_dump.forward_context(
                is_decode_role=False,
                is_fake_stream=False,
                model_name="DeepSeekV4Model",
            ), topk_dump.cp_context(cp_ctx):
                topk_dump.maybe_dump(self.strategy, self.topk, token_count=2)

        gather.assert_called_once()
        output = (
            self.prefill_dir
            / "query_trace-c/step_000/DeepSeekV4Model/layer_007/all_tokens.pt"
        )
        torch.testing.assert_close(torch.load(output, weights_only=True), gathered)

        self.query_file.write_text("trace-d")
        other_strategy = SimpleNamespace(cfg=SimpleNamespace(layer_id=7))
        cp_ctx.cp_rank = 1
        with mock.patch(
            "rtp_llm.models_py.modules.dsv4.cp.cp_all_gather_full",
            return_value=gathered,
        ) as gather:
            with topk_dump.forward_context(
                is_decode_role=False,
                is_fake_stream=False,
                model_name="DeepSeekV4Model",
            ), topk_dump.cp_context(cp_ctx):
                topk_dump.maybe_dump(other_strategy, self.topk, token_count=2)

        gather.assert_called_once()
        self.assertFalse((self.prefill_dir / "query_trace-d").exists())

    def test_prefill_cp_accumulates_kernel_chunks_before_gather(self) -> None:
        self.query_file.write_text("trace-chunked")
        cp_ctx = SimpleNamespace(cp_size=2, cp_rank=0, chunk_length=4)
        second_chunk = self.topk.add(20)
        gathered = torch.cat((self.topk, second_chunk), dim=0)

        with mock.patch(
            "rtp_llm.models_py.modules.dsv4.cp.cp_all_gather_full",
            return_value=gathered,
        ) as gather:
            with topk_dump.forward_context(
                is_decode_role=False,
                is_fake_stream=False,
                model_name="DeepSeekV4Model",
            ), topk_dump.cp_context(cp_ctx):
                topk_dump.maybe_dump(self.strategy, self.topk, token_count=2)
                gather.assert_not_called()
                topk_dump.maybe_dump(self.strategy, second_chunk, token_count=2)

        local_arg = gather.call_args.args[0]
        torch.testing.assert_close(local_arg, gathered)
        output = (
            self.prefill_dir
            / "query_trace-chunked/step_000/DeepSeekV4Model/layer_007/all_tokens.pt"
        )
        torch.testing.assert_close(torch.load(output, weights_only=True), gathered)


if __name__ == "__main__":
    unittest.main()
