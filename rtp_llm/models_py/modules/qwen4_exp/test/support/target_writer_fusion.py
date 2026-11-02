from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_compressor import (
    restore_indexer_cache,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_decode_triton import (
    write_target_window_with_undo_,
)

MODULE = "rtp_llm.models_py.modules.qwen4_exp.qsa.target_writer_triton"


class TargetWriterFixture:
    def case(self, prefixes, query, strided=False):
        batch = len(prefixes)
        columns = max(prefixes) // 128 + 2
        device = "cuda"
        raw = torch.randn(
            batch * query, 640 if strided else 128, device=device, dtype=torch.bfloat16
        )[:, -128:]
        p = torch.tensor(prefixes, device=device, dtype=torch.int32)
        cos = torch.randn(batch * query, 64, device=device, dtype=torch.bfloat16)
        sin = torch.randn_like(cos)
        gamma = torch.randn(128, device=device, dtype=torch.bfloat16) * 0.1
        table = torch.zeros(batch, columns, device=device, dtype=torch.int32)
        for b, prefix in enumerate(prefixes):
            table[b, prefix // 128 : prefix // 128 + 2] = torch.tensor(
                [1 + 2 * b, 2 + 2 * b], device=device, dtype=torch.int32
            )
        kv = torch.randn(1 + 2 * batch, 32, 128, device=device, dtype=torch.bfloat16)
        state = torch.randn(1 + 2 * batch, 8, 128, device=device).bfloat16().float()
        return [raw, p, cos, sin, gamma, kv, table, state, table.clone()]

    def compare(self, args):
        original_kv = args[5].clone()
        original_state = args[7].clone()
        expected = self.write(args, False)
        reference_kv = args[5].clone()
        reference_state = args[7].clone()
        args[5].copy_(original_kv)
        args[7].copy_(original_state)
        actual = self.write(args, True)
        self.assertTrue(torch.equal(args[5], reference_kv))
        self.assertTrue(torch.equal(args[7], reference_state))
        for name in ("state_slots", "original_state", "kv_slots", "original_kv"):
            self.assertTrue(
                torch.equal(getattr(actual, name), getattr(expected, name)), name
            )
        restore_indexer_cache(actual)
        self.assertTrue(torch.equal(args[5], original_kv))
        self.assertTrue(torch.equal(args[7], original_state))

    def write(self, args, fused):
        if fused:
            return write_target_window_with_undo_(*args, norm_eps=1.0e-6)
        with patch(MODULE + ".try_write_target_window", return_value=None):
            return write_target_window_with_undo_(*args, norm_eps=1.0e-6)
