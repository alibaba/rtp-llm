"""N5 bounded serial/parallel projections, state bytes and resource reuse."""

import unittest
from types import SimpleNamespace as NS

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv41.ppu.kv_codec import encode_kv
from rtp_llm.models_py.modules.dsv41.ppu.projection_streams import QKVStreams, is_supported


class SharedStreamsDevice(unittest.TestCase):
    @torch.inference_mode()
    def test_serial_parallel_and_refusal(self):
        self.assertTrue(torch.cuda.is_available(), "PPU execution required")
        torch.manual_seed(5106)
        x = torch.randn(1, 512, device="cuda", dtype=torch.bfloat16)
        wq = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        wk = torch.randn_like(wq)
        state = NS(prefill=False, idle=False, is_graph=False, lengths=(1,),
                   context=None, attn=NS(is_target_verify=False),
                   cache=NS(policy=NS(ced=False)),
                   runtime=NS(policy=NS(mode="full", max_rollback_tokens=0),
                              model=NS(_is_speculative=False)))
        self.assertTrue(is_supported(state, x))
        resources = QKVStreams(x.device)
        identities = tuple(id(v) for v in (resources.side_stream, resources.producer, resources.done))

        def query(value):
            projected = F.linear(value, wq)
            return projected, projected.square()

        def key(value):
            projected = F.linear(value, wk)
            v = projected.float()
            return (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + 1e-6)).bfloat16()

        # A second live input reuses the same events and exposes stale data.
        for token in range(2):
            x.add_(token)
            reference_q, reference_k = query(x), key(x)
            actual_q, actual_k = resources.run(x, query, key)
            torch.cuda.synchronize()
            for a, b in zip(actual_q, reference_q):
                self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(actual_k, reference_k))
            self.assertTrue(torch.equal(
                encode_kv(actual_k, pool_tag="swa_kv"),
                encode_kv(reference_k, pool_tag="swa_kv"),
            ))
            self.assertEqual(identities, tuple(id(v) for v in (
                resources.side_stream, resources.producer, resources.done
            )))
        self.assertEqual(resources.calls, 2)
        for field in ("prefill", "idle", "is_graph", "verify", "compacted", "restoring_aux"):
            setattr(state, field, True)
            self.assertFalse(is_supported(state, x), field)
            setattr(state, field, False)
        state.attn.is_target_verify = True
        self.assertFalse(is_supported(state, x))
        state.attn.is_target_verify = False
        with torch.cuda.stream(torch.cuda.Stream()):
            self.assertFalse(is_supported(state, x))


if __name__ == "__main__":
    unittest.main()
