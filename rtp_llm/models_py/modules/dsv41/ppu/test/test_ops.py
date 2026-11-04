"""Contract-level tests: PPUOps assembles the eight frozen V41BackendOps."""

import unittest
from unittest import mock

import torch

from rtp_llm.config.v41_runtime import PPU_STRATEGY
from rtp_llm.models_py.modules.dsv41.backend import (
    OP_NAMES,
    V41Backend,
    V41MHCResult,
    V41QuantizedTensor,
)
from rtp_llm.models_py.modules.dsv41.ppu.engram import (
    EngramShardLayout,
    HostShard,
    LocalEnpmGroup,
    dequant_rows,
)
from rtp_llm.models_py.modules.dsv41.ppu.linear import dequant_fp8_block32_ue8m0
from rtp_llm.models_py.modules.dsv41.ppu.mhc import mhc_head, mhc_post, mhc_pre
from rtp_llm.models_py.modules.dsv41.ppu.moe import (
    expert_forward_reference,
)
from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps


class _FakeModel:
    pass


class _FakeWeight:
    pass


def _make_expert_weights(e, i, k, seed, lo=124, hi=132):
    g = torch.Generator().manual_seed(seed)
    w13_w = torch.randint(0, 256, (e, 2 * i, k // 2), generator=g).to(torch.uint8)
    w13_s = torch.randint(lo, hi, (e, 2 * i, k // 32), generator=g).to(torch.uint8)
    w2_w = torch.randint(0, 256, (e, k, i // 2), generator=g).to(torch.uint8)
    w2_s = torch.randint(lo, hi, (e, k, i // 32), generator=g).to(torch.uint8)
    return (w13_w, w13_s), (w2_w, w2_s)


def _make_routing(t, num_experts, top_k, seed, slots=8):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, num_experts, (t, top_k), generator=g)
    weights = torch.rand((t, top_k), generator=g) + 0.1
    weights = weights / weights.sum(-1, keepdim=True) * 1.5
    pad_ids = torch.full((t, slots - top_k), -1, dtype=torch.int64)
    pad_w = torch.zeros((t, slots - top_k))
    return torch.cat([ids, pad_ids], dim=1), torch.cat([weights, pad_w], dim=1)


def _small_ops():
    return PPUOps(
        ep_rank=0,
        ep_size=1,
        num_routed_experts=8,
        num_local_experts=8,
        hidden_size=64,
        inter_dim=32,
    )


class OpsContractTest(unittest.TestCase):
    def test_ops_satisfy_backend_validation(self):
        backend = V41Backend(
            "ppu", PPU_STRATEGY, _small_ops(), _FakeModel, _FakeWeight, lambda r: None
        )
        backend.validate()
        self.assertEqual(backend.name, "ppu")

    def test_all_eight_ops_are_callable(self):
        ops = _small_ops()
        for name in OP_NAMES:
            self.assertTrue(callable(getattr(ops, name)), name)


class LinearOpTest(unittest.TestCase):
    def test_bf16_weight_uses_plain_linear(self):
        ops = _small_ops()
        weight = V41QuantizedTensor(
            torch.randn(16, 64).to(torch.bfloat16), None, "bf16"
        )
        x = torch.randn(4, 64).to(torch.bfloat16)
        out = ops.linear(x, weight)
        self.assertEqual(tuple(out.shape), (4, 16))
        expected = torch.nn.functional.linear(x, weight.values)
        self.assertTrue(torch.equal(out, expected))

    def test_fp8_block32_dequant_matches_oracle(self):
        ops = _small_ops()
        g = torch.Generator().manual_seed(3)
        raw = torch.randint(-120, 120, (32, 64), generator=g).to(torch.float8_e4m3fn)
        scale = torch.randint(120, 130, (1, 2), generator=g).to(torch.uint8)
        weight = V41QuantizedTensor(raw, scale, "fp8_32x32_ue8m0")
        x = torch.randn(3, 64)
        out = ops.linear(x, weight)
        dequant = dequant_fp8_block32_ue8m0(raw, scale)
        expected = torch.nn.functional.linear(x.to(dequant.dtype), dequant)
        self.assertTrue(torch.equal(out, expected))

    def test_fp4_weight_is_rejected(self):
        ops = _small_ops()
        weight = V41QuantizedTensor(
            torch.zeros(32, 32, dtype=torch.uint8), None, "fp4_ue8m0"
        )
        with self.assertRaises(ValueError):
            ops.linear(torch.zeros(1, 32), weight)


class MoeOpTest(unittest.TestCase):
    def _bind(self, ops, seed=11):
        (w13_w, w13_s), (w2_w, w2_s) = _make_expert_weights(8, 32, 64, seed)
        ops.bind_expert_weights(0, (w13_w, w13_s), (w2_w, w2_s))
        return (w13_w, w13_s), (w2_w, w2_s)

    def test_local_path_matches_reference(self):
        ops = _small_ops()
        (w13_w, w13_s), (w2_w, w2_s) = self._bind(ops)
        ids, weights = _make_routing(6, 8, 4, 5)
        x = torch.randn(6, 64).to(torch.bfloat16)
        out = ops.moe(x, ids, weights, layer_id=0, context=None)
        manual = torch.zeros(6, 64, dtype=torch.float32)
        for t in range(6):
            for slot in range(ids.shape[1]):
                expert = int(ids[t, slot])
                if expert < 0:
                    continue
                expert_out = expert_forward_reference(
                    x[t : t + 1],
                    w13_w[expert],
                    w13_s[expert],
                    w2_w[expert],
                    w2_s[expert],
                )
                manual[t] += expert_out[0].float() * float(weights[t, slot])
        torch.testing.assert_close(
            out.float(), manual.to(out.dtype).float(), rtol=0, atol=0
        )

    def test_empty_tokens_return_empty(self):
        ops = _small_ops()
        self._bind(ops)
        ids, weights = _make_routing(0, 8, 4, 5)
        x = torch.zeros(0, 64, dtype=torch.bfloat16)
        out = ops.moe(x, ids, weights, layer_id=0, context=None)
        self.assertEqual(tuple(out.shape), (0, 64))

    def test_unbound_layer_raises(self):
        ops = _small_ops()
        ids, weights = _make_routing(2, 8, 4, 5)
        with self.assertRaises(RuntimeError):
            ops.moe(torch.zeros(2, 64), ids, weights, layer_id=7, context=None)


class EngramOpTest(unittest.TestCase):
    def _table(self, rows, dim, seed=7):
        g = torch.Generator().manual_seed(seed)
        payload = torch.randint(-120, 120, (rows, dim), generator=g).to(
            torch.float8_e4m3fn
        )
        scale = torch.randint(120, 130, (rows, dim // 32), generator=g).to(torch.uint8)
        return payload, scale

    def test_lookup_returns_owned_dequantized_rows(self):
        rows, dim = 600, 64
        payload, scale = self._table(rows, dim)
        layout = EngramShardLayout.from_table(rows, enp_size=8)
        shards = [HostShard.from_tensors(payload, scale, layout, r) for r in range(8)]
        group = LocalEnpmGroup(shards, layout)
        ops = _small_ops()
        ops.bind_engram_shard(1, shards[2], layout, group=group)

        g = torch.Generator().manual_seed(9)
        query = torch.randint(0, rows, (12,), generator=g)
        query[3] = -1
        query[7] = rows + 500  # global-invalid
        out = ops.engram_lookup(query, layer_id=1, context=None)
        self.assertEqual(tuple(out.shape), (12, dim))
        expected = dequant_rows(payload, scale)
        for i, row in enumerate(query.tolist()):
            if row < 0 or row >= rows:
                self.assertTrue(torch.equal(out[i], torch.zeros(dim)))
            else:
                self.assertTrue(torch.equal(out[i], expected[row]))

    def test_unbound_layer_raises(self):
        ops = _small_ops()
        with self.assertRaises(RuntimeError):
            ops.engram_lookup(torch.tensor([0]), layer_id=3, context=None)


class MhcOpTest(unittest.TestCase):
    def _parameters(self, seed=13):
        g = torch.Generator().manual_seed(seed)
        fn = torch.randn(24, 20480, generator=g)
        base = torch.randn(24, generator=g)
        scale = torch.randn(3, generator=g)
        return {"fn": fn, "base": base, "scale": scale}

    def test_pre_post_head_chain_matches_direct_calls(self):
        ops = _small_ops()
        parameters = self._parameters()
        residual = torch.randn(5, 4, 5120)
        branch = torch.randn(5, 5120)

        pre = ops.mhc(residual, None, phase="pre", parameters=parameters, context=None)
        self.assertIsInstance(pre, V41MHCResult)
        direct_pre = mhc_pre(residual, parameters)
        self.assertTrue(torch.equal(pre.hidden, direct_pre.hidden))
        self.assertTrue(torch.equal(pre.post_mix, direct_pre.post_mix))

        post = ops.mhc(
            residual,
            branch,
            phase="post",
            parameters=parameters,
            context=None,
            state=pre,
        )
        direct_post = mhc_post(residual, branch, state=direct_pre)
        self.assertTrue(torch.equal(post.hidden, direct_post.hidden))

        head = ops.mhc(
            residual, None, phase="head", parameters=parameters, context=None, state=pre
        )
        direct_head = mhc_head(residual, state=direct_pre)
        self.assertTrue(torch.equal(head.hidden, direct_head.hidden))

    def test_invalid_phase_raises(self):
        ops = _small_ops()
        with self.assertRaises(ValueError):
            ops.mhc(
                torch.zeros(1, 4, 5120),
                None,
                phase="mix",
                parameters=self._parameters(),
            )


class KvCodecOpTest(unittest.TestCase):
    def test_global_roundtrip_delegates_to_kv_codec(self):
        from rtp_llm.models_py.modules.dsv41.ppu import kv_codec

        ops = _small_ops()
        kv = torch.randn(4, 512).to(torch.bfloat16)
        packed = ops.encode_kv(kv, pool_tag="global_kv_2", descriptor=None)
        direct_packed = kv_codec.encode_kv(kv, pool_tag="global_kv_2", descriptor=None)
        self.assertTrue(torch.equal(packed, direct_packed))
        self.assertEqual(tuple(packed.shape), (4, 288))
        out = torch.zeros(4, 512, dtype=torch.bfloat16)
        returned = ops.decode_kv(
            packed, pool_tag="global_kv_2", descriptor=None, out=out
        )
        self.assertIs(returned, out)
        direct_out = torch.zeros(4, 512, dtype=torch.bfloat16)
        kv_codec.decode_kv(
            packed, pool_tag="global_kv_2", descriptor=None, out=direct_out
        )
        self.assertTrue(torch.equal(out, direct_out))
        # FP4 quantization noise bound: group-16 E4M3 scales keep the
        # roundtrip within half a top-grid step per group.
        torch.testing.assert_close(out.float(), kv.float(), rtol=0.3, atol=0.6)

    def test_unknown_pool_tag_raises(self):
        ops = _small_ops()
        with self.assertRaises(Exception):
            ops.encode_kv(torch.zeros(2, 512), pool_tag="nope", descriptor=None)


class IndexerOpTest(unittest.TestCase):
    def test_delegates_to_indexer_module(self):
        ops = _small_ops()
        q = V41QuantizedTensor(torch.randn(3, 2, 128), None, "bf16")
        k = V41QuantizedTensor(torch.randn(10, 128), None, "bf16")
        head_weights = torch.rand(3, 2) + 0.5
        key_positions = torch.arange(10)
        candidate_blocks = torch.tensor([[0, 1, -1]] * 3, dtype=torch.int32)
        out = ops.index_score(
            q,
            k,
            head_weights,
            layer_id=2,
            candidate_blocks=candidate_blocks,
            key_positions=key_positions,
            context=None,
        )
        from rtp_llm.models_py.modules.dsv41.ppu.indexer import index_score

        direct = index_score(
            q,
            k,
            head_weights,
            layer_id=2,
            candidate_blocks=candidate_blocks,
            key_positions=key_positions,
            context=None,
        )
        self.assertTrue(torch.equal(out, direct))


class SparseAttentionOpTest(unittest.TestCase):
    def test_delegates_with_default_window(self):
        ops = _small_ops()
        q = torch.zeros(2, 64, 512, dtype=torch.bfloat16)
        blocks = torch.zeros(2, 4, dtype=torch.int32)
        cache = object()
        sentinel = object()
        with mock.patch(
            "rtp_llm.models_py.modules.dsv41.ppu.ops._sparse_attention",
            return_value=sentinel,
        ) as patched:
            result = ops.sparse_attention(
                q, blocks, cache=cache, layer_id=2, context=None
            )
        self.assertIs(result, sentinel)
        patched.assert_called_once()
        args, kwargs = patched.call_args
        self.assertIs(args[0], q)
        self.assertIs(args[1], blocks)
        self.assertIs(kwargs["cache"], cache)
        self.assertEqual(kwargs["layer_id"], 2)
        self.assertEqual(kwargs["window"], 128)


if __name__ == "__main__":
    unittest.main()
