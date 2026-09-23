"""Real BF16 MLA prefix-cache and default-policy regression checks."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.kimi_k3.mla_prefill import KimiK3MlaPrefillImpl
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import MlaFlashInferPrefillImpl
from rtp_llm.ops import KvCacheDataType
from rtp_llm.ops.compute_ops import PyAttentionInputs, LayerKVCache
from rtp_llm.utils.model_weight import W


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class KimiK3MlaPrefillTest(unittest.TestCase):
    def test_cached_suffix_and_legacy_defaults(self):
        torch.manual_seed(4921)
        heads, rank, nope, suffix, value, page = 16, 512, 128, 64, 128, 128
        prefix, tail = 128, 7
        tokens = prefix + tail
        attention = SimpleNamespace(
            head_num=heads, kv_lora_rank=rank, rope_head_dim=suffix,
            nope_head_dim=nope, v_head_dim=value, kernel_tokens_per_block=page,
            softmax_extra_scale=1., use_mla=True, is_sparse=False, kv_cache_dtype=KvCacheDataType.BASE,
        )
        config = SimpleNamespace(
            getAttentionConfigs=lambda tp: attention, quant_config=None,
            max_seq_len=256, headwise_config=None,
        )
        parallel = SimpleNamespace(get_attn_tp_size=lambda: 1)
        weights = SimpleNamespace(weights=[{
            W.mla_kv_b_w: (torch.randn(rank, heads*(nope+value), device='cuda')*.025).bfloat16(),
        }])
        q = (torch.randn(tokens, heads, nope+suffix, device='cuda')*.125).bfloat16()
        latent = (torch.randn(tokens, rank, device='cuda')*.125).bfloat16()
        k_suffix = (torch.randn(tokens, suffix, device='cuda')*.125).bfloat16()
        cache = LayerKVCache()
        cache.kv_cache_base = torch.zeros(3, page, rank+suffix, dtype=torch.bfloat16, device='cuda')

        def inputs(count, reused):
            result = PyAttentionInputs()
            result.is_prefill = True
            result.input_lengths = torch.tensor([count], dtype=torch.int32)
            result.prefix_lengths = torch.tensor([reused], dtype=torch.int32)
            result.sequence_lengths = torch.empty(0, dtype=torch.int32)
            table = torch.tensor([[1, 2]], dtype=torch.int32)
            result.kv_cache_block_id = table
            result.kv_cache_block_id_device = table.cuda()
            result.kv_cache_kernel_block_id = table
            result.kv_cache_kernel_block_id_device = table.cuda()
            return result

        full_inputs = inputs(tokens, 0)
        full = KimiK3MlaPrefillImpl(config, parallel, weights, full_inputs, None, False)
        self.assertIsNone(full.absorb_fmha)
        captures = {}
        def capture_forward(name, impl, q_arg, latent_arg, suffix_arg):
            original = impl.fmha_impl.prefill_wrapper.run
            def run(q, k, v, *args, **kwargs):
                captures[name] = (q.clone(), k.clone(), v.clone())
                return original(q, k, v, *args, **kwargs)
            with patch.object(impl.fmha_impl.prefill_wrapper, 'run', side_effect=run):
                return impl.forward(q_arg, latent_arg, suffix_arg, cache, 0)
        expected = capture_forward('full', full, q, latent, k_suffix)
        self.assertTrue(torch.equal(cache.kv_cache_base[1, :, :rank], latent[:prefix]))
        self.assertTrue(torch.equal(cache.kv_cache_base[1, :, rank:], k_suffix[:prefix]))
        prefix_snapshot = cache.kv_cache_base[1].clone()
        cache.kv_cache_base[2].fill_(float('nan'))
        reuse_inputs = inputs(tail, prefix)
        reused = KimiK3MlaPrefillImpl(config, parallel, weights, reuse_inputs, None, False)
        self.assertIsNone(reused.absorb_fmha)
        with patch.object(reused.fmha_impl.prefill_wrapper, 'plan', wraps=reused.fmha_impl.prefill_wrapper.plan) as plan:
            reused.prepare(reuse_inputs)
            self.assertIs(plan.call_args.kwargs['q_data_type'], torch.bfloat16)
        actual = capture_forward('reuse', reused, q[prefix:], latent[prefix:], k_suffix[prefix:])
        full_q, full_k, full_v = captures['full']
        reuse_q, reuse_k, reuse_v = captures['reuse']
        logits = torch.einsum('qhd,khd->hqk', reuse_q.double(), reuse_k.double()) / (nope+suffix)**.5
        mask = torch.arange(tokens, device='cuda')[None,:] <= (prefix+torch.arange(tail, device='cuda'))[:,None]
        probs = logits.masked_fill(~mask[None,:,:], float('-inf')).softmax(-1)
        oracle = torch.einsum('hqk,khd->qhd', probs, reuse_v.double())
        def metrics(x, y):
            delta=x.double()-y.double()
            return {'equal': torch.equal(x,y), 'max_abs': delta.abs().max().item(), 'relative_l2': (delta.norm()/y.double().norm()).item(), 'changed_elements': int((x!=y).sum())}
        import json, os
        report = {'q_equal':torch.equal(full_q[prefix:],reuse_q), 'k_equal':torch.equal(full_k,reuse_k), 'v_equal':torch.equal(full_v,reuse_v),
                  'full_vs_reuse':metrics(actual, expected[prefix:]), 'full_vs_fp64':metrics(expected[prefix:], oracle), 'reuse_vs_fp64':metrics(actual, oracle),
                  'bf16_output_rounding_floor':metrics(oracle.bfloat16(), oracle)}
        print(json.dumps(report,indent=2),flush=True)
        report_path=os.environ.get('K3_MLA_TEST_REPORT')
        if report_path:
            from pathlib import Path
            with Path(report_path).open('x') as out: json.dump(report,out,indent=2)
        self.assertTrue(torch.equal(actual, expected[prefix:]))
        self.assertTrue(torch.equal(cache.kv_cache_base[1], prefix_snapshot))
        self.assertTrue(torch.equal(cache.kv_cache_base[2, :tail, :rank], latent[prefix:]))
        self.assertTrue(torch.equal(cache.kv_cache_base[2, :tail, rank:], k_suffix[prefix:]))
        self.assertEqual(actual.dtype, torch.bfloat16)

        legacy = MlaFlashInferPrefillImpl(attention, reuse_inputs, weights.weights, None)
        self.assertIsNotNone(legacy.absorb_fmha)
        with patch.object(legacy.fmha_impl.prefill_wrapper, 'plan', wraps=legacy.fmha_impl.prefill_wrapper.plan) as plan:
            legacy.prepare(reuse_inputs)
            self.assertNotIn('disable_split_kv', plan.call_args.kwargs)

    def test_ordinary_prefill_does_not_claim_graph_support(self):
        with self.assertRaisesRegex(ValueError, 'eager planning'):
            KimiK3MlaPrefillImpl(None, None, None, None, None, True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
