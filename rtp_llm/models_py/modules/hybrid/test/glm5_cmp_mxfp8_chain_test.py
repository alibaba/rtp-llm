import os
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main

import torch

from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp
from rtp_llm.models_py.modules.hybrid.glm5_cmp import Glm5Cmp, should_enable_glm5_cmp
from rtp_llm.ops.compute_ops import LayerKVCache
from rtp_llm.utils.model_weight import W


class Glm5CmpMxfp8ChainTest(TestCase):
    def setUp(self) -> None:
        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(torch.cuda.get_device_capability()[0], 10)
        torch.manual_seed(5342)

    def _fixture(self, width: int):
        import deep_gemm

        rows, pages = 16, 64
        requests = rows // width

        def rand(shape, scale=1.0, dtype=torch.bfloat16):
            return (torch.randn(shape, device="cuda") * scale).to(dtype)

        def linear(n, k):
            scales = torch.full(
                (k // 512, n), 0x7F7F7F7F, device="cuda", dtype=torch.int32
            ).T
            return SimpleNamespace(
                weight=rand((n, k), 0.01, torch.float8_e4m3fn), weight_scales=scales
            )

        norm = lambda n: SimpleNamespace(
            weight=torch.ones(n, device="cuda", dtype=torch.bfloat16),
            variance_epsilon=1e-5,
        )
        op = IndexerOp(32, 128, 2048, 64)
        indexer = SimpleNamespace(
            indexer_op=op,
            wk=linear(128, 6144),
            wq_b=linear(4096, 2048),
            k_norm=norm(128),
            weights_proj=SimpleNamespace(weight=rand((32, 6144), 0.01, torch.float32)),
        )
        indexer.k_norm.beta = torch.zeros(128, device="cuda", dtype=torch.bfloat16)
        attention = SimpleNamespace(
            has_indexer=True,
            indexer=indexer,
            fused_qkv_a_proj=linear(2624, 6144),
            q_b_proj=linear(16384, 2048),
            o_proj=linear(6144, 16384),
            q_a_layernorm=norm(2048),
            kv_a_layernorm=norm(512),
        )
        cmp = Glm5Cmp(
            layer_idx=0,
            config=SimpleNamespace(
                model_type="glm_5",
                moe_layer_index=(0,),
                attn_config=SimpleNamespace(use_mla=True, kernel_tokens_per_block=64),
            ),
            parallelism_config=SimpleNamespace(tp_size=1, get_attn_tp_size=lambda: 1),
            self_attn=attention,
            input_layernorm=norm(6144),
            mlp=SimpleNamespace(),
            post_attention_layernorm=norm(6144),
        )
        cache = LayerKVCache()
        blocks = requests * pages
        cache.kv_cache_base = torch.zeros(
            blocks, 64, 656, device="cuda", dtype=torch.uint8
        )
        cache.kv_scale_base = torch.zeros(
            blocks, 64, 132, device="cuda", dtype=torch.uint8
        )
        slots = torch.arange(blocks * 64, device="cuda", dtype=torch.int64)
        op.quant_k_only(rand((blocks * 64, 128)), cache, slots)
        table = (
            torch.arange(blocks, device="cuda", dtype=torch.int32)
            .roll(3)
            .view(requests, pages)
        )
        lengths = torch.arange(rows, device="cuda", dtype=torch.int32) + 3073
        positions = lengths - 1
        request_ids = torch.arange(rows, device="cuda") // width
        write_slots = (
            table[request_ids, positions.long() // 64].long() * 64 + positions % 64
        )
        params = SimpleNamespace(
            positions_d=positions,
            slot_mapping=write_slots,
            expanded_seq_lens=lengths,
            kvlen_d=lengths.view(requests, width)[:, -1].contiguous(),
        )
        indices = torch.arange(rows, device="cuda", dtype=torch.int32)
        params.schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
            lengths.view(-1, 1), 64, deep_gemm.get_num_sms(), indices=indices
        )
        inputs = SimpleNamespace(
            is_prefill=False,
            is_target_verify=width > 1,
            is_draft_extend=False,
            kv_cache_block_id_device=table,
            kv_cache_kernel_block_id_device=table,
            decode_cu_seqlens_d=torch.arange(
                0, rows + 1, width, device="cuda", dtype=torch.int32
            ),
        )
        angles = (
            torch.arange(8192, device="cuda")[:, None]
            * torch.exp(-torch.arange(32, device="cuda").float() / 8)[None, :]
        )
        implementation = SimpleNamespace(
            fmha_params=params,
            attn_inputs=inputs,
            weights=[{W.mla_kc: rand((64, 192, 512), 0.01)}],
            _cos_sin_cache=torch.cat((angles.cos(), angles.sin()), -1),
        )
        hidden, residual = rand((rows, 6144)), rand((rows, 6144))
        cache_manager = SimpleNamespace(get_layer_cache=lambda _: cache)
        self.assertTrue(
            should_enable_glm5_cmp(
                [SimpleNamespace(cmp=cmp)], 1, hidden, implementation, cache_manager
            )
        )
        self.assertIsNotNone(cmp._mxfp8_ops)
        return cmp, implementation, cache, hidden, residual

    def _run_case(self, width: int) -> None:
        import deep_gemm

        cmp, impl, cache, hidden, residual = self._fixture(width)
        op = cmp.self_attn.indexer.indexer_op
        topk = op._paged_topk_op
        observed = []
        score = op._get_topk_paged
        score_inputs = []

        def observe_score(*args, **kwargs):
            score_inputs[:] = [args[0], args[1], kwargs["q_scale"]]
            return score(*args, **kwargs)

        op._get_topk_paged = observe_score

        def consume(logits, *args):
            self.assertEqual(logits.dtype, torch.bfloat16)
            observed[:] = [logits]
            return topk(logits, *args)

        op._paged_topk_op = consume

        def forward():
            return cmp.mla_prologue(hidden, residual, impl, cache, None)

        for _ in range(2):
            forward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = forward()
        captured_logits, captured_inputs = observed[0], tuple(score_inputs)
        clone = cmp.clone_for_cuda_graph(mlp=cmp.mlp)
        self.assertIs(clone._mxfp8_ops, cmp._mxfp8_ops)
        for step in (1, 2):
            hidden.normal_()
            residual.normal_()
            impl.fmha_params.expanded_seq_lens.add_(step)
            impl.fmha_params.positions_d.add_(step)
            impl.attn_inputs.kv_cache_block_id_device.copy_(
                impl.attn_inputs.kv_cache_block_id_device.roll(1, 1)
            )
            positions = impl.fmha_params.positions_d
            request_ids = torch.arange(hidden.size(0), device="cuda") // width
            table = impl.attn_inputs.kv_cache_block_id_device
            impl.fmha_params.slot_mapping.copy_(
                table[request_ids, positions.long() // 64].long() * 64 + positions % 64
            )
            impl.fmha_params.kvlen_d.copy_(
                impl.fmha_params.expanded_seq_lens.view(-1, width)[:, -1]
            )
            metadata = deep_gemm.get_paged_mqa_logits_metadata(
                impl.fmha_params.expanded_seq_lens.view(-1, 1),
                64,
                deep_gemm.get_num_sms(),
                indices=torch.arange(hidden.size(0), device="cuda", dtype=torch.int32),
            )
            impl.fmha_params.schedule_metadata.copy_(metadata)
            graph.replay()
            actual = tuple(value.clone() for value in captured)
            graph_cache = cache.kv_scale_base.clone()
            graph_logits = captured_logits.clone()
            graph_score_inputs = tuple(value.clone() for value in captured_inputs)
            expected = forward()
            for a, b in zip(actual[:2], expected[:2]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(graph_cache, cache.kv_scale_base, rtol=0, atol=0)
            for graph_value, eager_value in zip(graph_score_inputs, score_inputs):
                torch.testing.assert_close(
                    graph_value.view(torch.uint8),
                    eager_value.view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
            lengths = impl.fmha_params.expanded_seq_lens
            valid = (
                torch.arange(graph_logits.size(1), device="cuda")[None, :]
                < lengths[:, None]
            )
            torch.testing.assert_close(
                graph_logits[valid], observed[0][valid], rtol=0, atol=0
            )
            graph_selected = (
                graph_logits.float().gather(1, actual[2].long()).sort(-1).values
            )
            eager_selected = (
                observed[0].float().gather(1, expected[2].long()).sort(-1).values
            )
            torch.testing.assert_close(graph_selected, eager_selected, rtol=0, atol=0)
            print(
                f"Graph/eager Q, SF, gates, valid logits and selected values identical: width={width}, step={step}"
            )
            for indices in (actual[2], expected[2]):
                self.assertTrue(
                    bool(((indices >= 0) & (indices < lengths[:, None])).all())
                )
                self.assertTrue(bool((indices.sort(-1).values.diff(dim=-1) > 0).all()))
            logits = observed[0].float()
            lengths = impl.fmha_params.expanded_seq_lens
            logits.masked_fill_(
                torch.arange(logits.size(1), device="cuda")[None, :]
                >= lengths[:, None],
                -torch.inf,
            )
            selected = logits.gather(1, expected[2].long()).sort(-1).values
            torch.testing.assert_close(
                selected, logits.topk(2048, -1).values.sort(-1).values, rtol=0, atol=0
            )
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            forward()
            torch.cuda.synchronize()
        output = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp"))
        profile.export_chrome_trace(str(output / f"cmp_mxfp8_width{width}.json"))
        events = profile.events()
        self.assertFalse(any(event.name == "aten::_to_copy" for event in events))
        print(
            f"CMP real three-stream prologue, BF16 MQA→TopK, graph changed-input replay PASS: width={width}"
        )

    def test_decode_chain(self) -> None:
        self._run_case(1)

    def test_mtp_chain(self) -> None:
        self._run_case(4)


if __name__ == "__main__":
    main()
