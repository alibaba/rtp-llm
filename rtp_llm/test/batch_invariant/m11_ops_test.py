"""Bitwise operator coverage for M11's SM103 text path."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_ROPE_POS_TAG,
    INDEXER_STATE_TAG,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_context import (
    Qwen4ExpQSARuntimeContext,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_selection import (
    select_qsa_paged_tokens,
)
from rtp_llm.models_py.modules.base.common.zero_centered_rms_norm import (
    ZeroCenteredFusedQKRMSNorm,
)
from rtp_llm.models_py.modules.hybrid.gated_residual import GatedResidual
from rtp_llm.models_py.modules.qwen4_exp import gdn_prefill_adapter
from rtp_llm.models_py.modules.qwen4_exp.moe.topk import Qwen4ExpMoeTopk
from rtp_llm.models_py.modules.qwen4_exp.ple.ple import Qwen4ExpPLELayer
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import Qwen4ExpQSAIndexer
from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_gqa_impl import SparseGqaFmhaImpl
from rtp_llm.models_py.triton_kernels.moe.mega_moe_input_pack import (
    fused_pack_mega_moe_inputs,
)
from rtp_llm.ops import AttentionConfigs, RopeStyle
from rtp_llm.ops.compute_ops import get_typemeta
from rtp_llm.test.batch_invariant.gdn_test import bit_mismatches


class M11BatchInvariantOpsTest(unittest.TestCase):
    def setUp(self):
        self.assertEqual(torch.cuda.get_device_capability(), (10, 3))
        self.generator = torch.Generator(device="cuda").manual_seed(1215)

    def bf16(self, *shape):
        return torch.randn(
            shape, device="cuda", dtype=torch.bfloat16, generator=self.generator
        )

    def assert_bits(self, expected, actual, label):
        diffs = bit_mismatches(expected, actual)
        print(f"{label} {diffs} unequal elements", flush=True)
        self.assertEqual(diffs, 0, label)

    def test_bi_gdn_keeps_native_backend_with_flashinfer_requested(self):
        prefill = SimpleNamespace(
            linear_attn_config=object(),
            parallelism_config=object(),
            weights=object(),
            batch_invariant=True,
        )
        decode = object()
        attention = SimpleNamespace(prefill_gdn=prefill, decode_gdn=decode)
        with (
            patch.dict(os.environ, {"QWEN4_GDN_BACKEND": "flashinfer"}),
            patch.object(gdn_prefill_adapter, "Qwen4Sm120GdnPrefill") as native,
        ):
            gdn_prefill_adapter.install_qwen4_gdn_backends(attention)
        native.assert_called_once_with(
            prefill.linear_attn_config,
            prefill.parallelism_config,
            prefill.weights,
            batch_invariant=True,
        )
        self.assertIs(attention.prefill_gdn, native.return_value)
        self.assertIs(attention.decode_gdn, decode)

    @torch.inference_mode()
    def test_hc_row_counts(self):
        hc = GatedResidual(
            self.bf16(32768) * 0.01,
            self.bf16(1024, 32768) * 0.01,
            self.bf16(32768, 1024) * 0.01,
            self.bf16(4, 32768) * 0.01,
            hc_mult=4,
            norm_eps=1e-6,
            batch_invariant=True,
        )
        inputs = self.bf16(33, 32768)
        ref, _, ref_gates = hc(inputs[:1])
        for n in (31, 32, 33):
            actual, _, gates = hc(inputs[:n].roll(n - 1, dims=0))
            self.assert_bits(ref, actual[-1:], f"HC mixed {n} rows")
            self.assert_bits(ref_gates, gates[-1:], f"HC inject {n} rows")
        hc.merged_down_inject = None
        split, _, split_gates = hc(inputs[:33])
        self.assert_bits(ref, split[:1], "HC split projection")
        self.assert_bits(ref_gates, split_gates[:1], "HC split injection")

    @torch.inference_mode()
    def test_moe_routing_and_input_quantization(self):
        router = Qwen4ExpMoeTopk(
            SimpleNamespace(
                config=SimpleNamespace(expert_num=768, moe_k=10, has_moe_norm=True)
            ),
            batch_invariant=True,
        )
        logits = self.bf16(129, 768)
        logits[0].zero_()
        x = self.bf16(129, 8192)

        def pack(rows, scores):
            n = rows.shape[0]
            weights = torch.empty((n, 10), device="cuda", dtype=torch.float32)
            ids = torch.empty((n, 10), device="cuda", dtype=torch.int64)
            router(scores, ids, weights)
            fp8 = torch.empty_like(rows, dtype=torch.float8_e4m3fn)
            scales = torch.empty((n, 64), device="cuda", dtype=torch.int32)
            fused_pack_mega_moe_inputs(
                rows,
                weights,
                ids,
                fp8,
                scales,
                torch.empty_like(ids),
                torch.empty_like(weights),
            )
            return ids, weights, fp8, scales

        baseline = pack(x[:1], logits[:1])
        self.assertEqual(baseline[0][0].tolist(), list(range(10)))
        for size in (31, 32, 33, 129):
            mixed = pack(
                x[:size].roll(size - 1, dims=0), logits[:size].roll(size - 1, dims=0)
            )
            for label, ref, actual in zip(
                ("IDs", "weights", "FP8", "scales"), baseline, mixed
            ):
                self.assert_bits(ref, actual[-1:], f"MoE {size} rows {label}")

    @torch.inference_mode()
    def test_ple_page_boundary(self):
        ple = Qwen4ExpPLELayer(
            SimpleNamespace(ngram_size=3, context_len=2),
            self.bf16(32768, 8192) * 0.01,
            self.bf16(8192, 8192) * 0.01,
            self.bf16(32768, 1, 4) * 0.01,
            self.bf16(32768) * 0.01,
            self.bf16(32768) * 0.01,
            self.bf16(32768) * 0.01,
            hc_mult=4,
            hidden_size=8192,
            conv_kernel_size=4,
            norm_eps=1e-6,
            batch_invariant=True,
        )
        hidden, embeddings = self.bf16(1, 129, 32768), self.bf16(1, 129, 8192)
        history = torch.zeros((1, 131), device="cuda", dtype=torch.int64)
        reference, snapshots = ple.prefill_page_snapshots(
            hidden, history, 128, embeddings=embeddings, add_residual=True
        )
        for length in (127, 128):
            output, states = ple.prefill_page_snapshots(
                hidden[:, :length],
                history[:, : length + 2],
                128,
                embeddings=embeddings[:, :length],
                add_residual=True,
            )
            self.assert_bits(
                reference[:, :length], output, f"PLE first {length} tokens"
            )
            if length == 128:
                tail, updated = ple.prefill_page_snapshots(
                    hidden[:, 128:],
                    history[:, 128:],
                    128,
                    states[-1],
                    embeddings=embeddings[:, 128:],
                    add_residual=True,
                )
                self.assert_bits(reference[:, 128:], tail, "PLE page continuation")
                self.assert_bits(snapshots[-1], updated[-1], "PLE continuation state")
                for batch in (1, 3):
                    decode_hidden = hidden[:, 128:].expand(batch, -1, -1).contiguous()
                    decode_history = history[:, 128:].expand(batch, -1).contiguous()
                    decode_buffer = states[-1].expand(batch, -1, -1).contiguous()
                    decode_embed = embeddings[:, 128:].expand(batch, -1, -1)
                    output, state = ple.decode_step(
                        decode_hidden,
                        decode_history,
                        decode_buffer,
                        embeddings=decode_embed,
                        add_residual=True,
                    )
                    self.assert_bits(tail, output[-1:], f"PLE decode rows {batch}")
                    self.assert_bits(
                        updated[-1], state[-1:], f"PLE decode state rows {batch}"
                    )

    @torch.inference_mode()
    @patch.dict(
        os.environ,
        {
            "QWEN4_QSA_BACKEND": "flashinfer",
            "QWEN4_QSA_INDEXER_BACKEND": "flashinfer",
            "QWEN4_QSA_FUSED_INDEXER_PREP": "1",
        },
    )
    def test_qsa_completed_blocks_tail_and_paged_attention(self):
        tokens = 2052
        rows = torch.tensor([2047, 2050, 2051], device="cuda")

        def ints(values, device="cuda"):
            return torch.tensor(values, dtype=torch.int32, device=device)

        config = AttentionConfigs()
        config.head_num, config.kv_head_num, config.size_per_head = 64, 4, 256
        config.kernel_tokens_per_block, config.max_seq_len = 128, 8192
        config.need_rope_kv_cache = True
        rope = config.rope_config
        rope.style, rope.index_factor, rope.dim = RopeStyle.Mrope, 3, 64
        rope.base, rope.scale = 10_000_000, 1.0
        rope.mrope_dim1, rope.mrope_dim2, rope.mrope_dim3 = 11, 11, 10
        rope.mrope_interleaved = True

        hidden = self.bf16(tokens, 8192)
        qkv = self.bf16(tokens, (64 + 2 * 4) * 256)
        gate = self.bf16(tokens, 64 * 256)
        norm = ZeroCenteredFusedQKRMSNorm(
            self.bf16(256), self.bf16(256), 64, 4, 256, batch_invariant=True
        )
        indexer = Qwen4ExpQSAIndexer(
            self.bf16(1152, 8192),
            self.bf16(128),
            self.bf16(128),
            n_heads=8,
            kv_heads=1,
            head_dim=128,
            token_budget=2048,
            compress_ratio=4,
            norm_eps=1e-6,
            rope_config=rope,
            batch_invariant=True,
        )

        def side_cache(tag, width, dtype):
            return SimpleNamespace(
                tag=tag,
                seq_size_per_block=128,
                kv_cache_base=torch.zeros((18, width), device="cuda", dtype=dtype),
            )

        def run(spans, pages, *, fake=False):
            table = ints([pages])
            main = SimpleNamespace(
                tag="full",
                kv_cache_base=hidden.new_zeros((18, 2 * 4 * 128 * 256)),
                kv_scale_base=None,
            )
            kv = side_cache(INDEXER_KV_TAG, 2 * 32 * 128, torch.uint8)
            state = side_cache(INDEXER_STATE_TAG, 8 * 128, torch.float32)
            positions = side_cache(INDEXER_ROPE_POS_TAG, 8 * 3, torch.int32)
            selected_rows, outputs = [], []
            for begin, end, prefill in spans:
                length = end - begin
                metadata = dict(
                    is_prefill=prefill,
                    is_target_verify=False,
                    is_cuda_graph=False,
                    is_s_padded=fake,
                    context_parallel_info=None,
                    cache_store_inputs=None,
                    input_lengths=ints([length], "cpu"),
                    prefix_lengths=ints([begin] if prefill else [], "cpu"),
                    prefix_lengths_device=ints([begin] if prefill else []),
                    sequence_lengths=ints([] if prefill else [begin], "cpu"),
                    sequence_lengths_plus_1_device=ints([end]),
                    cu_seqlens_device=ints([0, length]),
                    cu_kv_seqlens_device=ints([0, end]),
                    combo_position_ids=torch.arange(
                        begin, end, device="cuda", dtype=torch.int32
                    ).repeat_interleave(3),
                    padding_offset=ints([0] * length),
                    context_total_kv_length=end,
                    dtype=get_typemeta(hidden),
                    kv_cache_kernel_block_id_device=table,
                    kv_cache_block_id_device=table,
                )
                main_inputs = SimpleNamespace(**metadata)
                impl = SparseGqaFmhaImpl(config, main_inputs, batch_invariant=True)
                runtime = Qwen4ExpQSARuntimeContext(
                    main,
                    main_inputs,
                    kv,
                    SimpleNamespace(**metadata),
                    state,
                    SimpleNamespace(**metadata),
                    position_state_cache=positions,
                    position_state_inputs=SimpleNamespace(**metadata),
                )
                x = hidden[begin:end]
                impl.begin_qsa_cache_transaction()
                impl.validate_qsa_before_side_write(runtime, indexer, x)
                self.assertIsNone(impl._flashinfer_qtoken)
                q, raw = indexer.project(x)
                if prefill:
                    selected = runtime.select_prefix_reuse_prefill_tokens(
                        q, raw, indexer=indexer, rope_config=rope
                    )
                    impl.set_trusted_selected_indices(
                        selected, cu_seqlens=main_inputs.cu_seqlens_device
                    )
                else:
                    selected = runtime.select_decode_tokens(
                        q, raw, indexer=indexer, rope_config=rope
                    )
                    impl.set_selected_indices(selected)
                output = impl.forward_with_qk_norm_and_output_gate(
                    qkv[begin:end].clone(), main, 3, gate[begin:end], norm
                )
                runtime.finalize_side_cache()
                selected_rows.append(selected)
                outputs.append(output)
            return torch.cat(selected_rows), torch.cat(outputs)

        pages = list(range(1, 18))
        cold_ids, cold_output = run([(0, tokens, True)], pages)
        boundary_ids = cold_ids[rows]
        self.assertEqual((boundary_ids >= 0).sum(-1).tolist(), [2048, 2051, 2048])
        self.assertEqual(boundary_ids[1, -3:].tolist(), [2048, 2049, 2050])
        self.assertEqual(boundary_ids[2, -3:].tolist(), [-1, -1, -1])
        tied = select_qsa_paged_tokens(
            torch.zeros((1, 513), device="cuda"),
            ints([2052]),
            compress_ratio=4,
            token_budget=2048,
            batch_invariant=True,
        )
        self.assertEqual(tied[0, :2048].tolist(), list(range(2048)))
        for name, spans in (
            ("aligned prefill", [(0, 2048, True), (2048, tokens, True)]),
            (
                "paged decode",
                [(0, 2048, True), *((i, i + 1, False) for i in range(2048, tokens))],
            ),
        ):
            ids, output = run(spans, pages[::-1])
            self.assert_bits(boundary_ids, ids[rows], f"QSA {name} ordered indices")
            self.assert_bits(cold_output[rows], output[rows], f"QSA {name} attention")

        # Idle ranks enter the MoE collective with an all-zero cache page table.
        fake_ids, fake_output = run([(1, 2, False)], [0], fake=True)
        self.assertTrue((fake_ids == -1).all().item())
        self.assertTrue((fake_output == 0).all().item())


if __name__ == "__main__":
    unittest.main()
