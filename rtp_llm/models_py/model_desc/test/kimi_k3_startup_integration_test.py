"""Exercise startup metadata through the production KDA and DSA consumers."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.model_factory import ModelFactory
from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3Model
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_page_rr_params import (
    DsaPageRRParams,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
    SparseMlaImpl,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_page_rr_impl import (
    PageRRSparseMlaOp,
    SparseMlaPageRRImpl,
)
from rtp_llm.models_py.modules.factory.attention.kda_impl import KdaMetadata, KdaPrefill
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.models_py.modules.hybrid.query_layout import AttentionInputView
from rtp_llm.models_py.modules.kimi_k3.dsa import KimiK31DSA
from rtp_llm.models_py.triton_kernels.causal_conv1d.paged_short_conv_prefill import (
    paged_short_conv_prefill,
    prepare_paged_short_conv_metadata,
)
from rtp_llm.models_py.triton_kernels.causal_conv1d.prefill_state_alias import (
    snapshot_prefill_conv_input,
)
from rtp_llm.ops import CPRotateMethod, ParallelismConfig, RoleType
from rtp_llm.ops.compute_ops import PyAttentionInputs
from rtp_llm.utils.model_weight import W


def query_cp_parallel(rank=0):
    parallel = ParallelismConfig()
    parallel.tp_size = parallel.world_size = 2
    parallel.tp_rank = rank
    parallel.role_type = RoleType.PREFILL
    parallel.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
    parallel.prefill_cp_config.prefill_cp_size = 2
    parallel.prefill_cp_config.kv_cache_sharded = True
    parallel.prefill_cp_config.preserve_tp_layout = True
    return parallel


class KimiK3StartupIntegrationTest(unittest.TestCase):
    def test_query_cp_model_retains_decode_padding_capability(self):
        def base_init(instance, config, *args):
            nn.Module.__init__(instance)
            instance.config, instance.layer_num = config, 0

        parallel = query_cp_parallel()
        parallel.ep_size = 2
        config = NS(
            hidden_size=8,
            is_mtp=False,
            layernorm_eps=1e-6,
            k3_runtime_config=NS(dsa=object(), attn_res_block_size=0),
        )
        weights = NS(weights=[], get_global_weight=lambda _: torch.ones(8))
        with (
            patch.object(GptModelBase, "__init__", base_init),
            patch.object(KimiK3Model, "_setup_dsa"),
            patch(
                "rtp_llm.models_py.model_desc.kimi_k3.Embedding",
                return_value=nn.Identity(),
            ),
        ):
            model = KimiK3Model(config, parallel, weights, 2, max_moe_tokens=16)
        self.assertTrue(model.prefill_query_cp)
        self.assertTrue(model.requires_sequence_parallel_padding)

    def test_device_input_prefix_warmup_runs_shared_kda_forward(self):
        config = NS(
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_key_heads=1,
            linear_num_value_heads=1,
            linear_conv_kernel_dim=4,
            ssm_state_dtype="fp32",
            conv_state_dtype="bf16",
        )
        weights = {
            W.linear_attn_conv1d_w: torch.ones(
                (384, 1, 4), dtype=torch.bfloat16, device="cuda"
            ),
            W.linear_attn_dt_b_kda: torch.zeros(128, device="cuda"),
            W.linear_attn_alog: torch.zeros(1, device="cuda"),
        }
        prefill = KdaPrefill(config, ParallelismConfig(), weights)
        # Native C++ supplies the read-only device mirrors. Recreate their
        # published values here while exercising the real shared KDA forward.
        inputs = AttentionInputView(
            PyAttentionInputs(),
            is_prefill=True,
            is_warmup=True,
            cache_store_inputs=None,
            cache_store_writer=None,
        )
        inputs.input_lengths = inputs.input_lengths_device = torch.tensor(
            [64], dtype=torch.int32, device="cuda"
        )
        inputs.prefix_lengths = inputs.prefix_lengths_device = torch.tensor(
            [64], dtype=torch.int32, device="cuda"
        )
        inputs.cu_seqlens = torch.empty(0, dtype=torch.int32)
        inputs.cu_seqlens_device = torch.tensor(
            [0, 64], dtype=torch.int32, device="cuda"
        )
        inputs.kv_cache_kernel_block_id = inputs.kv_cache_kernel_block_id_device = (
            torch.zeros((1, 2), dtype=torch.int32, device="cuda")
        )
        backing = torch.ones(
            (2, prefill.linear_cache_converter.block_size_bytes // 2),
            dtype=torch.bfloat16,
            device="cuda",
        )
        output = prefill(
            torch.zeros((64, 384), dtype=torch.bfloat16, device="cuda"),
            torch.zeros((64, 128), dtype=torch.bfloat16, device="cuda"),
            torch.zeros((64, 1), dtype=torch.bfloat16, device="cuda"),
            inputs,
            NS(kv_cache_base=backing, seq_size_per_block=64),
            KdaMetadata(),
        )
        self.assertEqual(output.shape, (64, 1, 128))
        self.assertTrue(torch.isfinite(output).all().item())
        self.assertTrue(torch.all(backing == 1).item())

    def test_noncausal_chunk_configuration_fails_before_engine_construction(self):
        engine = EngineConfig.create(PyEnvConfigs())
        engine.runtime_config.fifo_scheduler_config.prefill_chunk_size = 128
        model = NS(
            num_layers=1,
            max_seq_len=4096,
            model_name="noncausal",
            attn_config=NS(tokens_per_block=64, is_causal=False),
        )
        with self.assertRaisesRegex(ValueError, "[Nn]on.causal|is_causal"):
            ModelFactory.update_engine_config_from_model_config(engine, model)

    def test_full_length_warmup_reaches_paged_convolution(self):
        torch.manual_seed(1)
        values = torch.randn((192, 384), dtype=torch.bfloat16, device="cuda")
        weights = torch.randn((384, 4), dtype=torch.bfloat16, device="cuda")
        for table_device in ("cpu", "cuda"):
            with self.subTest(table_device=table_device):
                inputs = PyAttentionInputs()
                inputs.is_prefill = inputs.is_warmup = True
                inputs.prefix_lengths = torch.zeros(1, dtype=torch.int32)
                inputs.cu_seqlens = torch.tensor([0, 192], dtype=torch.int32)
                inputs.kv_cache_kernel_block_id = torch.zeros(
                    (1, 3), dtype=torch.int32, device=table_device
                )
                original = torch.ones((2, 3 * 384), dtype=torch.bfloat16, device="cuda")
                view, scratch = KdaPrefill._warmup_cache_inputs(inputs, original)
                states = scratch.view(-1, 3, 384)
                source, mask = snapshot_prefill_conv_input(
                    states, view, 64, KdaMetadata()
                )
                metadata = prepare_paged_short_conv_metadata(inputs.cu_seqlens, "cuda")
                args = (
                    view.prefix_lengths.cuda(),
                    inputs.cu_seqlens.cuda(),
                    64,
                    metadata,
                )
                actual = paged_short_conv_prefill(
                    values,
                    weights,
                    states,
                    view.kv_cache_kernel_block_id_device,
                    *args,
                    current_conv_state=source,
                    continuation_mask=mask
                )
                expected = paged_short_conv_prefill(values, weights, None, None, *args)
                for got, want in zip(actual[:3], expected[:3]):
                    torch.testing.assert_close(got, want, rtol=0, atol=0)
                torch.testing.assert_close(original, torch.ones_like(original))

    def test_real_prefill_still_rejects_multiple_writers(self):
        inputs = PyAttentionInputs()
        inputs.is_prefill = True
        inputs.prefix_lengths = torch.zeros(1, dtype=torch.int32)
        inputs.cu_seqlens = torch.tensor([0, 128], dtype=torch.int32)
        inputs.kv_cache_kernel_block_id = torch.ones((1, 2), dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "alias one physical state"):
            snapshot_prefill_conv_input(
                torch.zeros((2, 3, 384), device="cuda"), inputs, 64, KdaMetadata()
            )

    def test_query_cp_decode_constructor_keeps_complete_head_geometry(self):
        config = NS(
            max_seq_len=4096,
            getAttentionConfigs=lambda tp: NS(
                head_num=8 // tp,
                mla_fp8_compute=False,
                tokens_per_block=4096,
                kernel_tokens_per_block=128,
            ),
        )
        owner = NS(prefill_query_cp=True, sfa_use_fp8=False)
        observed = []

        def backend_init(instance, attention, *args, **kwargs):
            # Stop at the architecture-specific sparse kernel boundary. Its
            # producer must agree with the complete-head Query-CP weights.
            observed.append(attention.head_num)
            instance.kv_cache_write_op = NS()

        with patch.object(SparseMlaImpl, "__init__", backend_init):
            SparseMlaPageRRImpl(
                config,
                query_cp_parallel(),
                NS(weights=[]),
                NS(is_prefill=False),
                None,
                False,
                workspace_owner=owner,
            )
        self.assertEqual(observed, [8])

    def test_sparse_page_rr_decode_graph_refreshes_metadata(self):
        lengths = torch.tensor([1, 64], dtype=torch.int32, device="cuda")
        pages = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32, device="cuda")
        inputs = AttentionInputView(
            PyAttentionInputs(),
            is_prefill=False,
            is_cuda_graph=True,
            physical_token_count=2,
            input_lengths=torch.ones(2, dtype=torch.int32),
            sequence_lengths_plus_1_device=lengths,
            kv_cache_kernel_block_id_device=pages,
        )
        impl = SparseMlaPageRRImpl.__new__(SparseMlaPageRRImpl)
        impl.execution_phase = None
        impl.fmha_impl = PageRRSparseMlaOp.__new__(PageRRSparseMlaOp)
        impl.page_rr_params = DsaPageRRParams(
            NS(tokens_per_block=64, kernel_tokens_per_block=64),
            ParallelismConfig(),
            max_seq_len=128,
            is_cuda_graph=True,
        )
        with (
            patch.object(impl.page_rr_params, "prepare_execution"),
            patch.object(impl.fmha_impl, "plan"),
        ):
            impl.create_params(inputs)
            slots = impl.page_rr_params.slot_mapping
            self.assertEqual(slots.tolist(), [64, 255])
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                impl.prepare_cuda_graph(inputs)
            lengths.copy_(torch.tensor([65, 3], device="cuda", dtype=torch.int32))
            pages.copy_(
                torch.tensor([[5, 6], [7, 8]], device="cuda", dtype=torch.int32)
            )
            graph.replay()
            self.assertIs(impl.page_rr_params.slot_mapping, slots)
            self.assertEqual(slots.tolist(), [384, 450])

    def test_real_dsa_constructor_initializes_output_row_owner(self):
        config = NS(
            attn_config=NS(
                head_num=4,
                q_lora_rank=8,
                kv_lora_rank=8,
                rope_head_dim=8,
                nope_head_dim=8,
                v_head_dim=8,
            ),
            layernorm_eps=1e-6,
            hidden_size=8,
            is_mtp=True,
            indexer_projection_quant_config=object(),
            k3_runtime_config=NS(
                mla_use_nope=True,
                mla_use_output_gate=True,
                dsa=NS(indexer_use_fp8=True),
            ),
        )
        weights = {
            W.mla_fusedqkrope_w: torch.ones((8, 56), dtype=torch.bfloat16),
            W.mla_q_b_w: torch.ones((8, 64), dtype=torch.bfloat16),
            W.attn_o_w: torch.ones((32, 8), dtype=torch.bfloat16),
            W.mla_q_a_ln_gamma: torch.ones(8, dtype=torch.bfloat16),
            W.mla_kv_a_ln_gamma: torch.ones(8, dtype=torch.bfloat16),
        }
        with (
            patch(
                "rtp_llm.models_py.modules.kimi_k3.attention.create_kimi_projection",
                side_effect=lambda config, weights, name, *args: CudaF16Linear(
                    weights[name]
                ),
            ),
            patch("rtp_llm.models_py.modules.kimi_k3.dsa.Indexer"),
            patch("rtp_llm.models_py.modules.base.cuda.fp8_indexer_op.Fp8IndexerOp"),
        ):
            attention = KimiK31DSA(config, query_cp_parallel(1), weights, 0)
        self.assertEqual(attention.tp_rank, 1)


if __name__ == "__main__":
    unittest.main()
