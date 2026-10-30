"""Regression checks for K3 cache metadata, phase gates and workspace limits."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3
from rtp_llm.models_py.distributed.collective_torch import Fp8Activation
from rtp_llm.models_py.model_desc.block_map import ensure_host_prefill_metadata
from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3DecoderLayer, KimiK3Model
from rtp_llm.models_py.modules.factory.attention.cuda_kda_impl import (
    plan_state_sequences,
)
from rtp_llm.models_py.modules.factory.attention.kda_impl import KdaPrefill
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
    MegaMoeExecutor,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_comm_fusion import (
    Fp8CollectiveProjection,
)
from rtp_llm.models_py.modules.kimi_k3.attention import KimiK3MLA
from rtp_llm.models_py.modules.kimi_k3.residual import KimiK3AttentionResidual
from rtp_llm.models_py.triton_kernels.causal_conv1d.paged_short_conv_prefill import (
    paged_short_conv_prefill,
    prepare_paged_short_conv_metadata,
)
from rtp_llm.ops import HybridAttentionType, RoleType
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs


class _LinearMetadataProbe(nn.Module):
    layer_type = HybridAttentionType.LINEAR

    def forward(self, hidden, *args):
        return hidden


class K3HostMetadataReproTest(unittest.TestCase):
    def test_prefill_device_metadata_keeps_device_inputs_and_plans_on_host(self):
        model = KimiK3Model.__new__(KimiK3Model)
        nn.Module.__init__(model)
        model.tp_size = 1
        model.tp_rank = 0
        model.num_blocks = 0
        model.kv_cache = None
        model.layers = nn.ModuleList([_LinearMetadataProbe()])
        hidden = torch.ones((8, 128), device="cuda", dtype=torch.bfloat16)
        attn = PyAttentionInputs()
        attn.is_prefill = True
        attn.is_target_verify = False
        # PyWrappedModel's CUDA input_lengths branch supplies a device cumsum
        # and an empty host cu_seqlens; exercise the actual K3 consumer.
        attn.cu_seqlens = torch.empty(0, dtype=torch.int32)
        attn.cu_seqlens_device = torch.tensor(
            [0, 3, 8], device="cuda", dtype=torch.int32
        )
        inputs = NS(input_ids=torch.zeros(8, device="cuda"), attention_inputs=attn)
        actual = model._forward_layers(hidden, inputs, fmha_impl={})
        torch.testing.assert_close(actual, hidden)
        self.assertEqual(attn.cu_seqlens.tolist(), [0, 3, 8])
        self.assertTrue(attn.cu_seqlens.is_pinned())
        self.assertTrue(attn.cu_seqlens_device.is_cuda)

    def test_paged_convolution_requires_prefix_and_table_host_mirrors(self):
        attn = PyAttentionInputs()
        attn.is_prefill = True
        for device_prefix, device_table in ((True, False), (False, True)):
            with self.subTest(device_prefix=device_prefix, device_table=device_table):
                attn.prefix_lengths = torch.tensor(
                    [64], dtype=torch.int32, device="cuda" if device_prefix else "cpu"
                )
                attn.kv_cache_kernel_block_id = torch.tensor(
                    [[1, 2]],
                    dtype=torch.int32,
                    device="cuda" if device_table else "cpu",
                )
                ensure_host_prefill_metadata(NS(attention_inputs=attn))
                KdaPrefill._validate_prefix_state(attn, 64)
                self.assertEqual(attn.prefix_lengths.tolist(), [64])
                self.assertEqual(attn.kv_cache_kernel_block_id.tolist(), [[1, 2]])
                if device_prefix:
                    self.assertTrue(attn.prefix_lengths.is_pinned())
                if device_table:
                    self.assertTrue(attn.kv_cache_kernel_block_id.is_pinned())

    def test_tagged_host_mirrors_are_shared_and_prepared_only_once(self):
        attn = PyAttentionInputs()
        attn.is_prefill = True
        attn.cu_seqlens_device = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
        attn.prefix_lengths = torch.tensor([64], device="cuda", dtype=torch.int32)
        attn.kv_cache_kernel_block_id = torch.tensor(
            [[1, 2]], device="cuda", dtype=torch.int32
        )
        attn.kv_cache_kernel_block_id_device = attn.kv_cache_kernel_block_id
        inputs = PyModelInputs(attention_inputs={"linear": attn, "full": attn})
        ensure_host_prefill_metadata(inputs)
        linear, full = (
            inputs.attention_inputs["linear"],
            inputs.attention_inputs["full"],
        )
        self.assertEqual(linear.cu_seqlens.tolist(), [0, 4])
        self.assertEqual(linear.cu_seqlens.data_ptr(), full.cu_seqlens.data_ptr())
        self.assertEqual(
            linear.kv_cache_kernel_block_id_device.data_ptr(),
            attn.kv_cache_kernel_block_id.data_ptr(),
        )
        with patch("torch.cat", side_effect=AssertionError("metadata copied twice")):
            ensure_host_prefill_metadata(inputs)


class K3FakeWarmupReproTest(unittest.TestCase):
    def test_ping_pong_warmup_state_matches_whole_sequence_convolution(self):
        torch.manual_seed(0)
        values = torch.randn((256, 384), device="cuda", dtype=torch.bfloat16)
        weights = torch.randn((384, 4), device="cuda", dtype=torch.bfloat16)
        states = torch.zeros((3, 3, 384), device="cuda", dtype=torch.bfloat16)
        outputs = []
        for prefix, table in ((0, [-1, 2, -1, -1]), (128, [-1, 2, -1, 1])):
            cu = torch.tensor([0, 128], dtype=torch.int32)
            attn = PyAttentionInputs()
            attn.prefix_lengths = torch.tensor([prefix], dtype=torch.int32)
            attn.kv_cache_kernel_block_id = torch.tensor([table], dtype=torch.int32)
            KdaPrefill._validate_prefix_state(attn, 64)
            plan_state_sequences([0, 128], [prefix], [table], 64)
            outputs.append(
                paged_short_conv_prefill(
                    values[prefix : prefix + 128],
                    weights,
                    states,
                    attn.kv_cache_kernel_block_id.cuda(),
                    attn.prefix_lengths.cuda(),
                    cu.cuda(),
                    64,
                    prepare_paged_short_conv_metadata(cu, values.device),
                )
            )
        cu = torch.tensor([0, 256], dtype=torch.int32)
        expected = paged_short_conv_prefill(
            values,
            weights,
            None,
            None,
            torch.zeros(1, device="cuda", dtype=torch.int32),
            cu.cuda(),
            64,
            prepare_paged_short_conv_metadata(cu, values.device),
        )
        for plane in range(3):
            torch.testing.assert_close(
                torch.cat([output[plane] for output in outputs]),
                expected[plane],
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(states[2], values[125:128], rtol=0, atol=0)
        torch.testing.assert_close(states[1], values[253:256], rtol=0, atol=0)

    def test_zero_block_warmup_fails_only_after_a_prefix_is_present(self):
        # Zero still means a virtual request. Cached prefixes must carry real
        # state IDs, including synthetic warmup prefixes.
        attn = PyAttentionInputs()
        attn.kv_cache_kernel_block_id = torch.zeros((1, 4), dtype=torch.int32)
        attn.prefix_lengths = torch.tensor([0], dtype=torch.int32)
        KdaPrefill._validate_prefix_state(attn, 64)
        self.assertIsNone(
            plan_state_sequences([0, 64], [0], [[0] * 4], 64)[0].initial_block
        )
        for prefix in (64, 128):
            with self.subTest(prefix=prefix):
                attn.prefix_lengths = torch.tensor([prefix], dtype=torch.int32)
                with self.assertRaisesRegex(
                    ValueError, "missing a cached prefix state"
                ) as conv_error:
                    KdaPrefill._validate_prefix_state(attn, 64)
                with self.assertRaisesRegex(
                    ValueError, "Virtual KDA request cannot have cached history"
                ) as state_error:
                    plan_state_sequences([0, 64], [prefix], [[0] * 4], 64)
                print(
                    f"fake warmup prefix={prefix}:",
                    conv_error.exception,
                    ";",
                    state_error.exception,
                )

                # Valid nonzero physical state IDs satisfy both consumers.
                attn.kv_cache_kernel_block_id.fill_(1)
                KdaPrefill._validate_prefix_state(attn, 64)
                plan_state_sequences([0, 64], [prefix], [[1] * 4], 64)
                attn.kv_cache_kernel_block_id.zero_()


class K3DecodeFp8PhaseReproTest(unittest.TestCase):
    def test_fp8_production_matches_prefill_and_verify_consumers(self):
        rows, hidden_size = 4096, 7168
        collective = Fp8CollectiveProjection.__new__(Fp8CollectiveProjection)
        collective.enable_ag = True
        collective.decode_staging = True
        collective.world_size = 2
        collective.max_m = rows * collective.world_size
        # No communicator is needed: rejection precedes communication/GEMM.
        attention = KimiK3MLA.__new__(KimiK3MLA)
        nn.Module.__init__(attention)
        attention._fp8_collective = collective
        layer = KimiK3DecoderLayer.__new__(KimiK3DecoderLayer)
        nn.Module.__init__(layer)
        layer.index = 0
        layer.block_size = 1
        layer.attention = attention
        weight = torch.ones(hidden_size, device="cuda", dtype=torch.bfloat16)
        layer.attention_residual = KimiK3AttentionResidual(weight, weight, 1e-6)
        layer.attention_norm = NS(weight=weight, variance_epsilon=1e-6)
        hidden = torch.ones((rows, hidden_size), device="cuda", dtype=torch.bfloat16)
        anchors = torch.zeros(
            (rows, 1, hidden_size), device="cuda", dtype=torch.bfloat16
        )
        inputs = NS(is_prefill=True, is_mtp_draft_update=False)
        metadata = NS(is_target_verify=False)

        class ProjectionBoundary(Exception):
            pass

        for decode_staging, verify, quantized in (
            (True, False, False),
            (False, False, True),
            (True, True, True),
        ):
            with self.subTest(decode_staging=decode_staging, verify=verify):
                collective.decode_staging = decode_staging
                metadata.is_target_verify = verify
                layer.attention_residual.configure_verify_fp8(decode_staging)

                def consume(value, *args):
                    self.assertEqual(isinstance(value, Fp8Activation), quantized)
                    self.assertEqual(
                        collective.eligible_ag(value, inputs, metadata), quantized
                    )
                    raise ProjectionBoundary

                with patch.object(attention, "forward", side_effect=consume):
                    with self.assertRaises(ProjectionBoundary):
                        layer(hidden, anchors, None, None, inputs, metadata, None)
        fp8 = Fp8Activation(
            torch.empty((1, 128), device="cuda", dtype=torch.float8_e4m3fn),
            torch.empty((1, 4), device="cuda", dtype=torch.int32),
        )
        metadata.is_target_verify = False
        self.assertFalse(collective.eligible_ag(fp8, inputs, metadata))
        self.assertTrue(collective.eligible_ag(fp8, inputs, NS(is_target_verify=True)))
        collective.decode_staging = False
        with patch("torch.cuda.is_current_stream_capturing", return_value=False):
            self.assertTrue(collective.eligible_ag(fp8, inputs, metadata))


class K3CapacityReproTest(unittest.TestCase):
    @staticmethod
    def model(role, tp, batch, width, max_seq_len=4096, token_budget=128):
        model = object.__new__(KimiK3)
        model.model_config = NS(
            max_seq_len=max_seq_len,
            gen_num_per_cycle=width - 1,
            mm_model_config=NS(is_multimodal=False),
            moe_prefill_max_tokens_per_rank=token_budget,
        )
        model.runtime_config = NS(
            fifo_scheduler_config=NS(
                prefill_chunk_size=0,
                max_context_batch_size=1,
                max_batch_tokens_size=token_budget,
            ),
            warm_up=False,
            warm_up_with_loss=False,
            use_batch_decode_scheduler=False,
            batch_decode_scheduler_config=NS(
                batch_decode_scheduler_warmup_type=0,
                batch_decode_scheduler_batch_size=64,
            ),
        )
        model.parallelism_config = NS(role_type=role, tp_size=tp)
        model.hw_kernel_config = NS(enable_cuda_graph=False)
        model.kv_cache_config = NS(multi_task_prompt_tokens={})
        model.max_generate_batch_size = batch
        model.weight = model.fmha_config = model.moe_config = None
        return model

    def executor(self, model):
        import deep_gemm

        with patch("rtp_llm.models_py.model_desc.kimi_k3.KimiK3Model") as ctor, patch(
            "torch.cuda.empty_cache"
        ):
            model._create_python_model()
        global_rows = ctor.call_args.kwargs["max_moe_tokens"]
        executor = object.__new__(MegaMoeExecutor)
        nn.Module.__init__(executor)
        local_rows = global_rows // model.parallelism_config.tp_size
        alignment = int(deep_gemm._C.get_token_alignment_for_mega_moe_k3())
        capacity = (local_rows + alignment - 1) // alignment * alignment
        executor._mega_buf = NS(num_max_tokens_per_rank=capacity)
        executor._mega_y = torch.empty((capacity, 1))
        print(f"MoE requested={local_rows}, alignment={alignment}, capacity={capacity}")
        return executor, global_rows

    def test_fixed_scheduler_batch_is_included_in_decode_budget(self):
        model = self.model(RoleType.DECODE, tp=8, batch=16, width=3)
        model.runtime_config.use_batch_decode_scheduler = True
        executor, budget = self.executor(model)
        self.assertEqual(budget, 192)
        self.assertEqual(model.runtime_config.max_generate_batch_size, 64)
        self.assertEqual(model.max_generate_batch_size, 64)
        actual_rows = (
            model.runtime_config.batch_decode_scheduler_config.batch_decode_scheduler_batch_size
            * 3
            // 8
        )
        executor._validate_capacity(actual_rows)
        collective = Fp8CollectiveProjection.__new__(Fp8CollectiveProjection)
        collective.enable_ag = collective.decode_staging = True
        collective.world_size = model.parallelism_config.tp_size
        collective.max_m = budget
        self.assertTrue(
            collective.eligible_ag(
                torch.empty((actual_rows, 1)),
                NS(is_mtp_draft_update=False),
                NS(is_target_verify=True),
            )
        )

    def test_declared_prefill_capacity_precedes_the_executor_guard(self):
        model = self.model(
            RoleType.PDFUSION,
            tp=8,
            batch=16,
            width=1,
            max_seq_len=4096,
            token_budget=4096,
        )
        executor, budget = self.executor(model)
        self.assertEqual(budget, 4096)
        self.assertEqual(model.model_config.max_prefill_tokens, budget)
        prompt_len, returns = 3072, 3
        local_rows = prompt_len * returns // model.parallelism_config.tp_size
        executor._validate_capacity(prompt_len // model.parallelism_config.tp_size)
        with self.assertRaisesRegex(
            RuntimeError, r"input tokens=1152 exceeds num_max_tokens_per_rank="
        ) as error:
            executor._validate_capacity(local_rows)
        print(
            "raw / singleton, non-MTP, TP8, L3072, n3, max_seq_len4096:",
            error.exception,
        )


if __name__ == "__main__":
    unittest.main()
