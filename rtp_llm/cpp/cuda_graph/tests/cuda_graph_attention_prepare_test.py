"""Capture the production Page-RR metadata hooks with their consuming graph."""

import os
import unittest
from types import SimpleNamespace

import torch
from rtp_llm.cpp.cuda_graph.tests.cuda_graph_tagged_cache_test import (
    GROUP_TAGS,
    HIDDEN_SIZE,
    KERNEL_BLOCK_TABLE_WIDTH,
    CudaGraphRunner,
    _build_decode_inputs,
    _build_target_verify_inputs,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_kv_workspace import (
    DsaWorkspaceOwner,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_page_rr_params import (
    DsaPageRRParams,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_page_rr_impl import (
    PageRRSparseMlaOp,
    SparseMlaPageRRImpl,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.page_rr_mla_metadata import (
    PageRRMlaMetadata,
)
from rtp_llm.models_py.modules.kimi_k3.mla import KimiK3MlaImpl
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import PyModelOutputs


class PrepareProbe:
    def __init__(self, impl, capture, tokens):
        self.impl = impl
        self.tokens = tokens
        self.requires_host_metadata = impl.requires_host_metadata
        self.can_capture_cuda_graph_prepare = (
            capture and impl.can_capture_cuda_graph_prepare
        )
        self.calls = 0
        self.capture_calls = 0

    def prepare_cuda_graph(self, inputs):
        self.calls += 1
        self.capture_calls += int(torch.cuda.is_current_stream_capturing())
        inputs.is_cuda_graph = True
        inputs.physical_token_count = self.tokens
        self.impl.prepare_cuda_graph(inputs)


class MetadataModel:
    def __init__(self, backend, capture=(True, True), max_seq_len=512):
        self.backend = backend
        self.capture = capture
        self.max_seq_len = max_seq_len
        self.probes = []
        self.owner = DsaWorkspaceOwner(
            tp_size=2,
            heads=64,
            indexer_use_fp8=backend == "dsa_fp8",
            sfa_use_fp8=backend == "dsa_fp8",
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        # init_decode's eager dtype probe reserves 128 request slots.
        self.owner.graph_max_requests = 128
        self.owner.graph_max_queries = 2
        self.owner.graph_cache_groups = len(GROUP_TAGS)

    def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
        result = {}
        for tag, capture in zip(GROUP_TAGS, self.capture):
            attn = inputs.attention_inputs[tag]
            attn.is_cuda_graph = is_cuda_graph
            attn.physical_token_count = inputs.input_ids.numel()
            if self.backend.startswith("mla"):
                impl = object.__new__(KimiK3MlaImpl)
                impl._paged = True
                impl.batch = attn.input_lengths.numel()
                impl.tokens = attn.physical_token_count
                impl.q_replicated = impl.fp8_compute = self.backend == "mla_compact"
                impl.page_rr_metadata = PageRRMlaMetadata(128, 128, 2, 0)
            else:
                parallel = ParallelismConfig()
                parallel.tp_size = 2
                parallel.tp_rank = 0
                parallel.prefill_cp_config.kv_cache_sharded = True
                impl = object.__new__(SparseMlaPageRRImpl)
                impl.page_rr_params = DsaPageRRParams(
                    SimpleNamespace(tokens_per_block=128, kernel_tokens_per_block=128),
                    parallel,
                    max_seq_len=self.max_seq_len,
                    workspace_owner=self.owner,
                    is_cuda_graph=True,
                )
                op = object.__new__(PageRRSparseMlaOp)
                op.workspace_owner = self.owner
                op.kernel_heads = 64
                op.decode_kernel_heads = 64
                op.sfa_decode_backend = "sched"
                op.top_k = self.owner.topk
                op._fp8_schedules = {}
                impl.fmha_impl = op
            # Materialize real buffers and native scheduler state before capture.
            impl.prepare(attn)
            probe = PrepareProbe(impl, capture, attn.physical_token_count)
            self.probes.append(probe)
            result[tag] = probe
        return result

    def forward(self, inputs, fmha_impl):
        signatures = []
        for probe in fmha_impl.values():
            impl = probe.impl
            if self.backend.startswith("mla"):
                metadata = impl.page_rr_metadata
                extra = metadata.query_block_tables[:, 0]
                if extra.numel() != metadata.positions_d.numel():
                    extra = extra.repeat_interleave(
                        metadata.positions_d.numel() // extra.numel()
                    )
            else:
                params = impl.page_rr_params
                metadata = params.decode
                extra = metadata.query_block_tables[:, 0]
                for phase in (params.sfa_execution, params.indexer_execution):
                    extra = extra + phase.block_ids[:, 0].repeat_interleave(
                        phase.queries
                    )
                    for block in phase.blocks:
                        if block.native_context is not None:
                            extra = extra + block.native_context.reshape(-1)
                            extra = extra + block.native_table[:, 0].repeat_interleave(
                                phase.queries
                            )
            signatures.append(
                torch.stack(
                    (
                        metadata.positions_d,
                        metadata.slot_mapping,
                        metadata.local_causal_lens.reshape(-1),
                        extra,
                    ),
                    dim=1,
                ).to(torch.bfloat16)
            )
        return PyModelOutputs(sum(signatures) + inputs.input_hiddens)


def make_runner(model, role):
    runner = CudaGraphRunner()
    if role == "draft_update":
        runner.init_mtp_draft_prefill(model, 2, 2, 1, 1)
    else:
        runner.init_decode(
            model,
            HIDDEN_SIZE,
            512,
            KERNEL_BLOCK_TABLE_WIDTH,
            [2],
            GROUP_TAGS,
            role == "verify",
            2 if role == "verify" else 1,
        )
    return runner


def make_inputs(role, batch, width):
    values = {"full": width + 1, "aux": width + 2}
    if role == "decode":
        inputs = _build_decode_inputs(
            GROUP_TAGS, values, batch_size=batch, block_count=width
        )
    else:
        inputs = _build_target_verify_inputs(
            GROUP_TAGS,
            values,
            batch_size=batch,
            query_len=2,
            prefix_len=128 * width - 1,
        )
        # The input builder uses the test runner's block geometry. This
        # component probe uses real MLA kernel pages instead.
        tagged = inputs.attention_inputs
        for attn in tagged.values():
            table = attn.kv_cache_kernel_block_id[:, :width].contiguous()
            attn.kv_cache_kernel_block_id = table
            attn.kv_cache_kernel_block_id_device = table.cuda()
            if role == "draft_update":
                attn.is_target_verify = False
                attn.is_mtp_draft_update = True
        inputs.attention_inputs = tagged
    inputs.combo_position_ids = torch.arange(
        inputs.input_ids.numel(), dtype=torch.int32, device="cuda"
    )
    tagged = inputs.attention_inputs
    for attn in tagged.values():
        if role == "decode":
            attn.sequence_lengths.fill_(128 * width - 1)
            attn.sequence_lengths_plus_1_device.fill_(128 * width)
    inputs.attention_inputs = tagged
    return inputs


class CapturedAttentionPrepareTest(unittest.TestCase):
    def test_production_metadata_replay_matches_eager_prepare(self):
        for backend in ("mla", "mla_compact", "dsa", "dsa_fp8"):
            for role in ("decode", "verify", "draft_update"):
                with self.subTest(backend=backend, role=role):
                    captured = MetadataModel(backend)
                    eager = MetadataModel(backend, capture=(False, False))
                    graph_runner = make_runner(captured, role)
                    eager_runner = make_runner(eager, role)
                    self.assertTrue(any(p.capture_calls for p in captured.probes))
                    calls = sum(p.calls for p in captured.probes)
                    for batch, width in ((2, 2), (1, 1), (2, 3), (2, 1)):
                        inputs = make_inputs(role, batch, width)
                        with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CPU]
                        ) as prof:
                            self.assertTrue(
                                graph_runner.prepare(
                                    inputs, skip_forward_event_sync=True
                                )
                            )
                        scopes = {e.key for e in prof.key_averages()}
                        self.assertNotIn(
                            "cuda_graph.prepareAttentionInputs(prepare_cuda_graph)",
                            scopes,
                        )
                        self.assertEqual(
                            "cuda_graph.prepareAttentionInputs(wait_forward_event)"
                            in scopes,
                            os.environ.get("RTP_LLM_STREAM_ASYNC") != "1",
                        )
                        self.assertTrue(
                            eager_runner.prepare(inputs, skip_forward_event_sync=True)
                        )
                        actual = graph_runner.forward(inputs).hidden_states.clone()
                        expected = eager_runner.forward(inputs).hidden_states
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        self.assertEqual(sum(p.calls for p in captured.probes), calls)

    def test_mixed_capabilities_keep_eager_prepare(self):
        model = MetadataModel("mla", capture=(True, False))
        runner = make_runner(model, "decode")
        self.assertFalse(any(p.capture_calls for p in model.probes))
        for width in (2, 1, 3):
            calls = sum(p.calls for p in model.probes)
            inputs = make_inputs("decode", 2, width)
            self.assertTrue(runner.prepare(inputs))
            self.assertEqual(
                sum(p.calls for p in model.probes), calls + len(GROUP_TAGS)
            )
            self.assertTrue(runner.canRun(inputs))
            runner.forward(inputs)
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
