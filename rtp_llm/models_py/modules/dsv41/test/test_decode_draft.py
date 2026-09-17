"""Fixed draft graph geometry, official proposal scores and TAIL compact bytes."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch import nn

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.models.deepseek_v41_dspark import DeepSeekV41DSparkWeight
from rtp_llm.models_py.model_desc.deepseek_v41_dspark_model import (
    DeepSeekV41DSparkModel,
    V41DraftFmhaImpl,
    _draft_query_width,
    _native_cp_width,
)
from rtp_llm.models_py.modules.dsv41.cache_layout import CacheLayout, CacheRegion
from rtp_llm.models_py.modules.dsv41.compact_reader import CompactPages
from rtp_llm.models_py.modules.dsv41.decode_draft import (
    V41DraftModel,
    draft_flashmla_attention,
)
from rtp_llm.models_py.modules.dsv41.test.fixture import flash_config
from rtp_llm.models_py.modules.dsv41.test.test_compact_writer import _official_gpu
from rtp_llm.models_py.modules.dsv41.test import test_draft
from rtp_llm.ops.compute_ops import KVCacheRegionName
from rtp_llm.utils.model_weight import W


class DraftDescriptorTest(unittest.TestCase):
    def test_native_cp_width_mirrors_cache_config_helper(self):
        cases = (
            # (kv_cache_sharded, role, tp_size, prefill_cp_size, expected)
            (False, "PREFILL", 4, 4, 1),
            (False, "DECODE", 1, 4, 1),
            (True, "PREFILL", 4, 4, 4),
            (True, "PREFILL", 8, 8, 8),
            (True, "PREFILL", 16, 16, 16),
            (True, "DECODE", 1, 4, 4),
            (True, "DECODE", 1, 8, 8),
            (True, "DECODE", 1, 16, 16),
        )
        for sharded, role, tp_size, cp_size, expected in cases:
            parallelism = SimpleNamespace(
                role_type=role,
                tp_size=tp_size,
                prefill_cp_config=SimpleNamespace(
                    kv_cache_sharded=sharded, prefill_cp_size=cp_size
                ),
            )
            self.assertEqual(_native_cp_width(parallelism), expected)

    def test_prefill_initialization_does_not_bind_cp_shards_as_decode_pages(self):
        model = DeepSeekV41DSparkModel.__new__(DeepSeekV41DSparkModel)
        nn.Module.__init__(model)
        model.layout = CacheLayout(cp_size=8, speculative_tokens=5, draft_enabled=True)
        model._prefill_only = True
        model._pages, model._groups = {}, {}
        model.draft = None
        region = int(KVCacheRegionName.SWA_KV)
        for layers in (3, 43):
            mapping = [[-1] * (region + 1) for _ in range(layers)]
            for stage in range(layers - 3, layers):
                mapping[stage][region] = 0
            cache = SimpleNamespace(
                kv_cache_base_by_layer=[],
                kv_scale_base_by_layer=[],
                kv_cache_base_by_layer_region=[[] for _ in range(layers)],
                layer_region_to_group_id=mapping,
                group_seq_size_per_block=[model.layout.reuse_unit],
                get_raw_pool_tensor=Mock(
                    side_effect=AssertionError("P must not bind complete D pages")
                ),
            )
            self.assertTrue(model.initialize(SimpleNamespace(kv_cache=cache)))
            cache.get_raw_pool_tensor.assert_not_called()
            self.assertEqual(model._pages, {})

    def test_capture_geometry_does_not_use_preallocated_aux(self):
        for width in (5, 6):
            inputs = SimpleNamespace(
                input_ids=torch.zeros(4 * width, dtype=torch.int32),
                input_hiddens=torch.zeros((4 * width, 15360), dtype=torch.bfloat16),
                attention_inputs=SimpleNamespace(input_lengths=torch.full((4,), width)),
            )
            self.assertEqual(_draft_query_width(inputs), width)
        inputs.input_ids = torch.zeros(4, dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "PROPOSE5 or TAIL6"):
            _draft_query_width(inputs)

    def test_full_draft_weights_include_each_stage_and_only_local_experts(self):
        descriptor = DeepSeekV41DSparkWeight.__new__(DeepSeekV41DSparkWeight)
        descriptor.model_config = SimpleNamespace(
            dsv41_config=V41Config.from_dict(flash_config())
        )
        descriptor.tp_size, descriptor.ep_size, descriptor.ep_rank = 1, 8, 3
        descriptor.role_type = "DECODE"
        weights = descriptor._get_weight_info()
        self.assertEqual(len(weights.layer_weights), 3)
        for stage in weights.layer_weights:
            installed = {weight.name: weight for weight in stage}
            for name in (
                "attn.wq_a.weight",
                "attn.wq_b.scale",
                "attn.wo_b.weight",
                "attn_norm.weight",
                "ffn_norm.weight",
                "hc_attn_fn",
                "hc_ffn_scale",
                "ffn.gate.bias_vl",
            ):
                self.assertIn("v41." + name, installed)
            self.assertEqual(
                installed["v41.attn.wo_a.weight"].data_type, torch.bfloat16
            )
            self.assertEqual(len(installed["v41.attn.wo_a.weight"].specs), 2)
            experts = {
                int(name.split(".")[3])
                for name in installed
                if name.startswith("v41.ffn.experts.")
            }
            self.assertEqual(experts, set(range(48, 64)))
        globals_ = {weight.name for weight in weights.weights}
        self.assertTrue(
            {
                W.embedding,
                W.lm_head,
                W.final_ln_gamma,
                W.v4_dspark_markov_w1,
                W.v4_dspark_markov_w2,
                "v41.mtp.0.main_proj.weight",
                "v41.mtp.0.main_norm.weight",
            }
            <= globals_
        )
        descriptor.role_type = "PREFILL"
        selective = descriptor._get_weight_info()
        for stage in selective.layer_weights:
            self.assertEqual(
                {weight.name for weight in stage},
                {
                    "v41.attn.wkv.weight",
                    "v41.attn.wkv.scale",
                    "v41.attn.kv_norm.weight",
                },
            )


class DraftDecodeGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_draft.PrefillDraftGpuTest.setUpClass()
        cls.reference = test_draft.PrefillDraftGpuTest()
        cls.device = cls.reference.module.main_norm.device

    def fixture(self, width, starts, counts):
        layout = CacheLayout(cp_size=1, speculative_tokens=5, draft_enabled=True)
        batch = len(starts)
        stride = ((layout.swa_entries * 528 + 511) // 512) * 512
        model = SimpleNamespace(
            layout=layout,
            config=SimpleNamespace(max_seq_len=384, dspark_noise_token_id=7),
            _prefill_only=False,
            kv_cache=SimpleNamespace(),
            device=self.device,
            _groups={stage: stage for stage in range(3)},
            _pages={
                stage: CompactPages(
                    torch.zeros(
                        (batch * 3 + 1, stride), dtype=torch.uint8, device=self.device
                    ),
                    CacheRegion.SWA,
                    layout.swa_entries,
                )
                for stage in range(3)
            },
        )
        table = torch.arange(
            1, batch * 3 + 1, dtype=torch.int32, device=self.device
        ).view(batch, 3)
        fake = [count == 0 for count in counts]
        table[torch.tensor(fake, device=self.device)] = 0
        valid = (
            torch.arange(width, device=self.device)[None, :]
            < torch.tensor(counts, device=self.device)[:, None]
        )
        inputs = SimpleNamespace(
            input_ids=torch.arange(
                batch * width, dtype=torch.int32, device=self.device
            ),
            input_hiddens=torch.zeros(
                (batch * width, 15360), dtype=torch.bfloat16, device=self.device
            ),
            request_id=torch.tensor(
                [100 + i if not masked else -1 for i, masked in enumerate(fake)],
                dtype=torch.int64,
            ),
            v41_is_fake=torch.tensor(fake, dtype=torch.bool),
            v41_state_ready=torch.tensor(
                [not masked for masked in fake], dtype=torch.bool
            ),
            v41_token_valid=valid.flatten(),
            v41_execution_context=torch.tensor(
                [[0, start, start, start] for start in starts], dtype=torch.int64
            ),
            v41_swa_ranges=torch.tensor(
                [[[max(start - 128, 0), start, 0]] * 43 for start in starts],
                dtype=torch.int64,
            ),
            attention_inputs=SimpleNamespace(
                context_parallel_info=None,
                input_lengths=torch.full((batch,), width, dtype=torch.int32),
                prefix_lengths=torch.tensor(starts, dtype=torch.int32),
                kv_cache_kernel_block_id_device_by_group=[
                    table.clone() for _ in range(3)
                ],
            ),
        )
        return model, inputs, V41DraftFmhaImpl(model, inputs, query_width=width)

    @staticmethod
    def capture(run):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = run()
        return graph, output

    def test_proposal_replay_uses_full_history_and_query_block_without_persistent_writes(
        self,
    ):
        with torch.inference_mode():
            starts, counts = (3, 128, 135, 0), (5, 5, 2, 0)
            model, inputs, context = self.fixture(5, starts, counts)
            batch = len(starts)
            generator = torch.Generator(device=self.device).manual_seed(419)
            query = (
                torch.randn(
                    (batch * 5, 64, 512), generator=generator, device=self.device
                )
                * 0.2
            ).bfloat16()
            query_kv = torch.randn(
                (batch * 5, 512), generator=generator, device=self.device
            ).bfloat16()
            sinks = torch.linspace(-1, 1, 64, device=self.device)
            histories = []
            pages = model._pages[0]
            table = inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group[0]
            for request, (start, count) in enumerate(zip(starts, counts)):
                positions = torch.arange(max(start - 128, 0), start, device=self.device)
                history = torch.randn(
                    (positions.numel(), 512), generator=generator, device=self.device
                ).bfloat16()
                histories.append(history)
                if count:
                    page = int(table[request, (start - 1) // model.layout.reuse_unit])
                    raw = pages.data[page, : pages.entries_per_page * 528].view(
                        pages.entries_per_page, 528
                    )
                    raw[positions % pages.entries_per_page] = _official_gpu(
                        self.reference.kernel, history, CacheRegion.SWA
                    )
            before = pages.data.clone()

            def run():
                context.begin_forward()
                result = draft_flashmla_attention(query, query_kv, context, 0, sinks)
                context.finish_forward()
                return result

            graph, output = self.capture(run)
            torch.testing.assert_close(pages.data, before, rtol=0, atol=0)
            addresses = [
                tensor.data_ptr()
                for tensor in (
                    context.proposal_buffers.pages.data,
                    context.proposal_buffers.encoded,
                    context.proposal_buffers.indices,
                    context.proposal_buffers.lengths,
                )
            ]
            for multiplier in (1.0, -0.5):
                query.mul_(multiplier)
                context.prepare_model_inputs(inputs)
                graph.replay()
                context.check(inputs)
                expected_kernel_style = torch.zeros_like(output)
                for request, count in enumerate(counts):
                    if not count:
                        continue
                    values = torch.cat(
                        (
                            histories[request],
                            query_kv.view(batch, 5, 512)[request, :count],
                        )
                    )
                    kv = _official_gpu(
                        self.reference.kernel, values, CacheRegion.SWA, inplace=True
                    ).float()
                    queries = query.view(batch, 5, 64, 512)[request, :count].float()
                    scores = torch.einsum("qhd,kd->qhk", queries, kv) / math.sqrt(512)
                    # Match the pinned SM100 kernel: FP32 max/exp2 and sink
                    # normalizer, with BF16 score and KV tiles entering PV.
                    scale = math.log2(math.e)
                    maximum = scores.amax(dim=-1, keepdim=True)
                    exp2_scores = torch.exp2((scores - maximum) * scale)
                    exp2_sink = torch.exp2((sinks[None, :, None] - maximum) * scale)
                    weights = exp2_scores.bfloat16().float()
                    output_acc = torch.einsum(
                        "qhk,kd->qhd", weights, kv.bfloat16().float()
                    )
                    expected_kernel_style[request * 5 : request * 5 + count] = (
                        output_acc / (exp2_scores.sum(dim=-1, keepdim=True) + exp2_sink)
                    ).bfloat16()
                torch.testing.assert_close(
                    output, expected_kernel_style, rtol=8e-3, atol=1e-3
                )
                torch.testing.assert_close(pages.data, before, rtol=0, atol=0)
                self.assertEqual(
                    addresses,
                    [
                        tensor.data_ptr()
                        for tensor in (
                            context.proposal_buffers.pages.data,
                            context.proposal_buffers.encoded,
                            context.proposal_buffers.indices,
                            context.proposal_buffers.lengths,
                        )
                    ],
                )
                indices = context.proposal_buffers.indices.view(batch, 5, -1)
                for request, count in enumerate(counts):
                    length = min(starts[request], 128) + count if count else 0
                    self.assertEqual(
                        int(context.proposal_buffers.lengths[request * 5]), length
                    )
                    if count:
                        self.assertEqual(
                            int(indices[request, 0, 0]), (request + 1) * 192
                        )
                    self.assertTrue(bool((indices[request, :, length:] == -1).all()))

    def test_tail_replay_matches_official_projection_bytes_and_masks_padding(self):
        with torch.inference_mode():
            starts, counts = (127, 128, 250, 0), (6, 3, 1, 0)
            model, inputs, context = self.fixture(6, starts, counts)
            draft = V41DraftModel.__new__(V41DraftModel)
            nn.Module.__init__(draft)
            draft.committer = self.reference.module
            inputs.input_hiddens.fill_(float("nan"))
            for request, (start, count) in enumerate(zip(starts, counts)):
                if count:
                    hidden, _ = self.reference.aux(tuple(range(start, start + count)))
                    inputs.input_hiddens[request * 6 : request * 6 + count].copy_(
                        hidden
                    )

            def run():
                result = draft.commit(context)
                context.finish_forward()
                return result

            before = [pages.data.clone() for pages in model._pages.values()]
            graph, _ = self.capture(run)
            for actual, expected in zip(model._pages.values(), before):
                torch.testing.assert_close(actual.data, expected, rtol=0, atol=0)
            context.prepare_model_inputs(inputs)
            expected = [pages.data.clone() for pages in model._pages.values()]
            graph.replay()
            context.check(inputs)
            for request, (start, count) in enumerate(zip(starts, counts)):
                if not count:
                    continue
                positions = torch.arange(start, start + count, device=self.device)
                aux = inputs.input_hiddens[request * 6 : request * 6 + count]
                _, encoded = self.reference.reference_rows(aux, positions)
                for stage, pages in model._pages.items():
                    page_id = int(context.swa[stage].page_ids[request])
                    raw = expected[stage][page_id, : pages.entries_per_page * 528].view(
                        pages.entries_per_page, 528
                    )
                    raw[positions % pages.entries_per_page] = encoded[stage]
                    self.assertEqual(
                        int(context.swa[stage].valid_ends[request]), start + count
                    )
            for stage, pages in model._pages.items():
                torch.testing.assert_close(pages.data, expected[stage], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
