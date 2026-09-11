"""Decode metadata ownership, source identity and changing Graph inputs."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.platform_provider import (
    DefaultDsv4PlatformProvider,
    build_dsv4_decode_metadata,
)
from rtp_llm.platforms.ppu.models.dsv4.ppu_decode_provider import PpuDecodeProvider
from rtp_llm.platforms.ppu.models.dsv4.manifest import DECODE_EXECUTION_OPTIONS


class MetadataFactoryTest(unittest.TestCase):
    def test_shared_rope_options_and_late_materialization(self):
        from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_fmha_impl import (
            DSv4DecodeFmhaImplFP8,
        )

        options = dict(DECODE_EXECUTION_OPTIONS)
        provider = PpuDecodeProvider(options)
        options["DSV4_PPU_DECODE_ROPE"] = "layer"
        with self.assertRaisesRegex(ValueError, "constructed"):
            provider.build_decode_metadata(DSv4DecodeFmhaImplFP8)

        class Attention:
            def __init__(self, **kwargs):
                self.freqs_cis = torch.zeros((8, 2), dtype=torch.complex64)

        first = provider.build_attention(Attention)
        second = provider.build_attention(Attention)
        # Model materialization can replace the constructor's CPU table.
        table = torch.ones((8, 2), dtype=torch.complex64)
        first.freqs_cis = second.freqs_cis = table
        with patch(
            "rtp_llm.platforms.ppu.models.dsv4.ppu_decode_metadata.PpuDecodeMetadataGraph"
        ) as factory:
            provider.build_decode_metadata(DSv4DecodeFmhaImplFP8)
            tables = factory.call_args.kwargs["shared_rope_tables"]
            self.assertEqual(len(tables), 1)
            self.assertIs(tables[0], table)
        del first, second
        with self.assertRaisesRegex(RuntimeError, "released"):
            provider.build_decode_metadata(DSv4DecodeFmhaImplFP8)

    def test_shared_rope_rejects_a_replaced_attention_source(self):
        from rtp_llm.platforms.ppu.models.dsv4.ppu_decode_attention import (
            decode_attention_overlap,
        )

        old = torch.ones((8, 2), dtype=torch.complex64)
        attention = SimpleNamespace(
            freqs_cis=old.clone(), _ensure_freqs_cis_bound=lambda: None
        )
        metadata = SimpleNamespace(
            start_pos=torch.zeros(1, dtype=torch.int32),
            position_ids=torch.zeros(1, dtype=torch.int32),
            rope_freqs_by_source={id(old): old[:1]},
        )
        with self.assertRaisesRegex(RuntimeError, "RoPE source changed"):
            decode_attention_overlap(attention, torch.zeros((1, 1, 4)), metadata, {})

    def test_default_and_explicit_factory_contract(self):
        value = object()
        self.assertIs(
            build_dsv4_decode_metadata(
                lambda: value, platform_provider=DefaultDsv4PlatformProvider()
            ),
            value,
        )
        with self.assertRaisesRegex(ValueError, "FP8 Decode factory"):
            build_dsv4_decode_metadata(
                lambda: value,
                platform_provider=PpuDecodeProvider(DECODE_EXECUTION_OPTIONS),
            )


def _tensors(value, prefix=""):
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _tensors(item, prefix + "/" + str(key))


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_name() == "ZW-M890P",
    "requires a PPU M890P",
)
class MetadataGraphTest(unittest.TestCase):
    @torch.inference_mode()
    def test_actual_model_factory_and_dynamic_metadata(self):
        self._check_model_metadata("graph")

    @torch.inference_mode()
    def test_actual_model_factory_and_fused_metadata(self):
        self._check_model_metadata("graph_fused")

    @torch.inference_mode()
    def test_shared_rope_dynamic_positions_and_batch_ownership(self):
        self._check_model_metadata("graph_fused", shared_rope=True)

    def _check_model_metadata(self, mode, shared_rope=False):
        from rtp_llm.models_py.model_desc.deepseek_v4_model import DeepSeekV4Model
        from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_fmha_impl import (
            DSv4DecodeFmhaImplFP8,
        )
        from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
            CSA_KV,
            CSA_STATE,
            HCA_KV,
            HCA_STATE,
            INDEXER_KV,
            INDEXER_STATE,
            SWA_KV,
        )
        from rtp_llm.platforms.ppu.models.dsv4.ppu_decode_metadata import (
            PpuDecodeMetadataGraph,
        )

        specs = {
            SWA_KV: (128, 16384, 1),
            CSA_KV: (64, 256, 64),
            HCA_KV: (2, 256, 64),
            INDEXER_KV: (64, 256, 64),
            CSA_STATE: (8, 16384, 1),
            HCA_STATE: (128, 16384, 1),
            INDEXER_STATE: (8, 16384, 1),
        }
        constants = {
            "/req_id_per_token",
            "/req_id_per_token_long",
            "/decode_cu_seq_per_req",
            *(
                "/pool_write_slot_mappings/" + tag
                for tag in (CSA_STATE, HCA_STATE, INDEXER_STATE)
            ),
        }
        torch.manual_seed(890931)
        rope_tables = []
        if shared_rope:
            rope_tables = [
                torch.randn((16384, 32), device="cuda", dtype=torch.complex64),
                torch.randn((16384, 64), device="cuda", dtype=torch.complex64)[:, ::2],
            ]
        previous_batches = []
        for batch in (1, 3, 8, 32, 128):
            positions = torch.zeros(batch, dtype=torch.int32, pin_memory=True)
            inputs = {
                tag: SimpleNamespace(
                    sequence_lengths=positions,
                    input_lengths=torch.ones(batch, dtype=torch.int32),
                    is_prefill=False,
                    is_target_verify=False,
                    # Strided sources test that identity includes layout.
                    kv_cache_kernel_block_id_device=torch.ones(
                        (batch, count * 2), device="cuda", dtype=torch.int32
                    )[:, ::2],
                )
                for tag, (_, _, count) in specs.items()
            }
            if shared_rope:
                provider = PpuDecodeProvider(DECODE_EXECUTION_OPTIONS)
            else:
                # Keep unfused state-slot and per-layer RoPE numerical references
                # without exposing additional production provider modes.
                provider = SimpleNamespace(
                    capabilities=PpuDecodeProvider.capabilities,
                    build_decode_metadata=lambda factory, *args, **kwargs: PpuDecodeMetadataGraph(
                        *args, fused_state_slots=mode == "graph_fused", **kwargs
                    ),
                )
            attention_owners = []
            if shared_rope:

                class Attention:
                    def __init__(self, freqs_cis, **kwargs):
                        self.freqs_cis = freqs_cis

                attention_owners = [
                    provider.build_attention(Attention, freqs_cis=table)
                    for table in (*rope_tables, rope_tables[0])
                ]
            model = SimpleNamespace(
                _platform_provider=provider,
                kv_cache=SimpleNamespace(group_tags=list(specs)),
                fp8_kv_cache=True,
                _should_capture_cuda_graph=lambda *args: True,
                v4=SimpleNamespace(
                    embed=SimpleNamespace(weight=torch.empty(1, device="cuda"))
                ),
                _v4_args=SimpleNamespace(
                    max_seq_len=16384,
                    window_size=128,
                    head_dim=512,
                    compress_ratios=[0, 4, 128],
                    n_layers=3,
                    index_topk=512,
                ),
            )
            with patch(
                "rtp_llm.models_py.model_desc.deepseek_v4_model.build_paged_pool_specs",
                return_value=specs,
            ):
                candidate = DeepSeekV4Model.prepare_fmha_impl(
                    model, SimpleNamespace(attention_inputs=inputs), is_cuda_graph=True
                )
            self.assertIsInstance(candidate, PpuDecodeMetadataGraph)
            self.assertIsNone(candidate._metadata_graph)
            if mode == "graph_fused":
                from rtp_llm.platforms.ppu.kernels.ppu_decode_state_slots import (
                    update_compressor_state_slots,
                )

                self.assertIs(
                    candidate._state_slot_updater, update_compressor_state_slots
                )
            else:
                self.assertIsNone(candidate._state_slot_updater)
            reference = DSv4DecodeFmhaImplFP8(
                candidate.config, candidate.device, inputs[SWA_KV]
            )
            # Construction preserves the FlashMLA capture schedule's full widths.
            self.assertTrue(
                torch.equal(
                    candidate.metadata.swa_topk_length,
                    torch.full_like(candidate.metadata.swa_topk_length, 128),
                )
            )
            actual = dict(_tensors(vars(candidate.metadata)))
            pointers = {name: tensor.data_ptr() for name, tensor in actual.items()}
            graph = None
            for step, position in enumerate(
                (
                    0,
                    1,
                    2,
                    3,
                    4,
                    126,
                    127,
                    128,
                    129,
                    255,
                    256,
                    8191,
                    8192,
                    16383,
                    16384,
                    -1,
                )
            ):
                positions.copy_(
                    torch.arange(batch, dtype=torch.int32) * (step % 3) + position
                )
                for value in inputs.values():
                    table = value.kv_cache_kernel_block_id_device
                    table.random_(1, max(2, batch * table.shape[1] + 1))
                    if step % 3 == 0:
                        table[::2].zero_()
                    elif step % 3 == 1:
                        table[:, -1].fill_(-1)
                reference.prepare_cuda_graph(inputs)
                for name, tensor in actual.items():
                    if name not in constants:
                        tensor.fill_(-99)
                candidate.prepare_cuda_graph(inputs)
                if graph is None:
                    graph = candidate._metadata_graph
                self.assertIs(candidate._metadata_graph, graph)
                for name, expected in _tensors(vars(reference.metadata)):
                    self.assertEqual(actual[name].data_ptr(), pointers[name])
                    self.assertTrue(
                        torch.equal(actual[name], expected), (batch, position, name)
                    )
                for table in rope_tables:
                    name = "/rope_freqs_by_source/" + str(id(table))
                    self.assertEqual(actual[name].data_ptr(), pointers[name])
                    self.assertTrue(
                        torch.equal(
                            actual[name],
                            table.index_select(0, reference.metadata.position_ids_long),
                        ),
                        (batch, position, "shared RoPE"),
                    )
            if shared_rope:
                self.assertEqual(len(candidate.metadata.rope_freqs_by_source), 2)
                for previous, saved in previous_batches:
                    for key, value in saved.items():
                        prior = previous.metadata.rope_freqs_by_source[key]
                        self.assertNotEqual(
                            prior.data_ptr(),
                            candidate.metadata.rope_freqs_by_source[key].data_ptr(),
                        )
                        self.assertTrue(torch.equal(prior, value))
                previous_batches.append(
                    (
                        candidate,
                        {
                            key: value.clone()
                            for key, value in candidate.metadata.rope_freqs_by_source.items()
                        },
                    )
                )
                key = id(rope_tables[0])
                original_rows = candidate.metadata.rope_freqs_by_source[key]
                candidate.metadata.rope_freqs_by_source[key] = original_rows.clone()
                with self.assertRaisesRegex(ValueError, "RoPE table or output storage"):
                    candidate.prepare_cuda_graph(inputs)
                candidate.metadata.rope_freqs_by_source[key] = original_rows
                rope_tables[0].transpose_(0, 1)
                with self.assertRaisesRegex(ValueError, "RoPE table or output storage"):
                    candidate.prepare_cuda_graph(inputs)
                rope_tables[0].transpose_(0, 1)
            # Replacing a table with an equal-valued allocation must fail before
            # replay can keep reading its stale captured address.
            original = inputs[SWA_KV].kv_cache_kernel_block_id_device
            inputs[SWA_KV].kv_cache_kernel_block_id_device = original.clone()
            with self.assertRaisesRegex(ValueError, "storage changed"):
                candidate.prepare_cuda_graph(inputs)
            inputs[SWA_KV].kv_cache_kernel_block_id_device = original
            for value in inputs.values():
                value.sequence_lengths = positions.to(torch.int64)
            with self.assertRaisesRegex(ValueError, "int32 batch"):
                candidate.prepare_cuda_graph(inputs)
            for value in inputs.values():
                value.sequence_lengths = positions
            with self.assertRaisesRegex(ValueError, "every configured cache tag"):
                candidate.prepare_cuda_graph({SWA_KV: inputs[SWA_KV]})


if __name__ == "__main__":
    unittest.main()
