"""Regressions for host-side stalls in variable-shape ViT batches."""

import io
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from PIL import Image
from triton.runtime.jit import JITFunction

from rtp_llm.models_py.modules.dsv41 import vision_preprocess_triton as preprocess
from rtp_llm.models_py.modules.dsv41._vision_rope_triton import apply_vision_qk_rope
from rtp_llm.models_py.modules.dsv41.vision_residual_add import (
    apply_vision_residual_rmsnorm,
)
from rtp_llm.models_py.utils.cutlass import setup_cutlass_import_path
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41 import (
    deepseek_v41_vision as vision,
)
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_processor import (
    V41ImageProcessorConfig,
    preprocess_image,
)
from rtp_llm.utils.base_model_datatypes import MMUrlType


class DynamicBatchGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this regression")
        if torch.cuda.get_device_capability()[0] != 10:
            raise unittest.SkipTest("V4.1 native kernels require SM10x")
        cls.device = torch.device("cuda", torch.cuda.current_device())
        setup_cutlass_import_path()

    @torch.inference_mode()
    def test_varlen_new_lengths_reuse_startup_warmup(self):
        from flash_attn.cute import interface

        from rtp_llm.models_py.modules.dsv41.vision_attention import (
            apply_vision_fa4_varlen,
            warmup_vision_fa4_varlen,
        )

        warmup_vision_fa4_varlen(self.device)
        with mock.patch.object(
            interface.cute,
            "compile",
            side_effect=AssertionError("new batch compiled FA4"),
        ):
            for lengths in ((1, 7, 15), (513, 1025, 769)):
                rows = sum(lengths)
                qk = torch.randn(rows, 32, 64, device=self.device, dtype=torch.bfloat16)
                qkv = torch.randn(rows, 3072, device=self.device, dtype=torch.bfloat16)
                v = qkv[:, 2048:].view(rows, 16, 64)
                cu = torch.tensor(
                    [0, lengths[0], sum(lengths[:2]), rows],
                    device=self.device,
                    dtype=torch.int32,
                )
                actual = apply_vision_fa4_varlen(
                    qk[:, :16], qk[:, 16:], v, cu, max(lengths)
                )
                self.assertTrue(torch.isfinite(actual).all())

    @torch.inference_mode()
    def test_mixed_attention_avoids_regrouping_and_isolates_images(self):
        from rtp_llm.models_py.modules.dsv41.vision_attention import apply_vision_fa4
        from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.packed_vision import (
            _packed_fa4_attention,
        )

        lengths = [128, 189, 128, 1]
        grids = [(8, 16), (9, 21), (16, 8), (1, 1)]
        rows = sum(lengths)
        qk = torch.randn(rows, 32, 64, device=self.device, dtype=torch.bfloat16)
        q, k = qk[:, :16], qk[:, 16:]
        qkv = torch.randn(rows, 3072, device=self.device, dtype=torch.bfloat16)
        v = qkv[:, 2048:].view(rows, 16, 64)

        def independent_reference():
            parts = [t.split(lengths) for t in (q, k, v)]
            return torch.cat(
                [
                    apply_vision_fa4(*(p[i].unsqueeze(0) for p in parts)).reshape(
                        length, -1
                    )
                    for i, length in enumerate(lengths)
                ]
            )

        expected = independent_reference()
        with mock.patch.object(
            torch,
            "stack",
            side_effect=AssertionError("attention copied each geometry group"),
        ):
            actual = _packed_fa4_attention(q, k, v, grids, lengths)
        self.assertTrue(torch.equal(actual, expected))
        retained = actual.clone()
        v[lengths[0] : lengths[0] + lengths[1]].add_(3)
        changed = _packed_fa4_attention(q, k, v, grids, lengths)
        self.assertTrue(torch.equal(changed, independent_reference()))
        self.assertTrue(torch.equal(actual, retained))
        for index, (previous, current) in enumerate(
            zip(actual.split(lengths), changed.split(lengths))
        ):
            self.assertEqual(torch.equal(previous, current), index != 1)

    @torch.inference_mode()
    def test_varlen_graph_replay_uses_changed_inputs(self):
        from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.packed_vision import (
            _packed_fa4_attention,
            prepare_fa4_cu_seqlens,
        )

        grids, lengths = [(3, 5), (7, 9)], [15, 63]
        values = [
            torch.randn(sum(lengths), 16, 64, device=self.device, dtype=torch.bfloat16)
            for _ in range(3)
        ]
        cu = prepare_fa4_cu_seqlens(grids, self.device)

        def forward():
            return _packed_fa4_attention(*values, grids, lengths, cu_seqlens=cu)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                forward()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = forward()
        torch.cuda.current_stream().wait_stream(stream)
        original = result.clone()
        values[2].add_(2)
        expected = forward()
        graph.replay()
        self.assertTrue(torch.equal(result, expected))
        self.assertFalse(torch.equal(result, original))

    @torch.inference_mode()
    def test_prepared_tables_reject_other_image_geometry(self):
        from rtp_llm.models_py.modules.dsv41.vision_preprocess_tables import (
            prepare_resize_tables,
        )

        tables = prepare_resize_tables((29, 43), (28, 42))
        rgb = torch.zeros(30, 43, 3, device=self.device, dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, "do not match"):
            preprocess.preprocess_rgb_u8(rgb, (28, 42), tables=tables)

    @torch.inference_mode()
    def test_rotary_new_row_counts_do_not_compile(self):
        def run(rows):
            qkv = torch.randn(rows, 3072, device=self.device, dtype=torch.bfloat16)
            q, k, _ = [part.view(rows, 16, 64) for part in qkv.chunk(3, -1)]
            phase = torch.randn(rows, 1, 32, device=self.device)
            cos, sin = phase.cos(), phase.sin()
            actual = apply_vision_qk_rope(q, k, cos, sin)
            self.assertIsNotNone(actual)
            for source, result in zip((q, k), actual):
                a, b = source.float().chunk(2, -1)
                expected = torch.cat((a * cos - b * sin, a * sin + b * cos), -1)
                self.assertTrue(torch.equal(result, expected.bfloat16()))

        run(8)
        with mock.patch.object(
            JITFunction,
            "_do_compile",
            side_effect=AssertionError("new row count compiled RoPE"),
        ):
            for rows in (1, 7, 16, 37, 8193):
                run(rows)

    @torch.inference_mode()
    def test_residual_new_row_counts_do_not_compile(self):
        weight = torch.ones(1024, device=self.device, dtype=torch.bfloat16)

        def run(rows):
            x = torch.randn(rows, 1024, device=self.device, dtype=torch.bfloat16)
            residual = torch.randn_like(x)
            actual = apply_vision_residual_rmsnorm(x, residual, weight, 1e-6)
            self.assertIsNotNone(actual)
            summed, normalized = actual
            self.assertTrue(torch.equal(summed, x + residual))
            values = summed.float()
            expected = (
                values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
            ).bfloat16()
            self.assertLessEqual(
                float((normalized.float() - expected.float()).abs().max()), 0.03125
            )

        run(8)
        with mock.patch.object(
            JITFunction,
            "_do_compile",
            side_effect=AssertionError("new row count compiled RMSNorm"),
        ):
            for rows in (1, 7, 16, 37, 8193):
                run(rows)

    @torch.inference_mode()
    def test_cpu_prepared_tables_survive_upload_and_cache_eviction(self):
        adapter = object.__new__(vision.DeepSeekV41VisionEmbedding)
        torch.nn.Module.__init__(adapter)
        adapter.processor_config = V41ImageProcessorConfig(vision_min_pixels=1764)
        adapter.image_start = torch.nn.Parameter(
            torch.zeros(4, device=self.device, dtype=torch.bfloat16),
            requires_grad=False,
        )
        adapter._rgb_uploader = None
        adapter._staging_max_bytes = 16 * 1024**2
        adapter._resolve_resource_budget = mock.Mock()
        adapter.estimate_work = mock.Mock(
            return_value=SimpleNamespace(fits_within=lambda _: True)
        )
        image = Image.frombytes(
            "RGB", (333, 221), bytes(range(256)) * (333 * 221 * 3 // 256 + 1)
        )
        payload = io.BytesIO()
        image.save(payload, format="PNG")
        expected, _, _, _, _ = preprocess_image(image, adapter.processor_config)
        media = SimpleNamespace(url="test-image", mm_type=MMUrlType.IMAGE)
        with mock.patch.object(
            vision, "load_image_source", return_value=io.BytesIO(payload.getvalue())
        ), mock.patch.object(
            torch.cuda, "init", side_effect=AssertionError("CPU hook initialized CUDA")
        ):
            decoded = adapter.preprocess_input(
                [media],
                SimpleNamespace(download_headers="{}", mm_image_max_file_size_kb=1024),
                processor_config=adapter.processor_config,
            )

        # Retaining the actual tables per item is required: warming an LRU in
        # the CPU hook is insufficient once other image shapes evict them.
        preprocess._coefficients.cache_clear()
        with mock.patch.object(
            preprocess, "_DEVICE_TABLES", preprocess._DeviceTableCache(max_entries=0)
        ), mock.patch.object(
            preprocess,
            "_coefficients",
            side_effect=AssertionError("GPU thread rebuilt CPU resize tables"),
        ):
            staged = adapter.prepare_embedding_batch(
                [decoded, [decoded, decoded]], [MMUrlType.IMAGE] * 2
            )
            self.assertIsNotNone(staged)
            try:
                staged.wait()
                for data in staged.data_list:
                    for item in adapter._image_group(data):
                        actual = adapter._materialize_image(item).patches
                        self.assertTrue(torch.equal(actual.cpu(), expected))
            finally:
                staged.close()


if __name__ == "__main__":
    unittest.main()
