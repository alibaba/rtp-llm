from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MultimodalOutputPB,
    TensorPB,
)
from rtp_llm.metrics.kmonitor_metric_reporter import GaugeMetrics
from rtp_llm.multimodal.vit_metrics import (
    collect_vit_preprocess_metrics,
    record_vit_preprocess_value,
    video_resized_pixel_count,
    vit_preprocess_timer,
)
from rtp_llm.server.vit_rpc_server import _report_output_metrics, _tensor_pb_bytes


class VitMetricsTest(TestCase):
    def test_tensor_pb_bytes_counts_data_fields(self):
        tensor = TensorPB(
            fp32_data=b"1234",
            int32_data=b"12",
            fp16_data=b"1",
            bf16_data=b"",
        )
        self.assertEqual(_tensor_pb_bytes(tensor), 7)

    def test_multimodal_output_bytes_and_split_size_are_available(self):
        output = MultimodalOutputPB(
            multimodal_embedding=TensorPB(bf16_data=b"1234"),
            split_size=[3, 5],
        )
        self.assertGreater(output.ByteSize(), 0)
        self.assertEqual(sum(output.split_size), 8)

    def test_report_output_metrics_counts_extra_input_bytes(self):
        output = MultimodalOutputPB(
            multimodal_embedding=TensorPB(bf16_data=b"1234"),
            multimodal_pos_id=TensorPB(int32_data=b"12345678"),
            split_size=[3, 5],
        )
        output.multimodal_extra_input.extend(
            [TensorPB(fp16_data=b"12"), TensorPB(bf16_data=b"123")]
        )

        with patch("rtp_llm.server.vit_rpc_server.kmonitor.report") as report:
            _report_output_metrics(output, {"source": "test"})

        samples = {call.args[0]: call.args[1] for call in report.call_args_list}
        self.assertEqual(report.call_count, 5)
        self.assertGreater(samples[GaugeMetrics.VIT_RPC_RESPONSE_BYTES_METRIC], 0)
        self.assertEqual(samples[GaugeMetrics.VIT_RESPONSE_EMBEDDING_BYTES_METRIC], 4)
        self.assertEqual(samples[GaugeMetrics.VIT_RESPONSE_POS_BYTES_METRIC], 8)
        self.assertEqual(samples[GaugeMetrics.VIT_RESPONSE_DEEPSTACK_BYTES_METRIC], 5)
        self.assertEqual(samples[GaugeMetrics.VIT_OUTPUT_TOKEN_COUNT_METRIC], 8)

    def test_collect_vit_preprocess_metrics_records_values_and_timers(self):
        with collect_vit_preprocess_metrics() as metrics:
            record_vit_preprocess_value(
                GaugeMetrics.VIT_RESIZED_PIXEL_COUNT_METRIC, 1024
            )
            with vit_preprocess_timer(GaugeMetrics.VIT_IMAGE_RESIZE_RT_US_METRIC):
                pass

        names = [sample.metric for sample in metrics.samples]
        self.assertIn(GaugeMetrics.VIT_RESIZED_PIXEL_COUNT_METRIC, names)
        self.assertIn(GaugeMetrics.VIT_IMAGE_RESIZE_RT_US_METRIC, names)

    def test_video_resized_pixel_count_includes_all_frames(self):
        self.assertEqual(video_resized_pixel_count(8, 336, 448), 8 * 336 * 448)

    def test_qwen3_video_preprocess_passes_metrics_tags_to_loader(self):
        import torch

        from rtp_llm.multimodal.multimodal_mixins import qwen3_vl_mixin
        from rtp_llm.utils.base_model_datatypes import MMUrlType

        mm_input = SimpleNamespace(
            mm_type=MMUrlType.VIDEO,
            url="memory://video",
            mm_preprocess_config=SimpleNamespace(),
        )
        vit_config = SimpleNamespace(download_headers={})
        processor = SimpleNamespace(
            video_processor=MagicMock(
                return_value={
                    "pixel_values_videos": torch.zeros((1, 3, 10, 12)),
                    "video_grid_thw": torch.tensor([[1, 1, 1]]),
                }
            )
        )
        video = torch.zeros((2, 3, 10, 12))

        with patch.object(
            qwen3_vl_mixin, "get_bytes_io_from_url", return_value=b"video"
        ), patch.object(
            qwen3_vl_mixin.Qwen3_VLImageEmbedding,
            "load_video",
            return_value=video,
        ) as load_video:
            pixel_values, video_grid_thw = (
                qwen3_vl_mixin.Qwen3_VLImageEmbedding.preprocess_input(
                    [mm_input], vit_config, processor
                )
            )

        self.assertEqual(tuple(pixel_values.shape), (1, 3, 10, 12))
        self.assertEqual(video_grid_thw.tolist(), [[1, 1, 1]])
        load_video.assert_called_once_with(
            b"video",
            mm_input.mm_preprocess_config,
            vit_metrics_tags={"model": "qwen3_vl", "mm_type": "video"},
        )
        processor.video_processor.assert_called_once_with(
            video, return_tensors="pt", do_resize=True
        )

    def test_qwen3_image_preprocess_uses_image_media_tag(self):
        import torch

        from rtp_llm.multimodal.multimodal_mixins import qwen3_vl_mixin
        from rtp_llm.utils.base_model_datatypes import MMUrlType

        image = object()
        mm_input = SimpleNamespace(
            mm_type=MMUrlType.IMAGE,
            url="memory://image",
            mm_preprocess_config=SimpleNamespace(
                height=-1, width=-1, min_pixels=-1, max_pixels=-1
            ),
        )
        vit_config = SimpleNamespace(download_headers={})
        processor = SimpleNamespace(
            image_processor=MagicMock(
                return_value={
                    "pixel_values": torch.zeros((1, 3, 10, 12)),
                    "image_grid_thw": torch.tensor([[1, 1, 1]]),
                }
            )
        )

        with patch.object(
            qwen3_vl_mixin, "get_bytes_io_from_url", return_value=b"image"
        ), patch.object(qwen3_vl_mixin.Image, "open", return_value=image):
            with collect_vit_preprocess_metrics() as metrics:
                qwen3_vl_mixin.Qwen3_VLImageEmbedding.preprocess_input(
                    [mm_input], vit_config, processor
                )

        samples = {sample.metric: sample for sample in metrics.samples}
        self.assertEqual(
            samples[GaugeMetrics.VIT_IMAGE_FETCH_RT_US_METRIC].tags,
            {"model": "qwen3_vl", "mm_type": "image"},
        )
        self.assertEqual(
            samples[GaugeMetrics.VIT_IMAGE_DECODE_RT_US_METRIC].tags,
            {"model": "qwen3_vl", "mm_type": "image"},
        )
        self.assertEqual(
            samples[GaugeMetrics.VIT_IMAGE_PROCESSOR_RT_US_METRIC].tags,
            {"model": "qwen3_vl", "mm_type": "image"},
        )

    def test_qwen2_image_preprocess_uses_image_media_tag(self):
        import torch

        from rtp_llm.multimodal.multimodal_mixins.qwen2_vl import qwen2_vl_mixin
        from rtp_llm.utils.base_model_datatypes import MMUrlType

        image = object()
        mm_input = SimpleNamespace(
            mm_type=MMUrlType.IMAGE,
            url="memory://image",
            mm_preprocess_config=SimpleNamespace(),
        )
        vit_config = SimpleNamespace(download_headers={})
        processor = MagicMock(
            return_value={
                "pixel_values": torch.zeros((1, 3, 10, 12)),
                "image_grid_thw": torch.tensor([[1, 1, 1]]),
            }
        )

        with patch.object(
            qwen2_vl_mixin, "get_bytes_io_from_url", return_value=b"image"
        ), patch.object(
            qwen2_vl_mixin.Qwen2_VLImageEmbedding,
            "load_image",
            return_value=image,
        ) as load_image:
            with collect_vit_preprocess_metrics() as metrics:
                qwen2_vl_mixin.Qwen2_VLImageEmbedding.preprocess_input(
                    [mm_input], vit_config, processor
                )

        expected_tags = {"model": "qwen2_vl", "mm_type": "image"}
        load_image.assert_called_once_with(
            b"image", mm_input.mm_preprocess_config, vit_metrics_tags=expected_tags
        )
        samples = {sample.metric: sample for sample in metrics.samples}
        self.assertEqual(
            samples[GaugeMetrics.VIT_IMAGE_FETCH_RT_US_METRIC].tags, expected_tags
        )
        self.assertEqual(
            samples[GaugeMetrics.VIT_IMAGE_PROCESSOR_RT_US_METRIC].tags,
            expected_tags,
        )

    def test_qwen2_5_image_preprocess_passes_model_and_media_tags(self):
        import torch

        from rtp_llm.multimodal.multimodal_mixins.qwen2_5_vl import qwen2_5_vl_mixin
        from rtp_llm.utils.base_model_datatypes import MMUrlType

        image = object()
        mm_input = SimpleNamespace(
            mm_type=MMUrlType.IMAGE,
            url="memory://image",
            mm_preprocess_config=SimpleNamespace(),
        )
        vit_config = SimpleNamespace(download_headers={})
        processor = MagicMock(
            return_value={
                "pixel_values": torch.zeros((1, 3, 10, 12)),
                "image_grid_thw": torch.tensor([[1, 1, 1]]),
            }
        )

        with patch.object(
            qwen2_5_vl_mixin, "get_bytes_io_from_url", return_value=b"image"
        ), patch.object(
            qwen2_5_vl_mixin.Qwen2_VLImageEmbedding,
            "load_image",
            return_value=image,
        ) as load_image:
            pixel_values, image_grid_thw = (
                qwen2_5_vl_mixin.Qwen2_5_VLImageEmbedding.preprocess_input(
                    [mm_input], vit_config, processor
                )
            )

        self.assertEqual(tuple(pixel_values.shape), (1, 3, 10, 12))
        self.assertEqual(image_grid_thw.tolist(), [[1, 1, 1]])
        load_image.assert_called_once_with(
            b"image",
            mm_input.mm_preprocess_config,
            vit_metrics_tags={"model": "qwen2_5_vl", "mm_type": "image"},
        )
        processor.assert_called_once_with(
            images=image, videos=None, return_tensors="pt"
        )

    def test_qwen2_5_video_preprocess_passes_model_and_media_tags(self):
        import torch

        from rtp_llm.multimodal.multimodal_mixins.qwen2_5_vl import qwen2_5_vl_mixin
        from rtp_llm.utils.base_model_datatypes import MMUrlType

        video = torch.zeros((2, 3, 10, 12))
        mm_input = SimpleNamespace(
            mm_type=MMUrlType.VIDEO,
            url="memory://video",
            mm_preprocess_config=SimpleNamespace(),
        )
        vit_config = SimpleNamespace(download_headers={})
        processor = MagicMock(
            return_value={
                "pixel_values_videos": torch.zeros((1, 3, 10, 12)),
                "video_grid_thw": torch.tensor([[1, 1, 1]]),
            }
        )

        with patch.object(
            qwen2_5_vl_mixin, "get_bytes_io_from_url", return_value=b"video"
        ), patch.object(
            qwen2_5_vl_mixin.Qwen2_5_VLImageEmbedding,
            "load_video",
            return_value=video,
        ) as load_video:
            pixel_values, video_grid_thw = (
                qwen2_5_vl_mixin.Qwen2_5_VLImageEmbedding.preprocess_input(
                    [mm_input], vit_config, processor
                )
            )

        self.assertEqual(tuple(pixel_values.shape), (1, 3, 10, 12))
        self.assertEqual(video_grid_thw.tolist(), [[1, 1, 1]])
        load_video.assert_called_once_with(
            b"video",
            mm_input.mm_preprocess_config,
            vit_metrics_tags={"model": "qwen2_5_vl", "mm_type": "video"},
        )
        processor.assert_called_once_with(
            images=None, videos=video, return_tensors="pt"
        )

    def test_qwen2_video_load_records_decode_resize_and_pixels(self):
        import torch

        from rtp_llm.multimodal.multimodal_mixins.qwen2_vl import qwen2_vl_mixin
        from rtp_llm.multimodal.multimodal_mixins.qwen2_vl.qwen2_vl_mixin import (
            Qwen2_VLImageEmbedding,
        )

        class FakeBatch:
            def __init__(self, frame_count):
                self.frame_count = frame_count

            def asnumpy(self):
                return [
                    [[[0, 0, 0] for _ in range(6)] for _ in range(4)]
                    for _ in range(self.frame_count)
                ]

        class FakeVideoReader:
            def __init__(self, *args, **kwargs):
                pass

            def __len__(self):
                return 8

            def get_avg_fps(self):
                return 1

            def __getitem__(self, idx):
                return SimpleNamespace(shape=(4, 6, 3))

            def get_batch(self, idx):
                return FakeBatch(len(idx))

        class Config:
            fps = 1
            min_frames = 2
            max_frames = 8
            min_pixels = -1
            max_pixels = -1
            height = -1
            width = -1

        tags = {"model": "qwen2_vl", "mm_type": "video"}

        def fake_resize(video, size, interpolation=None, antialias=None):
            return torch.zeros((video.shape[0], 3, size[0], size[1]))

        with patch.object(qwen2_vl_mixin, "VideoReader", FakeVideoReader), patch.object(
            qwen2_vl_mixin, "cpu", lambda _: "cpu"
        ), patch.object(
            qwen2_vl_mixin, "smart_resize", return_value=(10, 12)
        ), patch.object(
            qwen2_vl_mixin.transforms.functional,
            "resize",
            side_effect=fake_resize,
        ):
            with collect_vit_preprocess_metrics() as metrics:
                video = Qwen2_VLImageEmbedding.load_video(b"video", Config())

        self.assertEqual(tuple(video.shape[-2:]), (10, 12))
        samples = {sample.metric: sample for sample in metrics.samples}
        self.assertIn(GaugeMetrics.VIT_IMAGE_DECODE_RT_US_METRIC, samples)
        self.assertIn(GaugeMetrics.VIT_IMAGE_RESIZE_RT_US_METRIC, samples)
        self.assertEqual(
            samples[GaugeMetrics.VIT_RESIZED_PIXEL_COUNT_METRIC].value,
            video.shape[0] * 10 * 12,
        )
        self.assertEqual(
            samples[GaugeMetrics.VIT_RESIZED_PIXEL_COUNT_METRIC].tags, tags
        )

    def test_qwen3_video_load_records_decode_resize_and_pixels(self):
        import torch

        from rtp_llm.multimodal.multimodal_mixins.qwen2_5_vl import qwen2_5_vl_mixin
        from rtp_llm.multimodal.multimodal_mixins.qwen2_5_vl.qwen2_5_vl_mixin import (
            Qwen2_5_VLImageEmbedding,
        )

        class FakeBatch:
            def __init__(self, frame_count):
                self.frame_count = frame_count

            def asnumpy(self):
                return [
                    [[[0, 0, 0] for _ in range(6)] for _ in range(4)]
                    for _ in range(self.frame_count)
                ]

        class FakeVideoReader:
            def __init__(self, *args, **kwargs):
                pass

            def __len__(self):
                return 8

            def get_avg_fps(self):
                return 1

            def __getitem__(self, idx):
                return SimpleNamespace(shape=(4, 6, 3))

            def get_batch(self, idx):
                return FakeBatch(len(idx))

        class Config:
            fps = 1
            min_frames = 2
            max_frames = 8
            min_pixels = -1
            max_pixels = -1
            height = -1
            width = -1

        tags = {"model": "qwen3_vl", "mm_type": "video"}

        def fake_resize(video, size, interpolation=None, antialias=None):
            return torch.zeros((video.shape[0], 3, size[0], size[1]))

        with patch.object(
            qwen2_5_vl_mixin, "VideoReader", FakeVideoReader
        ), patch.object(qwen2_5_vl_mixin, "cpu", lambda _: "cpu"), patch.object(
            qwen2_5_vl_mixin, "smart_resize", return_value=(10, 12)
        ), patch.object(
            qwen2_5_vl_mixin.transforms.functional,
            "resize",
            side_effect=fake_resize,
        ):
            with collect_vit_preprocess_metrics() as metrics:
                video = Qwen2_5_VLImageEmbedding.load_video(
                    b"video", Config(), vit_metrics_tags=tags
                )

        self.assertEqual(tuple(video.shape[-2:]), (10, 12))
        samples = {sample.metric: sample for sample in metrics.samples}
        self.assertIn(GaugeMetrics.VIT_IMAGE_DECODE_RT_US_METRIC, samples)
        self.assertIn(GaugeMetrics.VIT_IMAGE_RESIZE_RT_US_METRIC, samples)
        self.assertEqual(
            samples[GaugeMetrics.VIT_RESIZED_PIXEL_COUNT_METRIC].value,
            video.shape[0] * 10 * 12,
        )
        self.assertEqual(
            samples[GaugeMetrics.VIT_RESIZED_PIXEL_COUNT_METRIC].tags, tags
        )


if __name__ == "__main__":
    main()
