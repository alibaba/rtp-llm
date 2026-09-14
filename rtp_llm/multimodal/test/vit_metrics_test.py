from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

from rtp_llm.metrics.kmonitor_metric_reporter import GaugeMetrics
from rtp_llm.multimodal.vit_metrics import (
    collect_vit_preprocess_metrics,
    record_vit_preprocess_value,
    video_resized_pixel_count,
    vit_preprocess_timer,
)


class VitMetricsTest(TestCase):
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

    def test_qwen3_video_preprocess_records_decode_resize_and_processor_metrics(self):
        import torch

        from rtp_llm.multimodal.multimodal_mixins import qwen3_vl_mixin
        from rtp_llm.utils.base_model_datatypes import MMUrlType

        item = SimpleNamespace(
            mm_type=MMUrlType.VIDEO,
            url="memory://video",
            mm_preprocess_config=SimpleNamespace(),
        )
        config = SimpleNamespace(
            download_headers={},
            mm_video_max_file_size_kb=2048,
            mm_video_total_min_pixels=0,
            mm_video_total_max_pixels=0,
        )
        video = torch.zeros((3, 3, 10, 12))
        processor = SimpleNamespace(
            video_processor=MagicMock(
                return_value={
                    "pixel_values_videos": torch.zeros((1, 3, 10, 12)),
                    "video_grid_thw": torch.tensor([[2, 1, 1]]),
                }
            )
        )
        processor.video_processor.size = {
            "shortest_edge": 4096,
            "longest_edge": 25165824,
        }
        with (
            patch.object(
                qwen3_vl_mixin, "get_bytes_io_from_url", return_value=b"video"
            ) as fetch,
            patch.object(
                qwen3_vl_mixin, "decode_video", return_value=(video, None)
            ) as decode,
            patch.object(qwen3_vl_mixin, "resize_video", return_value=video) as resize,
        ):
            with collect_vit_preprocess_metrics() as metrics:
                pixels, grid = qwen3_vl_mixin.Qwen3_VLImageEmbedding.preprocess_input(
                    [item], config, processor
                )
        self.assertEqual(grid.tolist(), [[2, 1, 1]])
        fetch.assert_called_once_with("memory://video", {}, max_file_size_kb=2048)
        decode.assert_called_once_with(b"video", item.mm_preprocess_config)
        resize.assert_called_once_with(
            video,
            item.mm_preprocess_config,
            processor.video_processor,
            {"shortest_edge": 4096, "longest_edge": 25165824},
        )
        processor.video_processor.assert_called_once_with(
            video,
            return_tensors="pt",
            do_resize=False,
            do_sample_frames=False,
            video_metadata=None,
        )
        samples = {sample.metric: sample for sample in metrics.samples}
        for name in (
            GaugeMetrics.VIT_IMAGE_FETCH_RT_US_METRIC,
            GaugeMetrics.VIT_IMAGE_DECODE_RT_US_METRIC,
            GaugeMetrics.VIT_IMAGE_RESIZE_RT_US_METRIC,
            GaugeMetrics.VIT_IMAGE_PROCESSOR_RT_US_METRIC,
            GaugeMetrics.VIT_RESIZED_PIXEL_COUNT_METRIC,
        ):
            self.assertIn(name, samples)
            self.assertEqual(
                samples[name].tags, {"model": "qwen3_vl", "mm_type": "video"}
            )
        self.assertEqual(
            samples[GaugeMetrics.VIT_RESIZED_PIXEL_COUNT_METRIC].value, 3 * 10 * 12
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
        vit_config = SimpleNamespace(
            download_headers={},
            mm_image_max_file_size_kb=1024,
            mm_video_max_file_size_kb=2048,
        )
        processor = SimpleNamespace(
            image_processor=MagicMock(
                return_value={
                    "pixel_values": torch.zeros((1, 3, 10, 12)),
                    "image_grid_thw": torch.tensor([[1, 1, 1]]),
                }
            )
        )

        with (
            patch.object(
                qwen3_vl_mixin, "get_bytes_io_from_url", return_value=b"image"
            ) as get_bytes,
            patch.object(qwen3_vl_mixin.Image, "open", return_value=image),
        ):
            with collect_vit_preprocess_metrics() as metrics:
                qwen3_vl_mixin.Qwen3_VLImageEmbedding.preprocess_input(
                    [mm_input], vit_config, processor
                )

        get_bytes.assert_called_once_with(
            "memory://image",
            {},
            max_file_size_kb=vit_config.mm_image_max_file_size_kb,
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
        vit_config = SimpleNamespace(
            download_headers={},
            mm_image_max_file_size_kb=1024,
            mm_video_max_file_size_kb=2048,
        )
        processor = MagicMock(
            return_value={
                "pixel_values": torch.zeros((1, 3, 10, 12)),
                "image_grid_thw": torch.tensor([[1, 1, 1]]),
            }
        )

        with (
            patch.object(
                qwen2_vl_mixin, "get_bytes_io_from_url", return_value=b"image"
            ) as get_bytes,
            patch.object(
                qwen2_vl_mixin.Qwen2_VLImageEmbedding,
                "load_image",
                return_value=image,
            ) as load_image,
        ):
            with collect_vit_preprocess_metrics() as metrics:
                qwen2_vl_mixin.Qwen2_VLImageEmbedding.preprocess_input(
                    [mm_input], vit_config, processor
                )

        expected_tags = {"model": "qwen2_vl", "mm_type": "image"}
        get_bytes.assert_called_once_with(
            "memory://image",
            {},
            max_file_size_kb=vit_config.mm_image_max_file_size_kb,
        )
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
        vit_config = SimpleNamespace(
            download_headers={},
            mm_image_max_file_size_kb=1024,
            mm_video_max_file_size_kb=2048,
        )
        processor = MagicMock(
            return_value={
                "pixel_values": torch.zeros((1, 3, 10, 12)),
                "image_grid_thw": torch.tensor([[1, 1, 1]]),
            }
        )

        with (
            patch.object(
                qwen2_5_vl_mixin, "get_bytes_io_from_url", return_value=b"image"
            ) as get_bytes,
            patch.object(
                qwen2_5_vl_mixin.Qwen2_VLImageEmbedding,
                "load_image",
                return_value=image,
            ) as load_image,
        ):
            pixel_values, image_grid_thw = (
                qwen2_5_vl_mixin.Qwen2_5_VLImageEmbedding.preprocess_input(
                    [mm_input], vit_config, processor
                )
            )

        get_bytes.assert_called_once_with(
            "memory://image",
            {},
            max_file_size_kb=vit_config.mm_image_max_file_size_kb,
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
        vit_config = SimpleNamespace(
            download_headers={},
            mm_image_max_file_size_kb=1024,
            mm_video_max_file_size_kb=2048,
        )
        processor = MagicMock(
            return_value={
                "pixel_values_videos": torch.zeros((1, 3, 10, 12)),
                "video_grid_thw": torch.tensor([[1, 1, 1]]),
            }
        )

        with (
            patch.object(
                qwen2_5_vl_mixin, "get_bytes_io_from_url", return_value=b"video"
            ) as get_bytes,
            patch.object(
                qwen2_5_vl_mixin.Qwen2_5_VLImageEmbedding,
                "load_video",
                return_value=video,
            ) as load_video,
        ):
            pixel_values, video_grid_thw = (
                qwen2_5_vl_mixin.Qwen2_5_VLImageEmbedding.preprocess_input(
                    [mm_input], vit_config, processor
                )
            )

        get_bytes.assert_called_once_with(
            "memory://video",
            {},
            max_file_size_kb=vit_config.mm_video_max_file_size_kb,
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

        with (
            patch.object(qwen2_vl_mixin, "VideoReader", FakeVideoReader),
            patch.object(qwen2_vl_mixin, "cpu", lambda _: "cpu"),
            patch.object(qwen2_vl_mixin, "smart_resize", return_value=(10, 12)),
            patch.object(
                qwen2_vl_mixin.transforms.functional,
                "resize",
                side_effect=fake_resize,
            ),
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
                return torch.zeros(
                    (self.frame_count, 4, 6, 3), dtype=torch.uint8
                ).numpy()

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

        with (
            patch.object(qwen2_5_vl_mixin, "VideoReader", FakeVideoReader),
            patch.object(qwen2_5_vl_mixin, "cpu", lambda _: "cpu"),
            patch.object(qwen2_5_vl_mixin, "smart_resize", return_value=(10, 12)),
            patch.object(
                qwen2_5_vl_mixin.transforms.functional,
                "resize",
                side_effect=fake_resize,
            ),
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


    def test_qwen3_video_load_falls_back_to_pyav(self):
        import numpy as np
        import torch
        from rtp_llm.multimodal.multimodal_mixins.qwen2_5_vl import qwen2_5_vl_mixin
        from rtp_llm.multimodal.multimodal_mixins.qwen2_5_vl.qwen2_5_vl_mixin import (
            Qwen2_5_VLImageEmbedding,
        )

        class FakeFrame:
            def __init__(self, index):
                self.index = index

            def to_ndarray(self, format):
                self.assert_format = format
                return np.full((4, 6, 3), self.index, dtype=np.uint8)

        class FakeContainer:
            def __init__(self):
                self.stream = SimpleNamespace(
                    frames=8, average_rate=1, guessed_rate=None
                )
                self.streams = SimpleNamespace(video=[self.stream])

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def decode(self, stream):
                self.assert_stream = stream
                return iter(FakeFrame(index) for index in range(8))

        class Config:
            fps = 1
            min_frames = 4
            max_frames = 4
            min_pixels = -1
            max_pixels = -1
            height = -1
            width = -1

        fake_av = SimpleNamespace(open=MagicMock(return_value=FakeContainer()))

        def fake_resize(video, size, interpolation=None, antialias=None):
            return video

        with patch.object(qwen2_5_vl_mixin, "VideoReader", None), patch.object(
            qwen2_5_vl_mixin, "av", fake_av
        ), patch.object(
            qwen2_5_vl_mixin, "smart_resize", return_value=(4, 6)
        ), patch.object(
            qwen2_5_vl_mixin.transforms.functional,
            "resize",
            side_effect=fake_resize,
        ):
            video = Qwen2_5_VLImageEmbedding.load_video(b"video", Config())

        fake_av.open.assert_called_once_with(b"video", mode="r")
        self.assertEqual(tuple(video.shape), (4, 3, 4, 6))
        self.assertEqual(video[:, 0, 0, 0].tolist(), [0, 2, 5, 7])


if __name__ == "__main__":
    main()
