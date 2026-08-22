from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MultimodalOutputPB,
    TensorPB,
)
from rtp_llm.metrics.kmonitor_metric_reporter import GaugeMetrics
from rtp_llm.multimodal.mm_output_transport.base import MMOutputResult
from rtp_llm.multimodal.mm_output_transport.grpc.backend import TRANSPORT_BYTES
from rtp_llm.multimodal.mm_output_transport.metrics import report_output_metrics
from rtp_llm.multimodal.mm_output_transport.rdma.backend import TRANSPORT_RDMA
from rtp_llm.multimodal.vit_metrics import (
    collect_vit_preprocess_metrics,
    record_vit_preprocess_value,
    video_resized_pixel_count,
    vit_preprocess_timer,
)


def _result(transport: str, receipt: MultimodalOutputPB, **payload) -> MMOutputResult:
    return MMOutputResult(
        receipt=receipt,
        transport=transport,
        payload_embedding_bytes=payload.get("embedding", 0),
        payload_pos_bytes=payload.get("pos", 0),
        payload_extra_bytes=payload.get("extra", 0),
    )


class VitMetricsTest(TestCase):
    def test_multimodal_output_bytes_and_split_size_are_available(self):
        output = MultimodalOutputPB(
            multimodal_embedding=TensorPB(bf16_data=b"1234"),
            split_size=[3, 5],
        )
        self.assertGreater(output.ByteSize(), 0)
        self.assertEqual(sum(output.split_size), 8)

    def test_report_output_metrics_takes_payload_bytes_from_the_result(self):
        # The reporter no longer inspects the receipt to work out how many payload bytes went
        # out: each backend hands them over, because only the implementation can count them.
        receipt = MultimodalOutputPB(
            multimodal_embedding=TensorPB(bf16_data=b"1234"),
            multimodal_pos_id=TensorPB(int32_data=b"12345678"),
            split_size=[3, 5],
        )

        with patch(
            "rtp_llm.multimodal.mm_output_transport.metrics.kmonitor.report"
        ) as report:
            report_output_metrics(
                _result(TRANSPORT_BYTES, receipt, embedding=4, pos=8, extra=5)
            )

        samples = {call.args[0]: call.args[1] for call in report.call_args_list}
        self.assertEqual(report.call_count, 5)
        self.assertGreater(samples[GaugeMetrics.VIT_RPC_RESPONSE_BYTES_METRIC], 0)
        self.assertEqual(samples[GaugeMetrics.VIT_RESPONSE_EMBEDDING_BYTES_METRIC], 4)
        self.assertEqual(samples[GaugeMetrics.VIT_RESPONSE_POS_BYTES_METRIC], 8)
        self.assertEqual(samples[GaugeMetrics.VIT_RESPONSE_DEEPSTACK_BYTES_METRIC], 5)
        self.assertEqual(samples[GaugeMetrics.VIT_OUTPUT_TOKEN_COUNT_METRIC], 8)
        for call in report.call_args_list:
            self.assertEqual(
                call.args[2], {"source": "vit_server", "transport": TRANSPORT_BYTES}
            )

    def test_transport_tag_comes_from_the_result_not_the_receipt_shape(self):
        # On the RDMA path VIT_RPC_RESPONSE_BYTES still covers only the receipt's wire size,
        # and the tag is stated by the backend rather than inferred from output_rdma_slots.
        receipt = MultimodalOutputPB(split_size=[2])
        receipt.output_rdma_slots.add(handle="one", nbytes=35)

        with patch(
            "rtp_llm.multimodal.mm_output_transport.metrics.kmonitor.report"
        ) as report:
            report_output_metrics(
                _result(TRANSPORT_RDMA, receipt, embedding=16, pos=8, extra=11)
            )

        values = [(call.args[1], call.args[2]) for call in report.call_args_list]
        rdma_tags = {"source": "vit_server", "transport": TRANSPORT_RDMA}
        self.assertIn((16, rdma_tags), values)
        self.assertIn((8, rdma_tags), values)
        self.assertIn((11, rdma_tags), values)
        self.assertIn((receipt.ByteSize(), rdma_tags), values)

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
        vit_config = SimpleNamespace(
            download_headers={},
            mm_image_max_file_size_kb=1024,
            mm_video_max_file_size_kb=2048,
        )
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
        ) as get_bytes, patch.object(
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
        get_bytes.assert_called_once_with(
            "memory://video",
            {},
            max_file_size_kb=vit_config.mm_video_max_file_size_kb,
        )
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

        with patch.object(
            qwen3_vl_mixin, "get_bytes_io_from_url", return_value=b"image"
        ) as get_bytes, patch.object(qwen3_vl_mixin.Image, "open", return_value=image):
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

        with patch.object(
            qwen2_vl_mixin, "get_bytes_io_from_url", return_value=b"image"
        ) as get_bytes, patch.object(
            qwen2_vl_mixin.Qwen2_VLImageEmbedding,
            "load_image",
            return_value=image,
        ) as load_image:
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

        with patch.object(
            qwen2_5_vl_mixin, "get_bytes_io_from_url", return_value=b"image"
        ) as get_bytes, patch.object(
            qwen2_5_vl_mixin.Qwen2_VLImageEmbedding,
            "load_image",
            return_value=image,
        ) as load_image:
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

        with patch.object(
            qwen2_5_vl_mixin, "get_bytes_io_from_url", return_value=b"video"
        ) as get_bytes, patch.object(
            qwen2_5_vl_mixin.Qwen2_5_VLImageEmbedding,
            "load_video",
            return_value=video,
        ) as load_video:
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
