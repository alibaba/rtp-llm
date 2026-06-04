from unittest import TestCase, main

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MultimodalOutputPB,
    TensorPB,
)
from rtp_llm.metrics.kmonitor_metric_reporter import GaugeMetrics
from rtp_llm.multimodal.vit_metrics import (
    collect_vit_preprocess_metrics,
    record_vit_preprocess_value,
    vit_preprocess_timer,
)
from rtp_llm.server.vit_rpc_server import _tensor_pb_bytes


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


if __name__ == "__main__":
    main()
