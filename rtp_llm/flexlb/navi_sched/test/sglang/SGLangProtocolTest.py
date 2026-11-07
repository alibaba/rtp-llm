import dataclasses
import unittest
from unittest.mock import patch

import msgspec
from navi_sched.adapters.sglang.SGLangForwardProtocol import SGLangForwardCodec
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.utils import ImageData, VideoData


class SGLangProtocolTest(unittest.TestCase):
    def setUp(self):
        self.codec = SGLangForwardCodec()

    def test_media_references_round_trip_without_fetch(self):
        image = ImageData(
            "https://invalid.example/image.png",
            detail="high",
            max_dynamic_patch=8,
            preprocess_kwargs={"min_pixels": 256},
            content_hash="sha256:" + "a" * 64,
            max_long_side_pixel=2048,
        )
        video = VideoData("https://invalid.example/video.mp4", {"fps": 2})
        request = GenerateReqInput(
            text="Describe this",
            image_data=[image],
            video_data=video,
            sampling_params={"max_new_tokens": 8},
            rid="original",
            http_worker_ipc="ipc:///source/worker",
            lora_id="source-process-id",
        )
        with patch(
            "requests.get", side_effect=AssertionError("source downloaded media")
        ):
            payload = self.codec.encode_request(request)
            result = self.codec.decode_request(payload)
        self.assertLess(len(payload), 1024)
        self.assertEqual(result.text, request.text)
        self.assertEqual(result.rid, "original")
        self.assertEqual(
            dataclasses.asdict(result.image_data[0]), dataclasses.asdict(image)
        )
        self.assertEqual(
            dataclasses.asdict(result.video_data), dataclasses.asdict(video)
        )
        self.assertIsNone(result.http_worker_ipc)
        self.assertIsNone(result.lora_id)

    def test_limits_include_inline_bytes_and_envelope(self):
        codec = SGLangForwardCodec(128)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            codec.encode_request(GenerateReqInput(text="test", image_data=b"a" * 129))
        payload = self.codec.encode_request(GenerateReqInput(text="a" * 129))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            codec.decode_request(payload)
        small = self.codec.encode_request(GenerateReqInput(text="hi"))
        exact = SGLangForwardCodec(len(small))
        self.assertEqual(exact.encode_request(GenerateReqInput(text="hi")), small)
        with self.assertRaises(ValueError):
            SGLangForwardCodec(len(small) - 1).encode_request(
                GenerateReqInput(text="hi")
            )

    def test_rejects_prepared_media_and_process_local_fields(self):
        with self.assertRaisesRegex(ValueError, "raw media"):
            self.codec.encode_request(
                GenerateReqInput(text="hi", image_data={"features": [1, 2]})
            )
        payload = msgspec.msgpack.encode(
            [1, "request", {"text": "hi", "http_worker_ipc": "ipc:///remote"}]
        )
        with self.assertRaisesRegex(ValueError, "process-local"):
            self.codec.decode_request(payload)

    def test_normalized_sampling_and_batch_are_lossless(self):
        request = GenerateReqInput(
            text=["first", "second"],
            rid=["a", "b"],
            sampling_params=[{"temperature": 0, "max_new_tokens": 3}] * 2,
            cache_salt=["one", "two"],
            extra_key=["x", "y"],
        )
        result = self.codec.decode_request(self.codec.encode_request(request))
        result.normalize_batch_and_arguments()
        self.assertEqual(result.rid, ["a", "b"])
        self.assertEqual(result[1].cache_salt, "two")
        self.assertEqual(result[0].extra_key, "x")
        self.assertEqual(result[0].sampling_params["max_new_tokens"], 3)

    def test_response_protocol_and_errors(self):
        response = {
            "text": "hello",
            "meta_info": {"id": "a", "prompt_tokens": 3, "cached_tokens": 2},
            "output_ids": [1, 2],
        }
        self.assertEqual(
            self.codec.decode_response(self.codec.encode_response(response)), response
        )
        for envelope in (
            [2, "response", {}],
            [1, "request", {}],
            [True, "response", {}],
            [1, "response", []],
        ):
            with self.assertRaisesRegex(ValueError, "unsupported"):
                self.codec.decode_response(msgspec.msgpack.encode(envelope))


if __name__ == "__main__":
    unittest.main()
