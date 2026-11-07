from __future__ import annotations

import dataclasses
from functools import lru_cache
from typing import Any

import msgspec

PROTOCOL_VERSION = 1
MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
_IMAGE_REFERENCE = 41
_VIDEO_REFERENCE = 42
_LOCAL_REQUEST_FIELDS = frozenset(
    {
        "http_worker_ipc",
        "received_time",
        "lora_id",
        "need_wait_for_mm_inputs",
        "num_items_assigned",
        "encoder_urls",
    }
)


@lru_cache(maxsize=4)
def _request_fields(request_type):
    return {
        field.name: field
        for field in dataclasses.fields(request_type)
        if field.init and field.name not in _LOCAL_REQUEST_FIELDS
    }


class SGLangForwardCodec:
    def __init__(self, max_payload_bytes: int = MAX_PAYLOAD_BYTES):
        if type(max_payload_bytes) is not int or max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be a positive integer")
        self.max_payload_bytes = max_payload_bytes

    def _encode(self, kind: str, body: dict) -> bytes:
        self._check_inline_size(body)
        payload = msgspec.msgpack.encode([PROTOCOL_VERSION, kind, body])
        if len(payload) > self.max_payload_bytes:
            raise ValueError("SGLang forwarding payload exceeds max_payload_bytes")
        return payload

    def _check_inline_size(self, value: Any) -> None:
        remaining = self.max_payload_bytes

        def visit(item):
            nonlocal remaining
            if isinstance(item, str):
                remaining -= len(item.encode("utf-8"))
            elif isinstance(item, (bytes, bytearray, memoryview)):
                remaining -= len(item)
            elif isinstance(item, msgspec.msgpack.Ext):
                remaining -= len(item.data)
            elif isinstance(item, dict):
                for key, child in item.items():
                    visit(key)
                    visit(child)
            elif isinstance(item, (list, tuple)):
                for child in item:
                    visit(child)
            else:
                remaining -= 1
            if remaining < 0:
                raise ValueError("SGLang forwarding payload exceeds max_payload_bytes")

        visit(value)

    def _decode(self, payload: bytes, kind: str) -> dict:
        if len(payload) > self.max_payload_bytes:
            raise ValueError("SGLang forwarding payload exceeds max_payload_bytes")
        value = msgspec.msgpack.decode(payload)
        if (
            not isinstance(value, list)
            or len(value) != 3
            or type(value[0]) is not int
            or value[0] != PROTOCOL_VERSION
            or value[1] != kind
            or not isinstance(value[2], dict)
        ):
            raise ValueError("unsupported SGLang forwarding protocol")
        return value[2]

    def _encode_media(self, value):
        from sglang.srt.utils import ImageData, VideoData

        if isinstance(value, (ImageData, VideoData)):
            # Preserve the URL and processing options without loading its contents.
            fields = dataclasses.asdict(value)
            self._check_inline_size(fields)
            code = (
                _IMAGE_REFERENCE if isinstance(value, ImageData) else _VIDEO_REFERENCE
            )
            return msgspec.msgpack.Ext(code, msgspec.msgpack.encode(fields))
        if isinstance(value, (list, tuple)):
            return [self._encode_media(item) for item in value]
        if value is None or isinstance(value, (str, bytes)):
            return value
        if isinstance(value, dict):
            # Tensor/processor outputs must stay on the node that prepared them.
            raise ValueError(
                "forward raw media references instead of processor outputs"
            )
        raise TypeError("forward media URLs or encoded bytes instead of decoded media")

    def _decode_media(self, value):
        from sglang.srt.utils import ImageData, VideoData

        if isinstance(value, msgspec.msgpack.Ext):
            media_type = {
                _IMAGE_REFERENCE: ImageData,
                _VIDEO_REFERENCE: VideoData,
            }.get(value.code)
            if media_type is None:
                raise ValueError("unsupported SGLang media reference")
            return media_type(**msgspec.msgpack.decode(value.data))
        if isinstance(value, list):
            return [self._decode_media(item) for item in value]
        if value is None or isinstance(value, (str, bytes)):
            return value
        raise ValueError("invalid SGLang media reference")

    def encode_request(self, request: Any) -> bytes:
        from sglang.srt.managers.io_struct import GenerateReqInput

        if not isinstance(request, GenerateReqInput):
            raise TypeError("NaviSched requires GenerateReqInput")
        values = {}
        for name, field in _request_fields(type(request)).items():
            value = getattr(request, name)
            if value is field.default or (
                type(value) in (str, int, float, bool)
                and type(value) is type(field.default)
                and value == field.default
            ):
                continue
            if name in ("image_data", "video_data", "audio_data"):
                value = self._encode_media(value)
            values[name] = value
        return self._encode("request", values)

    def decode_request(self, payload: bytes):
        from sglang.srt.managers.io_struct import GenerateReqInput

        fields = self._decode(payload, "request")
        if fields.keys() - _request_fields(GenerateReqInput).keys():
            raise ValueError("unknown or process-local SGLang request fields")
        for name in ("image_data", "video_data", "audio_data"):
            if name in fields:
                fields[name] = self._decode_media(fields[name])
        return GenerateReqInput(**fields)

    def encode_response(self, response: dict) -> bytes:
        if not isinstance(response, dict):
            raise TypeError("SGLang response must be an object")
        return self._encode("response", response)

    def decode_response(self, payload: bytes) -> dict:
        return self._decode(payload, "response")
