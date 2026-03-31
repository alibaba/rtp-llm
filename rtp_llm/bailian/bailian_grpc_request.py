"""Parse Bailian gRPC ModelInferRequest (wire format from predict_v2.proto): input_ids and sampling."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

# Defaults aligned with ``rtp_llm.config.generate_config.GenerateConfig`` (same field names).


@dataclass(frozen=True)
class OtherParams:
    """其他请求参数（与 ``input_ids``、sampling 张量并列，由 ``parse_other_params`` 填充）。"""

    return_input_ids: bool = False


@dataclass(frozen=True)
class SamplingParams:
    """Sampling / generation options from ``request.inputs`` (+ legacy ``top_k`` in ``request.parameters``)."""

    max_new_tokens: int = 32000
    num_return_sequences: int = 0
    top_p: float = 1.0
    top_k: int = 0
    temperature: float = 1.0
    min_new_tokens: int = 0
    random_seed: int | None = None
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_words_list: tuple[tuple[int, ...], ...] = field(default_factory=tuple)

    @property
    def n(self) -> int:
        """Alias for ``num_return_sequences`` (same as HuggingFace ``n``)."""
        return self.num_return_sequences

    def stop_words_list_py(self) -> list[list[int]]:
        """``GenerateConfig.stop_words_list`` shape: ``List[List[int]]``."""
        return [list(group) for group in self.stop_words_list]

    def to_generate_config(self, *, other: OtherParams | None = None):
        """Build ``GenerateConfig`` from sampling fields; ``other`` 提供 ``return_input_ids`` 等。"""
        from rtp_llm.config.generate_config import GenerateConfig

        return_input_ids = other.return_input_ids if other is not None else False
        return GenerateConfig(
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=self.num_return_sequences,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
            min_new_tokens=self.min_new_tokens,
            random_seed=self.random_seed,
            repetition_penalty=self.repetition_penalty,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop_words_list=self.stop_words_list_py(),
            return_input_ids=return_input_ids,
            is_streaming=True,
        )


def _find_input_raw(request, tensor_name: str):
    """Return ``(InferInputTensor | None, raw bytes | None)`` for ``tensor_name``."""
    for i, inp in enumerate(request.inputs):
        if inp.name != tensor_name:
            continue
        if i >= len(request.raw_input_contents):
            return inp, None
        return inp, request.raw_input_contents[i]
    return None, None


def _unpack_int32_list(raw: bytes) -> list[int]:
    return [struct.unpack_from("<i", raw, j * 4)[0] for j in range(len(raw) // 4)]


def _unpack_int64_list(raw: bytes) -> list[int]:
    return [int(struct.unpack_from("<q", raw, j * 8)[0]) for j in range(len(raw) // 8)]


def _parse_int_tensor_flat(inp, raw: bytes | None) -> list[int] | None:
    if raw is None or not raw:
        return None
    dt = inp.datatype
    if dt == "INT32":
        if len(raw) % 4 != 0:
            return None
        return _unpack_int32_list(raw)
    if dt == "INT64":
        if len(raw) % 8 != 0:
            return None
        return _unpack_int64_list(raw)
    return None


def _parse_optional_scalar_int(request, tensor_name: str) -> int | None:
    inp, raw = _find_input_raw(request, tensor_name)
    if inp is None or raw is None:
        return None
    ids = _parse_int_tensor_flat(inp, raw)
    if not ids:
        return None
    return int(ids[0])


def _parse_optional_scalar_float(request, tensor_name: str) -> float | None:
    inp, raw = _find_input_raw(request, tensor_name)
    if inp is None or raw is None or not raw:
        return None
    dt = inp.datatype
    if dt == "FP32" and len(raw) >= 4:
        return float(struct.unpack_from("<f", raw, 0)[0])
    if dt == "FP64" and len(raw) >= 8:
        return float(struct.unpack_from("<d", raw, 0)[0])
    # Allow integer tensors as scalar floats (e.g. top_p sent as INT32 1)
    if dt == "INT32" and len(raw) >= 4:
        return float(struct.unpack_from("<i", raw, 0)[0])
    if dt == "INT64" and len(raw) >= 8:
        return float(struct.unpack_from("<q", raw, 0)[0])
    return None


def _parse_stop_words_list_input(request) -> tuple[tuple[int, ...], ...] | None:
    """Input name ``stop_words_list`` -> ``GenerateConfig.stop_words_list`` (groups of token ids)."""
    inp, raw = _find_input_raw(request, "stop_words_list")
    if inp is None or raw is None:
        return None
    flat = _parse_int_tensor_flat(inp, raw)
    if flat is None:
        return None
    shape = [int(x) for x in inp.shape]
    if not shape or (len(shape) == 1 and shape[0] <= 0):
        return tuple()
    if len(shape) == 1:
        return (tuple(flat),) if flat else tuple()
    if len(shape) == 2:
        rows, cols = shape[0], shape[1]
        if rows * cols != len(flat):
            return None
        return tuple(tuple(flat[r * cols : (r + 1) * cols]) for r in range(rows))
    # Higher rank: treat as single flat sequence
    return (tuple(flat),) if flat else tuple()


def parse_input_ids_from_request(request) -> list[int] | None:
    """Read ``input_ids`` from ``raw_input_contents`` (INT32 / INT64, little-endian).

    Returns ``None`` if tensor missing, index mismatch, or unsupported datatype.
    """
    inp, raw = _find_input_raw(request, "input_ids")
    if inp is None or raw is None:
        return None
    return _parse_int_tensor_flat(inp, raw)


def parse_sampling_params(request) -> SamplingParams:
    """Read sampling fields from ``request.inputs`` (tensor names below).

    Tensor names: ``max_new_tokens``, ``num_return_sequences``, ``top_p``, ``top_k``,
    ``stop_words_list`` (token id groups; maps to ``stop_words_list`` / stop token ids),
    ``temperature``, ``min_new_tokens``, ``seed``, ``repetition_penalty``,
    ``frequency_penalty``, ``presence_penalty``.

    Legacy: if there is no ``top_k`` input, ``request.parameters["top_k"].int64_param`` is used.
    """
    max_new_tokens = 32000
    num_return_sequences = 0
    top_p = 1.0
    top_k = 0
    temperature = 1.0
    min_new_tokens = 0
    random_seed: int | None = None
    repetition_penalty = 1.0
    frequency_penalty = 0.0
    presence_penalty = 0.0
    stop_words_list: tuple[tuple[int, ...], ...] = tuple()

    v = _parse_optional_scalar_int(request, "max_new_tokens")
    if v is not None:
        max_new_tokens = max(0, v)

    v = _parse_optional_scalar_int(request, "num_return_sequences")
    if v is not None:
        num_return_sequences = max(0, v)

    vf = _parse_optional_scalar_float(request, "top_p")
    if vf is not None:
        top_p = vf

    v = _parse_optional_scalar_int(request, "top_k")
    if v is not None:
        top_k = v
    elif "top_k" in request.parameters:
        p = request.parameters["top_k"]
        if p.HasField("int64_param"):
            top_k = int(p.int64_param)

    vf = _parse_optional_scalar_float(request, "temperature")
    if vf is not None:
        temperature = vf

    v = _parse_optional_scalar_int(request, "min_new_tokens")
    if v is not None:
        min_new_tokens = max(0, v)

    v = _parse_optional_scalar_int(request, "seed")
    if v is not None:
        random_seed = v

    vf = _parse_optional_scalar_float(request, "repetition_penalty")
    if vf is not None:
        repetition_penalty = vf

    vf = _parse_optional_scalar_float(request, "frequency_penalty")
    if vf is not None:
        frequency_penalty = vf

    vf = _parse_optional_scalar_float(request, "presence_penalty")
    if vf is not None:
        presence_penalty = vf

    sw = _parse_stop_words_list_input(request)
    if sw is not None:
        stop_words_list = sw

    return SamplingParams(
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        min_new_tokens=min_new_tokens,
        random_seed=random_seed,
        repetition_penalty=repetition_penalty,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop_words_list=stop_words_list,
    )


def parse_other_params(request) -> OtherParams:
    """解析其他参数（非 ``input_ids``、非 sampling 张量）。

    当前支持：请求中的 ``return_input_ids`` 张量——BOOL 单字节，或 INT32/INT64/FP32/FP64
    标量（非 0 为 True）。未提供时 ``return_input_ids`` 为 False。
    """
    return_input_ids = False
    inp, raw = _find_input_raw(request, "return_input_ids")
    if inp is not None and raw:
        if inp.datatype == "BOOL" and len(raw) >= 1:
            return_input_ids = raw[0] != 0
        else:
            v = _parse_optional_scalar_int(request, "return_input_ids")
            if v is not None:
                return_input_ids = v != 0
            else:
                vf = _parse_optional_scalar_float(request, "return_input_ids")
                if vf is not None:
                    return_input_ids = vf != 0.0
    return OtherParams(return_input_ids=return_input_ids)


def parse_bailian_grpc_request(
    request,
) -> tuple[list[int] | None, SamplingParams | None, OtherParams | None]:
    """Parse one ``ModelInferRequest``: ``input_ids``, sampling tensors, ``other`` params.

    Returns ``(None, None, None)`` when ``input_ids`` cannot be read; otherwise
    ``(ids, sampling, other)`` with non-None ``SamplingParams`` and ``OtherParams``.
    """
    ids = parse_input_ids_from_request(request)
    if ids is None:
        return None, None, None
    return ids, parse_sampling_params(request), parse_other_params(request)
