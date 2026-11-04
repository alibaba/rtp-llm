#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import pathlib
import struct


MAGIC = "navi_sched_latency_state"
STATE_VERSION = 2
FORMULA_NONLINEAR_PREFILL = 1
BINARY_VERSION = 19
PARAMETER_COUNTS = {11, 18}


def _encode_varuint(value):
    if value < 0:
        raise ValueError("varuint cannot encode a negative value")
    result = bytearray()
    while value >= 0x80:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


def _encode_int64(value):
    return _encode_varuint((value << 1) ^ (value >> 63))


def _decode_varuint(content, offset):
    value = 0
    shift = 0
    while offset < len(content) and shift <= 63:
        current = content[offset]
        offset += 1
        value |= (current & 0x7F) << shift
        if current < 0x80:
            return value, offset
        shift += 7
    raise ValueError("invalid varuint")


def _decode_int64(content, offset):
    value, offset = _decode_varuint(content, offset)
    return (value >> 1) ^ -(value & 1), offset


def encode_latency_model_state(
        parameters, revision=1, adam_moment_1=None, adam_moment_2=None,
        optimizer_step=1):
    parameters = [float(value) for value in parameters]
    if adam_moment_1 is None:
        adam_moment_1 = [0.0] * len(parameters)
    if adam_moment_2 is None:
        adam_moment_2 = [0.0] * len(parameters)
    adam_moment_1 = [float(value) for value in adam_moment_1]
    adam_moment_2 = [float(value) for value in adam_moment_2]
    if (len(parameters) not in PARAMETER_COUNTS or revision <= 0 or
            len(adam_moment_1) != len(parameters) or
            len(adam_moment_2) != len(parameters) or optimizer_step <= 0 or
            not all(math.isfinite(value) for value in parameters) or
            not all(math.isfinite(value) for value in adam_moment_1) or
            not all(math.isfinite(value) and value >= 0.0
                    for value in adam_moment_2)):
        raise ValueError(
            "state requires valid parameters, Adam state, and positive steps")
    magic = MAGIC.encode("ascii")
    result = bytearray()
    result.extend(_encode_varuint(len(magic)))
    result.extend(magic)
    result.extend(struct.pack("<ii", STATE_VERSION, FORMULA_NONLINEAR_PREFILL))
    result.extend(_encode_int64(BINARY_VERSION))
    result.extend(_encode_int64(revision))
    result.extend(_encode_varuint(len(parameters)))
    result.extend(struct.pack("<{}d".format(len(parameters)), *parameters))
    result.extend(_encode_varuint(len(adam_moment_1)))
    result.extend(struct.pack(
        "<{}d".format(len(adam_moment_1)), *adam_moment_1))
    result.extend(_encode_varuint(len(adam_moment_2)))
    result.extend(struct.pack(
        "<{}d".format(len(adam_moment_2)), *adam_moment_2))
    result.extend(_encode_int64(optimizer_step))
    result.extend(_encode_varuint(0))
    return bytes(result)


def decode_latency_model_state(content):
    offset = 0
    magic_size, offset = _decode_varuint(content, offset)
    magic = content[offset:offset + magic_size].decode("ascii")
    offset += magic_size
    state_version, formula = struct.unpack_from("<ii", content, offset)
    offset += struct.calcsize("<ii")
    binary_version, offset = _decode_int64(content, offset)
    revision, offset = _decode_int64(content, offset)
    parameter_count, offset = _decode_varuint(content, offset)
    parameters = list(struct.unpack_from(
        "<{}d".format(parameter_count), content, offset))
    offset += struct.calcsize("<{}d".format(parameter_count))
    moment_1_count, offset = _decode_varuint(content, offset)
    adam_moment_1 = list(struct.unpack_from(
        "<{}d".format(moment_1_count), content, offset))
    offset += struct.calcsize("<{}d".format(moment_1_count))
    moment_2_count, offset = _decode_varuint(content, offset)
    adam_moment_2 = list(struct.unpack_from(
        "<{}d".format(moment_2_count), content, offset))
    offset += struct.calcsize("<{}d".format(moment_2_count))
    optimizer_step, offset = _decode_int64(content, offset)
    history_size, offset = _decode_varuint(content, offset)
    if (magic != MAGIC or state_version != STATE_VERSION or
            formula != FORMULA_NONLINEAR_PREFILL or
            binary_version != BINARY_VERSION or
            parameter_count not in PARAMETER_COUNTS or
            moment_1_count != parameter_count or
            moment_2_count != parameter_count or optimizer_step <= 0 or
            history_size != 0 or
            offset != len(content)):
        raise ValueError("invalid empty latency model state")
    return {
        "revision": revision,
        "parameters": parameters,
        "adam_moment_1": adam_moment_1,
        "adam_moment_2": adam_moment_2,
        "optimizer_step": optimizer_step,
    }


def write_latency_model_state(path, parameters, revision=1):
    content = encode_latency_model_state(parameters, revision)
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(target)
    return hashlib.sha256(content).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Create an empty NaviSched latency model state")
    parser.add_argument("--output", required=True)
    parser.add_argument("--parameters-json", required=True)
    parser.add_argument("--revision", type=int, default=1)
    options = parser.parse_args()
    parameters = json.loads(options.parameters_json)
    sha256 = write_latency_model_state(
        options.output, parameters, options.revision)
    decoded = decode_latency_model_state(pathlib.Path(options.output).read_bytes())
    print(json.dumps({
        "binary_version": BINARY_VERSION,
        "formula": FORMULA_NONLINEAR_PREFILL,
        "output": str(pathlib.Path(options.output).resolve()),
        "parameters": decoded["parameters"],
        "revision": decoded["revision"],
        "sha256": sha256,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
