"""Audit pinned checkpoint metadata/global scales using bounded HTTP ranges."""

import argparse
import base64
import hashlib
import json
import math
import struct
import urllib.request

MODEL = "RedHatAI/Qwen3-8B-NVFP4"
REVISION = "e391349c110709b87bfc2ad2fde3f50dc5839fd8"
BASE = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/"


def bounded_range(shard, start, end):
    request = urllib.request.Request(
        BASE + shard + f"?audit_range={start}-{end}",
        headers={"Range": f"bytes={start}-{end}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 206:
            raise ValueError("Server must honor Range; refusing a whole-shard download")
        data = response.read(end - start + 2)
    if len(data) != end - start + 1:
        raise ValueError("Unexpected range response length")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with urllib.request.urlopen(BASE + "config.json", timeout=30) as response:
        config_bytes = response.read(1024 * 1024)
    config = json.loads(config_bytes)
    with urllib.request.urlopen(
        BASE + "model.safetensors.index.json", timeout=30
    ) as response:
        index = json.load(response)
    layers = config["num_hidden_layers"]
    values, global_shapes, local_shapes, sources = {}, {}, {}, []
    for shard in sorted(set(index["weight_map"].values())):
        length = struct.unpack("<Q", bounded_range(shard, 0, 7))[0]
        if length > 4 * 1024 * 1024:
            raise ValueError("Unreasonably large safetensors header")
        raw = bounded_range(shard, 8, length + 7)
        header = json.loads(raw)
        globals_ = {
            name: meta
            for name, meta in header.items()
            if name.endswith((".weight_global_scale", ".input_global_scale"))
        }
        first = min(meta["data_offsets"][0] for meta in globals_.values())
        last = max(meta["data_offsets"][1] for meta in globals_.values())
        if last - first > 1024 * 1024:
            raise ValueError("Global scalars are not in a small bounded region")
        payload = bounded_range(shard, 8 + length + first, 8 + length + last - 1)
        sources.append(
            dict(
                shard=shard,
                header_sha256=hashlib.sha256(raw).hexdigest(),
                header_length=length,
                payload_data_offsets=[first, last],
                payload_sha256=hashlib.sha256(payload).hexdigest(),
                payload_base64=base64.b64encode(payload).decode(),
            )
        )
        for name, meta in globals_.items():
            if meta["dtype"] != "F32" or meta["shape"] != [1]:
                raise ValueError(f"Unexpected global metadata: {name}: {meta}")
            offset = meta["data_offsets"][0] - first
            value = struct.unpack_from("<f", payload, offset)[0]
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid global divisor: {name}")
            values[name] = value
            global_shapes[name] = dict(
                dtype=meta["dtype"],
                shape=meta["shape"],
                shard=shard,
                data_offsets=meta["data_offsets"],
            )
        for name, meta in header.items():
            if not name.endswith(".weight_scale"):
                continue
            packed = header[name.removesuffix(".weight_scale") + ".weight_packed"]
            if (
                meta["dtype"] != "F8_E4M3"
                or packed["dtype"] != "U8"
                or packed["shape"][0] != meta["shape"][0]
                or packed["shape"][1] != meta["shape"][1] * 8
            ):
                raise ValueError(f"Unexpected local scale metadata: {name}")
            local_shapes[name] = dict(
                dtype=meta["dtype"],
                shape=meta["shape"],
                packed_shape=packed["shape"],
                shard=shard,
            )
    groups = []
    for layer in range(layers):
        for suffix in ("weight_global_scale", "input_global_scale"):
            for names in (
                ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
                ("mlp.gate_proj", "mlp.up_proj"),
            ):
                keys = [f"model.layers.{layer}.{name}.{suffix}" for name in names]
                numbers = [values[name] for name in keys]
                if len(set(numbers)) != 1:
                    raise ValueError(
                        f"Unequal fused globals: {dict(zip(keys, numbers))}"
                    )
                groups.append(dict(keys=keys, value=numbers[0]))
    if len(values) != layers * 7 * 2 or len(local_shapes) != layers * 7:
        raise ValueError("Incomplete layer/global/local-scale audit")
    report = dict(
        model=MODEL,
        revision=REVISION,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        layers_checked=layers,
        global_scalars=len(values),
        fused_global_checks=len(groups),
        local_scale_tensors=len(local_shapes),
        values=values,
        global_storage=dict(dtype="F32", shape=[1], count=len(global_shapes)),
        layer0_local_metadata={
            key: value
            for key, value in local_shapes.items()
            if key.startswith("model.layers.0.")
        },
        fused_groups_all_equal=True,
        range_sources=sources,
        limitation="Metadata and scalar audit only; not kernel/quality validation",
    )
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "model",
                    "revision",
                    "layers_checked",
                    "global_scalars",
                    "fused_global_checks",
                    "local_scale_tensors",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
