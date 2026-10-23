"""从数据 envelope 识别 lineage codec；调用方版本仅作一致性断言。"""
from __future__ import annotations

import json
from traffic import prefix_lineage, prefix_lineage_v3
from traffic.lineage_model import LineageModel
from traffic.lineage_envelope import decompress

CODECS = {2: prefix_lineage.BlockAlignedLineageCodec,
          3: prefix_lineage_v3.ExactLengthLineageCodec}

def codec_for_version(version):
    if type(version) is not int or version not in CODECS:
        raise ValueError(f"unknown prefix_lineage codec version: {version!r}")
    return CODECS[version]()


def decode_model(raw: bytes, manifest=None, *, expected_version: str | None = None) -> LineageModel:
    if manifest is not None:
        codec = manifest.get("codec", {})
        if codec.get("name") != "prefix_lineage":
            raise ValueError("manifest codec.name must be prefix_lineage")
        module = codec_for_version(codec.get("version"))
    else:
        # 生成 sidecar / 内存 API 从自描述 envelope 读取版本；文件消费方传入 manifest。
        data = decompress(raw, "invalid or oversized lineage envelope")
        if data.startswith(prefix_lineage.MAGIC):
            module = codec_for_version(2)
        else:
            try:
                version = json.loads(data)["metadata"]["version"]
            except (ValueError, KeyError, TypeError):
                raise ValueError("missing lineage codec version") from None
            module = codec_for_version(version)
    model = module.decode(raw) if manifest is not None else module._parse(data)
    metadata = model.metadata.to_dict()
    if expected_version is not None and expected_version != str(model.metadata.version):
        raise ValueError("source version disagrees with model contents")
    if manifest is not None:
        for field in ("version", "block_size"):
            if manifest.get(field) != metadata[field]:
                raise ValueError(f"manifest {field} disagrees with model contents")
    return model


def decode(raw, manifest=None):
    return decode_model(raw, manifest).to_wire()
