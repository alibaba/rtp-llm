"""按显式版本选择 lineage codec；不以异常回退或最新版本猜测。"""
import json
import lzma
from traffic import prefix_lineage, prefix_lineage_v3


def codec_for_version(version):
    if type(version) is not int or version not in (2, 3):
        raise ValueError(f"unknown prefix_lineage codec version: {version!r}")
    return {2: prefix_lineage, 3: prefix_lineage_v3}[version]


def decode(raw, manifest=None):
    if manifest is not None:
        codec = manifest.get("codec", {})
        if codec.get("name") != "prefix_lineage":
            raise ValueError("manifest codec.name must be prefix_lineage")
        module = codec_for_version(codec.get("version"))
    else:
        # 生成 sidecar / 内存 API 从自描述 envelope 读取版本；文件消费方传入 manifest。
        decoder = lzma.LZMADecompressor(memlimit=128 * 1024 * 1024)
        data = decoder.decompress(raw, max_length=128 * 1024 * 1024)
        if not decoder.eof or decoder.unused_data:
            raise ValueError("invalid or oversized lineage envelope")
        if data.startswith(prefix_lineage.MAGIC):
            module = codec_for_version(2)
        else:
            try:
                version = json.loads(data)["metadata"]["version"]
            except (ValueError, KeyError, TypeError):
                raise ValueError("missing lineage codec version") from None
            module = codec_for_version(version)
    metadata, events = module.decode(raw)
    if manifest is not None:
        for field in ("version", "block_size"):
            if manifest.get(field) != metadata[field]:
                raise ValueError(f"manifest {field} disagrees with model contents")
    return metadata, events


def block_events(metadata, events):
    """将 v3 投影为结构分析的完整块和私有尾块预留，不伪造精确长度。"""
    if metadata["version"] == 2:
        return events
    # 结构画像仍按完整块建模；短请求占一个私有标签，精确长度分布由调用方另存。
    block = metadata["block_size"]
    return [(ts, max(1, length // block), parent, shared, int(length >= block and length % block != 0))
            for ts, length, parent, shared in events]
