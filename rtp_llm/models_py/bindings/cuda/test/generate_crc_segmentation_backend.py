#!/usr/bin/env python3
"""Generate a benchmark-only policy switch from the production CRC backend."""

import argparse
import hashlib
import json
from pathlib import Path

PREFIX = "rtp_llm/models_py/bindings/cuda/test/"


def replace_once(text, old, new):
    count = text.count(old)
    if count != 1:
        raise ValueError(f"expected one occurrence, found {count}: {old!r}")
    return text.replace(old, new, 1)


def generate(header, source, internal):
    class_start = "class CrcBlockCopyBatch {"
    class_end = "\n};\n"
    if header.count(class_start) != 1:
        raise ValueError("production class declaration changed")
    begin = header.index(class_start)
    end = header.index(class_end, begin) + len(class_end)
    declaration = header[begin:end].replace(
        "CrcBlockCopyBatch", "ForcedCrcBlockCopyBatch"
    )
    declaration = replace_once(
        declaration,
        "    size_t deviceMemoryBytes() const;",
        "    size_t deviceMemoryBytes() const;\n"
        "    void setSegmented(bool enabled);\n"
        "    size_t lastMessageCount() const;",
    )
    generated_header = (
        "// Generated for the segmentation benchmark; never linked into production.\n"
        '#pragma once\n#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"\n\n'
        "namespace rtp_llm {\n" + declaration + "\n}  // namespace rtp_llm\n"
    )

    # Keep every constructor allocation, including segmentation scratch, intact.
    capacity = "const bool can_segment = crc_block_copy_internal::shouldSegment(items, payload);"
    if source.count(capacity) != 1:
        raise ValueError("production workspace capacity decision changed")
    original_capacity = capacity
    source = replace_once(
        source,
        "        prepared.segmented = crc_block_copy_internal::shouldSegment(prepared.count, largest);",
        "        (void)largest;\n        prepared.segmented = force_segmented;",
    )
    source = replace_once(
        source,
        "    std::mutex                     mutex;",
        "    bool                           force_segmented{false};\n"
        "    size_t                         last_message_count{0};\n"
        "    std::mutex                     mutex;",
    )
    source = replace_once(
        source,
        "        return prepared;",
        "        last_message_count = prepared.messages;\n        return prepared;",
    )
    source = replace_once(
        source,
        "size_t CrcBlockCopyBatch::deviceMemoryBytes() const {",
        "void CrcBlockCopyBatch::setSegmented(bool enabled) {\n"
        "    std::lock_guard<std::mutex> lock(impl_->mutex);\n"
        "    if (enabled && impl_->max_segments <= 1)\n"
        '        throw std::invalid_argument("forced segmentation requires preallocated segment capacity");\n'
        "    impl_->force_segmented = enabled;\n"
        "}\n\n"
        "size_t CrcBlockCopyBatch::lastMessageCount() const {\n"
        "    std::lock_guard<std::mutex> lock(impl_->mutex);\n"
        "    return impl_->last_message_count;\n"
        "}\n\n"
        "size_t CrcBlockCopyBatch::deviceMemoryBytes() const {",
    )
    if source.count(original_capacity) != 1:
        raise ValueError("workspace capacity decision must remain unchanged")
    source = replace_once(
        source,
        '#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"',
        f'#include "{PREFIX}ForcedCrcBlockCopy.h"',
    )
    source = replace_once(
        source,
        '#include "rtp_llm/models_py/bindings/cuda/CrcBlockCopyInternal.cuh"',
        f'#include "{PREFIX}ForcedCrcBlockCopyInternal.cuh"',
    )
    # Both symbols are otherwise exported by the production library, which is
    # also a transitive dependency of the existing benchmark support library.
    source = source.replace("CrcBlockCopyBatch", "ForcedCrcBlockCopyBatch")
    source = source.replace("crc_block_copy_internal", "crc_segmentation_ab_internal")
    internal = internal.replace(
        "crc_block_copy_internal", "crc_segmentation_ab_internal"
    )
    banner = "// Generated benchmark-only backend: production copy pipeline, forced CRC policy.\n"
    return generated_header, banner + source, banner + internal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "header",
        "source",
        "internal",
        "output_header",
        "output_source",
        "output_internal",
        "manifest",
    ):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    args = parser.parse_args()
    inputs = {
        name: getattr(args, name).read_bytes()
        for name in ("header", "source", "internal")
    }
    outputs = generate(
        *(inputs[name].decode() for name in ("header", "source", "internal"))
    )
    manifest = {
        "generator": Path(__file__).name,
        "input_sha256": {
            name: hashlib.sha256(data).hexdigest() for name, data in inputs.items()
        },
        "changes": [
            "rename public backend class and internal namespace to avoid production symbol collisions",
            "force only prepareBatch.segmented; retain constructor capacity, DMA and kernel code",
            "add a mutex-protected policy setter outside timing without clearing caches",
            "record actual prepared.messages for path verification after timing",
        ],
        "output_sha256": {},
    }
    for name, output in zip(
        ("output_header", "output_source", "output_internal"), outputs
    ):
        path = getattr(args, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output)
        manifest["output_sha256"][path.name] = hashlib.sha256(
            output.encode()
        ).hexdigest()
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
