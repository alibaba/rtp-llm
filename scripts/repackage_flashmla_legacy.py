#!/usr/bin/env python3
"""Give an existing FlashMLA wheel an isolated legacy namespace.

The CUDA binary and license remain byte-identical. This permits existing MLA
and DSv4 cache formats to coexist with latest upstream's V4.1-only FlashMLA.
Rebuilds wheel RECORD and records the original artifact checksum for auditing.
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path


def repackage(source: Path, destination: Path) -> Path:
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with zipfile.ZipFile(source) as original:
        metadata_path = next(
            name for name in original.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = original.read(metadata_path).decode()
        version = re.search(r"^Version: (.+)$", metadata, re.M).group(1)
        new_version = version + (".rtp1" if "+" in version else "+rtp1")
        old_info = metadata_path.rsplit("/", 1)[0]
        new_info = f"flash_mla_legacy-{new_version}.dist-info"
        suffix = source.name.split("-", 2)[2]
        result = destination / f"flash_mla_legacy-{new_version}-{suffix}"
        files = {}
        for name in original.namelist():
            if (
                name.endswith("/RECORD")
                or name.endswith("/RECORD.jws")
                or name.endswith("/RECORD.p7s")
            ):
                continue
            content = original.read(name)
            target = name.replace(old_info + "/", new_info + "/", 1)
            if target.startswith("flash_mla/"):
                target = "flash_mla_legacy/" + target[len("flash_mla/") :]
                if name.endswith(".py"):
                    content = content.replace(b"flash_mla.", b"flash_mla_legacy.")
            if name == metadata_path:
                updated = re.sub(
                    r"^Name: .+$", "Name: flash-mla-legacy", metadata, flags=re.M
                )
                updated = re.sub(
                    r"^Version: .+$", "Version: " + new_version, updated, flags=re.M
                )
                content = updated.encode()
            elif name.endswith("/top_level.txt"):
                content = b"flash_mla_legacy\n"
            files[target] = content
        files[new_info + "/UPSTREAM_ARTIFACT.json"] = (
            json.dumps(
                {
                    "original_wheel": source.name,
                    "original_sha256": source_hash,
                    "original_version": version,
                    "changes": [
                        "Python package namespace",
                        "distribution name/version",
                        "wheel RECORD",
                    ],
                    "native_binaries_modified": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
        rows = []
        for name, content in sorted(files.items()):
            digest = (
                base64.urlsafe_b64encode(hashlib.sha256(content).digest())
                .rstrip(b"=")
                .decode()
            )
            rows.append((name, "sha256=" + digest, str(len(content))))
        record = new_info + "/RECORD"
        rows.append((record, "", ""))
        out = io.StringIO()
        csv.writer(out, lineterminator="\n").writerows(rows)
        files[record] = out.getvalue().encode()
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(result, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in sorted(files.items()):
                info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, content)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(repackage(args.source, args.destination))
