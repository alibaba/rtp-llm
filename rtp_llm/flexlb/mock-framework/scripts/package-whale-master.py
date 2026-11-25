#!/usr/bin/env python3
"""Add the Mock Master entry point to a matching FlexLB Whale distribution."""

import argparse
import io
import os
import re
import tarfile
import tempfile
import zipfile
from pathlib import Path

MANIFEST = "FlexLB/META-INF/MANIFEST.MF"
MAIN_CLASS = "org.flexlb.WhaleMockMasterMain"
BOOT_CLASSES = "FlexLB/BOOT-INF/classes/"
MOCK_CLASSES = (
    "org/flexlb/WhaleMockMasterMain",
    "org/flexlb/MockDpServiceDiscoveryProvider",
)
VIP_PROVIDER = "org/flexlb/discovery/VipServerDiscovery.class"
PROVIDER_INTERFACE = b"org/flexlb/discovery/ServiceDiscoveryProvider"


def package(master_archive: Path, mock_jar: Path, output: Path) -> None:
    if master_archive.resolve() == output.resolve():
        raise ValueError("Use a separate output path for the Mock Master distribution")
    with zipfile.ZipFile(mock_jar) as jar:
        names = jar.namelist()
        classes = {}
        for base in MOCK_CLASSES:
            if base + ".class" not in names:
                raise ValueError(f"{mock_jar} does not contain {base}.class")
            for name in names:
                if name == base + ".class" or (
                    name.startswith(base + "$") and name.endswith(".class")
                ):
                    classes[BOOT_CLASSES + name] = jar.read(name)

    with tarfile.open(master_archive, "r:gz") as source:
        names = source.getnames()
        if classes.keys() & set(names):
            raise ValueError(
                "Input distribution already contains the Mock Master overlay"
            )
        vip_jars = [
            name
            for name in names
            if re.fullmatch(r"FlexLB/BOOT-INF/lib/vipserver-\d[^/]*\.jar", name)
        ]
        if len(vip_jars) != 1:
            raise ValueError(
                f"Expected one internal VIP provider JAR, found {vip_jars}"
            )
        with source.extractfile(vip_jars[0]) as payload:
            with zipfile.ZipFile(io.BytesIO(payload.read())) as vip:
                if (
                    VIP_PROVIDER not in vip.namelist()
                    or PROVIDER_INTERFACE not in vip.read(VIP_PROVIDER)
                ):
                    raise ValueError(
                        "Input VIP provider uses an incompatible discovery interface"
                    )
        with source.extractfile(MANIFEST) as original:
            manifest, count = re.subn(
                rb"(?m)^Start-Class: org\.flexlb\.Application(?=\r?$)",
                f"Start-Class: {MAIN_CLASS}".encode(),
                original.read(),
            )
        if count != 1:
            raise ValueError("Expected the standard FlexLB Application entry point")

        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as temp:
            staged = Path(temp.name)
        try:
            with tarfile.open(staged, "w:gz") as target:
                for entry in source:
                    if entry.name == MANIFEST:
                        entry.size = len(manifest)
                        target.addfile(entry, io.BytesIO(manifest))
                    elif entry.isfile():
                        with source.extractfile(entry) as payload:
                            target.addfile(entry, payload)
                    else:
                        target.addfile(entry)
                for name, data in classes.items():
                    entry = tarfile.TarInfo(name)
                    entry.size = len(data)
                    target.addfile(entry, io.BytesIO(data))
            os.replace(staged, output)
        finally:
            staged.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("master_archive", type=Path)
    parser.add_argument("mock_jar", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    package(args.master_archive, args.mock_jar, args.output)


if __name__ == "__main__":
    main()
