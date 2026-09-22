#!/usr/bin/env python3
"""One portable, compressed artifact for a local case/scenario/stress run.

The manifest records exact source hashes and any size-limited raw evidence.
Structured results are never silently truncated. Whale may use KMonitor only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

SCHEMA_VERSION = 1
STRUCTURED = {".json", ".yaml", ".yml", ".html", ".csv"}
RAW_LIMIT = 2 * 1024 * 1024
HEAD_TAIL = 256 * 1024
SECRET_WORDS = ("api_token", "auth_token", "access_token", "refresh_token",
                "api_key", "access_key", "secret", "credential", "password",
                "private_key")


def _secret_like(path: Path) -> bool:
    name = path.name.lower()
    return name in {"token", "token.txt", "token.json", "token.yaml",
                    "token.yml", ".env"} or any(
        word in name for word in SECRET_WORDS
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_archive(output: Path, sources: dict[str, Path], *, kind: str,
                   status: str = "complete", metadata: dict | None = None) -> dict:
    """Archive named run directories/files; never follow symlinks or include output."""
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if status not in {"complete", "incomplete"}:
        raise ValueError("status must be complete or incomplete")
    manifest = {"schema_version": SCHEMA_VERSION, "kind": kind, "status": status,
                "metadata": metadata or {}, "files": [], "omitted": []}
    fd, temp_name = tempfile.mkstemp(prefix=".experiment-", suffix=".zip", dir=output.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temp_name, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as archive:
            for label, root in sources.items():
                if not label or "/" in label or label in (".", ".."):
                    raise ValueError(f"invalid source label: {label!r}")
                root = Path(root).resolve()
                if not root.exists():
                    raise FileNotFoundError(root)
                files = sorted(root.rglob("*")) if root.is_dir() else [root]
                for path in files:
                    if (path.is_symlink() or not path.is_file()
                            or path.resolve() in {output, Path(temp_name).resolve()}):
                        continue
                    relative = path.relative_to(root) if root.is_dir() else Path(path.name)
                    name = f"{label}/{relative.as_posix()}"
                    if _secret_like(relative):
                        manifest["omitted"].append({"path": name, "reason": "secret-like filename"})
                        continue
                    size = path.stat().st_size
                    record = {"path": name, "size": size, "sha256": _sha256(path)}
                    if path.suffix.lower() in STRUCTURED or size <= RAW_LIMIT:
                        archive.write(path, name)
                        record["completeness"] = "full"
                    else:
                        with path.open("rb") as fh:
                            first = fh.read(HEAD_TAIL)
                            fh.seek(max(0, size - HEAD_TAIL))
                            last = fh.read(HEAD_TAIL)
                        archive.writestr(name + ".head", first)
                        archive.writestr(name + ".tail", last)
                        record["completeness"] = "head_tail"
                        record["included_bytes"] = len(first) + len(last)
                    manifest["files"].append(record)
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False,
                                                          indent=2).encode("utf-8"))
        os.replace(temp_name, output)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create or inspect an experiment ZIP")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--out", type=Path, required=True)
    create.add_argument("--kind", choices=("case", "scenario", "stress", "ab"), required=True)
    create.add_argument("--source", action="append", required=True,
                        help="label=directory or label=file; repeat as needed")
    create.add_argument("--status", choices=("complete", "incomplete"), default="complete")
    create.add_argument("--metadata-json", default="{}", help="small provenance/verdict object")
    inspect = sub.add_parser("inspect")
    inspect.add_argument("archive", type=Path)
    args = parser.parse_args(argv)
    if args.command == "create":
        sources = {}
        for item in args.source:
            label, sep, path = item.partition("=")
            if not sep or label in sources:
                parser.error("--source needs a distinct label=path")
            sources[label] = Path(path)
        metadata = json.loads(args.metadata_json)
        if not isinstance(metadata, dict):
            parser.error("--metadata-json must be an object")
        manifest = create_archive(args.out, sources, kind=args.kind,
                                  status=args.status, metadata=metadata)
    else:
        with zipfile.ZipFile(args.archive) as archive:
            manifest = json.loads(archive.read("manifest.json"))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
