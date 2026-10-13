"""Prepare an isolated, source-checked vLLM package for full-state P/D tests."""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


def patched_files(patch: str, source: Path, allowed: set) -> dict:
    """Apply exact unified hunks in memory; reject fuzz, extra paths and offsets."""
    sections = re.split(r"(?=^--- a/vllm/)", patch, flags=re.MULTILINE)
    outputs = {}
    for section in sections:
        if not section:
            continue
        lines = section.splitlines(keepends=True)
        name = lines[0][len("--- a/vllm/") :].rstrip("\n")
        if name not in allowed or name in outputs:
            raise ValueError(f"Unexpected or duplicate patch path: {name}")
        if lines[1] != f"+++ b/vllm/{name}\n":
            raise ValueError("Patch must preserve the source path")
        original = (source / name).read_text().splitlines(keepends=True)
        result, cursor, index = [], 0, 2
        while index < len(lines):
            header = re.fullmatch(r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@.*\n", lines[index])
            if header is None:
                raise ValueError("Unsupported unified hunk header")
            start, old_count, new_start, new_count = map(int, header.groups())
            if start - 1 < cursor:
                raise ValueError("Overlapping patch hunks")
            result.extend(original[cursor : start - 1])
            cursor = start - 1
            if len(result) != new_start - 1:
                raise ValueError("Patch output offset differs")
            index += 1
            removed = added = 0
            while index < len(lines) and not lines[index].startswith("@@ "):
                line = lines[index]
                if line[0] in " -":
                    if cursor >= len(original) or original[cursor] != line[1:]:
                        raise ValueError("Patch context differs from pinned source")
                    cursor += 1
                    removed += 1
                if line[0] in " +":
                    result.append(line[1:])
                    added += 1
                if line[0] not in " +-":
                    raise ValueError("Unsupported patch line")
                index += 1
            if (removed, added) != (old_count, new_count):
                raise ValueError("Patch hunk length differs")
        result.extend(original[cursor:])
        outputs[name] = "".join(result).encode()
    if set(outputs) != allowed:
        raise ValueError("Patch inventory differs from contract")
    return outputs


def prepare_overlay(source: Path, destination: Path) -> dict:
    package = Path(__file__).parent
    contract = json.loads((package / "source_contract.json").read_text())
    modified = {row["path"]: row for row in contract["files"]}
    if destination.exists():
        raise FileExistsError(destination)
    # Validate everything before creating the candidate or changing any file.
    for name, row in modified.items():
        actual = hashlib.sha256((source / name).read_bytes()).hexdigest()
        if actual != row["source_sha256"]:
            raise ValueError(f"vLLM source differs from the pinned contract: {name}")
    outputs = patched_files(
        (package / "vllm_full_state.patch").read_text(), source, set(modified)
    )
    for name, data in outputs.items():
        if hashlib.sha256(data).hexdigest() != modified[name]["patched_sha256"]:
            raise ValueError(f"Patched source checksum differs: {name}")
    destination.mkdir(parents=True)

    def clone(directory: Path, target: Path, relative: str = ""):
        target.mkdir()
        for entry in directory.iterdir():
            if entry.name == "__pycache__":
                continue
            name = f"{relative}/{entry.name}".lstrip("/")
            output = target / entry.name
            if name in modified:
                shutil.copyfile(entry, output)
            elif any(key.startswith(name + "/") for key in modified):
                clone(entry, output, name)
            else:
                output.symlink_to(entry.resolve(), target_is_directory=entry.is_dir())

    clone(source, destination / "vllm")
    for name, data in outputs.items():
        (destination / "vllm" / name).write_bytes(data)
    for name, row in modified.items():
        path = destination / "vllm" / name
        if path.is_symlink():
            raise ValueError(f"Patched file must be independent: {name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != row["patched_sha256"]:
            raise ValueError(f"Patched source checksum differs: {name}")
    manifest = dict(
        source=str(source),
        overlay=str(destination),
        files=contract["files"],
        scope=contract["scope"],
        disable_bytecode_writes=True,
    )
    (destination / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_overlay(args.source, args.destination), indent=2))
