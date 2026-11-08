"""Retain bounded ELF evidence before smoke core cleanup."""

import hashlib
import json
import os
import stat
import struct
from pathlib import Path

MAX_EVIDENCE_BYTES = 8 * 1024**2


def preserve_core_evidence(task):
    task = Path(task)
    records = []
    for core in sorted(task.glob("core-*")):
        before = core.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Core evidence requires a regular file: {core.name}")
        destination = task / "crash-evidence" / core.name
        destination.mkdir(parents=True, exist_ok=False)
        record = {
            "name": core.name,
            "logical_bytes": before.st_size,
            "allocated_bytes": before.st_blocks * 512,
            "partial_core_evidence": True,
            "core_dump_remains_a_failure": True,
            "fragments": [],
        }
        retained = 0

        def retain(stream, offset, size, label):
            nonlocal retained
            if offset < 0 or size < 0 or offset + size > before.st_size:
                raise ValueError("Core fragment exceeds source file")
            if retained + size > MAX_EVIDENCE_BYTES:
                raise ValueError("Core evidence exceeds its per-file budget")
            stream.seek(offset)
            data = stream.read(size)
            if len(data) != size:
                raise ValueError("Core changed during evidence collection")
            (destination / label).write_bytes(data)
            retained += size
            record["fragments"].append(
                {
                    "file": label,
                    "source_offset": offset,
                    "bytes": size,
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            return data

        with core.open("rb") as stream:
            header = stream.read(64)
            if len(header) != 64 or header[:6] != b"\x7fELF\x02\x01":
                retain(stream, 0, min(before.st_size, 65536), "prefix.bin")
                record["format"] = "unsupported-prefix-only"
            else:
                elf = struct.unpack("<16sHHIQQQIHHHHHH", header)
                record["elf_machine"] = elf[2]
                if elf[1] != 4 or elf[9] != 56 or not 0 < elf[10] <= 8192:
                    raise ValueError("Invalid ELF core program header table")
                table = retain(stream, 0, elf[5] + elf[9] * elf[10], "headers.bin")
                segments = [
                    struct.unpack_from("<IIQQQQQQ", table, elf[5] + i * elf[9])
                    for i in range(elf[10])
                ]
                threads, main_pid = [], None
                for index, segment in enumerate(segments):
                    if segment[0] != 4:
                        continue
                    notes = retain(stream, segment[2], segment[5], f"notes-{index}.bin")
                    offset = 0
                    while offset + 12 <= len(notes):
                        namesz, descsz, kind = struct.unpack_from("<III", notes, offset)
                        offset += 12
                        owner = notes[offset : offset + namesz].rstrip(b"\0")
                        offset += (namesz + 3) // 4 * 4
                        desc = notes[offset : offset + descsz]
                        offset += (descsz + 3) // 4 * 4
                        if owner == b"CORE" and kind == 1 and elf[2] == 62:
                            threads.append(
                                {
                                    "pid": struct.unpack_from("<i", desc, 32)[0],
                                    "signal": struct.unpack_from("<i", desc)[0],
                                    "rip": struct.unpack_from("<Q", desc, 240)[0],
                                    "rsp": struct.unpack_from("<Q", desc, 264)[0],
                                }
                            )
                        if owner == b"CORE" and kind == 3:
                            main_pid = struct.unpack_from("<i", desc, 24)[0]
                record["machine"] = elf[2]
                record["threads"] = threads
                for index, thread in enumerate(threads):
                    if index != 0 and thread["pid"] != main_pid:
                        continue
                    for segment in segments:
                        if (
                            segment[0] == 1
                            and segment[3] <= thread["rsp"] < segment[3] + segment[5]
                        ):
                            begin = max(segment[3], thread["rsp"] - 4096)
                            end = min(segment[3] + segment[5], thread["rsp"] + 65536)
                            retain(
                                stream,
                                segment[2] + begin - segment[3],
                                end - begin,
                                f"stack-{thread['pid']}.bin",
                            )
                            break
        after = core.lstat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("Core is still changing; keeping the original")
        record["retained_bytes"] = retained
        records.append(record)
    manifest = {"version": 1, "core_dump_detected": bool(records), "cores": records}
    temporary = task / "crash-evidence.json.next"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, task / "crash-evidence.json")
    return manifest
