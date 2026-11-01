"""Check process-wide native state before allocating complete model weights."""
from pathlib import Path


def assert_single_compute_library(maps_text=None):
    """ELF segments/aliases of one inode are fine; independent copies are not.

    register_comm_ops writes library-local state. An engine bound to a second
    compute object cannot see those callbacks even when both files have the
    same bytes. /proc maps' device + inode identify the loaded backing files.
    """
    if maps_text is None:
        maps_text = Path("/proc/self/maps").read_text()
    copies = {}
    for line in maps_text.splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) != 6:
            continue
        name = Path(parts[5].removesuffix(" (deleted)")).name
        if not name.startswith("librtp_compute_ops") or ".so" not in name:
            continue
        if parts[4] == "0" or parts[5].endswith(" (deleted)"):
            raise RuntimeError("compute library has invalid/deleted backing")
        copies.setdefault((parts[3], parts[4]), set()).add(parts[5])
    if len(copies) != 1:
        raise RuntimeError(
            f"expected one loaded compute library, found {len(copies)}: "
            + repr(sorted(path for paths in copies.values() for path in paths))
        )
    return sorted(next(iter(copies.values())))
