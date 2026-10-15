"""Initialize the original gfx942 target list without appending duplicates."""

import fcntl
from pathlib import Path


def initialize_target_list(path: Path = Path("/opt/rocm/bin/target.lst")) -> None:
    # Lock the opened file so two CI setup processes cannot both append to an
    # initially empty file. An existing unsupported target is never overwritten.
    with path.open("a+") as target_file:
        fcntl.flock(target_file, fcntl.LOCK_EX)
        target_file.seek(0)
        targets = [line.strip() for line in target_file if line.strip()]
        if not targets:
            target_file.write("gfx942\n")
            target_file.flush()
        elif targets != ["gfx942"]:
            raise RuntimeError(
                f"ROCm target list mismatch in {path}: "
                f"expected only 'gfx942', got {targets}."
            )


if __name__ == "__main__":
    initialize_target_list()
