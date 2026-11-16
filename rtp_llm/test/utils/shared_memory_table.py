"""Reclaim abandoned host tables on shared GPU test workers."""

import fcntl
import logging
import os
import re
import stat
from pathlib import Path

logger = logging.getLogger(__name__)


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        # A killed worker can remain as a zombie until its parent reaps it.
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        return state not in {"Z", "X"}
    except FileNotFoundError:
        return not Path("/proc").is_dir()
    except (OSError, IndexError):
        return True


def _live_leases(data_path: Path):
    leases = []
    for lease in data_path.parent.glob(data_path.name + ".lease.*"):
        try:
            pid = int(lease.name.split(".lease.", 1)[1].split(".", 1)[0])
        except ValueError:
            leases.append(lease)
        else:
            if _process_alive(pid):
                leases.append(lease)
            else:
                lease.unlink(missing_ok=True)
    return leases


def _reclaim_table(data_path, lock_directory):
    uid = os.getuid()
    fd = os.open(
        lock_directory / (data_path.name + ".lock"),
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
        0o600,
    )
    with os.fdopen(fd, "a+b") as lock:
        info = os.fstat(lock.fileno())
        if info.st_uid != uid or not stat.S_ISREG(info.st_mode):
            return 0
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        paths = list(data_path.parent.glob(data_path.name + "*"))
        metadata = [path.lstat() for path in paths]
        if any(
            item.st_uid != uid or not stat.S_ISREG(item.st_mode) for item in metadata
        ):
            return 0
        if _live_leases(data_path):
            return 0
        partials = list(data_path.parent.glob(data_path.name + ".partial.*"))
        if any(_process_alive(int(p.name.rsplit(".", 1)[1])) for p in partials):
            return 0
        reclaimed = 0
        for path in [data_path, *partials]:
            if path.exists():
                reclaimed += path.stat().st_blocks * 512
                path.unlink()
        return reclaimed


def reclaim_abandoned_tables(
    directory=Path("/dev/shm"), lock_directory=Path("/tmp"), *, prefix="rtp_llm_engram"
):
    """Reclaim this UID's tables left by killed workers, across checkpoints.

    Creation, reader registration and deletion use the same per-table lock.
    A busy writer or any live reader preserves the entire entry. Lock files
    remain in place so concurrent processes always lock the same inode.
    """
    uid = os.getuid()
    pattern = re.compile(
        rf"({re.escape(prefix)}_{uid}_[0-9a-f]{{24}})"
        r"(?:\.partial\.[0-9]+|\.lease\.[0-9]+\.[0-9a-f]{32})?"
    )
    entries = set()
    for path in directory.glob(f"{prefix}_{uid}_*"):
        match = pattern.fullmatch(path.name)
        if match:
            entries.add(match[1])
    reclaimed = 0
    for name in entries:
        try:
            reclaimed += _reclaim_table(directory / name, lock_directory)
        except (FileNotFoundError, BlockingIOError):
            continue
        except (OSError, ValueError):
            logger.warning(
                "Could not reclaim abandoned shared host table %s", name, exc_info=True
            )
    if reclaimed:
        logger.info("Reclaimed %.2f GiB abandoned shared host RAM", reclaimed / 1024**3)
    return reclaimed
