"""Pod-local Prefill stall detection and CPU/GPU core collection."""

import logging
import math
import mmap
import os
import select
import shlex
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional
from uuid import uuid4

_PHASES = ("idle", "prepare", "layer", "graph", "tail", "gpu_wait", "gpu_done")
_CONFIG_PHASES = _PHASES[1:-1]
_WORD = struct.Struct("<Q")
_MIN_FREE_BYTES = 128 * 1024 * 1024
_DEFAULT_TIMEOUT_S = 10.0
_KILL_REAP_GRACE_S = 1.0
_CUDA_GDB = "/usr/local/cuda/bin/cuda-gdb"
_GPU_CORE_READABLE = "Prefill GPU core opened successfully"
_LOG = logging.getLogger(__name__)


def _parse_thresholds(value: str) -> dict[str, float]:
    thresholds = {}
    if value.strip():
        for item in value.split(","):
            name, seconds = (part.strip() for part in item.split("=", 1))
            if name not in _CONFIG_PHASES or name in thresholds:
                raise ValueError(f"invalid or duplicate Prefill phase {name!r}")
            deadline = float(seconds)
            if not math.isfinite(deadline) or deadline <= 0:
                raise ValueError(
                    f"Prefill phase {name} needs a positive finite deadline"
                )
            thresholds[name] = deadline
    return thresholds


def _pid_start_ticks(pid: int) -> Optional[int]:
    try:
        # comm can contain spaces or parentheses. starttime is field 22.
        fields = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
        if fields[0] in {"Z", "X"}:
            return None
        return int(fields[19])
    except (OSError, IndexError, ValueError):
        return None


def _is_elf_core(path: Path) -> bool:
    with path.open("rb") as core:
        size = os.fstat(core.fileno()).st_size
        header = core.read(64)
        if header[:4] != b"\x7fELF" or len(header) < 52:
            return False
        elf_class, data_encoding = header[4:6]
        if data_encoding not in (1, 2):
            return False
        order = "<" if data_encoding == 1 else ">"
        if struct.unpack_from(f"{order}H", header, 16)[0] != 4:  # ET_CORE
            return False

        if elf_class == 1:
            phoff, shoff = struct.unpack_from(f"{order}II", header, 28)
            phsize, phnum, shsize = struct.unpack_from(f"{order}HHH", header, 42)
            program = struct.Struct(f"{order}IIIIIIII")
            offset_index, size_index, info_offset, header_size = 1, 4, 28, 52
        elif elf_class == 2 and len(header) == 64:
            phoff, shoff = struct.unpack_from(f"{order}QQ", header, 32)
            phsize, phnum, shsize = struct.unpack_from(f"{order}HHH", header, 54)
            program = struct.Struct(f"{order}IIQQQQQQ")
            offset_index, size_index, info_offset, header_size = 2, 5, 44, 64
        else:
            return False

        if phnum == 0xFFFF:  # PN_XNUM, the actual count is in section 0 sh_info.
            if shoff < header_size or shsize < info_offset + 4 or shoff + shsize > size:
                return False
            core.seek(shoff + info_offset)
            phnum = struct.unpack(f"{order}I", core.read(4))[0]
        if not phnum or phsize != program.size or phoff < header_size:
            return False
        if phoff + phnum * phsize > size:
            return False

        # GDB can return zero with a truncated file, for example under a file
        # size limit. Check declared data ranges before publishing the core.
        core.seek(phoff)
        has_notes = False
        for _ in range(phnum):
            entry = program.unpack(core.read(phsize))
            offset, contents_size = entry[offset_index], entry[size_index]
            if contents_size and offset + contents_size > size:
                return False
            if entry[0] == 4 and contents_size:  # PT_NOTE
                has_notes = True
        return has_notes


@dataclass
class _RankSlot:
    path: Path
    memory: mmap.mmap
    word: int = 0
    since: float = 0.0
    warned: bool = False

    def read(self) -> int:
        return _WORD.unpack_from(self.memory)[0]


@dataclass(frozen=True)
class _RankIdentity:
    pid: int
    start_ticks: int

    def is_current(self) -> bool:
        return _pid_start_ticks(self.pid) == self.start_ticks


@dataclass(frozen=True)
class _CoreTarget:
    rank: int
    identity: _RankIdentity
    word: Optional[int]
    gpu: bool = False

    @property
    def device(self) -> str:
        return "GPU" if self.gpu else "CPU"


def _gdb_capture_command(identity: _RankIdentity) -> str:
    # Once the incident is admitted, peers may be idle or change their
    # progress before their turn. Guard the process identity, not its phase.
    script = f"""
import gdb
import os
import resource
import signal

def is_current():
    with open("/proc/{identity.pid}/stat") as stat_file:
        fields = stat_file.read().rpartition(")")[2].split()
    return fields[0] not in ("Z", "X") and int(fields[19]) == {identity.start_ticks}

disk = os.statvfs(".")
max_bytes = disk.f_bavail * disk.f_frsize - {_MIN_FREE_BYTES}
if max_bytes <= 0:
    raise gdb.GdbError("insufficient free space for Prefill CPU core")
soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
if soft != resource.RLIM_INFINITY:
    max_bytes = min(max_bytes, soft)
resource.setrlimit(resource.RLIMIT_FSIZE, (max_bytes, hard))
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)

if not is_current():
    raise gdb.GdbError("rank identity changed before attach")
gdb.execute("attach {identity.pid}")
try:
    if not is_current():
        raise gdb.GdbError("rank identity changed during attach")
    gdb.execute("generate-core-file core.partial")
finally:
    gdb.execute("detach")
"""
    return f"python exec({script!r})"


def _cuda_gdb_commands(identity: _RankIdentity, staging: Path) -> str:
    check = (
        "from pathlib import Path; "
        f"fields = Path('/proc/{identity.pid}/stat').read_text().rpartition(')')[2].split(); "
        f"raise SystemExit(fields[0] in ('Z', 'X') or int(fields[19]) != {identity.start_ticks})"
    )
    check_identity = "shell " + shlex.join((sys.executable, "-c", check))
    # The CUDA driver writes from the rank's cwd, not the debugger's cwd.
    core_file = shlex.quote(str(staging / "gpu-core-stream"))
    # Keep CUDA local memory available for device stack unwinding.
    flags = (
        "skip_nonrelocated_elf_images,skip_global_memory,"
        "skip_shared_memory,skip_constbank_memory"
    )
    return f"""set confirm off
set pagination off
{check_identity}
if $_shell_exitcode != 0
    echo rank identity changed before CUDA debugger attach\\n
    quit 1
end
attach {identity.pid}
{check_identity}
if $_shell_exitcode != 0
    detach
    echo rank identity changed during CUDA debugger attach\\n
    quit 1
end
generate-core-file --no-cpu --cuda-flags {flags} --cuda-core-file {core_file}
detach
"""


_GPU_READBACK_COMMANDS = f"""target cudacore gpu.nvcudmp.partial
info cuda devices
info cuda contexts
info cuda kernels
info cuda threads kernel all
echo {_GPU_CORE_READABLE}\\n
bt
"""


def _ensure_private_core_dir(path: Path) -> None:
    """Validate ancestors before trusting a directory used by the debugger."""
    owners = (0, os.geteuid())
    for directory in (*reversed(path.parents), path):
        try:
            inode = directory.lstat()
        except FileNotFoundError:
            directory.mkdir(mode=0o700, exist_ok=True)
            inode = directory.lstat()
        mode = inode.st_mode
        if not stat.S_ISDIR(mode) or inode.st_uid not in owners:
            raise OSError(f"untrusted Prefill core directory {directory}")
        if directory == path:
            if inode.st_uid != os.geteuid() or mode & 0o077:
                raise OSError(
                    f"Prefill core directory must be private to the service: {path}"
                )
        elif mode & 0o022 and not mode & stat.S_ISVTX:
            raise OSError(f"writable Prefill core directory ancestor {directory}")


def _publish_core(staging: Path, code: int, gpu: bool) -> Path:
    filename = "gpu.nvcudmp" if gpu else "core"
    partial = staging / f"{filename}.partial"
    if gpu:
        # Scan for the readback marker independent of how long bt output is.
        # Newer versions may succeed in opening a core with no active threads.
        with (staging / "cuda-gdb.log").open(errors="replace") as log:
            lines = (line.strip() for line in log)
            opened = False
            for line in lines:
                if "Failed to read core file:" in line or line == "No CUDA devices.":
                    raise ValueError("CUDA core could not be read back")
                if line == _GPU_CORE_READABLE:
                    opened = True
                    break
        if not opened or code < 0 or not partial.stat().st_size:
            raise ValueError("CUDA core could not be read back")
    elif code != 0 or not _is_elf_core(partial):
        raise ValueError("debugger did not produce a complete ELF core")
    partial.chmod(0o600)
    partial.rename(staging / filename)
    final = staging.with_name(staging.name[1:])
    staging.rename(final)
    return final / filename


def _collect_gpu_core(args, staging: Path, log, limit: int) -> None:
    """The CUDA driver writes to a FIFO. Limit the bytes saved by the helper."""
    fifo = staging / "gpu-core-stream"
    os.mkfifo(fifo, 0o600)
    reader = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    writer = None
    cancelled = False

    def cancel(_signum, _frame):
        nonlocal cancelled
        cancelled = True

    previous_handler = signal.signal(signal.SIGTERM, cancel)
    failure = None
    try:
        # Keep the pipe open until CUDA-GDB exits, including when CUDA was
        # never initialized and the rank does not open the pipe at all.
        writer = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
        with (
            (staging / "gpu.nvcudmp.partial").open("xb", buffering=0) as core,
            subprocess.Popen(
                args,
                cwd=staging,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            ) as debugger,
        ):
            written = 0
            while True:
                if cancelled and debugger.poll() is None:
                    debugger.kill()
                if writer is not None and debugger.poll() is not None:
                    os.close(writer)
                    writer = None
                try:
                    data = os.read(reader, 256 * 1024)
                except BlockingIOError:
                    select.select([reader], [], [], 0.05)
                    continue
                if not data:
                    break
                # Keep draining even after a write error or cancellation so
                # the rank's CUDA driver does not lose its FIFO reader.
                if not cancelled and failure is None:
                    try:
                        if written + len(data) > limit:
                            raise OSError("GPU core exceeds available capture space")
                        if core.write(data) != len(data):
                            raise OSError("incomplete GPU core write")
                        written += len(data)
                    except OSError as exc:
                        failure = exc
            if cancelled:
                raise InterruptedError("GPU core capture cancelled")
            if failure is not None:
                raise failure
            if debugger.wait() != 0 or not written:
                raise ValueError("CUDA debugger did not write a GPU core")
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        if writer is not None:
            os.close(writer)
        os.close(reader)
        fifo.unlink(missing_ok=True)


def _limit_capture_files(max_bytes: int) -> int:
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    if soft != resource.RLIM_INFINITY:
        max_bytes = min(max_bytes, soft)
    resource.setrlimit(resource.RLIMIT_FSIZE, (max_bytes, hard))
    return max_bytes


def _run_capture(staging: Path, identity: _RankIdentity, gpu: bool) -> None:
    name = "cuda-gdb" if gpu else "gdb"
    debugger = shutil.which(name)
    if gpu and debugger is None:
        debugger = shutil.which(_CUDA_GDB)
    if debugger is None:
        raise FileNotFoundError(f"{name} is unavailable")

    _ensure_private_core_dir(staging.parent)
    available = shutil.disk_usage(staging.parent).free - _MIN_FREE_BYTES
    if available <= 0:
        raise OSError("insufficient free space for Prefill core")
    limit = _limit_capture_files(available)

    staging.mkdir(mode=0o700)
    log_path = staging / f"{name}.log"
    try:
        args = [
            debugger,
            "-nx",
            "-nh",
            "-batch",
            "-iex",
            "set auto-load off",
            "-iex",
            "set debuginfod enabled off",
        ]
        with log_path.open("xb") as log:
            if not identity.is_current():
                raise OSError("process identity changed before debugger attach")
            if gpu:
                (staging / "capture.gdb").write_text(
                    _cuda_gdb_commands(identity, staging)
                )
                _collect_gpu_core(args + ["-x", "capture.gdb"], staging, log, limit)
                (staging / "readback.gdb").write_text(_GPU_READBACK_COMMANDS)
                args += ["-x", "readback.gdb"]
            else:
                args += ["-ex", _gdb_capture_command(identity)]
            code = subprocess.run(
                args,
                cwd=staging,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            ).returncode
        saved = _publish_core(staging, code, gpu)
        _LOG.warning("Prefill %s core saved at %s", "GPU" if gpu else "CPU", saved)
        if gpu and code != 0:
            _LOG.warning(
                "Prefill GPU backtrace unavailable. See %s",
                saved.with_name(log_path.name),
            )
    except (OSError, ValueError) as exc:
        tail = ""
        try:
            with log_path.open("rb") as log:
                log.seek(max(0, os.fstat(log.fileno()).st_size - 2048))
                tail = log.read().decode(errors="replace").strip()
        except OSError:
            pass
        raise RuntimeError(f"{exc}; {tail}") from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)


@dataclass
class _Dump:
    target: _CoreTarget
    process: subprocess.Popen
    staging: Path
    deadline: float
    preexisting_stalls: dict[int, tuple[_RankIdentity, int]]
    stop_reason: Optional[str] = None
    traced_when_reaping: bool = False

    def owns_trace(self) -> bool:
        if not self.target.identity.is_current():
            return False
        task_dir = Path(f"/proc/{self.target.identity.pid}/task")
        for status in task_dir.glob("*/status"):
            try:
                tracer = next(
                    int(line.split()[1])
                    for line in status.read_text().splitlines()
                    if line.startswith("TracerPid:")
                )
                if tracer and os.getpgid(tracer) == self.process.pid:
                    return True
            except (OSError, ValueError, StopIteration):
                continue
        return False

    def cancel(self, reason: str) -> None:
        if self.stop_reason is None:
            self.stop_reason = reason
            # Only the helper and its debugger belong to this group. Leave
            # the helper alive to drain the GPU FIFO or clean up partial files.
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


class PrefillWatchdog:
    """Called from ProcessManager's local rank health loop.

    The writer publishes one aligned little-endian uint64 per rank. The monitor
    timestamps changes when it observes them, so delayed polls cannot make a
    phase look older than it is. GDB runs independently and is reaped by poll().
    """

    @classmethod
    def from_config(
        cls, thresholds: str, core_dir: str, gpu_core: bool
    ) -> Optional["PrefillWatchdog"]:
        try:
            deadlines = _parse_thresholds(thresholds)
            if not deadlines:
                return None
            if sys.platform != "linux" or sys.byteorder != "little":
                raise ValueError("Prefill watchdog requires little-endian Linux")
            return cls(deadlines, Path(core_dir), _DEFAULT_TIMEOUT_S, gpu_core=gpu_core)
        except (OSError, ValueError) as exc:
            _LOG.warning("Prefill watchdog disabled: %s", exc)
            return None

    def __init__(
        self,
        thresholds: dict[str, float],
        core_dir: Path,
        timeout: float,
        *,
        gpu_core: bool = False,
    ) -> None:
        self._thresholds = thresholds
        self._unarmed_deadline = max(thresholds.values())
        self._core_dir = Path(os.path.abspath(core_dir))
        self._timeout = timeout
        self._gpu_core = gpu_core
        self._temp = tempfile.TemporaryDirectory(prefix="rtp_prefill_progress_")
        self._slots: dict[int, _RankSlot] = {}
        self._ranks: dict[int, _RankIdentity] = {}
        self._attempted: dict[int, int] = {}
        self._incident = ""
        self._pending: deque[_CoreTarget] = deque()
        self._dump: Optional[_Dump] = None
        self._reaping: list[_Dump] = []
        self._closed = False

    def create_slot(self, world_rank: int) -> str:
        if world_rank not in self._slots:
            path = Path(self._temp.name) / f"rank{world_rank}.slot"
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            try:
                os.ftruncate(fd, _WORD.size)
                memory = mmap.mmap(fd, _WORD.size, access=mmap.ACCESS_READ)
            except OSError:
                path.unlink(missing_ok=True)
                raise
            finally:
                os.close(fd)
            self._slots[world_rank] = _RankSlot(path, memory)
        return str(self._slots[world_rank].path)

    def register_process(self, world_rank: int, pid: Optional[int]) -> None:
        self._attempted.pop(world_rank, None)
        start_ticks = _pid_start_ticks(pid) if pid is not None else None
        if pid is not None and start_ticks is not None:
            self._ranks[world_rank] = _RankIdentity(pid, start_ticks)
        else:
            self._ranks.pop(world_rank, None)

        slot = self._slots.get(world_rank)
        if slot is not None:
            slot.word = 0
            slot.since = 0.0
            slot.warned = False

    def poll(self) -> None:
        if self._closed:
            return
        now = time.monotonic()
        collective_pauses = self._poll_reaping(now)
        if self._dump is not None:
            collective_pauses.append(self._dump)
        self._poll_dump(now)
        reaping_identities = {dump.target.identity for dump in self._reaping}

        expired = []
        for rank, slot in self._slots.items():
            identity = self._ranks.get(rank)
            if identity is None:
                continue
            word = slot.read()
            if word != slot.word:
                slot.word, slot.since = word, now
                slot.warned = False
                self._attempted.pop(rank, None)
            if any(
                capture.preexisting_stalls.get(rank) != (identity, word)
                for capture in collective_pauses
            ):
                # Attaching to one rank can also pause its collective peers.
                # Retain already-expired words and start fresh for new ones.
                slot.since = now
            phase = word & 0xF
            if not 0 < phase < len(_PHASES):
                continue
            if identity in reaping_identities:
                continue
            name = _PHASES[phase]
            deadline = self._deadline(word)
            elapsed = now - slot.since
            if deadline is not None:
                if (
                    elapsed >= deadline
                    and self._attempted.get(rank) != word
                    and identity.is_current()
                    and slot.read() == word
                ):
                    expired.append((slot.since, rank, slot, word))
            elif (
                elapsed >= self._unarmed_deadline
                and not slot.warned
                and identity.is_current()
                and slot.read() == word
            ):
                _LOG.warning(
                    "Suspected Prefill stall without a configured deadline: "
                    "rank=%d pid=%d phase=%s seq=%d observed_seconds=%.2f",
                    rank,
                    identity.pid,
                    name,
                    word >> 36,
                    elapsed,
                )
                slot.warned = True
        if expired and not self._pending and self._dump is None:
            since, rank, slot, word = min(expired)
            identity = self._ranks[rank]
            if identity.is_current() and slot.read() == word:
                for _, stalled_rank, _, stalled_word in expired:
                    self._attempted[stalled_rank] = stalled_word
                self._begin_incident(rank, word, now - since)

        while self._pending and self._dump is None:
            self._launch(self._pending.popleft(), now)

    def _begin_incident(self, rank: int, word: int, elapsed: float) -> None:
        self._incident = uuid4().hex
        targets = [_CoreTarget(rank, self._ranks[rank], word)]
        for peer_rank, identity in sorted(self._ranks.items()):
            if peer_rank != rank:
                slot = self._slots.get(peer_rank)
                targets.append(
                    _CoreTarget(
                        peer_rank, identity, slot.read() if slot is not None else None
                    )
                )
        for target in targets:
            if self._gpu_core:
                self._pending.append(replace(target, gpu=True))
            self._pending.append(target)
        _LOG.warning(
            "Suspected Prefill stall: rank=%d pid=%d tid=%d phase=%s layer=%d "
            "seq=%d observed_seconds=%.2f incident=%s capture_ranks=%s",
            rank,
            self._ranks[rank].pid,
            (word >> 14) & 0x3FFFFF,
            _PHASES[word & 0xF],
            ((word >> 4) & 0x3FF) - 1,
            word >> 36,
            elapsed,
            self._incident,
            [target.rank for target in targets],
        )

    def _launch(self, target: _CoreTarget, now: float) -> None:
        rank, identity = target.rank, target.identity
        if self._ranks.get(rank) != identity or not identity.is_current():
            _LOG.warning(
                "Prefill %s core unavailable for rank %d incident=%s: process identity changed",
                target.device,
                rank,
                self._incident,
            )
            return
        if any(dump.target.identity == identity for dump in self._reaping):
            _LOG.warning(
                "Prefill %s core unavailable for rank %d incident=%s: previous debugger has not exited",
                target.device,
                rank,
                self._incident,
            )
            return

        word = target.word
        phase = word & 0xF if word is not None else len(_PHASES)
        name = _PHASES[phase] if phase < len(_PHASES) else "unknown"
        sequence = word >> 36 if word is not None else 0
        staging = self._core_dir / (
            f".rank{rank}-pid{identity.pid}-start{identity.start_ticks}-{name}-seq{sequence}-"
            f"incident{self._incident}-{uuid4().hex[:12]}"
        )
        preexisting_stalls = {}
        for observed_rank, slot in self._slots.items():
            observed_identity = self._ranks.get(observed_rank)
            if observed_identity is None:
                continue
            current = slot.read()
            deadline = self._deadline(current)
            if (
                current == slot.word
                and deadline is not None
                and now - slot.since >= deadline
            ):
                preexisting_stalls[observed_rank] = (observed_identity, current)
                if observed_rank == rank:
                    self._attempted[rank] = current
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    __file__,
                    str(staging),
                    str(identity.pid),
                    str(identity.start_ticks),
                    target.device,
                ],
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            self._dump = _Dump(
                target,
                process,
                staging,
                time.monotonic() + self._timeout,
                preexisting_stalls,
            )
        except OSError as exc:
            _LOG.warning(
                "Prefill %s core unavailable for rank %d incident=%s: %s",
                target.device,
                rank,
                self._incident,
                exc,
            )

    def _deadline(self, word: int) -> Optional[float]:
        phase = word & 0xF
        if 0 < phase < len(_PHASES):
            name = _PHASES[phase]
            return self._thresholds.get("tail" if name == "gpu_done" else name)
        return None

    def _poll_dump(self, now: float) -> None:
        dump = self._dump
        if dump is None:
            return
        code = dump.process.poll()
        if code is not None:
            self._dump = None
            if dump.stop_reason is not None or code != 0:
                _LOG.warning(
                    "Prefill %s core unavailable for %s (%s, exit=%d)",
                    dump.target.device,
                    dump.staging.name,
                    dump.stop_reason or "capture failed",
                    code,
                )
        elif now >= dump.deadline:
            if dump.stop_reason is None:
                dump.cancel("debugger deadline exceeded")
                dump.deadline = now + _KILL_REAP_GRACE_S
            else:
                _LOG.warning(
                    "Prefill %s core unavailable for %s: debugger deadline exceeded; "
                    "any partial output remains private at %s",
                    dump.target.device,
                    dump.staging.name,
                    dump.staging,
                )
                dump.traced_when_reaping = dump.owns_trace()
                self._reaping.append(dump)
                self._dump = None

    def _poll_reaping(self, now: float) -> list[_Dump]:
        remaining = []
        collective_pauses = []
        for dump in self._reaping:
            running = dump.process.poll() is None
            traced = running and dump.owns_trace()
            if traced or dump.traced_when_reaping:
                # Also credit the poll that first observes detach.
                collective_pauses.append(dump)
            dump.traced_when_reaping = traced
            if running:
                remaining.append(dump)
                continue
            rank, identity = dump.target.rank, dump.target.identity
            slot = self._slots.get(rank)
            if (
                slot is not None
                and self._ranks.get(rank) == identity
                and dump.preexisting_stalls.get(rank) != (identity, slot.read())
            ):
                slot.since = now
        self._reaping = remaining
        return collective_pauses

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pending.clear()
        self._poll_reaping(time.monotonic())
        dump, self._dump = self._dump, None
        if dump is not None and dump.process.poll() is None:
            dump.cancel("watchdog closed")
            try:
                dump.process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                _LOG.warning(
                    "Capture helper has not exited; partial output at %s",
                    dump.staging,
                )
        for slot in self._slots.values():
            slot.memory.close()
        self._temp.cleanup()


def _interrupt_capture(_signum, _frame) -> None:
    raise InterruptedError("capture cancelled")


def _capture_main() -> int:
    import resource

    staging, pid, start_ticks, device = sys.argv[1:]
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, _interrupt_capture)
    try:
        _run_capture(
            Path(staging), _RankIdentity(int(pid), int(start_ticks)), device == "GPU"
        )
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        _LOG.warning("Prefill %s core %s failed: %s", device, Path(staging).name, exc)
        return 1


if __name__ == "__main__":
    sys.exit(_capture_main())
