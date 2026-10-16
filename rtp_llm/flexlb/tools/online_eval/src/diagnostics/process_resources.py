"""Optional process RSS and explicitly scoped cgroup v2 accounting."""
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class ResourceSnapshot:
    cpu_usec: int | None = None
    throttled_usec: int | None = None
    memory_bytes: int | None = None
    rss_bytes: int | None = None
    cpu_max: str | None = None
    memory_max: str | None = None
    cgroup_path: str | None = None
    accounting_scope: str = 'cgroup_v2'
    unavailable: str | None = None


def process_resources(pid, *, proc_root=Path('/proc'), cgroup_root=Path('/sys/fs/cgroup')):
    """Unsupported accounting is explicit missing evidence, never zero usage.

    CPU and memory.current belong to the cgroup; only VmRSS belongs to the
    selected process. A v1 hierarchy is not interpreted as unified accounting.
    """
    snapshot = ResourceSnapshot()
    process = Path(proc_root) / str(pid)
    try:
        status = (process/'status').read_text().splitlines()
        rss = next((int(line.split()[1])*1024 for line in status if line.startswith('VmRSS:')), None)
        snapshot = replace(snapshot, rss_bytes=rss)
        memberships = (process/'cgroup').read_text().splitlines()
        path = next((line.split('::', 1)[1] for line in memberships if line.startswith('0::')), None)
        if path is None:
            return asdict(replace(snapshot, unavailable='cgroup v2 accounting unavailable'))
        snapshot = replace(snapshot, cgroup_path=path)
        group = Path(cgroup_root) / path.lstrip('/')
        cpu = {k: int(v) for k, v in (line.split() for line in (group/'cpu.stat').read_text().splitlines())}
        return asdict(replace(snapshot, cpu_usec=cpu['usage_usec'], throttled_usec=cpu.get('throttled_usec'),
            memory_bytes=int((group/'memory.current').read_text()), cpu_max=(group/'cpu.max').read_text().strip(),
            memory_max=(group/'memory.max').read_text().strip()))
    except FileNotFoundError:
        return asdict(replace(snapshot, unavailable='process or cgroup accounting files unavailable'))
