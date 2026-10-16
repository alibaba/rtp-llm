"""Validate the two bundle JVM artifacts before publishing the CPU test image."""
import io
from pathlib import Path
import sys
import zipfile


REQUIRED = {"org/flexlb/monitor/KMonitorAdapter.class", "com/taobao/kmonitor/KMonitor.class"}


def verify(path):
    found = set()
    with zipfile.ZipFile(path) as jar:
        names = set(jar.namelist())
        found.update(REQUIRED & names)
        found.update(name for name in REQUIRED if "BOOT-INF/classes/" + name in names)
        for name in names:
            if name.startswith("BOOT-INF/lib/") and name.endswith(".jar"):
                with zipfile.ZipFile(io.BytesIO(jar.read(name))) as dependency:
                    found.update(REQUIRED & set(dependency.namelist()))
    if missing := REQUIRED - found:
        raise ValueError(f"{path.name} missing KMonitor classes: {sorted(missing)}")


if __name__ == "__main__":
    for filename in ("master.jar", "mock.jar"):
        verify(Path(sys.argv[1]) / filename)
    print("Master and Mock KMonitor artifacts verified")
