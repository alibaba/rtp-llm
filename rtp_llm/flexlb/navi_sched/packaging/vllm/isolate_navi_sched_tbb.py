"""Keep NaviSched's legacy TBB independent of the PPU runtime's TBB."""

from pathlib import Path
import subprocess
import sys
import sysconfig


def isolate(package, patchelf):
    original = package / "libtbb.so"
    private = package / "libnavi_sched_tbb.so"
    core = package / "_navi_sched_core.so"
    needed = subprocess.check_output(
        [patchelf, "--print-needed", str(core)], text=True
    ).splitlines()
    if private.name in needed:
        assert private.is_file() and not original.exists()
        return
    assert original.name in needed and original.is_file()
    original.rename(private)
    subprocess.run([patchelf, "--set-soname", private.name, str(private)], check=True)
    subprocess.run(
        [patchelf, "--replace-needed", original.name, private.name, str(core)],
        check=True,
    )
    assert (
        private.name
        in subprocess.check_output(
            [patchelf, "--print-needed", str(core)], text=True
        ).splitlines()
    )


if __name__ == "__main__":
    isolate(Path(sysconfig.get_path("purelib")) / "navi_sched", sys.argv[1])
