import glob
import os
import sys
import sysconfig


def _find_python_shared_library(
    libdir: str | None, ldlibrary: str | None, version: str
) -> str:
    candidates: list[str] = []
    if libdir:
        if ldlibrary and (".so" in ldlibrary or ldlibrary.endswith(".dylib")):
            candidates.append(os.path.join(libdir, ldlibrary))
        for suffix in (".so", ".dylib"):
            candidates.extend(
                sorted(glob.glob(os.path.join(libdir, f"libpython{version}{suffix}*")))
            )

    for path in dict.fromkeys(candidates):
        if os.path.isfile(path):
            return path
    raise RuntimeError(
        f"could not find a Python {version} shared library in {libdir or '<unset>'}"
    )


def find_python_shared_library() -> str:
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    return _find_python_shared_library(
        sysconfig.get_config_var("LIBDIR"),
        sysconfig.get_config_var("LDLIBRARY"),
        version,
    )
