"""Read native Triton groups after the managed cache tree has moved."""

import json
from pathlib import Path

from triton.runtime.cache import FileCacheManager


def _is_filename(name: object) -> bool:
    return (
        isinstance(name, str)
        and bool(name)
        and name not in (".", "..")
        and Path(name).name == name
        and "\\" not in name
        and "\0" not in name
    )


class RelocatableFileCacheManager(FileCacheManager):
    def get_group(self, filename: str) -> dict[str, str] | None:
        if not _is_filename(filename):
            return None
        group_path = self.get_file(f"__grp__{filename}")
        if group_path is None or Path(group_path).is_symlink():
            return None
        try:
            child_paths = json.loads(Path(group_path).read_text()).get("child_paths")
            if not isinstance(child_paths, dict) or not all(
                _is_filename(name) for name in child_paths
            ):
                return None
            # Native groups contain absolute producer paths. Only the filenames
            # identify members of this key; the archived JSON stays untouched.
            result = {}
            for name in child_paths:
                path = self.get_file(name)
                if path and Path(path).is_file() and not Path(path).is_symlink():
                    result[name] = path
            binaries = {
                name for name in child_paths if name.endswith((".cubin", ".hsaco"))
            }
            if filename not in result or not binaries or not binaries <= result.keys():
                return None
            return result
        except (OSError, ValueError, AttributeError):
            return None
