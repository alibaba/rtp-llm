"""Select the FA4 CuTe subpackage when FA2 and FA4 are installed together."""

import importlib.metadata
import sys
from pathlib import Path

from packaging.requirements import Requirement


def prefer_flash_attn_4_cute() -> None:
    try:
        distribution = importlib.metadata.distribution("flash-attn-4")
        package_dir = Path(distribution.locate_file("flash_attn"))
    except importlib.metadata.PackageNotFoundError:
        return
    if not (package_dir / "cute" / "interface.py").is_file():
        raise ImportError(f"flash-attn-4 has no CuTe interface at {package_dir}")

    # The private FA4 signature alone does not prove its JIT ABI is usable.
    # Check the installed package's declared compiler/kernel requirements before
    # selecting it; an incompatible installation keeps the native SWA reader.
    for text in getattr(distribution, "requires", None) or ():
        requirement = Requirement(text)
        if requirement.name not in ("nvidia-cutlass-dsl", "quack-kernels"):
            continue
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        try:
            version = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError as error:
            raise ImportError(f"FA4 requires {requirement}") from error
        if not requirement.specifier.contains(version, prereleases=True):
            raise ImportError(f"FA4 requires {requirement}, installed {version}")

    imported_cute = sys.modules.get("flash_attn.cute")
    imported_path = getattr(imported_cute, "__file__", None)
    if imported_cute is not None and (
        imported_path is None or not Path(imported_path).is_relative_to(package_dir)
    ):
        raise ImportError(
            "flash_attn.cute was imported from FlashAttention 2 before MiniMax-M3.1"
        )

    import flash_attn

    flash_attn.__path__ = [str(package_dir)] + [
        path for path in flash_attn.__path__ if path != str(package_dir)
    ]
