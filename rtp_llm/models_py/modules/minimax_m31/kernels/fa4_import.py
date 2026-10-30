"""Select the FA4 CuTe subpackage when FA2 and FA4 are installed together."""

import importlib.metadata
import sys
from pathlib import Path


def prefer_flash_attn_4_cute() -> None:
    try:
        package_dir = Path(
            importlib.metadata.distribution("flash-attn-4").locate_file("flash_attn")
        )
    except importlib.metadata.PackageNotFoundError:
        return
    if not (package_dir / "cute" / "interface.py").is_file():
        raise ImportError(f"flash-attn-4 has no CuTe interface at {package_dir}")

    imported_cute = sys.modules.get("flash_attn.cute")
    if imported_cute is not None and not Path(imported_cute.__file__).is_relative_to(
        package_dir
    ):
        raise ImportError(
            "flash_attn.cute was imported from FlashAttention 2 before MiniMax-M3.1"
        )

    import flash_attn

    flash_attn.__path__ = [str(package_dir)] + [
        path for path in flash_attn.__path__ if path != str(package_dir)
    ]
