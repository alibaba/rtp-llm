"""Check independent ownership and both import orders without allocating on a PPU."""

import importlib.metadata as metadata
import subprocess
import sys


def check_install():
    official = metadata.distribution("deep_gemm")
    mega = metadata.distribution("deep_gemm_mega")
    official_files = {p.locate().resolve() for p in official.files or ()}
    mega_files = {p.locate().resolve() for p in mega.files or ()}
    overlap = official_files & mega_files
    if overlap:
        raise RuntimeError(f"DeepGEMM packages own overlapping files: {overlap}")
    for order in (("deep_gemm", "deep_gemm_mega"), ("deep_gemm_mega", "deep_gemm")):
        subprocess.run(
            [sys.executable, "-c", "\n".join([
                f"import {order[0]}",
                f"import {order[1]}",
                "from pathlib import Path",
                "import deep_gemm_mega",
                "for name in ('get_symm_buffer_for_mega_moe', 'transform_weights_for_mega_moe', 'mega_moe_pre_dispatch', 'fp4_fp4_mega_moe'):",
                "    assert callable(getattr(deep_gemm_mega, name))",
                "assert (Path(deep_gemm_mega.__file__).parent / 'configs/dsv4_flash_bm64/ZW-M890P_fp4_megamoe.lut').is_file()",
            ])],
            check=True,
        )
    print(f"Independent DeepGEMM installations: official={official.version}, mega={mega.version}")


if __name__ == "__main__":
    check_install()
