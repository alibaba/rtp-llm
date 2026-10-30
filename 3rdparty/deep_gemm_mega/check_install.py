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
                "import ast",
                "from deep_gemm_mega import mega",
                "native = deep_gemm_mega.deep_gemm_mega_cpp",
                "assert mega._C is native",
                "tree = ast.parse(Path(mega.__file__).read_text())",
                "required = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == '_C'}",
                "missing = sorted(name for name in required if not callable(getattr(native, name, None)))",
                "assert not missing, f'MegaMoE native APIs missing from {native.__file__}: {missing}'",
                "num_bytes, slice_buffers = native.get_symm_buffer_size_for_mega_moe(8, 256, 128, 6, 4096, 2048, 'fp4xfp4', 'swiglu', 0)",
                "assert isinstance(num_bytes, int) and num_bytes > 0 and callable(slice_buffers)",
                "print(f'Validated MegaMoE native APIs: {sorted(required)}; buffer_bytes={num_bytes}')",
                "for name in ('get_symm_buffer_for_mega_moe', 'transform_weights_for_mega_moe', 'mega_moe_pre_dispatch', 'fp4_fp4_mega_moe'):",
                "    assert callable(getattr(deep_gemm_mega, name))",
                "assert (Path(deep_gemm_mega.__file__).parent / 'configs/dsv4_flash_bm64/ZW-M890P_fp4_megamoe.lut').is_file()",
                "import ctypes",
                "for kind, entries in (('sparse', ('create', 'begin', 'finish', 'destroy')), ('signal', ('create', 'exchange', 'destroy'))):",
                "    library = ctypes.CDLL(str(Path(deep_gemm_mega.__file__).parent / 'lib' / f'lib{kind}_gin.so'))",
                "    for entry in entries:",
                "        assert getattr(library, f'{kind}_gin_{entry}') is not None",
            ])],
            check=True,
        )
    print(f"Independent DeepGEMM installations: official={official.version}, mega={mega.version}")


if __name__ == "__main__":
    check_install()
