import fcntl
import hashlib
import importlib.util
import logging
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

PROGRAM_NAME = "tipc"


def _source_signature(root: Path, build_args: list) -> str:
    digest = hashlib.sha256(repr(build_args).encode())
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()[:16]


class __CompileHelper__:
    def __init__(self) -> None:
        self.__CUDA_EXTENTION__ = None

    def compile(self):
        """
        Compiles a CUDA extension from all source files in the source directory.

        This function automatically finds all .c, .cc, .cpp, and .cu files
        in the specified source directory and compiles them using JIT compilation.
        The compiled extension is a CUDAExtension object that can be called from Python.

        Requires CUDA and C++17 for compilation.
        """
        print(
            f"{PROGRAM_NAME} is currently compiling the code, which may take some time. "
            f"If any errors occur, please check your compilation environment: {PROGRAM_NAME} "
            "requires C++17 and CUDA."
        )
        source_dir = Path(__file__).with_name("csrc")
        sources = self._find_all_source_files(source_dir)
        cflags, cuda_cflags = ["-O3"], ["-O3", "-use_fast_math"]
        build_dir = (
            Path(
                os.environ.get("TORCH_EXTENSIONS_DIR")
                or Path(__file__).with_name("build")
            )
            / PROGRAM_NAME
            / _source_signature(source_dir, [cflags, cuda_cflags, None, True])
        )
        build_dir.mkdir(parents=True, exist_ok=True)
        so = build_dir / f"{PROGRAM_NAME}.so"
        # The kernel releases flock on exit, so a stale path cannot block builders.
        with (build_dir / ".load.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            # We hold the flock, so any leftover FileBaton `lock` is a dead
            # builder's corpse; drop it or load() would wait on it forever.
            (build_dir / "lock").unlink(missing_ok=True)
            if so.is_file():
                try:
                    spec = importlib.util.spec_from_file_location(PROGRAM_NAME, so)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self.__CUDA_EXTENTION__ = module
                    return module
                except Exception:
                    logging.warning("TIPC cache is unusable; rebuilding", exc_info=True)
                    so.unlink(missing_ok=True)
            self.__CUDA_EXTENTION__ = load(
                PROGRAM_NAME,
                sources,
                build_directory=str(build_dir),
                extra_include_paths=[str(source_dir)],
                with_cuda=True,
                extra_cuda_cflags=cuda_cflags,
                extra_cflags=cflags,
            )
        return self.__CUDA_EXTENTION__

    def _find_all_source_files(self, directory: Path) -> list[str]:
        return sorted(
            str(path)
            for path in directory.rglob("*")
            if path.suffix in (".c", ".cc", ".cpp", ".cu")
        )

    @property
    def CUDA_EXTENSION(self):
        if self.__CUDA_EXTENTION__ is None:
            self.compile()
        return self.__CUDA_EXTENTION__


CompileHelper = __CompileHelper__()


class CUDA:
    """Helper class for calling Compiled Methods."""

    @staticmethod
    def build_cuipc_meta(t: torch.Tensor) -> bytes:
        if not t.is_cuda:
            raise ValueError("Invalid tensor, not on cuda.")

        # Ensure the tensor is contiguous and synchronized before export.
        if not t.is_contiguous():
            t = t.contiguous()

        torch.cuda.synchronize(device=t.device)
        return CompileHelper.CUDA_EXTENSION.export_tensor_ipc(t)

    @staticmethod
    def build_tensor_from_meta(ipc: bytes) -> torch.Tensor:
        if not isinstance(ipc, bytes):
            raise TypeError("invalid input type, expected bytes.")
        return CompileHelper.CUDA_EXTENSION.import_tensor_ipc(ipc)


__all__ = ["CUDA"]
