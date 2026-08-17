import fcntl
import hashlib
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

PROGRAM_NAME = "tipc"


def _source_signature(root: Path, build_args: list) -> str:
    digest = hashlib.sha256(repr((str(root.resolve()), build_args)).encode())
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()[:16]


class __CompileHelper__:
    def __init__(self) -> None:
        self.__CUDA_EXTENSION__ = None

    def compile(self):
        """Compile (or warm-load) the CUDA extension; needs C++17 + CUDA."""
        from rtp_llm.utils.util import COMPILE_FLAG_ENVS, torch_abi_fingerprint

        source_dir = Path(__file__).with_name("csrc")
        sources = self._find_all_source_files(source_dir)
        cflags, cuda_cflags = ["-O3"], ["-O3", "-use_fast_math"]
        # Keys build_dir below, so a wrong arch would serve a wrong binary: fail
        # hard here, unlike the cache scope probe that only gates caching.
        major, minor = torch.cuda.get_device_capability()
        fingerprint = torch_abi_fingerprint()
        if fingerprint is None:
            raise RuntimeError("torch C++ ABI flag unavailable; cannot key TIPC build")
        build_args = [
            *cflags,
            *cuda_cflags,
            f"sm_{major}{minor}",
            torch.version.cuda or "",
            *map(str, fingerprint),
            # These envs change the binary without touching sources (e.g. load()
            # derives -gencode from TORCH_CUDA_ARCH_LIST).
            *(f"{name}={os.environ.get(name, '')}" for name in COMPILE_FLAG_ENVS),
        ]
        build_dir = (
            Path(
                os.environ.get("TORCH_EXTENSIONS_DIR")
                or Path(__file__).with_name("build")
            )
            / PROGRAM_NAME
            / _source_signature(source_dir, build_args)
        )
        build_dir.mkdir(parents=True, exist_ok=True)
        # Serialize load() and clear a stale FileBaton left by a killed builder.
        lock_path = build_dir / ".load.lock"
        with os.fdopen(
            os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o666), "r+"
        ) as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            (build_dir / "lock").unlink(missing_ok=True)
            self.__CUDA_EXTENSION__ = load(
                PROGRAM_NAME,
                sources,
                build_directory=str(build_dir),
                extra_include_paths=[str(source_dir)],
                with_cuda=True,
                extra_cuda_cflags=cuda_cflags,
                extra_cflags=cflags,
            )
        return self.__CUDA_EXTENSION__

    def _find_all_source_files(self, directory: Path) -> list[str]:
        return sorted(
            str(path)
            for path in directory.rglob("*")
            if path.suffix in (".c", ".cc", ".cpp", ".cu")
        )

    @property
    def CUDA_EXTENSION(self):
        if self.__CUDA_EXTENSION__ is None:
            self.compile()
        return self.__CUDA_EXTENSION__


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
