import os
from typing import List

import torch
from torch.utils.cpp_extension import CUDAExtension, load

SOURCE_DIR = "tipc/csrc"
EXTENSION_BUILD_DIR = "tipc/build"
PROGRAM_NAME = "tipc"

class __CompileHelper__:
    def __init__(self) -> None:
        self.BUILD_DIR = EXTENSION_BUILD_DIR
        self.__CUDA_EXTENTION__ = None

        if torch.__version__ < '1.6.0':
            raise RuntimeError(f"{PROGRAM_NAME} cannot finish compile; PyTorch version 1.6 or higher is required.")

    def compile(self) -> CUDAExtension:
        """
        Compiles a CUDA extension from all source files in the source directory.

        This function automatically finds all .c, .cc, .cpp, and .cu files
        in the specified source directory and compiles them using JIT compilation.
        The compiled extension is a CUDAExtension object that can be called from Python.

        Requires CUDA and C++17 for compilation.
        """
        print(f'{PROGRAM_NAME} is currently compiling the code, which may take some time. '
              f'If any errors occur, please check your compilation environment: {PROGRAM_NAME} '
              'requires C++17 and CUDA.')

        # delete lock file.
        lock_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), self.BUILD_DIR, 'lock')
        if os.path.exists(lock_file): 
            try: os.remove(lock_file)
            except Exception as e:
                raise PermissionError(f'Can not delete lock file at {lock_file}, delete it first!')

        sources = self._find_all_source_files(os.path.join(os.path.dirname(os.path.dirname(__file__)), SOURCE_DIR))

        self.__CUDA_EXTENTION__ = load(
            name=PROGRAM_NAME,
            sources=sources,
            extra_include_paths=[
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csrc'),
            ],
            build_directory=os.path.join(os.path.dirname(os.path.dirname(__file__)), self.BUILD_DIR),
            with_cuda=True,
            extra_cuda_cflags=['-O3', '-use_fast_math'],
            extra_cflags=['-O3'],
        )
        return self.__CUDA_EXTENTION__

    def _find_all_source_files(self, directory: str) -> List[str]:
        """ Recursively finds all C/C++ and CUDA source files in a directory. """
        source_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(('.c', '.cc', '.cpp', '.cu')):
                    source_files.append(os.path.join(root, file))
        return source_files

    @property
    def CUDA_EXTENSION(self):
        if self.__CUDA_EXTENTION__ is None:
            self.compile()
        return self.__CUDA_EXTENTION__


CompileHelper = __CompileHelper__()

def CONTIGUOUS_TENSOR(tensor: torch.Tensor):
    """ Helper function """
    if tensor.is_contiguous(): 
        return tensor
    else: 
        return tensor.contiguous()

class CUDA:
    """ Helper class for calling Compiled Methods. """
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