if __name__ == "__main__":
    import os
    import sys

    # Load torch shared libraries first so that the test .so (linked against
    # torch_deps) can resolve libtorch* (e.g. libtorch_nvshmem.so) without
    # relying on LD_LIBRARY_PATH (same pattern as rtp_llm/ops/__init__.py).
    import torch

    torch_lib_dir = os.path.join(torch.__path__[0], "lib")
    nvshmem_so = os.path.join(torch_lib_dir, "libtorch_nvshmem.so")
    if os.path.exists(nvshmem_so):
        from ctypes import cdll

        cdll.LoadLibrary(nvshmem_so)

    from sleep_service_unittest_lib import RunCppUnittest

    sys.exit(RunCppUnittest())
