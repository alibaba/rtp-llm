"""Capabilities of the locked packages used for native UE8M0 test fixtures."""

def checkpoint_reader_test_env():
    # Zero is intentional for wheel-less locks; CUDA/ROCm must exercise the
    # real reader even if an incorrectly packaged wheel is absent at runtime.
    base = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
    native = dict(base, NEWLOADER_TEST_NATIVE_FST = "1")
    return select({
        "@//:cuda_pre_12_9": native,
        "@//:using_cuda12_9_x86": native,
        "@//:using_cuda13_x86": native,
        "@//:using_cuda12_arm": native,
        "@//:using_cuda13_arm": native,
        "@//:using_rocm": native,
        "//conditions:default": dict(base, NEWLOADER_TEST_NATIVE_FST = "0"),
    })

def ue8m0_test_platforms(require_fst = False):
    return select({
        "@//:using_cuda13_arm": [],
        "@//:using_cuda13_x86": [],
        # PPU ships safetensors with UE8M0 support, but not the FST reader.
        "@//:using_ppu": ["@platforms//:incompatible"] if require_fst else [],
        "//conditions:default": ["@platforms//:incompatible"],
    })

def ue8m0_test_env():
    base = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    return select({
        "@//:using_ppu": dict(base, NEWLOADER_TEST_NATIVE_FST = "0"),
        "//conditions:default": dict(base, NEWLOADER_TEST_NATIVE_FST = "1"),
    })
