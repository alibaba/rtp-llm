"""Device-only DeepSeek newloader numerical tests.

Only these cases are exported by the GPU targets, so CUDA and ROCm lanes do
not repeat the CPU loader/configuration suite.
"""

import unittest

# Bazel puts the main script's directory on sys.path. Keep this deliberate
# sibling import: the fully qualified path executes deepseek_v3/__init__.py
# before the shared test module initializes its imports, which makes the ROCm
# lane eagerly import CUDA-only flashinfer modules and fail during collection.
from test_deepseek_newloader import DeepSeekNewloaderTest as _SharedTests


class DeepSeekNewloaderGpuTest(unittest.TestCase):
    pass


for _name in dir(_SharedTests):
    if _name.startswith("_gpu_"):
        setattr(
            DeepSeekNewloaderGpuTest,
            _name.replace("_gpu_", "test_", 1),
            getattr(_SharedTests, _name),
        )


del _SharedTests


if __name__ == "__main__":
    unittest.main()
