import logging
import os
import sys
import unittest

import pytest
import torch

from rtp_llm.test.utils.platform_skip import skip_if_hip

# flashinfer/mla tests are CUDA-only; skip cleanly on ROCm during collection.
skip_if_hip("flashinfer JIT tests are skipped on ROCm")

pytest.importorskip("flashinfer")

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla import (
    warmup_flashinfer_python,
)

# 应用torch_patch来修复flashinfer JIT编译路径问题
from rtp_llm.utils import torch_patch  # noqa: F401


class FlashInferJitTest(unittest.TestCase):
    """测试flashinfer JIT功能是否在Bazel环境中正常工作"""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

        # 设置日志级别
        logging.basicConfig(level=logging.INFO)

    def test_flashinfer_jit_warmup(self):
        """测试flashinfer JIT预热和编译功能"""
        try:
            # 验证 torch 和 flashinfer 是从 .cache 导入而不是 Bazel runfiles
            import flashinfer

            torch_path = torch.__file__
            flashinfer_path = flashinfer.__file__

            logging.info(f"Torch imported from: {torch_path}")
            logging.info(f"FlashInfer imported from: {flashinfer_path}")

            # 断言：torch 路径不能包含 runfiles
            self.assertNotIn(
                "runfiles",
                torch_path,
                f"Torch should not be imported from Bazel runfiles, but got: {torch_path}",
            )
            # 断言：torch 路径应该包含 .cache
            self.assertIn(
                ".cache",
                torch_path,
                f"Torch should be imported from .cache, but got: {torch_path}",
            )

            # 断言：flashinfer 路径不能包含 runfiles
            self.assertNotIn(
                "runfiles",
                flashinfer_path,
                f"FlashInfer should not be imported from Bazel runfiles, but got: {flashinfer_path}",
            )
            # 断言：flashinfer 路径应该包含 .cache
            self.assertIn(
                ".cache",
                flashinfer_path,
                f"FlashInfer should be imported from .cache, but got: {flashinfer_path}",
            )

            logging.info("✓ Package import paths validated successfully")

            # 测试torch.utils.cpp_extension路径是否正确配置
            import torch.utils.cpp_extension as cpp_ext

            logging.info(
                f"warmup_flashinfer_python before update cpp_extension.include_paths: _HERE = {cpp_ext._HERE}"
            )
            logging.info(
                f"warmup_flashinfer_python before update cpp_extension.include_paths: _TORCH_PATH = {cpp_ext._TORCH_PATH}"
            )
            logging.info(
                f"warmup_flashinfer_python before update cpp_extension.include_paths: TORCH_LIB_PATH = {cpp_ext.TORCH_LIB_PATH}"
            )

            # 执行flashinfer预热，这会触发JIT编译
            warmup_flashinfer_python()

            logging.info(
                f"warmup_flashinfer_python cpp_extension.include_paths: _HERE = {cpp_ext._HERE}"
            )
            logging.info(
                f"warmup_flashinfer_python cpp_extension.include_paths: _TORCH_PATH = {cpp_ext._TORCH_PATH}"
            )
            logging.info(
                f"warmup_flashinfer_python cpp_extension.include_paths: TORCH_LIB_PATH = {cpp_ext.TORCH_LIB_PATH}"
            )

            logging.info("FlashInfer JIT warmup completed successfully")

        except ImportError as e:
            self.skipTest(f"FlashInfer not available: {e}")
        except Exception as e:
            self.fail(f"FlashInfer JIT warmup failed: {e}")


if __name__ == "__main__":
    unittest.main()
