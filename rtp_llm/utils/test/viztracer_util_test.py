#!/usr/bin/env python3

import glob
import importlib
import logging
import os
import time
from unittest import TestCase, main

from rtp_llm.config.log_config import LOGGING_CONFIG
from rtp_llm.config.py_config_modules import StaticConfig
from rtp_llm.utils.viztracer_util import trace_func, trace_scope


class VizTracerTest(TestCase):
    def setUp(self):
        os.environ["VIZTRACER_ENABLE"] = "1"
        os.environ["VIZTRACER_MIN_DURATION_MS"] = "50"
        os.environ["LOG_PATH"] = self._testdata_path()
        # 重新导入
        StaticConfig.update_from_env()
        import rtp_llm.utils.viztracer_util

        importlib.reload(rtp_llm.utils.viztracer_util)
        self.pid = os.getpid()
        print("Current PID:", self.pid)
        return super().setUp()

    @staticmethod
    def _testdata_path():
        return "rtp_llm/utils/test/testdata/"

    def test_trace_scope_file_exists(self):
        """测试 trace_scope 是否生成文件"""
        test_dir = self._testdata_path()
        os.makedirs(test_dir, exist_ok=True)

        with trace_scope("test_scope", output_dir=test_dir, min_duration_ms=10):
            time.sleep(1)

        files = glob.glob(os.path.join(test_dir, f"{self.pid}*test_scope.json"))
        self.assertTrue(files[0].endswith("test_scope.json"), files)
        # 检查文件是否存在，不强制要求
        self.assertGreater(os.path.getsize(files[0]), 0)

        """测试 trace_func 是否生成文件"""
        test_dir = self._testdata_path()

        @trace_func("test_func")
        def slow_func():
            time.sleep(1)
            return "done"

        result = slow_func()
        self.assertEqual(result, "done")

        files = glob.glob(os.path.join(test_dir, f"{self.pid}*test_func.json"))
        self.assertTrue(files[0].endswith("test_func.json"), files)
        self.assertGreater(os.path.getsize(files[0]), 0)

        """测试禁用时不生成文件"""
        os.environ["VIZTRACER_ENABLE"] = "0"
        StaticConfig.update_from_env()

        # 重新导入
        import importlib

        import rtp_llm.utils.viztracer_util

        importlib.reload(rtp_llm.utils.viztracer_util)

        from rtp_llm.utils.viztracer_util import trace_scope as new_trace_scope

        test_dir = self._testdata_path()
        old_count = len(
            glob.glob(os.path.join(test_dir, f"{self.pid}*disabled_test.json"))
        )

        with new_trace_scope("disabled_test", output_dir=test_dir):
            time.sleep(0.1)

        new_count = len(
            glob.glob(os.path.join(test_dir, f"{self.pid}*disabled_test.json"))
        )
        # 禁用时文件数量不应该增加
        self.assertEqual(old_count, new_count)

        # 恢复
        os.environ["VIZTRACER_ENABLE"] = "1"


if __name__ == "__main__":
    # if os.environ.get("FT_SERVER_TEST", None) is None:
    #     logging.config.dictConfig(LOGGING_CONFIG)
    main()
