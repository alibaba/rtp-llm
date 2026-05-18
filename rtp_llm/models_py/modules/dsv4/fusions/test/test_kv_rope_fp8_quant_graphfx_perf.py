from __future__ import annotations

import os
import sys
import unittest

from rtp_llm.models_py.modules.dsv4.fusions.test.test_kv_rope_fp8_quant_graphfx import (
    Dsv4KvRopeFp8QuantGraphFXTest,
)


def main() -> None:
    os.environ.setdefault("DSV4_GRAPHFX_RUN_PERF_IN_UT", "1")
    suite = unittest.TestSuite()
    suite.addTest(Dsv4KvRopeFp8QuantGraphFXTest("test_graphfx_rewrite_perf"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
