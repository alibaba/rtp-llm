"""Dedicated GLM5.x entry preserving upstream grid execution and reporting."""

import runpy
import sys

from rtp_llm.test.perf_test.glm5x_perf import normalize_model_paths, validation_policy

if __name__ == "__main__":
    sys.argv = normalize_model_paths(sys.argv)
    with validation_policy():
        runpy.run_module(
            "rtp_llm.test.perf_test.glm5x_legacy.test_entry", run_name="__main__"
        )
