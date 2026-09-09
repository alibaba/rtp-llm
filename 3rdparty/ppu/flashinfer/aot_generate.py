"""Run FlashInfer generators with Bazel's selected Python toolchain."""

import runpy
import sys

if __name__ == "__main__":
    runpy.run_module(sys.argv.pop(1), run_name="__main__")
