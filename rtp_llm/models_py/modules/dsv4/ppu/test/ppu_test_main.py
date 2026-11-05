"""Run PPU numerical tests with an allocated device count and no silent skips."""

import argparse
import importlib
from pathlib import Path
import sys
import unittest

import torch

from rtp_llm.device.device_type import is_ppu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=int, required=True)
    parser.add_argument("test_file")
    args = parser.parse_args()
    if not is_ppu() or torch.cuda.device_count() < args.devices:
        raise RuntimeError(f"requires {args.devices} allocated PPU devices")

    # An importable module name is required by the multi-rank spawn test.
    path = Path(args.test_file).resolve()
    sys.path.insert(0, str(path.parent))
    module = importlib.import_module(path.stem)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.testsRun == 0 or result.skipped:
        print("ERROR: PPU numerical CI requires execution without skipped tests")
        return 1
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
