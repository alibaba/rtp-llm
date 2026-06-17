import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jit_cache_manager import JitSnapshotBuilder


def parse_args():
    parser = argparse.ArgumentParser(description="Build RTP-LLM JIT cache snapshot")
    parser.add_argument(
        "--remote_jit_dir",
        default=os.environ.get("REMOTE_JIT_DIR", ""),
        help="Absolute mounted remote JIT cache directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = JitSnapshotBuilder(args.remote_jit_dir).build()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as e:
        print(json.dumps({"result": "failed", "message": str(e)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    sys.exit(main())
