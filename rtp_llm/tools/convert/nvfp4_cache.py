"""Export the manifest of portable pre-swizzle NVFP4 expert artifacts."""

import argparse
import json
from pathlib import Path

from rtp_llm.model_loader.nvfp4_cache import Nvfp4ExpertCache


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).write_text(
        json.dumps(Nvfp4ExpertCache(args.cache_dir).manifest(), indent=2)
    )
