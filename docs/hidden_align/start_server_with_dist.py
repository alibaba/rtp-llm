#!/usr/bin/env python3
"""Wrapper that initializes torch.distributed (world_size=1) before starting
the RTP-LLM server.  Needed because MegaMoE requires dist.is_initialized()
even for single-GPU runs, but backend_manager.py skips init when world_size==1.
"""
import os
import sys

# Ensure CWD (github-opensource/) is on sys.path so rtp_llm is importable
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

import torch
import torch.distributed as dist

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29501")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")

dist.init_process_group(backend="nccl", world_size=1, rank=0)

from rtp_llm.start_server import main  # noqa: E402

main()
