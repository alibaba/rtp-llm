"""Load unchanged definitions from the pinned official model for CUDA probes."""

import ast
import hashlib
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

OFFICIAL_MODEL_SHA256 = (
    "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
)


def load_official_compressor():
    configured = os.environ.get("DSV41_OFFICIAL_MODEL_PATH")
    if configured is None:
        model = os.environ.get("DSV41_MODEL_PATH")
        if model is None:
            raise RuntimeError("set DSV41_OFFICIAL_MODEL_PATH or DSV41_MODEL_PATH")
        configured = str(Path(model) / "inference/model.py")
    path = Path(configured)
    source = path.read_bytes()
    actual_hash = hashlib.sha256(source).hexdigest()
    if actual_hash != OFFICIAL_MODEL_SHA256:
        raise RuntimeError(
            f"official model hash mismatch: expected {OFFICIAL_MODEL_SHA256}, got {actual_hash}"
        )
    names = {
        "ModelArgs",
        "set_dtype",
        "linear",
        "Linear",
        "RMSNorm",
        "precompute_freqs_cis",
        "apply_rotary_emb",
        "Compressor",
        "Block",
    }
    syntax = ast.parse(source, filename=str(path))
    selected = [
        node
        for node in syntax.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    if {node.name for node in selected} != names:
        raise RuntimeError("the pinned official compressor definitions are incomplete")
    module = ModuleType("_dsv41_official_compressor_definitions")
    module.__dict__.update(
        torch=torch,
        nn=nn,
        F=F,
        math=math,
        Literal=Literal,
        dataclass=dataclass,
        contextmanager=contextmanager,
        lru_cache=lru_cache,
        default_dtype=torch.bfloat16,
        world_size=1,
        rank=0,
    )
    sys.modules[module.__name__] = module
    # Preserve the actual function/class ASTs. No model constructor, source edit,
    # monkeypatch of numerical operations, or unrelated official imports run.
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
        module.__dict__,
    )
    return module
