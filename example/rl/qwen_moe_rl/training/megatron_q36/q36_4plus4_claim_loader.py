"""Resolve the exact reviewed task-claim implementation from Q36_CODE_DIR."""
import hashlib
import importlib
import importlib.machinery
import os
from pathlib import Path
import sys

GUARD_SHA256 = 'd6faa983a2d404a33d106fec026f1049cc68aa0ceb392e0a6e948cdd7185197c'


def load_guard():
    root = Path(os.environ['Q36_CODE_DIR']).resolve(strict=True)
    name = 'q36_4plus4_claim_guard'
    source = root / (name + '.py')
    spec = importlib.machinery.PathFinder.find_spec(name, sys.path)
    loaded = sys.modules.get(name)
    if (source.is_symlink() or not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != GUARD_SHA256 or
        spec is None or not spec.origin or Path(spec.origin).resolve() != source or
        loaded is not None and Path(getattr(loaded, '__file__', '')).resolve() != source):
        raise RuntimeError('qualified 4+4 Ray claim source identity mismatch')
    return importlib.import_module(name)
