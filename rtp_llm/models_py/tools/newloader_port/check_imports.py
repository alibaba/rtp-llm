#!/usr/bin/env python3
"""Self-check the new-loader plumbing after a main merge.

Verifies, in the GPU container's python env (needs torch etc.):
  1. rtp_llm.models_py imports cleanly (catches broken imports from a main merge).
  2. Every model_type in MODEL_REGISTRY maps to an importable class.
  3. Prints the coverage gap: which old-factory model_types are NOT yet in the
     new registry -> the remaining porting work list.

Run from repo root inside the container:
  python rtp_llm/models_py/tools/newloader_port/check_imports.py
"""
import sys
import traceback


def _probe_tokenizer(path):
    """Reproduce exactly what BaseTokenizer.init_tokenizer does, to see why
    self.tokenizer ends up a bool. Set env PROBE_TOKENIZER=/path to a ckpt dir."""
    import os as _os

    print(f"== PROBE_TOKENIZER: {path}")
    print(
        f"   tokenizer_config.json exists: "
        f"{_os.path.exists(_os.path.join(path, 'tokenizer_config.json'))}"
    )
    print(
        f"   tokenizer.json exists: "
        f"{_os.path.exists(_os.path.join(path, 'tokenizer.json'))}"
    )
    import transformers
    from transformers import AutoTokenizer
    from transformers.models.llama.tokenization_llama import LlamaTokenizer

    print(f"   transformers version: {transformers.__version__}")

    variants = [
        ("Auto default", lambda: AutoTokenizer.from_pretrained(path)),
        (
            "Auto use_fast=True",
            lambda: AutoTokenizer.from_pretrained(path, use_fast=True),
        ),
        (
            "Auto use_fast=False",
            lambda: AutoTokenizer.from_pretrained(path, use_fast=False),
        ),
        (
            "Auto trust_remote_code=True",
            lambda: AutoTokenizer.from_pretrained(path, trust_remote_code=True),
        ),
        (
            "Auto verbose=False",
            lambda: AutoTokenizer.from_pretrained(path, verbose=False),
        ),
        (
            "Auto FULL (rtp current)",
            lambda: AutoTokenizer.from_pretrained(
                path, trust_remote_code=True, verbose=False, use_fast=True
            ),
        ),
        ("slow LlamaTokenizer", lambda: LlamaTokenizer.from_pretrained(path)),
    ]
    for label, fn in variants:
        try:
            tok = fn()
            print(
                f"   [{label}] -> type={type(tok).__name__} "
                f"eos={getattr(tok, 'eos_token_id', '<no attr>')!r}"
            )
        except Exception as e:
            print(f"   [{label}] -> EXCEPTION {type(e).__name__}: {e}")
    return 0


def main():
    import os as _os

    _probe = _os.environ.get("PROBE_TOKENIZER")
    if _probe:
        return _probe_tokenizer(_probe)

    print("== importing rtp_llm.models_py ...")
    try:
        import rtp_llm.models_py as mp  # noqa
        from rtp_llm.models_py.registry import MODEL_REGISTRY
    except Exception:
        print("FAIL: new-loader package did not import (likely a main-merge breakage):")
        traceback.print_exc()
        return 1
    print(f"   OK. new registry has {len(MODEL_REGISTRY)} model_type(s):")
    for k in sorted(MODEL_REGISTRY):
        print(f"     - {k:22s} -> {MODEL_REGISTRY[k].__name__}")

    print("== importing legacy models to compute coverage gap ...")
    try:
        # rtp_llm.models/__init__.py imports every model module, whose
        # @register_model decorators populate the global _model_factory dict.
        import rtp_llm.models  # noqa  (side-effect: registers all model_types)
        from rtp_llm.model_factory_register import _model_factory  # global dict

        old_types = set(_model_factory.keys())
    except Exception:
        print("WARN: could not introspect legacy factory; skipping gap report.")
        traceback.print_exc()
        old_types = set()

    if old_types:
        new_types = set(MODEL_REGISTRY)
        todo = sorted(old_types - new_types)
        done = sorted(old_types & new_types)
        print(f"   legacy factory: {len(old_types)} model_type(s)")
        print(f"   DONE in new loader ({len(done)}): {', '.join(done) or '(none)'}")
        print(f"   TODO  ({len(todo)}):")
        for t in todo:
            print(f"     - {t}")

    print("== OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
