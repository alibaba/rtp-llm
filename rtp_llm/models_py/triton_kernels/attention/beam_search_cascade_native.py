"""Independent RN/FP32-output CK prefix module for gfx942 cascade attention.

Load/build once during operator construction, before Graph capture. The regular
AITER BF16-output module and its compiler settings remain unchanged. Sources
and their upstream provenance ship as package data; no installed source patch
or experiment-directory library is required.
"""

import hashlib
import importlib.util
import json
import logging
import sys
from functools import lru_cache
from pathlib import Path

import torch

MODULE = "rtp_cascade_prefix_bf16_rn_fp32_d128_ps16_gfx942_v1"


@lru_cache(maxsize=1)
def load_native_prefix():
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Load the cascade FP32 module before Graph capture")
    from aiter.jit import core

    if core.AITER_REBUILD:
        raise RuntimeError("Cascade FP32 module requires AITER_REBUILD=0")
    if core.gfxs != ["gfx942"]:
        raise RuntimeError("Cascade FP32 module currently supports gfx942 only")
    library = Path(core.get_user_jit_dir()) / (MODULE + ".so")
    if not library.is_file():
        root = Path(__file__).with_name("native_cascade_fp32")
        provenance = json.loads((root / "provenance.json").read_text())
        for name, expected in provenance["sources"].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(
                    "Cascade FP32 source differs from its manifest: " + name
                )
        site = Path(core.CK_DIR).resolve().parents[2]
        for name, expected in provenance["dependency_sha256"].items():
            if hashlib.sha256((site / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(
                    "Cascade FP32 dependency requires revalidation: " + name
                )
        args = core.get_args_of_build("module_mha_batch_prefill")
        args.update(
            md_name=MODULE,
            srcs=[str(root / name) for name in provenance["sources"]],
            blob_gen_cmd=[],
            flags_extra_hip=[
                "-fbracket-depth=1024",
                "-DCK_TILE_FMHA_FWD_FAST_EXP2=1",
                "-DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=0",
                "-DCK_TILE_ATTENTION_LOGITS_SOFT_CAP_DEFAULT=0",
                "-DCK_TILE_ATTENTION_USE_SOFTSIGN_ASM=1",
            ],
        )
        args.pop("hip_clang_path", None)
        core.build_module(**args)
    spec = importlib.util.spec_from_file_location(MODULE, library)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[MODULE] = module
    logging.info(
        "Cascade FP32 prefix module: %s, sha256=%s",
        library,
        hashlib.sha256(library.read_bytes()).hexdigest(),
    )
    return module


def prefix_forward(
    module, q, k, v, qptr, indptr, indices, max_q, max_k, *, out, block_table, seqlen_k
):
    # Match AITER's existing batch-prefill ABI; do not cast any tensor. The
    # native wrapper enforces the narrower geometry and FP32 output contract.
    result = module.mha_batch_prefill(
        q,
        k,
        v,
        qptr,
        indptr,
        indices,
        max_q,
        max_k,
        0.0,
        q.shape[-1] ** -0.5,
        0.0,
        False,
        False,
        -1,
        -1,
        0,
        True,
        False,
        out,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        block_table,
        seqlen_k,
        None,
        None,
    )
    return result[0], result[1]
