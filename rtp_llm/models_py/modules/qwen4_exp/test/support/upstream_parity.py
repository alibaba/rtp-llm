import ast
import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

_UPSTREAM_NAMES = (
    "Qwen4ExpTextRMSNorm",
    "Qwen4ExpTextGatedResidual",
    "Qwen4ExpTextPLELayer",
    "Qwen4ExpTextNGramEmbedding",
    "Qwen4ExpTextQSAIndexer",
    "apply_rotary_pos_emb",
    "rotate_half",
    "_splitmix64",
    "_build_layer_multipliers",
    "_is_prime",
    "_find_nth_prime_after",
    "apply_mask_to_padding_states",
)


def _load_upstream(path):
    """Exec the named upstream definitions in an otherwise empty namespace."""
    tree = ast.parse(open(path).read())
    body = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            if node.name in _UPSTREAM_NAMES:
                body.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id.isupper() for t in node.targets
        ):
            # splitmix64 seeds and masks.
            body.append(node)
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "math": math,
        # Referenced only from annotations.
        "Cache": object,
        "Qwen4ExpTextConfig": object,
        "__name__": "qwen4_exp_upstream_ref",
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), namespace)
    missing = [name for name in _UPSTREAM_NAMES if name not in namespace]
    if missing:
        raise AssertionError(f"upstream source lacks {missing}; refresh the file")
    return SimpleNamespace(**{name: namespace[name] for name in _UPSTREAM_NAMES})


def _randomize(module, scale=0.05):
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.randn_like(param) * scale)


def _gamma(weight):
    """Gamma as our modules take it: the raw checkpoint tensor.

    ``grouped_rms_norm`` applies upstream's ``+1`` itself, in fp32 after the cast,
    so nothing is folded at load time.
    """
    return weight.detach()
