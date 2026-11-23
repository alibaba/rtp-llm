"""Check default/override dispatch without initializing a GPU."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import importlib.metadata

root = Path(importlib.metadata.distribution("atom").locate_file(""))
hc = ast.parse((root / "atom/models/qwen4_exp_mtp.py").read_text())
topk = ast.parse((root / "atom/plugin/rtpllm/utils/topk_compat.py").read_text())
hc_expr = next(n.value for n in ast.walk(hc) if isinstance(n, ast.keyword) and n.arg == "fused_hc")
topk_expr = next(n.test for n in ast.walk(topk) if isinstance(n, ast.If) and "ATOM_RTP_TOPK512" in ast.unparse(n.test))
for name, expr in [("ATOM_RTP_MTP_HC_FUSION", hc_expr), ("ATOM_RTP_TOPK512", topk_expr)]:
    for hip in [None, "7.2"]:
        for rtp in [None, False, True]:
            for value in [None, "0", "1"]:
                with patch.dict(os.environ, {}, clear=True):
                    if value is not None:
                        os.environ[name] = value
                    ns = dict(torch=SimpleNamespace(version=SimpleNamespace(hip=hip), bfloat16="bf16"),
                              atom_config=SimpleNamespace(plugin_config=None if rtp is None else SimpleNamespace(is_rtpllm=rtp)),
                              os=os, is_rtpllm=lambda: rtp, renormalize=True, topk=10,
                              token=4, shared=1, scoring="sigmoid",
                              gating_output=SimpleNamespace(dtype="bf16", shape=(4, 513), stride=lambda _: 1))
                    actual = bool(eval(compile(ast.Expression(expr), "dispatch", "eval"), ns))
                    assert actual == bool(hip and rtp and value != "0"), (name, hip, rtp, value)
print("PASS: both defaults, explicit overrides, non-ROCm and non-RTP guards")
