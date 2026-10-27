"""Generate the official d3 acceptance references through Bazel (GPU target).

Runs the staged snapshot's own pinned ``d3_runtime.py`` with the frozen
official dependency set (transformers 5.17.0 etc.) from an isolated venv, over
the shared acceptance corpus, and writes the precision-specific reference
files plus raw official outputs. The reference values stay independent of the
RTP implementation: this harness only reuses the corpus materialization from
``d3_acceptance_test`` so both sides send byte-identical requests.

Required environment (supplied via --test_env at the canonical bazel_run
invocation; GPU ownership is the gpu_lock wrapper's responsibility):

- D3_REFERENCE_MODEL: staged snapshot directory (publisher-hash verified).
- D3_REFERENCE_VENV: frozen venv providing transformers==5.17.0,
  flash-linear-attention 0.5.2 and the small official deps, plus a
  bazel_torch_cu130.pth that exposes the Bazel-materialized torch
  2.11.0+cu130 with its matching torchvision/triton/nvidia runtime deps
  (measured reuse; the same shared Torch runtime the smoke set uses).
- D3_REFERENCE_OUTPUT: writable output directory (fresh; not a runfiles dir).

Optional: D3_REFERENCE_DEVICE (default cuda:0), D3_REFERENCE_BATCH (default 8),
D3_REFERENCE_MAX_LENGTH (default 32768, the acceptance serving limit),
D3_REFERENCE_CORPUS (default: the committed corpus fixture).

Outputs: ``references/d3.{bf16,fp8_per_block,fp8_per_channel_compressed}.json``
(the official answers are BF16 backbone outputs; each file selects the
comparison tolerances of the precision under test), ``raw/<case>.json`` per
corpus case, and ``reference-source.json`` with exact dependency versions and
the actual GDN kernel binding (fla-core kernels vs the transformers PyTorch
reference fallback — recorded from the resolved module, not pip metadata).
"""

import datetime
import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

MODEL = Path(os.environ["D3_REFERENCE_MODEL"])
VENV = Path(os.environ["D3_REFERENCE_VENV"])
OUTPUT = Path(os.environ["D3_REFERENCE_OUTPUT"])
DEVICE = os.environ.get("D3_REFERENCE_DEVICE", "cuda:0")

# The frozen venv must win over the Bazel-provided transformers, while the
# Bazel-materialized torch 2.11.0+cu130 (plus the matching torchvision, triton
# and nvidia runtime deps) enters through the venv's bazel_torch_cu130.pth:
# site.addsitedir processes that .pth (no copies, Bazel repos stay read-only)
# and the explicit insert keeps the venv's transformers 5.17.0 first.
import site

_candidates = sorted(VENV.glob("lib/python*/site-packages"), reverse=True)
if not _candidates:
    raise AssertionError(f"no site-packages under {VENV}")
_SITE = str(_candidates[0])
site.addsitedir(_SITE)
sys.path.insert(0, _SITE)

import torch  # noqa: E402
import torchvision  # noqa: E402  (matching build, from the Bazel set)
import transformers  # noqa: E402  (from the venv)
from d3_acceptance_test import MAX_SEQ_LEN, _corpus_path, _materialize  # noqa: E402

sys.path.insert(0, str(MODEL))
import d3_runtime  # the snapshot's own pinned official runtime  # noqa: E402

TOLERANCES = {
    "BF16": {
        "max_abs_probability_error": 0.01,
        "require_same_choice": True,
        "max_confidence_error": 0.02,
        "max_score_error": 0.02,
        "max_abs_noul_error": 0.01,
    },
    "FP8_PER_BLOCK": {
        "max_abs_probability_error": 0.03,
        "require_same_choice": True,
        "max_confidence_error": 0.06,
        "max_score_error": 0.06,
        "max_abs_noul_error": 0.03,
    },
    "FP8_PER_CHANNEL_COMPRESSED": {
        "max_abs_probability_error": 0.03,
        "require_same_choice": True,
        "max_confidence_error": 0.06,
        "max_score_error": 0.06,
        "max_abs_noul_error": 0.03,
    },
}


def _gdn_kernel_binding():
    """Replicate the use_kernel_func_from_hub_with_fallback resolution EXACTLY
    (import package -> resolve_internal_import -> module fallback branch),
    capturing each step's outcome so a runtime fallback is distinguishable
    from a real kernel binding. This is runtime binding evidence, not pip
    metadata."""
    import importlib

    from transformers.integrations.hub_kernels import _KERNELS_INTERNAL_PATH_MAPPINGS
    from transformers.utils.import_utils import (
        is_causal_conv1d_available,
        is_flash_linear_attention_available,
        resolve_internal_import,
    )

    info = {
        "fla_available": bool(is_flash_linear_attention_available()),
        "causal_conv1d_available": bool(is_causal_conv1d_available()),
        "chunk_gated_delta_rule": None,
        "decorator_probe": {},
    }
    internal = _KERNELS_INTERNAL_PATH_MAPPINGS.get("chunk_gated_delta_rule")
    full_func_path = f"{internal}.chunk_gated_delta_rule"
    full_module_path = f"fla.{internal}"
    probe = info["decorator_probe"]
    resolved = None
    try:
        module = importlib.import_module("fla")
        probe["import_fla"] = "ok"
    except Exception as exc:  # noqa: BLE001
        import inspect
        import traceback

        probe["import_fla"] = repr(exc)
        probe["import_fla_traceback"] = [
            f"{frame.name} @ {frame.filename}:{frame.lineno}"
            for frame in traceback.extract_tb(exc.__traceback__)[-6:]
        ]
        # Decisive detail: the function triton was jitting when it failed,
        # and the source inspect serves for it at that moment.
        frames = []
        tb = exc.__traceback__
        while tb is not None:
            local_fn = tb.tb_frame.f_locals.get("fn")
            if local_fn is not None and hasattr(local_fn, "__code__"):
                code = local_fn.__code__
                entry = {
                    "frame": f"{tb.tb_frame.f_code.co_name} @ "
                    f"{code.co_filename}:{code.co_firstlineno}",
                }
                try:
                    raw, first_line = inspect.getsourcelines(local_fn)
                    entry["src_first_line"] = raw[0].rstrip()[:100] if raw else None
                    entry["src_lines"] = len(raw)
                    entry["has_def_line"] = any(line.startswith("def ") for line in raw)
                except Exception as inner:  # noqa: BLE001
                    entry["getsourcelines_error"] = repr(inner)
                frames.append(entry)
            tb = tb.tb_next
        probe["jitted_fn_detail"] = frames
        return info
    try:
        resolved = resolve_internal_import(module, full_func_path)
        probe["resolve_internal_import"] = "resolved" if resolved else "None"
    except Exception as exc:  # noqa: BLE001
        probe["resolve_internal_import"] = repr(exc)
    if resolved is None:
        try:
            submodule = importlib.import_module(full_module_path)
            resolved = getattr(submodule, "chunk_gated_delta_rule", None)
            probe["module_fallback_branch"] = "resolved" if resolved else "None"
        except Exception as exc:  # noqa: BLE001
            probe["module_fallback_branch"] = repr(exc)
    if resolved is not None:
        info["chunk_gated_delta_rule"] = (
            f"{resolved.__module__}.{resolved.__qualname__}"
        )
    return info


class D3ReferenceGenTest(unittest.TestCase):
    def test_generate_official_reference(self):
        self.assertEqual(
            transformers.__version__, "5.17.0", "frozen reference deps required"
        )
        if DEVICE.startswith("cuda"):
            self.assertTrue(
                torch.cuda.is_available(), "reference generation needs the GPU"
            )
        corpus_path = Path(os.environ.get("D3_REFERENCE_CORPUS") or _corpus_path())
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        self.assertEqual(corpus["schema"], "d3-acceptance-corpus/1")
        staged = json.loads((MODEL / "STAGED.json").read_text(encoding="utf-8"))
        self.assertEqual(staged["verified_files"], "all")
        # Early probe mirrors the decorator's resolution time (model load).
        gdn_early = _gdn_kernel_binding()
        (OUTPUT / "raw").mkdir(parents=True, exist_ok=True)
        (OUTPUT / "references").mkdir(parents=True, exist_ok=True)
        model = d3_runtime.D3.from_pretrained(
            MODEL,
            device=DEVICE,
            batch_size=int(os.environ.get("D3_REFERENCE_BATCH", "8")),
            verify="full",
            max_length=int(os.environ.get("D3_REFERENCE_MAX_LENGTH", str(MAX_SEQ_LEN))),
        )
        model.warmup()
        cases = {}
        for case in corpus["cases"]:
            if "request" not in case:
                continue
            request = _materialize(case, MODEL)
            response = model.system_one(
                state=request["state"],
                questions=request["questions"],
                images=request.get("images"),
                videos=request.get("videos"),
            )
            self.assertEqual(
                set(response["answers"]), set(request["questions"]), case["name"]
            )
            self.assertEqual(response["usage"]["output_tokens"], 0, case["name"])
            for qid, question in request["questions"].items():
                answer = response["answers"][qid]
                if "error" not in answer:
                    self.assertEqual(
                        answer["type"], question["type"], f"{case['name']}:{qid}"
                    )
                else:
                    self.assertEqual(
                        answer["error"], "max_length_exceeded", f"{case['name']}:{qid}"
                    )
            (OUTPUT / "raw" / f"{case['name']}.json").write_text(
                json.dumps(response, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            cases[case["name"]] = {
                "answers": response["answers"],
                "usage": response["usage"],
            }
        self.assertTrue({"short", "long", "changed_state"} <= set(cases))

        def sha(name):
            return hashlib.sha256((MODEL / name).read_bytes()).hexdigest()

        source = {
            "reference_revision": staged["source_revision"],
            "reference_runtime_sha256": sha("d3_runtime.py"),
            "model_manifest_sha256": sha("MODEL_MANIFEST.json"),
            "decision_config_sha256": sha("decision_config.json"),
            "readout_sha256": sha("readout.safetensors"),
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torch_git_version": torch.version.git_version,
            "torchvision_version": torchvision.__version__,
            "torch_source": (
                "Bazel-materialized pip_gpu_cuda13_torch set "
                "(torch 2.11.0+cu130, matching torchvision/triton/nvidia deps) "
                "exposed read-only via ref-venv-cu211 bazel_torch_cu130.pth"
            ),
            "venv_torch_paths": [
                line.strip()
                for line in (Path(_SITE) / "bazel_torch_cu130.pth")
                .read_text()
                .splitlines()
                if line.strip()
            ],
            "device": str(model.device),
            "dtype": "bfloat16",
            "batch_size": int(os.environ.get("D3_REFERENCE_BATCH", "8")),
            "max_length": int(
                os.environ.get("D3_REFERENCE_MAX_LENGTH", str(MAX_SEQ_LEN))
            ),
            "generated_at_utc": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "generator": "//rtp_llm/test/smoke:d3_reference_gen (official d3_runtime, frozen deps)",
            "gdn_kernel": _gdn_kernel_binding(),
            "gdn_kernel_early": gdn_early,
        }
        for precision, tolerances in TOLERANCES.items():
            payload = {
                "schema": "d3-acceptance-reference/1",
                "source": source,
                "tolerances": tolerances,
                "comparison_precision": precision,
                "note": (
                    "official reference answers are BF16 backbone outputs; "
                    "comparison_precision selects tolerances for the precision "
                    "the implementation under test runs"
                ),
                "cases": cases,
            }
            name = {
                "BF16": "d3.bf16.json",
                "FP8_PER_BLOCK": "d3.fp8_per_block.json",
                "FP8_PER_CHANNEL_COMPRESSED": "d3.fp8_per_channel_compressed.json",
            }[precision]
            (OUTPUT / "references" / name).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        (OUTPUT / "reference-source.json").write_text(
            json.dumps(source, indent=2) + "\n", encoding="utf-8"
        )
        print(f"reference generated: {len(cases)} cases -> {OUTPUT}/references")


if __name__ == "__main__":
    unittest.main()
