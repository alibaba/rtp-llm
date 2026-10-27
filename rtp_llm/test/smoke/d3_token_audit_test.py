"""Exact per-question token audit: official prepare vs RTP D3Executor prepare.

Equal ``input_tokens`` counts only prove equal length, not equal token
sequences. This audit captures, per question of every text corpus case, the
rendered prompt text and the token-id sequence from BOTH prepare paths using
the actual checkpoint tokenizer in the real RTP Bazel environment:

- official: the staged snapshot's own ``d3_runtime.normalize_question`` +
  ``d3_runtime.render`` (prompt family from ``decision_config.json``),
  tokenized with ``add_special_tokens=False`` exactly as ``D3.prepare`` does;
- RTP: ``rtp_llm.frontend.d3.D3Executor`` (real constructor validation of the
  tokenizer, answer codes and token ids against ``decision_config.json``) and
  its own ``_prompt_text`` render, tokenized the way the engine does (the
  constructor asserts the tokenizer adds no special tokens).

Text cases only: media cases follow a structurally different prepare path and
the video transport fix is pending (frontend GLM). The comparison asserts
exact text and token-sequence equality; any mismatch is a contract defect and
fails the target with the first divergence recorded. The full per-question
report (sha256 of text and ids, lengths, first divergence) is written to
``D3_TOKEN_AUDIT_OUTPUT`` when set.

Required environment: ``D3_TOKEN_AUDIT_MODEL`` (staged snapshot directory),
``D3_TOKEN_AUDIT_VENV`` (frozen venv with transformers==5.17.0). CPU only.
"""

import hashlib
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

MODEL = Path(os.environ["D3_TOKEN_AUDIT_MODEL"])
VENV = Path(os.environ["D3_TOKEN_AUDIT_VENV"])

# The frozen venv must win over the Bazel-provided transformers for the
# official tokenizer (transformers==5.17.0, the official requirements pin).
import site

_candidates = sorted(VENV.glob("lib/python*/site-packages"), reverse=True)
if not _candidates:
    raise AssertionError(f"no site-packages under {VENV}")
_SITE = str(_candidates[0])
site.addsitedir(_SITE)
sys.path.insert(0, _SITE)

import transformers  # noqa: E402  (from the venv)
from transformers import AutoTokenizer  # noqa: E402  (from the venv)

# The staged snapshot's own official runtime/format (publisher-hash verified).
sys.path.insert(0, str(MODEL))
import d3_runtime  # noqa: E402  (official, from the staged snapshot)
from d3_acceptance_test import _corpus_path, _materialize  # noqa: E402

from rtp_llm.frontend import d3 as rtp_d3  # noqa: E402  (Bazel runfiles)

TEXT_ONLY_CASES = (
    "short",
    "long",
    "changed_state",
    "empty_state",
    "chinese",
    "choice_255",
    "choice_255_semantic",
    "score_2",
    "score_10",
    "over_limit",
)


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_ids(ids):
    return hashlib.sha256(json.dumps(ids).encode("ascii")).hexdigest()


class D3TokenAuditTest(unittest.TestCase):
    def test_official_and_rtp_token_sequences_match(self):
        self.assertEqual(transformers.__version__, "5.17.0")
        config = json.loads(
            (MODEL / "decision_config.json").read_text(encoding="utf-8")
        )
        checkpoint = json.loads((MODEL / "config.json").read_text(encoding="utf-8"))
        hidden = (
            checkpoint.get("text_config", {}).get("hidden_size")
            or checkpoint["hidden_size"]
        )
        codes = list(config["codes"])
        self.assertEqual(config["prompt"], "d3")
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL))

        async def run_rows(rows):  # never called: the audit uses the prepare path only
            return [None] * len(rows)

        executor = rtp_d3.D3Executor(
            SimpleNamespace(**config, hidden_size=hidden),
            run_rows,
            tokenizer=tokenizer,
            model_name="d3-token-audit",
        )
        # The executor constructor already validated this tokenizer against
        # decision_config.json (answer codes + token ids) and asserted it adds
        # no special tokens: the exact contract the serving engine checks.

        corpus = json.loads(_corpus_path().read_text(encoding="utf-8"))
        cases = {case["name"]: case for case in corpus["cases"]}
        missing = [name for name in TEXT_ONLY_CASES if name not in cases]
        self.assertFalse(missing, f"corpus lost text cases: {missing}")

        report = {
            "schema": "d3-token-audit/1",
            "model_revision_marker": "STAGED.json",
            "official_side": "staged d3_runtime.prepare semantics "
            "(normalize_question + render + add_special_tokens=False)",
            "rtp_side": "rtp_llm.frontend.d3.D3Executor._prompt_text",
            "tokenizer": "AutoTokenizer.from_pretrained(staged snapshot); "
            "validated by D3Executor against decision_config.json",
            "cases": {},
        }
        mismatches = []
        for name in TEXT_ONLY_CASES:
            request = _materialize(cases[name], MODEL)
            state = request["state"]
            per_case = {}
            for qid, question in request["questions"].items():
                official = d3_runtime.normalize_question(question)
                official_text = d3_runtime.render(
                    tokenizer, config["prompt"], state, official.rendered, codes
                )
                official_ids = tokenizer(official_text, add_special_tokens=False)[
                    "input_ids"
                ]
                rtp_question = rtp_d3.normalize_question(question)
                rtp_text = executor._prompt_text(state, rtp_question, 0, ())
                rtp_ids = tokenizer(rtp_text, add_special_tokens=False)["input_ids"]
                row = {
                    "text_equal": official_text == rtp_text,
                    "official_text_sha256": _sha256_text(official_text),
                    "rtp_text_sha256": _sha256_text(rtp_text),
                    "ids_equal": official_ids == rtp_ids,
                    "official_len": len(official_ids),
                    "rtp_len": len(rtp_ids),
                    "official_ids_sha256": _sha256_ids(official_ids),
                    "rtp_ids_sha256": _sha256_ids(rtp_ids),
                }
                if not row["ids_equal"]:
                    first = next(
                        (
                            i
                            for i, (a, b) in enumerate(zip(official_ids, rtp_ids))
                            if a != b
                        ),
                        min(len(official_ids), len(rtp_ids)),
                    )
                    row["first_divergence"] = first
                    row["first_divergence_ids"] = [
                        official_ids[first : first + 4],
                        rtp_ids[first : first + 4],
                    ]
                    mismatches.append(f"{name}:{qid}")
                per_case[qid] = row
            report["cases"][name] = per_case

        output = os.environ.get("D3_TOKEN_AUDIT_OUTPUT", "")
        if output:
            path = Path(output)
            path.mkdir(parents=True, exist_ok=True)
            (path / "token-audit.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        summary = {
            name: {
                qid: {
                    "text_equal": row["text_equal"],
                    "ids_equal": row["ids_equal"],
                    "len": row["official_len"],
                }
                for qid, row in per_case.items()
            }
            for name, per_case in report["cases"].items()
        }
        print("token audit summary:", json.dumps(summary, ensure_ascii=False))
        self.assertFalse(
            mismatches,
            "token sequences differ from the official prepare for: "
            f"{mismatches}; first divergences recorded in the report",
        )


if __name__ == "__main__":
    unittest.main()
