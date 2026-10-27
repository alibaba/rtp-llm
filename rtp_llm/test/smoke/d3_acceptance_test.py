"""One-start HTTP acceptance for a staged d3 code-readout checkpoint (d3-mini).

Required environment: D3_ACCEPTANCE_MODEL_PATH, D3_ACCEPTANCE_REFERENCE.

The corpus fixture ``data/d3/corpus.json`` is the single request source shared
with the independent official reference generator, so the server under test and
the reference receive byte-identical requests. The reference JSON (computed
independently with the official d3_runtime.py + pinned deps; this test never
writes references)::

    {"schema": "d3-acceptance-reference/1",
     "source": {"reference_revision": "<40-hex>",
                "reference_runtime_sha256": "<sha256 of staged d3_runtime.py>",
                "model_manifest_sha256": "<sha256 of staged MODEL_MANIFEST.json>",
                "decision_config_sha256": "<sha256 of staged decision_config.json>",
                "readout_sha256": "<sha256 of staged readout.safetensors>",
                "transformers_version": "5.17.0", ...},
     "tolerances": {"max_abs_probability_error": 0.01,
                    "require_same_choice": true,
                    "max_confidence_error": 0.02,
                    "max_score_error": 0.02,
                    "max_abs_noul_error": 0.02},
     "cases": {"short": {"answers": {... official answer objects ...},
                          "usage": {"input_tokens": N, "output_tokens": 0}}, ...}}

The reference source must pin the staged checkpoint: its decision_config,
MODEL_MANIFEST, readout and d3_runtime hashes must match the served model
directory, otherwise the reference belongs to a different checkpoint and the
run stops. When the staged snapshot carries a STAGED.json provenance marker,
the reference revision must equal the staged source revision.

D3_ACCEPTANCE_QUANTIZATION is BF16 (default), FP8_PER_BLOCK (NVIDIA) or
FP8_PER_CHANNEL_COMPRESSED (AMD; PTPC is an alias). The reference's
comparison_precision, when present, must equal the requested precision.
Tolerances are read from the reference so each precision is judged with the
values selected for it.

Cases named short, long and changed_state are required. long spans many cache
blocks; over_limit must exceed the 32768-token serving limit the server below
starts with. Media cases inline staged-model assets (pinned by manifest hash)
as base64 data URLs, so text, image and video ride the same corpus on every
platform. cross_question_isolation carries set_a/set_b question groups: each
group is requested alone, combined, reordered, repeated and mutated, asserting
full-row semantics - every question is a complete independent prompt and no
cross-question language cache reuse leaks between rows (d3 noncausal layers
make state hidden depend on later tokens, so sharing would be observable).

The official error contract is asserted: a request with an invalid question is
rejected whole with 422 and ``{"detail": {"invalid_questions": {...}}}``;
over-limit questions are answered with per-question ``max_length_exceeded``
error objects in a 200 response; the server keeps answering after errors
(recovery). usage.output_tokens is always 0. The video/mixed cases are gated by
D3_ACCEPTANCE_VIDEO (see below): required by default, explicit interim skip
records PASS_PARTIAL_SCOPE.

D3_ACCEPTANCE_PROBABILITY_GATE defaults to strict; set it to report to run the
SYSTEM-correctness gate: a clean run then reports PASS_SYSTEM_CORRECTNESS
(never a bare/strict-numeric PASS), with system_gate/numerical_gate fields and
all reference probability/score-EV/noul/confidence deltas recorded as
diagnostics (score in EV units plus per-option probability units) including
tolerance exceedances. System checks are never relaxed in report mode:
finite/shape/schema answers, choice == argmax of the answer's own
probabilities, score/confidence/noul consistency with those probabilities,
usage accounting (output_tokens == 0), request order, repeated-request
determinism, full-row isolation, cancellation, the invalid-question/error
contract, no 500 for valid requests, and all modalities in scope. Against the
official reference, the choice gate is margin-aware: semantically decisive
questions (reference top-2 gap above twice the registered probability
tolerance) must match the official choice on every hardware/precision; a
reference near-tie difference is recorded with top-3 and margins and is never
treated as a system defect. Strict mode keeps all numeric assertions.
D3_ACCEPTANCE_USE_FLYDSL selects the ROCm FlyDSL Chunk-GDN bridge (default 0;
the MI308X target sets 1). D3_ACCEPTANCE_SMOKE_ARGS / _SMOKE_ENV append server
arguments and environment pairs. Each run's server/process logs are archived
(sha256) under <output>/evidence/ with the OBSERVED quantization/precision/
checkpoint identity extracted from the real server log, because requested
quantization alone is not execution proof.

D3_ACCEPTANCE_VIDEO is the full-modality gate and defaults to require: the
video and mixed cases must be served (a 422 "video inputs are not available"
gap answer FAILS the run - a valid-video 422 is never a passing full
acceptance, including the final ordinary-image gate). Set it to skip only for
explicit interim text/image bring-up: the video/mixed cases are then skipped,
and the run's status is PASS_PARTIAL_SCOPE with the partial scope recorded in
the summary - never a full PASS. With D3_ACCEPTANCE_ENDPOINT, only HTTP
assertions run against that endpoint; the caller owns that server's lifecycle
(including the final image smoke) and must record its launch configuration
separately.

Raw HTTP records and a summary go to D3_ACCEPTANCE_OUTPUT_DIR, otherwise
TEST_UNDECLARED_OUTPUTS_DIR. A single MagaServerManager owns startup and
teardown; GPU ownership is the Bazel test-execution wrapper's responsibility.
"""

import base64
import copy
import hashlib
import json
import math
import os
import re
import shlex
import struct
import tempfile
import time
import unittest
from pathlib import Path

import requests
from tokenizers import Tokenizer

FORMAT_VERSION = 1
FORMAT_ID = "d3-code-readout-v1"
MAX_OPTIONS = 255
HIDDEN_SIZE = 2560
READOUT_SHAPE = (MAX_OPTIONS, HIDDEN_SIZE)
MAX_SEQ_LEN = 32768

_QUANTIZATION_ALIASES = {
    "BF16": "BF16",
    "FP8_PER_BLOCK": "FP8_PER_BLOCK",
    "PTPC": "FP8_PER_CHANNEL_COMPRESSED",
    "FP8_PER_CHANNEL_COMPRESSED": "FP8_PER_CHANNEL_COMPRESSED",
}
_IMAGE_TYPES = {"png": "png", "jpeg": "jpeg", "jpg": "jpg", "webp": "webp"}
_VIDEO_TYPES = {"mp4": "mp4", "webm": "webm", "mov": "quicktime", "mkv": "x-matroska"}


def _confidence(values):
    """Official confidence: one minus normalized Shannon entropy, in [0, 1]."""
    n = len(values)
    if n < 2:
        return 1.0
    entropy = -sum(p * math.log(p) for p in values if p > 0)
    return max(0.0, min(1.0, 1 - entropy / math.log(n)))


def _canonical(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _corpus_path():
    override = os.environ.get("D3_ACCEPTANCE_CORPUS")
    if override:
        return Path(override)
    srcdir = os.environ.get("TEST_SRCDIR")
    workspace = os.environ.get("TEST_WORKSPACE")
    if srcdir and workspace:
        candidate = Path(srcdir) / workspace / "rtp_llm/test/smoke/data/d3/corpus.json"
        if candidate.is_file():
            return candidate
    return Path(__file__).resolve().parent / "data" / "d3" / "corpus.json"


def _materialize(case, model):
    """The concrete request of one corpus case; media is inlined as data URLs."""
    request = copy.deepcopy(case["request"])
    if "state_builder" in case:
        assert request.get("state") is None
        builder = case["state_builder"]
        request["state"] = builder["text"] * builder["times"]
    manifest = json.loads((model / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    files = manifest["files_sha256"]
    for field, types in (("images", _IMAGE_TYPES), ("videos", _VIDEO_TYPES)):
        items = request.get(field)
        if items is None:
            continue
        urls = []
        for item in items:
            if not (isinstance(item, str) and item.startswith("@model:")):
                raise AssertionError(
                    f"corpus {field} entries must be @model paths: {item!r}"
                )
            relative = item[len("@model:") :]
            data = (model / relative).read_bytes()
            assert hashlib.sha256(data).hexdigest() == files[relative], relative
            suffix = relative.rsplit(".", 1)[-1].lower()
            urls.append(
                f"data:{field[:-1]}/{types[suffix]};base64,"
                + base64.b64encode(data).decode("ascii")
            )
        request[field] = urls
    return request


class D3AcceptanceTest(unittest.TestCase):
    def setUp(self):
        self.failures = []
        self.records = []
        self.report = {"status": "NOT_RUN"}
        self.output = Path(
            os.environ.get("D3_ACCEPTANCE_OUTPUT_DIR")
            or os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
            or tempfile.mkdtemp(prefix="d3-acceptance-")
        )
        self.output.mkdir(parents=True, exist_ok=True)
        for name in ("requests.jsonl", "summary.json"):
            self.assertFalse(
                (self.output / name).exists(),
                f"use a fresh output directory: {self.output}",
            )
        self.addCleanup(self._write_report)
        self.model = Path(os.environ["D3_ACCEPTANCE_MODEL_PATH"])
        self._check_checkpoint()
        self.corpus = json.loads(_corpus_path().read_text(encoding="utf-8"))
        self.assertEqual(self.corpus.get("schema"), "d3-acceptance-corpus/1")
        self.assertEqual(self.corpus.get("format_id"), FORMAT_ID)
        self.cases = {item["name"]: item for item in self.corpus["cases"]}
        self.assertEqual(len(self.cases), len(self.corpus["cases"]), "duplicate cases")
        self.assertTrue({"short", "long", "changed_state"} <= self.cases.keys())
        for item in self.cases.values():
            if "request" in item:
                _materialize(
                    item, self.model
                )  # corpus must materialize against this model
        isolation = self.cases["cross_question_isolation"]
        self.assertEqual(set(isolation), {"name", "state", "set_a", "set_b"})
        reference_path = Path(os.environ["D3_ACCEPTANCE_REFERENCE"])
        reference_bytes = reference_path.read_bytes()
        reference = json.loads(reference_bytes)
        self.assertEqual(reference.get("schema"), "d3-acceptance-reference/1")
        source = reference["source"]
        revision = source["reference_revision"]
        self.assertIs(
            isinstance(revision, str)
            and len(revision) == 40
            and revision.strip("0123456789abcdef") == "",
            True,
            "independent reference must pin the official runtime revision",
        )
        if self.staged_revision is not None:
            self.assertEqual(revision, self.staged_revision)
        self.assertEqual(
            source["decision_config_sha256"], self._sha256("decision_config.json")
        )
        self.assertEqual(
            source["model_manifest_sha256"], self._sha256("MODEL_MANIFEST.json")
        )
        self.assertEqual(source["readout_sha256"], self._sha256("readout.safetensors"))
        self.assertEqual(
            source["reference_runtime_sha256"], self._sha256("d3_runtime.py")
        )
        self.tolerance = reference["tolerances"]
        for key in (
            "max_abs_probability_error",
            "max_confidence_error",
            "max_score_error",
            "max_abs_noul_error",
        ):
            value = self.tolerance[key]
            self.assertIs(type(value) in (float, int), True)
            self.assertTrue(math.isfinite(value) and 0 <= value < 1, key)
        self.same_choice = self.tolerance["require_same_choice"]
        self.assertIs(type(self.same_choice), bool)
        self.probability_gate = os.environ.get(
            "D3_ACCEPTANCE_PROBABILITY_GATE", "strict"
        )
        self.assertIn(self.probability_gate, ("strict", "report"))
        self.quantization = _QUANTIZATION_ALIASES[
            os.environ.get("D3_ACCEPTANCE_QUANTIZATION", "BF16").upper()
        ]
        reference_precision = reference.get("comparison_precision")
        if reference_precision is not None:
            self.assertEqual(
                _QUANTIZATION_ALIASES[reference_precision.upper()],
                self.quantization,
                "reference tolerances belong to a different requested precision",
            )
        self.reference_cases = reference["cases"]
        self.assertTrue(
            {"short", "long", "changed_state"} <= self.reference_cases.keys()
        )
        self.report.update(
            source=source,
            reference_sha256=hashlib.sha256(reference_bytes).hexdigest(),
            reference_path=str(reference_path),
            requested_quantization=self.quantization,
            probability_gate=self.probability_gate,
            tolerances=self.tolerance,
            lifecycle=(
                "external"
                if os.environ.get("D3_ACCEPTANCE_ENDPOINT")
                else "MagaServerManager"
            ),
            model_path=str(self.model),
            checkpoint_identity={
                "reference_revision": revision,
                "decision_config_sha256": self._sha256("decision_config.json"),
                "model_manifest_sha256": self._sha256("MODEL_MANIFEST.json"),
                "readout_sha256": self._sha256("readout.safetensors"),
                "d3_runtime_sha256": self._sha256("d3_runtime.py"),
            },
            language_prefix_cache="forbidden: started with --reuse_cache 0; "
            "full-row isolation asserted behaviorally (cross_question_isolation)",
        )
        self.report["probability_tolerance_exceedances"] = []
        # Full-modality gate: video/mixed must be served by default. "skip" is
        # an explicit interim mode and can only produce PASS_PARTIAL_SCOPE.
        self.video_gate = os.environ.get("D3_ACCEPTANCE_VIDEO", "require")
        self.assertIn(self.video_gate, ("require", "skip"))
        self.video_supported = None
        self.report["video_gate"] = self.video_gate
        self.report["quantization"] = self.quantization
        self.report["scope"] = (
            "full (text + image + video + mixed)"
            if self.video_gate == "require"
            else "PARTIAL: text + image (video skipped; explicit interim mode)"
        )
        self.endpoint = os.environ.get("D3_ACCEPTANCE_ENDPOINT", "").rstrip("/")
        if not self.endpoint:
            self._start_server()
        self.assertTrue(self.endpoint.startswith(("http://", "https://")))
        self.session = requests.Session()
        self.session.trust_env = False
        self.addCleanup(self.session.close)
        health = self.session.get(f"{self.endpoint}/health", timeout=30)
        self.assertEqual(health.status_code, 200, health.text)

    def _sha256(self, name):
        return hashlib.sha256((self.model / name).read_bytes()).hexdigest()

    def _check_checkpoint(self):
        config = json.loads(
            (self.model / "decision_config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config.get("format_version"), FORMAT_VERSION)
        self.assertEqual(config.get("format_id"), FORMAT_ID)
        self.assertEqual(config.get("prompt"), "d3")
        self.assertEqual(config.get("attention_mode"), "noncausal_full_attention")
        self.assertEqual(config.get("pooling"), "last")
        self.assertEqual(config.get("readout_dtype"), "float32")
        codes = config.get("codes")
        token_ids = config.get("token_ids")
        self.assertEqual(len(codes), MAX_OPTIONS)
        self.assertEqual(len(set(codes)), MAX_OPTIONS, "duplicate answer codes")
        self.assertEqual(len(token_ids), MAX_OPTIONS)
        self.assertEqual(len(set(token_ids)), MAX_OPTIONS, "duplicate readout tokens")
        self.assertTrue(
            math.isfinite(config.get("temperature", 1.0)) and config["temperature"] > 0
        )
        tokenizer = Tokenizer.from_file(str(self.model / "tokenizer.json"))
        for code, token_id in zip(codes, token_ids):
            alone = tokenizer.encode(code, add_special_tokens=False).ids
            self.assertEqual(alone, [token_id], f"code {code!r} is not a single token")
        with (self.model / "readout.safetensors").open("rb") as handle:
            header_length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_length))
        self.assertEqual(set(header), {"weight"})
        self.assertEqual(header["weight"]["dtype"], "F32")
        self.assertEqual(tuple(header["weight"]["shape"]), READOUT_SHAPE)
        manifest = json.loads(
            (self.model / "MODEL_MANIFEST.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest.get("schema"), "d3-package-manifest/1")
        self.assertEqual(manifest.get("format_id"), FORMAT_ID)
        self.assertEqual(
            manifest.get("runtime", {}).get("files_sha256", {}).get("d3_runtime.py"),
            self._sha256("d3_runtime.py"),
            "staged d3_runtime.py differs from the publisher manifest",
        )
        staged = self.model / "STAGED.json"
        self.staged_revision = None
        if staged.is_file():
            info = json.loads(staged.read_text(encoding="utf-8"))
            self.staged_revision = info.get("source_revision")
            self.assertEqual(
                info.get("verified_files"),
                "all",
                "staged snapshot marker must record full hash verification",
            )

    def _start_server(self):
        from rtp_llm.test.utils.maga_server_manager import MagaServerManager

        args = [
            "--act_type",
            "bf16",
            "--tp_size",
            "1",
            "--world_size",
            "1",
            "--enable_cuda_graph",
            "false",
            "--frontend_server_count",
            "1",
            "--warm_up",
            "0",
            "--reuse_cache",
            "0",
            "--seq_size_per_block",
            "64",
            "--kernel_seq_size_per_block",
            "16",
            "--max_seq_len",
            str(MAX_SEQ_LEN),
            "--max_context_batch_size",
            "16",
            "--max_batch_tokens_size",
            "65536",
            # KV capacity must match the advertised max_seq_len (32768) or a
            # bounded supported length -- accepting a request and then failing
            # it with 500 is an admission bug. Empirically 4096 MB held 5056
            # tokens (~0.81 MB/token on this model), so 32768 MB holds ~40k
            # tokens: every admitted request fits, including the corpus
            # over_limit case (~39k tokens) whose over-limit behaviour must
            # come from the max_seq_len check (official per-question
            # max_length_exceeded at 200), never from a KV 500. GPU7 is 72 GB
            # (model 9.1 GB + KV 32 GB + activations fits comfortably).
            "--kv_cache_mem_mb",
            "32768",
        ]
        args += shlex.split(os.environ.get("D3_ACCEPTANCE_SMOKE_ARGS", ""))
        # Keep the requested precision explicit even if the shell sets QUANTIZATION.
        args += [
            "--quantization",
            "" if self.quantization == "BF16" else self.quantization,
        ]
        env_args = {
            "LOAD_PYTHON_MODEL": "1",
            "USE_FLYDSL": os.environ.get("D3_ACCEPTANCE_USE_FLYDSL", "0"),
        }
        for pair in shlex.split(os.environ.get("D3_ACCEPTANCE_SMOKE_ENV", "")):
            key, sep, value = pair.partition("=")
            self.assertTrue(sep and key, f"bad D3_ACCEPTANCE_SMOKE_ENV entry: {pair!r}")
            env_args[key] = value
        self.server = MagaServerManager(
            env_args=env_args,
            smoke_args_str=shlex.join(args),
            role_name="d3_acceptance",
        )
        self.addCleanup(self.server.stop_server)
        self.assertTrue(
            self.server.start_server(
                model_path=str(self.model),
                model_type="",
                timeout=int(os.environ.get("D3_ACCEPTANCE_START_TIMEOUT", "1600")),
            ),
            "d3 server failed to become healthy",
        )
        self.endpoint = f"http://127.0.0.1:{self.server.port}"
        self.report["server_log"] = self.server.log_file_path
        self.report["server_args"] = args
        self.report["server_env"] = env_args

    def _archive_server_evidence(self):
        """Copy the mutable Bazel-target server logs into this run's unique
        output before the next run of the same target overwrites them, and
        extract the OBSERVED execution identity (quantization mode, precision
        configuration, checkpoint paths, kernel selection lines) from the
        real server log. Requested quantization alone is not execution proof."""
        server = getattr(self, "server", None)
        log_path = getattr(server, "log_file_path", None)
        if not log_path:
            return
        source_dir = Path(log_path).parent
        evidence = self.output / "evidence" / "d3_acceptance_logs"
        evidence.mkdir(parents=True, exist_ok=True)
        copied = {}
        for path in sorted(source_dir.glob("*")):
            if path.is_file():
                data = path.read_bytes()
                target = evidence / path.name
                target.write_bytes(data)
                copied[path.name] = hashlib.sha256(data).hexdigest()
        manifest = {
            "schema": "d3-acceptance-evidence/1",
            "source_dir": str(source_dir),
            "files_sha256": copied,
            "requested_quantization": self.quantization,
            "observed": {},
        }
        process_log = evidence / Path(log_path).name
        if process_log.is_file():
            text = process_log.read_text(encoding="utf-8", errors="replace")
            patterns = {
                "observed_quantization": r"need_load_quant by (\S+)",
                "final_precision_configuration": (
                    r"Final precision configuration - (.+)"
                ),
                "checkpoint_path": r"checkpoint_path: (\S+?),",
                "max_seq_len": r"max_seq_len: (\d+)",
                "use_fp32_to_compute_logit": r"use_fp32_to_compute_logit: (\d+)",
            }
            for key, pattern in patterns.items():
                found = re.findall(pattern, text)
                if found:
                    manifest["observed"][key] = sorted(set(found))
            kernel_lines = [
                line.strip()[:300]
                for line in text.splitlines()
                if re.search(r"gdn|gated_delta|chunk_delta|fla\b", line, re.IGNORECASE)
            ]
            if kernel_lines:
                manifest["observed"]["gdn_kernel_lines"] = kernel_lines[:20]
        (self.output / "evidence" / "evidence-manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        archived = evidence / Path(log_path).name
        if archived.is_file():
            self.report["server_log"] = str(archived)
            self.report["server_log_sha256"] = copied.get(Path(log_path).name)
            self.report["observed_quantization"] = manifest["observed"].get(
                "observed_quantization", []
            )

    def _write_report(self):
        self._archive_server_evidence()
        self.report["failures"] = self.failures
        self.report["http_records"] = len(self.records)
        (self.output / "requests.jsonl").write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n" for record in self.records
            ),
            encoding="utf-8",
        )
        (self.output / "summary.json").write_text(
            json.dumps(self.report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _post(self, label, path, request, status=200):
        started = time.perf_counter()
        record = {"case": label, "endpoint": path, "request": request}
        try:
            response = self.session.post(
                self.endpoint + path,
                json=request,
                headers={"x-typesafe-request-id": "d3-acceptance-" + label},
                timeout=float(os.environ.get("D3_ACCEPTANCE_HTTP_TIMEOUT", "600")),
            )
            record.update(
                status_code=response.status_code,
                headers=dict(response.headers),
                response_text=response.text,
            )
            self.assertEqual(response.status_code, status, response.text)
            actual = None
            try:
                actual = response.json()
                record["response"] = actual
            except ValueError:
                pass
            return actual
        except Exception as error:
            record["error"] = str(error)
            raise
        finally:
            record["http_elapsed_ms"] = (time.perf_counter() - started) * 1000
            self.records.append(record)
            with (self.output / "requests.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _probe(self, label, request):
        """POST /v1/systemone recording the raw outcome without asserting it."""
        started = time.perf_counter()
        response = self.session.post(
            self.endpoint + "/v1/systemone",
            json=request,
            headers={"x-typesafe-request-id": "d3-acceptance-" + label},
            timeout=float(os.environ.get("D3_ACCEPTANCE_HTTP_TIMEOUT", "600")),
        )
        try:
            body = response.json()
        except ValueError:
            body = None
        record = {
            "case": label,
            "status_code": response.status_code,
            "response": body,
            "response_text": response.text,
        }
        record["http_elapsed_ms"] = (time.perf_counter() - started) * 1000
        self.records.append(record)
        with (self.output / "requests.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
        return response.status_code, body

    # ------------------------------------------------------------ response schema

    def _check_answer(self, question, answer):
        """Schema and self-consistency of one answer; official semantics only."""
        kind = question["type"]
        self.assertEqual(answer.get("type"), kind)
        if kind == "noul":
            noul = answer["noul"]
            self.assertIs(type(noul) in (float, int), True)
            self.assertTrue(0 <= noul <= 1)
            return
        probabilities = answer["probabilities"]
        if kind == "choice":
            self.assertEqual(list(probabilities), list(question["criteria"]))
            shares = [probabilities[key] for key in question["criteria"]]
        else:
            levels = question["criteria"]
            self.assertEqual(list(probabilities), [str(i) for i in range(len(levels))])
            shares = [probabilities[str(i)] for i in range(len(levels))]
            self.assertEqual(
                answer["legend"],
                {str(i): _canonical(level) for i, level in enumerate(levels)},
            )
            score = answer["score"]
            self.assertIs(type(score) in (float, int), True)
            self.assertAlmostEqual(
                score,
                sum(level * p for level, p in enumerate(shares)),
                delta=0.01,
            )
        self.assertTrue(all(math.isfinite(p) and 0 <= p <= 1 for p in shares))
        self.assertAlmostEqual(sum(shares), 1.0, delta=1e-5)
        confidence = answer["confidence"]
        self.assertIs(type(confidence) in (float, int), True)
        self.assertTrue(0 <= confidence <= 1)
        self.assertAlmostEqual(confidence, _confidence(shares), delta=0.01)
        if kind == "choice":
            best = 0
            for i in range(1, len(shares)):
                if shares[i] > shares[best]:
                    best = i
            self.assertEqual(answer["choice"], list(question["criteria"])[best])

    def _check_response(self, questions, actual, label):
        self.assertTrue(isinstance(actual.get("model"), str) and actual["model"])
        self.assertEqual(set(actual["answers"]), set(questions), label)
        usage = actual["usage"]
        self.assertIs(type(usage.get("input_tokens")), int)
        self.assertGreaterEqual(usage["input_tokens"], 0)
        self.assertEqual(usage["output_tokens"], 0)
        runnable = False
        for key, question in questions.items():
            answer = actual["answers"][key]
            if "error" in answer:
                self.assertEqual(
                    answer["error"], "max_length_exceeded", f"{label}:{key}"
                )
                self.assertTrue(answer.get("message"))
                self.assertIn("type", answer)
                continue
            runnable = True
            self._check_answer(question, answer)
        if runnable:
            self.assertGreater(usage["input_tokens"], 0, label)

    def _compare_reference(self, label, questions, actual):
        expected = self.reference_cases.get(label)
        if expected is None:
            return
        expected_answers = expected["answers"]
        self.assertEqual(set(expected_answers), set(questions), label)
        for key, question in questions.items():
            want = expected_answers[key]
            got = actual["answers"][key]
            if "error" in want:
                self.assertEqual(got.get("error"), want["error"], f"{label}:{key}")
                continue
            self.assertEqual(got.get("type"), want["type"], f"{label}:{key}")
            if question["type"] == "noul":
                self._gate(
                    label,
                    key,
                    "noul",
                    abs(got["noul"] - want["noul"]),
                    self.tolerance["max_abs_noul_error"],
                )
                continue
            probabilities = want["probabilities"]
            self.assertEqual(list(got["probabilities"]), list(probabilities))
            self._gate(
                label,
                key,
                "probability",
                max(abs(got["probabilities"][k] - v) for k, v in probabilities.items()),
                self.tolerance["max_abs_probability_error"],
            )
            self._gate(
                label,
                key,
                "confidence",
                abs(got["confidence"] - want["confidence"]),
                self.tolerance["max_confidence_error"],
                slack=0.00005,
            )
            if question["type"] == "choice":
                if self.same_choice:
                    # Margin-aware choice gate: semantically decisive questions
                    # (reference top-2 gap above twice the registered
                    # probability tolerance) must match the official choice on
                    # every hardware/precision; only a reference near-tie may
                    # differ, and then the difference is recorded with its
                    # top-3 and margins. Never a global choice-gate disable.
                    ranked = sorted(probabilities.items(), key=lambda kv: -kv[1])
                    gap = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 1.0
                    tolerance = self.tolerance["max_abs_probability_error"]
                    near_tie = gap <= 2 * tolerance
                    if got["choice"] != want["choice"]:
                        actual_ranked = sorted(
                            got["probabilities"].items(), key=lambda kv: -kv[1]
                        )
                        self.report.setdefault("argmax_differences", []).append(
                            {
                                "case": label,
                                "question": key,
                                "reference_choice": want["choice"],
                                "actual_choice": got["choice"],
                                "reference_top3": [
                                    [name, round(value, 6)]
                                    for name, value in ranked[:3]
                                ],
                                "actual_top3": [
                                    [name, round(value, 6)]
                                    for name, value in actual_ranked[:3]
                                ],
                                "reference_top2_gap": gap,
                                "near_tie_threshold": 2 * tolerance,
                                "near_tie": near_tie,
                            }
                        )
                        if not near_tie:
                            self.fail(
                                f"{label}:{key}: choice differs from the official "
                                f"reference on a decisive question (reference "
                                f"top-2 gap {gap:.6f} exceeds 2x the tolerance "
                                f"{tolerance}); got {got['choice']!r}, want "
                                f"{want['choice']!r}"
                            )
            else:
                # Score expected value: gated diagnostic with BOTH units
                # (EV delta in score units and the max per-option probability
                # delta); strict mode keeps the assertion.
                ev_delta = abs(got["score"] - want["score"])
                option_delta = max(
                    abs(got["probabilities"][k] - v) for k, v in probabilities.items()
                )
                self._gate(
                    label,
                    key,
                    "score_ev",
                    ev_delta,
                    self.tolerance["max_score_error"],
                    slack=0.00005,
                )
                self.records[-1].setdefault("score_units", {})[key] = {
                    "ev_delta": ev_delta,
                    "max_option_prob_delta": option_delta,
                }
        want_usage = expected.get("usage")
        if want_usage is not None:
            self.assertEqual(actual["usage"]["output_tokens"], 0)
            self.assertEqual(
                actual["usage"]["input_tokens"],
                want_usage["input_tokens"],
                f"{label}: input token accounting differs from the official "
                "reference (expanded prompt tokens per runnable question)",
            )

    def _gate(self, label, question, kind, error, tolerance, slack=0.0):
        record = self.records[-1]
        record.setdefault("max_abs_errors", {})[question] = error
        within = error <= tolerance + slack
        record.setdefault("within_tolerance", {})[question] = within
        if not within:
            self.report["probability_tolerance_exceedances"].append(
                {"case": label, "question": question, "kind": kind, "error": error}
            )
        if self.probability_gate == "strict":
            self.assertLessEqual(error, tolerance + slack, f"{label}:{question}")

    # ------------------------------------------------------------------ requests

    def _ask(self, label, case):
        request = _materialize(case, self.model)
        actual = self._post(label, "/v1/systemone", request)
        self._check_response(request["questions"], actual, label)
        self._compare_reference(case["name"], request["questions"], actual)
        return actual

    def _ask_video(self, label, case_name):
        """Video cases: required by default; explicit interim skip is partial.

        require (default): the request must be served; a 422 "video inputs are
        not available" gap answer fails the run - a valid-video 422 is never a
        passing full acceptance. skip: the case is recorded as skipped and the
        final status is downgraded to PASS_PARTIAL_SCOPE.
        """
        case = self.cases[case_name]
        request = _materialize(case, self.model)
        if self.video_gate == "skip":
            if self.video_supported is None:
                status, body = self._probe(label, request)
                if status == 200:
                    self.video_supported = True
                    self.report["video_support"] = "supported"
                elif (
                    status == 422
                    and isinstance(body, dict)
                    and (
                        "video inputs are not available" in str(body.get("detail", ""))
                    )
                ):
                    self.video_supported = False
                    self.report["video_support"] = (
                        "NOT_SUPPORTED (interim skip; gap 422 recorded)"
                    )
                else:
                    # Any other failure is a real defect, not the known gap.
                    self.fail(f"{label}: unexpected video status {status}: {body}")
                    return
            if not self.video_supported:
                self.report.setdefault("video_cases_skipped", []).append(label)
                return
            self._ask(label, case)
            return
        if self.video_supported is None:
            status, body = self._probe(label, request)
            if (
                status == 422
                and isinstance(body, dict)
                and ("video inputs are not available" in str(body.get("detail", "")))
            ):
                self.video_supported = False
                self.report["video_support"] = (
                    "NOT_SUPPORTED: frontend reports the multimodal RPC video gap; "
                    "full acceptance requires text+image+video+mixed"
                )
                self.fail(
                    f"{label}: full acceptance requires text+image+video+mixed; "
                    "the frontend answered the official video gap 422. Use "
                    "D3_ACCEPTANCE_VIDEO=skip only for explicit interim "
                    "text/image bring-up (status PASS_PARTIAL_SCOPE, never full)"
                )
            if status != 200:
                # Keep the probe failure in the report before failing: the
                # summary must carry what the video path actually answered.
                self.report["video_support"] = (
                    f"FAILED: probe status {status}: {str(body)[:300]}"
                )
                self.fail(f"{label}: unexpected video status {status}: {body}")
            self.video_supported = True
            self._check_response(request["questions"], body, label)
            self._compare_reference(case_name, request["questions"], body)
            return
        if not self.video_supported:
            self.fail(f"{label}: video support was required but missing")
            return
        self._ask(label, case)

    def _case(self, name, function):
        try:
            function()
        except Exception as error:
            self.failures.append({"case": name, "error": str(error)})

    def test_d3_acceptance(self):
        # Each fixture's first and repeated requests share exact reference
        # input; a matching repeat also asserts deterministic recomputation.
        for name in self.cases:
            if name == "cross_question_isolation":
                continue
            for phase in ("first", "repeat"):
                label = f"{name}_{phase}"
                if name in ("video", "mixed"):
                    self._case(
                        label,
                        lambda name=name, label=label: self._ask_video(label, name),
                    )
                    continue
                self._case(
                    label,
                    lambda name=name, label=label: self._ask(label, self.cases[name]),
                )

        self._case("isolation", self._isolation)
        self._case("errors", self._errors)
        # Recovery after rejected requests is part of the same server lifecycle.
        self._case(
            "after_errors",
            lambda: self._ask("after_errors", self.cases["short"]),
        )
        skipped = self.report.get("video_cases_skipped", [])
        exceedances = self.report.get("probability_tolerance_exceedances", [])
        if self.failures:
            self.report["status"] = "FAIL"
        elif skipped and self.video_gate == "skip":
            # Interim text/image scope can never be reported as a full PASS.
            self.report["status"] = "PASS_PARTIAL_SCOPE"
        elif self.probability_gate == "report":
            # Report mode collects reference numerics as diagnostics: a clean
            # run certifies SYSTEM correctness, never a strict numeric PASS.
            self.report["status"] = "PASS_SYSTEM_CORRECTNESS"
        else:
            self.report["status"] = "PASS"
        self.report["system_gate"] = "FAIL" if self.failures else "PASS"
        self.report["numerical_gate"] = (
            "strict (assertions enforced)"
            if self.probability_gate == "strict"
            else (
                "PASS"
                if not exceedances
                else "EXCEEDED_RECORDED (diagnostics; not a system-correctness failure)"
            )
        )
        self.assertFalse(self.failures, json.dumps(self.failures, ensure_ascii=False))

    def _isolation(self):
        """Full rows: answers never depend on the other questions in a batch."""
        case = self.cases["cross_question_isolation"]
        state = case["state"]
        set_a = case["set_a"]
        set_b = case["set_b"]
        alone_a = self._post(
            "isolation_a", "/v1/systemone", {"state": state, "questions": set_a}
        )
        self._check_response(set_a, alone_a, "isolation_a")
        alone_b = self._post(
            "isolation_b", "/v1/systemone", {"state": state, "questions": set_b}
        )
        self._check_response(set_b, alone_b, "isolation_b")

        def compare(got, want, key):
            self.assertEqual(got["type"], want["type"], key)
            if "noul" in want:
                self.assertLessEqual(
                    abs(got["noul"] - want["noul"]),
                    self.tolerance["max_abs_noul_error"],
                    key,
                )
                return
            error = max(
                abs(got["probabilities"][k] - v)
                for k, v in want["probabilities"].items()
            )
            self.assertLessEqual(
                error, self.tolerance["max_abs_probability_error"], key
            )
            if self.same_choice and "choice" in want:
                self.assertEqual(got["choice"], want["choice"], key)

        for label, questions in (
            ("isolation_ab", {**set_a, **set_b}),
            ("isolation_ba", {**set_b, **set_a}),
        ):
            combined = self._post(
                label, "/v1/systemone", {"state": state, "questions": questions}
            )
            self._check_response(questions, combined, label)
            # Same question in different company: probabilities must not move
            # beyond tolerance, because d3 rows never share language KV/GDN
            # state across questions.
            for key, question in set_a.items():
                compare(combined["answers"][key], alone_a["answers"][key], key)
            for key, question in set_b.items():
                compare(combined["answers"][key], alone_b["answers"][key], key)

        # An identical repeated request must reproduce the same answers.
        repeat = self._post(
            "isolation_repeat", "/v1/systemone", {"state": state, "questions": set_a}
        )
        self._check_response(set_a, repeat, "isolation_repeat")
        for key, answer in alone_a["answers"].items():
            got = repeat["answers"][key]
            if "error" in answer:
                self.assertEqual(got.get("error"), answer["error"], key)
            elif "noul" in answer:
                self.assertEqual(got["noul"], answer["noul"], key)
            else:
                self.assertEqual(got["probabilities"], answer["probabilities"], key)

        # Mutating one row of a multi-question request must not disturb the rest.
        mutated = copy.deepcopy(set_a)
        mutated["a_extra"] = {
            "type": "choice",
            "instructions": "Pick the odd one out.",
            "criteria": {"one": "the first item", "two": "the second item"},
        }
        batch = {**set_b, **mutated}
        mixed = self._post(
            "isolation_mutated", "/v1/systemone", {"state": state, "questions": batch}
        )
        self._check_response(batch, mixed, "isolation_mutated")
        for key in set_b:
            compare(mixed["answers"][key], alone_b["answers"][key], key)

    def _errors(self):
        short = _materialize(self.cases["short"], self.model)
        state = short["state"]
        good = short["questions"]["kind"]
        invalid_questions = [
            {"type": "unknown_kind", "criteria": {"a": "x"}},
            {
                "type": "choice",
                "criteria": {f"opt{i}": f"option {i}" for i in range(MAX_OPTIONS + 1)},
            },
            {"type": "score", "criteria": [f"level {i}" for i in range(11)]},
            {"type": "score", "criteria": ["dup", "dup"]},
            {"type": "score", "criteria": ["only-one"]},
            {"type": "choice", "criteria": {}},
            {"type": "choice", "criteria": {"": "empty key"}},
            {"type": "noul", "criteria": {"maybe": "neither"}},
        ]
        for i, question in enumerate(invalid_questions):
            request = {"state": state, "questions": {f"bad_{i}": question}}

            def reject(request=request, i=i):
                response = self._post(
                    f"invalid_question_{i}", "/v1/systemone", request, 422
                )
                detail = (response or {}).get("detail")
                self.assertEqual(
                    isinstance(detail, dict) and set(detail) == {"invalid_questions"},
                    True,
                    "official servers reject the whole request with "
                    f'{{"detail": {{"invalid_questions": ...}}}}; got {detail!r}',
                )
                self.assertIn(f"bad_{i}", detail["invalid_questions"])

            self._case(f"invalid_question_{i}", reject)
        structural = [
            {"state": state, "questions": {}},
            {"state": state, "questions": [good]},
            {"state": state},
            {"questions": good},
            {"state": state, "questions": {"q": good}, "unknown_field": 1},
            {"state": state, "questions": {"q": good}, "images": "not-a-list"},
            {
                "state": state,
                "questions": {"q": good},
                "images": ["data:image/gif;base64,R0lGODlhAQABAAAAACw="],
            },
            {
                "state": state,
                "questions": {"q": good},
                "images": ["data:image/png;base64,!!not-base64!!"],
            },
            {
                "state": state,
                "questions": {"q": good},
                "videos": ["data:video/mp4;base64,AAAA"],
            },
        ]
        for i, request in enumerate(structural):

            def reject_structural(request=request):
                response = self._post(
                    f"invalid_request_{i}", "/v1/systemone", request, 422
                )
                self.assertIn("detail", response or {})

            self._case(f"invalid_request_{i}", reject_structural)

        # Not JSON at all.
        response = self.session.post(
            self.endpoint + "/v1/systemone",
            data=b"this is not json",
            headers={"content-type": "application/json"},
            timeout=float(os.environ.get("D3_ACCEPTANCE_HTTP_TIMEOUT", "600")),
        )
        self.assertEqual(response.status_code, 422, response.text)
        self.records.append(
            {
                "case": "invalid_body",
                "status_code": response.status_code,
                "response_text": response.text,
            }
        )
        # Over-limit questions are answered, not truncated and not rejected whole.
        over = _materialize(self.cases["over_limit"], self.model)
        actual = self._post("over_limit", "/v1/systemone", over)
        self._check_response(over["questions"], actual, "over_limit")
        for key, answer in actual["answers"].items():
            self.assertEqual(answer["error"], "max_length_exceeded", key)
            self.assertTrue(answer["message"])


if __name__ == "__main__":
    unittest.main()
