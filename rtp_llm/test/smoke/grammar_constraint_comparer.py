import json
import logging
import os
import re
from typing import Any, List, Union

import requests

from smoke.common_def import QueryStatus, SmokeException
from smoke.openai_comparer import OpenaiComparer
from smoke.utils import no_compare, save_response
from rtp_llm.openai.api_datatype import (
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
)


class GrammarConstraintComparer(OpenaiComparer):
    """Comparer for chat queries carrying a `response_format`.

    Dispatch rules:
      - `response_format.type == "regex"`:
            validate via `re.fullmatch(pattern, content)` per choice; skip
            byte-equal (content varies with model/prompt and is unbounded
            unless the pattern is bounded).
      - `response_format.type == "json_schema"` AND the query has
            `"grammar_constraint_only": true`:
            validate via `json.loads(content)` + recursive `required`/`type`
            walk; skip byte-equal. Use this when the task_info has no real
            golden content (e.g., PD-sep, MTP) and only wants to assert the
            structural constraint was enforced.
      - otherwise (json_schema without the opt-in flag, or no response_format):
            delegate to `OpenaiComparer` (byte-equal determinism, the
            existing cache-golden contract).
    """

    # -- run override for the error-path ------------------------------------

    def run(self):  # type: ignore[override]
        """For queries with `expect_grammar_error: true`, an *explicit* server
        rejection (HTTP 4xx/5xx body, or 200 with `error`/`detail`/`message`
        envelope) is the PASS signal. Connection-refused / read-timeout /
        process-exited are NOT — those indicate the server died or never
        came up, not a successful reject path. We only PASS the silent
        path when the response actually parses to a ChatCompletion-shaped
        envelope that we then explicitly validate.
        """
        if not self.qr_info.get("expect_grammar_error"):
            return super().run()

        # TODO(grammar): error-path queries that are Pydantic-invalid will be
        # rejected client-side before reaching the server. To truly test server
        # compile rejection, use schemas that pass local validation but fail on
        # the xgrammar backend (e.g. unsupported features). See review issue #2.
        query_info = self.format_query(self.qr_info["query"])
        self.tracer.query = query_info
        request_info = query_info.model_dump(exclude_defaults=True)
        self.maybe_set_concurrency(query_info)

        # Bypass server_manager.visit (which collapses every non-200 outcome to
        # `(False, None)` after retries) and post directly so we can tell apart
        # an explicit reject (4xx/5xx with body) from a hard failure (timeout /
        # connection-refused / process exit). Only the former is a valid PASS.
        try:
            url = (
                f"http://0.0.0.0:{int(self.server_manager._port)}"
                f"{self.request_endpoint}"
            )
            response = requests.post(url, json=request_info, timeout=30)
        except requests.exceptions.Timeout as e:
            self._raise(
                f"[grammar-error-path] request timed out: {e} — server may be hung; not an explicit reject"
            )
            return
        except requests.exceptions.ConnectionError as e:
            self._raise(
                f"[grammar-error-path] connection error: {e} — server may have died; not an explicit reject"
            )
            return

        # Verify the server is still alive before treating any non-200 as a
        # legitimate reject. A 5xx that came together with the process exiting
        # is a crash, not an "acceptable reject".
        proc = getattr(self.server_manager, "_server_process", None)
        if proc is not None and proc.poll() is not None:
            self._raise(
                f"[grammar-error-path] server process exited (rc={proc.poll()}) "
                "during error-path request — crash, not a reject"
            )
            return

        if response.status_code != 200:
            logging.info(
                "[grammar-error-path] server returned %d with body %.200s — explicit reject",
                response.status_code,
                response.text,
            )
            return

        res = (
            [x.encode("utf-8") for x in response.text.splitlines() if x]
            if response.headers.get("Transfer-Encoding") == "chunked"
            else response.text
        )

        # HTTP 200 — now inspect the body.
        try:
            actual_dict = self.curl_response_to_json(query_info, res)
        except Exception as e:
            logging.info(
                "[grammar-error-path] curl_response_to_json failed: %s — "
                "acceptable (malformed body in error envelope)",
                e,
            )
            return

        if isinstance(actual_dict, dict):
            for err_key in ("error", "detail", "message"):
                if err_key in actual_dict and actual_dict[err_key]:
                    logging.info(
                        "[grammar-error-path] response carries %r field: %.200s",
                        err_key,
                        str(actual_dict[err_key]),
                    )
                    return

        try:
            actual_result = self.format_result(actual_dict)
        except Exception as e:
            logging.info(
                "[grammar-error-path] format_result rejected body (=error envelope): %s",
                e,
            )
            return

        self.tracer.actual_result = actual_result
        self._dump_actual_to_artifact(actual_dict)
        # Mirror BaseComparer.run(): honor golden rewrite / no-compare before the
        # actual assertion so the error-path stays consistent with the normal path.
        if save_response():
            self.qr_info["result"] = actual_result.model_dump(exclude_defaults=True)
        if no_compare():
            return
        expect_result = self.format_result(self.qr_info["result"])
        self.tracer.expect_result = expect_result
        self._validate_grammar_error(expect_result, actual_result)

    # -- entry point ---------------------------------------------------------

    def compare_result(
        self,
        expect_result: Union[ChatCompletionResponse, ChatCompletionStreamResponse],
        actual_result: Union[ChatCompletionResponse, ChatCompletionStreamResponse],
    ) -> None:
        rf = self.qr_info.get("query", {}).get("response_format") or {}
        rf_type = rf.get("type")
        constraint_only = bool(self.qr_info.get("grammar_constraint_only"))

        if rf_type == "regex":
            self._validate_regex(expect_result, actual_result, rf)
            return

        if rf_type == "json_schema" and constraint_only:
            self._validate_json_schema(expect_result, actual_result, rf)
            return

        if rf_type == "ebnf":
            # EBNF grammar outputs are model-dependent; require a regex in
            # task_info that the grammar must produce (cheap structural check).
            validate_pattern = self.qr_info.get("ebnf_validate_regex")
            if validate_pattern:
                self._validate_regex(
                    expect_result, actual_result, {"pattern": validate_pattern}
                )
                return
            self._raise("EBNF grammar smoke case missing 'ebnf_validate_regex'")

        if self.qr_info.get("skip_content_check"):
            # Mixed-batch sidecar: a non-grammar query running alongside
            # grammar streams. We do NOT want a content-level golden (would
            # be brittle under concurrency_test batching), we only want to
            # assert the response is well-formed and non-empty — i.e. the
            # neighbouring grammar stream did not mask-poison this one.
            self._validate_well_formed(expect_result, actual_result)
            return

        if self.qr_info.get("expect_grammar_error"):
            # Safety net: run() handles the dedicated malformed-grammar path
            # before compare_result(), but keep this branch so direct callers of
            # compare_result() do not silently fall through to byte-equal checks.
            # Error-path smoke: the query deliberately carries a malformed
            # schema/regex/ebnf. The server MUST reject it gracefully —
            # either via an error-field in the response body or via a
            # non-stop finish_reason — rather than hang or crash. We
            # specifically do NOT require a particular error message
            # wording (that's xgrammar / renderer-dependent); we only
            # require evidence the request terminated with a failure
            # signal of some kind.
            self._validate_grammar_error(expect_result, actual_result)
            return

        super().compare_result(expect_result, actual_result)

    # -- regex path ---------------------------------------------------------

    def _validate_regex(self, expect_result, actual_result, rf: dict) -> None:
        pattern = rf.get("pattern")
        if not pattern:
            self._raise("regex response_format missing 'pattern'")

        try:
            compiled = re.compile(pattern)
        except re.error as e:
            self._raise(f"invalid regex in task_info: {pattern!r} ({e})")

        diffs: List[str] = []
        self._check_response_type(expect_result, actual_result, diffs)

        actual_choices = self._choices(actual_result)
        if not actual_choices:
            diffs.append("actual response has no choices — cannot validate grammar constraint")
            self._raise(self._format_all_diffs(diffs))

        for idx, choice in enumerate(actual_choices):
            content = self._extract_content(choice)
            finish_reason = self._extract_finish_reason(choice)
            if content is None:
                diffs.append(f"choice[{idx}]: content is None; regex cannot be validated")
                continue
            # Reasoner contract: the OpenAI endpoint splits the thinking segment
            # into `reasoning_content`, so `content` must be the post-</think>
            # answer only. A </think> marker leaking into `content` means the
            # reasoning split / grammar gating is broken — surface it as a failure
            # instead of masking it by stripping the suffix away.
            if "</think>" in content:
                diffs.append(
                    f"choice[{idx}]: content contains </think> — thinking segment "
                    f"was not split into reasoning_content (reasoning parser / "
                    f"grammar gating bug)\n"
                    f"  content({len(content)}B): {content[:500]!r}"
                )
                continue
            if compiled.fullmatch(content) is None:
                preview = content[:500]
                suffix = "..." if len(content) > len(preview) else ""
                diffs.append(
                    f"choice[{idx}]: regex NOT matched\n"
                    f"  pattern      : {pattern}\n"
                    f"  finish_reason: {finish_reason}\n"
                    f"  content({len(content)}B): {preview!r}{suffix}"
                )
            elif finish_reason not in ("stop", "length"):
                diffs.append(
                    f"choice[{idx}]: matched but unexpected finish_reason={finish_reason}"
                )

        if diffs:
            self._raise(self._format_all_diffs(diffs))

    # -- json_schema path ---------------------------------------------------

    def _validate_json_schema(self, expect_result, actual_result, rf: dict) -> None:
        schema_envelope = rf.get("json_schema") or {}
        schema = schema_envelope.get("schema") or {}
        if not schema:
            self._raise("json_schema response_format missing 'schema'")

        # Opt-in: some reasoner smoke cases assert reasoning_content stays free-form
        # during the thinking phase. Disabled by default to avoid flaky heuristics.
        check_reasoning_unconstrained = bool(
            self.qr_info.get("grammar_check_reasoning_unconstrained")
        )

        diffs: List[str] = []
        self._check_response_type(expect_result, actual_result, diffs)

        actual_choices = self._choices(actual_result)
        if not actual_choices:
            diffs.append("actual response has no choices — cannot validate grammar constraint")
            self._raise(self._format_all_diffs(diffs))

        for idx, choice in enumerate(actual_choices):
            content = self._extract_content(choice)
            finish_reason = self._extract_finish_reason(choice)
            if content is None:
                diffs.append(f"choice[{idx}]: content is None; schema cannot be validated")
                continue
            # Reasoner contract: thinking text belongs in `reasoning_content`;
            # `content` must be the post-</think> answer only. A leaked </think>
            # means the reasoning split / grammar gating is broken — fail instead
            # of stripping the suffix away and hiding the bug.
            if "</think>" in content:
                diffs.append(
                    f"choice[{idx}]: content contains </think> — thinking segment "
                    f"was not split into reasoning_content (reasoning parser / "
                    f"grammar gating bug)\n"
                    f"  content({len(content)}B): {content[:500]!r}"
                )
                continue
            clean = content.strip()
            try:
                parsed = json.loads(clean)
            except json.JSONDecodeError as e:
                diffs.append(
                    f"choice[{idx}]: content is not valid JSON\n"
                    f"  error        : {e}\n"
                    f"  finish_reason: {finish_reason}\n"
                    f"  content({len(content)}B): {content!r}"
                )
                continue
            err = _schema_check(parsed, schema, path="$")
            if err is not None:
                diffs.append(
                    f"choice[{idx}]: content does not conform to schema\n"
                    f"  error        : {err}\n"
                    f"  finish_reason: {finish_reason}\n"
                    f"  content({len(content)}B): {content!r}"
                )
            elif finish_reason not in ("stop", "length"):
                diffs.append(
                    f"choice[{idx}]: valid JSON but unexpected finish_reason={finish_reason}"
                )

            if check_reasoning_unconstrained:
                reasoning = self._extract_reasoning_content(choice) or ""
                # TODO(grammar): this heuristic is over-sensitive — model version
                # or template changes can produce short reasoning that triggers a
                # false failure. Convert to opt-in or weaken thresholds to avoid
                # smoke flakiness. See review issue #1 (PR grammar_logits_processor).
                # Heuristic: a free-form reasoning trace has whitespace or CJK
                # text and many letters. If the mask leaked into the thinking
                # phase, we'd see mostly schema tokens (braces, quotes, digits).
                has_text_separator = bool(re.search(r"\s", reasoning)) or any(
                    "\u4e00" <= c <= "\u9fff" for c in reasoning[:50]
                )
                alpha_count = sum(c.isalpha() for c in reasoning)
                if (
                    len(reasoning) < 50
                    or not has_text_separator
                    or alpha_count < 20
                ):
                    diffs.append(
                        f"choice[{idx}]: reasoning_content looks mask-constrained "
                        f"(expected free-form text during thinking)\n"
                        f"  len         : {len(reasoning)}\n"
                        f"  text_sep    : {has_text_separator}\n"
                        f"  alpha_count : {alpha_count}\n"
                        f"  preview     : {reasoning[:120]!r}"
                    )

        if diffs:
            self._raise(self._format_all_diffs(diffs))

    # -- invalid-grammar error path ----------------------------------------

    def _validate_grammar_error(self, expect_result, actual_result) -> None:
        """Assert the server rejected a malformed grammar request rather
        than crashed. We accept any of:
          (a) actual_result has a top-level `error` field / attr,
          (b) some choice has finish_reason == "error" (or similar),
          (c) some choice content is empty while finish_reason != "stop",
          (d) an InvalidGrammarError / schema error keyword in content.
        """
        # Response type check still applies — we expect a ChatCompletion*
        # object, not an arbitrary dict.
        diffs: List[str] = []
        self._check_response_type(expect_result, actual_result, diffs)
        if diffs:
            self._raise(self._format_all_diffs(diffs))

        # (a) top-level error field?
        err_field = getattr(actual_result, "error", None)
        if err_field:
            return

        choices = self._choices(actual_result)
        if not choices:
            # No choices at all is itself an error signal — server rejected
            # before producing any stream output. Acceptable.
            return

        for choice in choices:
            content = self._extract_content(choice) or ""
            finish_reason = self._extract_finish_reason(choice)

            # (b) explicit error-ish finish_reason
            if finish_reason not in ("stop", "length", None):
                return

            # (c) empty-content + non-stop finish_reason
            if len(content) == 0 and finish_reason != "stop":
                return

            # (d) content mentions a schema/grammar error keyword
            low = content.lower()
            for kw in (
                "failed to compile",
                "invalid grammar",
                "invalid schema",
                "invalidgrammar",
                "cannot parse",
                "xgrammar",
                "json schema error",
                "compile error",
            ):
                if kw in low:
                    return

        # None of the signals fired. The server responded as if the
        # malformed query were fine — that's the failure mode we want to
        # guard against.
        sample_choice = choices[0]
        content = self._extract_content(sample_choice)
        finish_reason = self._extract_finish_reason(sample_choice)
        self._raise(
            "malformed grammar query was accepted without any error "
            "signal — server would have produced unconstrained output.\n"
            f"  finish_reason: {finish_reason}\n"
            f"  content      : {content!r}"
        )

    # -- mixed-batch sidecar path ------------------------------------------

    def _validate_well_formed(self, expect_result, actual_result) -> None:
        """Sanity check for a non-grammar query sharing a batch with grammar
        streams. We assert the response is a parsed ChatCompletion* (not an
        exception), each choice has a non-empty string `content`, and
        finish_reason is one of the normal values. If a neighbouring grammar
        stream's bitmask leaked onto this stream, `content` would either be
        empty, missing, or crash parsing — any of which this catches."""
        diffs: List[str] = []
        self._check_response_type(expect_result, actual_result, diffs)

        actual_choices = self._choices(actual_result)
        if not actual_choices:
            diffs.append("actual response has no choices — cannot validate grammar constraint")
            self._raise(self._format_all_diffs(diffs))

        for idx, choice in enumerate(actual_choices):
            content = self._extract_content(choice)
            finish_reason = self._extract_finish_reason(choice)
            if content is None:
                diffs.append(f"choice[{idx}]: content is None")
                continue
            if not isinstance(content, str):
                diffs.append(
                    f"choice[{idx}]: content is not str, type={type(content).__name__}"
                )
                continue
            if len(content) == 0:
                diffs.append(
                    f"choice[{idx}]: content is empty (finish_reason={finish_reason})"
                )
                continue
            if finish_reason not in ("stop", "length"):
                diffs.append(
                    f"choice[{idx}]: unexpected finish_reason={finish_reason} "
                    f"(content={content[:60]!r})"
                )

        if diffs:
            self._raise(self._format_all_diffs(diffs))

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _choices(result: Any) -> list:
        return getattr(result, "choices", None) or []

    @staticmethod
    def _extract_content(choice: Any) -> Any:
        message = getattr(choice, "message", None)
        if message is not None:
            return getattr(message, "content", None)
        delta = getattr(choice, "delta", None)
        if delta is not None:
            return getattr(delta, "content", None)
        return None

    @staticmethod
    def _extract_reasoning_content(choice: Any) -> Any:
        message = getattr(choice, "message", None)
        if message is not None:
            return getattr(message, "reasoning_content", None)
        delta = getattr(choice, "delta", None)
        if delta is not None:
            return getattr(delta, "reasoning_content", None)
        return None

    @staticmethod
    def _extract_finish_reason(choice: Any) -> Any:
        reason = getattr(choice, "finish_reason", None)
        if reason is None:
            return None
        return getattr(reason, "value", reason)

    @staticmethod
    def _check_response_type(expect_result, actual_result, diffs: List[str]) -> None:
        if type(expect_result) != type(actual_result):
            diffs.append(
                "type not equal:\n"
                f"  expect: {type(expect_result).__name__}\n"
                f"  actual: {type(actual_result).__name__}"
            )

    def _raise(self, message: str) -> None:
        raise SmokeException(QueryStatus.COMPARE_FAILED, message)


# ---- stdlib schema walker (no jsonschema dep) ----------------------------


_PY_TYPE_MAP = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}


def _schema_check(value: Any, schema: dict, path: str) -> str | None:
    # Lightweight smoke-time validator. Covers the keywords actually used in
    # the grammar smoke datasets: type/properties/items/required/enum, plus
    # length/range/count bounds (maxLength, minLength, minimum, maximum,
    # exclusiveMinimum, exclusiveMaximum, maxItems, minItems). Does NOT
    # implement oneOf/anyOf/allOf/not, pattern, format, or additionalProperties
    # subschemas — if you add such a keyword to a smoke schema, extend this
    # function or the constraint will silently pass.
    if not isinstance(schema, dict):
        return None

    t = schema.get("type")
    if isinstance(t, list):
        # union; accept if any type matches
        for one in t:
            single = {"type": one, **{k: v for k, v in schema.items() if k != "type"}}
            if _schema_check(value, single, path) is None:
                return None
        return f"{path}: value {type(value).__name__} does not match any of type={t}"

    if t in ("integer", "number") and isinstance(value, bool):
        # In Python, bool is a subclass of int; JSON schema treats booleans as
        # distinct from integer/number, so reject True/False here explicitly.
        return f"{path}: expected {t}, got bool"

    py_t = _PY_TYPE_MAP.get(t) if t else None
    if py_t is not None and not isinstance(value, py_t):
        return f"{path}: expected type {t}, got {type(value).__name__}"

    enum_vals = schema.get("enum")
    if isinstance(enum_vals, list) and value not in enum_vals:
        return f"{path}: value {value!r} not in enum {enum_vals}"

    if t == "string" and isinstance(value, str):
        max_len = schema.get("maxLength")
        if isinstance(max_len, int) and len(value) > max_len:
            return f"{path}: string length {len(value)} > maxLength {max_len}"
        min_len = schema.get("minLength")
        if isinstance(min_len, int) and len(value) < min_len:
            return f"{path}: string length {len(value)} < minLength {min_len}"

    if t in ("integer", "number") and isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and value < minimum:
            return f"{path}: value {value} < minimum {minimum}"
        maximum = schema.get("maximum")
        if isinstance(maximum, (int, float)) and value > maximum:
            return f"{path}: value {value} > maximum {maximum}"
        excl_min = schema.get("exclusiveMinimum")
        if isinstance(excl_min, (int, float)) and value <= excl_min:
            return f"{path}: value {value} <= exclusiveMinimum {excl_min}"
        excl_max = schema.get("exclusiveMaximum")
        if isinstance(excl_max, (int, float)) and value >= excl_max:
            return f"{path}: value {value} >= exclusiveMaximum {excl_max}"

    if t == "object":
        props = schema.get("properties") or {}
        required = schema.get("required") or []
        if not isinstance(value, dict):
            return f"{path}: expected object"
        for req_key in required:
            if req_key not in value:
                return f"{path}.{req_key}: required field missing"
        for key, sub_schema in props.items():
            if key in value:
                err = _schema_check(value[key], sub_schema, path=f"{path}.{key}")
                if err is not None:
                    return err
        if schema.get("additionalProperties") is False:
            extra = set(value.keys()) - set(props.keys())
            if extra:
                return f"{path}: unexpected keys {sorted(extra)} (additionalProperties=false)"
    elif t == "array":
        items_schema = schema.get("items") or {}
        if not isinstance(value, list):
            return f"{path}: expected array"
        max_items = schema.get("maxItems")
        if isinstance(max_items, int) and len(value) > max_items:
            return f"{path}: array length {len(value)} > maxItems {max_items}"
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            return f"{path}: array length {len(value)} < minItems {min_items}"
        for i, item in enumerate(value):
            err = _schema_check(item, items_schema, path=f"{path}[{i}]")
            if err is not None:
                return err

    return None
