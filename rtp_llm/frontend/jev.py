"""JEV System1 protocol and bounded strategy execution, without model dependencies.

The bare prompt, option permutations and tournament reconstruction follow
autotrust/JEV-27B-VL serve_decide.py at
9e53b8a809d2ab98e3e6a55b2b2dbf94b4c1e84a. The engine callback returns compact
raw logits after one prefill; bias and temperature are applied exactly once here.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import random
import time
import zlib
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Sequence, TypeVar

from rtp_llm.frontend import systemone

PROTOCOL = "jev27-bare-v1"
KINDS = ("noul", "score", "choice")
STRATEGIES = ("auto", "single", "permute", "tournament")
T = TypeVar("T")
R = TypeVar("R")


class JevRequestError(ValueError):
    """Invalid client input, distinct from malformed model output."""


def _finite(value: Any, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True)
class JevKindSpec:
    row_indices: tuple[int, ...]
    label_names: tuple[str, ...]
    bias: tuple[float, ...]
    temperature: float


@dataclass(frozen=True)
class JevHeadSpec:
    num_labels: int
    kinds: dict[str, JevKindSpec]
    model: str = "jev-latest"
    model_type: str = "jev"
    default_strategy: str = "single"
    max_images: int = 8
    max_questions: int = 64

    @property
    def max_options(self) -> int:
        return len(self.kinds["choice"].row_indices)

    @classmethod
    def from_config(cls, config: dict, *, model: str | None = None) -> JevHeadSpec:
        if not isinstance(config, dict):
            raise ValueError("jev_config must be an object")
        if (
            type(config.get("schema_version")) is not int
            or config["schema_version"] != 1
            or config.get("mode") != "system1"
        ):
            raise ValueError("JEV requires schema_version=1 and mode=system1")
        if config.get("protocol", PROTOCOL) != PROTOCOL:
            raise ValueError("unsupported JEV protocol")
        if config.get("template_version", "bare-v1") != "bare-v1":
            raise ValueError("unsupported JEV template_version")
        if (
            config.get("prefix", "") != ""
            or config.get("head_includes_bias", False) is not False
        ):
            raise ValueError(
                "JEV requires an empty Qwen prefix and raw logits without bias"
            )
        count = config.get("num_labels")
        if type(count) is not int or not 24 <= count <= 264:
            raise ValueError("num_labels must be an integer in 24..264")
        kinds = config.get("kinds")
        if not isinstance(kinds, dict) or any(k not in kinds for k in KINDS):
            raise ValueError("jev_config requires noul, score and choice kinds")
        parsed = {}
        for kind in KINDS:
            item = kinds[kind]
            if not isinstance(item, dict):
                raise ValueError(f"kinds.{kind} must be an object")
            rows, names = item.get("row_indices"), item.get("label_names")
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"kinds.{kind}.row_indices must be nonempty")
            if any(type(i) is not int or not 0 <= i < count for i in rows):
                raise ValueError(f"kinds.{kind} contains an invalid row index")
            if len(set(rows)) != len(rows):
                raise ValueError(f"kinds.{kind} contains duplicate row indices")
            if kind != "choice" and len(rows) != (2 if kind == "noul" else 6):
                raise ValueError(f"invalid {kind} slot count")
            if kind == "choice" and not 16 <= len(rows) <= 256:
                raise ValueError("choice must provide 16..256 labels")
            if (
                not isinstance(names, list)
                or len(names) != len(rows)
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)
            ):
                raise ValueError(f"kinds.{kind}.label_names must match row_indices")
            biases = item.get("bias")
            lengths = (16, len(rows)) if kind == "choice" else (len(rows),)
            if not isinstance(biases, list) or len(biases) not in lengths:
                raise ValueError(f"kinds.{kind}.bias must match its slots")
            bias = [_finite(v, f"kinds.{kind}.bias") for v in biases]
            bias.extend([0.0] * (len(rows) - len(bias)))
            temperature = _finite(item.get("temperature"), f"{kind} temperature")
            if temperature <= 0:
                raise ValueError(f"{kind} temperature must be positive")
            if kind == "choice":
                if item.get("native_slots", 16) != 16:
                    raise ValueError("JEV native choice slot count must be 16")
                if item.get("max_options", len(rows)) != len(rows):
                    raise ValueError("choice max_options must match its labels")
            parsed[kind] = JevKindSpec(
                tuple(rows), tuple(names), tuple(bias), temperature
            )
        if parsed["noul"].label_names != ("false", "true"):
            raise ValueError("noul labels must be false/true in that order")
        if parsed["score"].label_names != tuple(str(i) for i in range(6)):
            raise ValueError("score labels must be 0..5 in that order")
        strategy = config.get("default_strategy", "single")
        if strategy not in STRATEGIES[1:]:
            raise ValueError("invalid JEV default_strategy")
        limits = {}
        for name, default in (("max_images", 8), ("max_questions", 64)):
            value = config.get(name, default)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            limits[name] = value
        model_name = model or config.get("model", "jev-latest")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model must be a nonempty string")
        return cls(
            count,
            parsed,
            model_name,
            config.get("model_type", "jev"),
            strategy,
            **limits,
        )


@dataclass(frozen=True)
class JevRequest:
    kind: str
    parts: tuple[dict, ...]
    has_images: bool
    question: str
    options: tuple[str, ...]
    strategy: str
    # Only the SystemOne compatibility adapter may construct a singleton.
    deterministic: bool = False


@dataclass(frozen=True)
class JevRow:
    content: list[dict]
    has_images: bool
    kind: str
    question: str
    options: tuple[str, ...]


@dataclass(frozen=True)
class JevRowResult:
    logits: Sequence[float]
    prompt_tokens: int
    vision_tokens: int = 0
    cache_reuse_tokens: int = 0


def _parts(state: Any, max_images: int) -> tuple[tuple[dict, ...], bool]:
    if isinstance(state, str):
        return ({"type": "text", "text": state},), False
    if isinstance(state, dict):
        return ({"type": "text", "text": json.dumps(state, ensure_ascii=False)},), False
    if not isinstance(state, list):
        raise ValueError("state must be a string, JSON object or list")
    out, images = [], 0
    for part in state:
        if isinstance(part, str):
            out.append({"type": "text", "text": part})
            continue
        if isinstance(part, dict) and "image" in part:
            item = {"type": "image_url", "image_url": {"url": part["image"]}}
        elif isinstance(part, dict) and part.get("type") in ("image_url", "text"):
            item = copy.deepcopy(part)
        elif isinstance(part, dict) and part.get("type") in (
            "video",
            "video_url",
            "audio",
            "input_audio",
            "audio_url",
        ):
            raise ValueError("JEV System1 supports text and images, not video/audio")
        else:
            out.append({"type": "text", "text": json.dumps(part, ensure_ascii=False)})
            continue
        if item["type"] == "text":
            if not isinstance(item.get("text"), str):
                raise ValueError("text parts require a string text field")
        else:
            image = item.get("image_url")
            if (
                not isinstance(image, dict)
                or not isinstance(image.get("url"), str)
                or not image["url"]
            ):
                raise ValueError("image parts require a nonempty image_url.url string")
            images += 1
        out.append(item)
    if images > max_images:
        raise ValueError(f"state exceeds max_images={max_images}")
    return tuple(out), bool(images)


def _system1_controls(request: dict) -> None:
    for field in ("return_reasoning", "debug"):
        if field in request and type(request[field]) is not bool:
            raise ValueError(f"{field} must be a boolean")
    if request.get("threshold") is not None:
        _finite(request["threshold"], "threshold")
    if request.get("chat_template_kwargs") is not None and not isinstance(
        request["chat_template_kwargs"], dict
    ):
        raise ValueError("chat_template_kwargs must be an object or null")
    thinking = request.get("thinking", "default")
    if thinking not in ("default", "off"):
        raise ValueError(
            "this JEV deployment supports System1 only: thinking must be default or off"
        )
    if type(request.get("system2_only", False)) is not bool:
        raise ValueError("system2_only must be a boolean")
    if request.get("system2_only", False):
        raise ValueError(
            "system2_only requires System2, which this deployment does not support"
        )
    # Explicit reasoning requests must never be silently discarded.
    if (
        any(request.get(k) is not None for k in ("think_budget", "reasoning_effort"))
        or request.get("chat_template_kwargs")
        or request.get("return_reasoning", False)
    ):
        raise ValueError(
            "reasoning controls require System2, which this deployment does not support"
        )


def parse_decide_request(request: dict, spec: JevHeadSpec) -> JevRequest:
    try:
        return _parse_decide_request(request, spec)
    except ValueError as error:
        raise JevRequestError(str(error)) from error


def _parse_decide_request(request: dict, spec: JevHeadSpec) -> JevRequest:
    if not isinstance(request, dict):
        raise ValueError("JEV decide request must be an object")
    systemone._check_json(request, "request", set())
    _system1_controls(request)
    kind, question = request.get("kind"), request.get("question")
    if kind not in KINDS:
        raise ValueError("kind must be noul, score or choice")
    if not isinstance(question, str):
        raise ValueError("question must be a string")
    strategy = request.get("strategy", "auto")
    if strategy not in STRATEGIES:
        raise ValueError("strategy must be auto, single, permute or tournament")
    strategy = spec.default_strategy if strategy == "auto" else strategy
    options = request.get("options")
    if options is not None and (
        not isinstance(options, list) or any(not isinstance(o, str) for o in options)
    ):
        raise ValueError("options must be an array of strings")
    if kind == "choice":
        if options is None or not 2 <= len(options) <= spec.max_options:
            raise ValueError(f"choice needs 2-{spec.max_options} options")
    else:
        options = ["false", "true"] if kind == "noul" else [str(i) for i in range(6)]
    parts, has_images = _parts(request.get("state", ""), spec.max_images)
    return JevRequest(kind, parts, has_images, question, tuple(options), strategy)


def parse_systemone_request(
    request: dict, spec: JevHeadSpec
) -> tuple[str, list[JevRequest], list[dict]]:
    try:
        return _parse_systemone_request(request, spec)
    except ValueError as error:
        raise JevRequestError(str(error)) from error


def _parse_systemone_request(
    request: dict, spec: JevHeadSpec
) -> tuple[str, list[JevRequest], list[dict]]:
    if not isinstance(request, dict):
        raise ValueError("SystemOne request must be an object")
    systemone._check_json(request, "request", set())
    _system1_controls(request)
    if "state" not in request:
        raise ValueError("state is required")
    state = request["state"]
    # Preserve legacy scalar states; list/dict states use JEV's native serializer.
    if not isinstance(state, (str, list, dict)):
        state = systemone._render(state)
    parts, has_images = _parts(state, spec.max_images)
    model = request.get("model", spec.model)
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a nonempty string")
    questions = request.get("questions")
    if not isinstance(questions, dict) or not 1 <= len(questions) <= spec.max_questions:
        raise ValueError(
            f"questions must be an object with 1..{spec.max_questions} entries"
        )
    records, metadata = [], []
    for qid, item in questions.items():
        record, meta = systemone._question(qid, item, spec.max_options)
        _system1_controls(item)
        strategy = item.get("strategy", request.get("strategy", "auto"))
        if strategy not in STRATEGIES:
            raise ValueError("strategy must be auto, single, permute or tournament")
        strategy = spec.default_strategy if strategy == "auto" else strategy
        kind, question, options = item["type"], record["instr"], record["options"]
        effective_kind = "choice" if kind == "score" else kind
        if kind == "noul":
            if item.get("criteria"):
                question += "\n[criteria]\n" + "\n".join(
                    systemone._option_text(key, item["criteria"].get(key))
                    for key in ("false", "true")
                )
            options = ["false", "true"]
        singleton = len(options) == 1
        meta = dict(meta, effective_kind=effective_kind)
        meta["adaptation"] = (
            "deterministic"
            if singleton
            else (
                "score_rubric_as_choice"
                if kind == "score"
                else (strategy if kind == "choice" and len(options) > 16 else "native")
            )
        )
        records.append(
            JevRequest(
                effective_kind,
                parts,
                has_images,
                question,
                tuple(options),
                strategy,
                singleton,
            )
        )
        metadata.append(meta)
    return model, records, metadata


def build_row(request: JevRequest, spec: JevHeadSpec, options: Sequence[str]) -> JevRow:
    opts = tuple(options)
    lines = (
        [
            f"{label}) {option}"
            for label, option in zip(spec.kinds["choice"].label_names, opts)
        ]
        if request.kind == "choice"
        else list(opts)
    )
    content = [{"type": "text", "text": f"[kind] {request.kind}\n[state] "}]
    content.extend(copy.deepcopy(request.parts))
    content.append(
        {
            "type": "text",
            "text": f"\n[question] {request.question}\n[options]\n"
            + "\n".join(lines)
            + "\n[decision]:",
        }
    )
    return JevRow(content, request.has_images, request.kind, request.question, opts)


def _softmax(values: Sequence[float]) -> list[float]:
    peak = max(values)
    exp = [math.exp(value - peak) for value in values]
    total = sum(exp)
    return [value / total for value in exp]


def probabilities(
    result: JevRowResult, kind: str, option_count: int, spec: JevHeadSpec
) -> list[float]:
    if len(result.logits) != spec.num_labels:
        raise ValueError("JEV engine logits do not match the compact head size")
    logits = [_finite(value, "JEV logit") for value in result.logits]
    item = spec.kinds[kind]
    if not 1 <= option_count <= len(item.row_indices):
        raise ValueError("invalid option count for JEV readout")
    values = [
        _finite((logits[row] + bias) / item.temperature, "calibrated JEV logit")
        for row, bias in zip(item.row_indices[:option_count], item.bias[:option_count])
    ]
    return _softmax(values)


def groups(count: int, size: int = 16) -> list[list[int]]:
    number = math.ceil(count / size)
    base, extra = divmod(count, number)
    result, index = [], 0
    for group in range(number):
        length = base + (1 if group < extra else 0)
        result.append(list(range(index, index + length)))
        index += length
    return result


async def _bounded_map(
    function: Callable[[T], Awaitable[R]], values: Sequence[T], limit: int
) -> list[R]:
    """Create at most limit tasks and drain every worker on error/cancellation."""
    iterator = iter(enumerate(values))
    results: list[Any] = [None] * len(values)

    async def worker() -> None:
        for index, value in iterator:
            results[index] = await function(value)

    tasks = [asyncio.create_task(worker()) for _ in range(min(limit, len(values)))]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return results


@dataclass
class _Usage:
    requests: int = 0
    prompt_tokens: int = 0
    vision_tokens: int = 0
    cache_reuse_tokens: int = 0

    def add(self, result: JevRowResult) -> None:
        for key in ("prompt_tokens", "vision_tokens", "cache_reuse_tokens"):
            value = getattr(result, key)
            if type(value) is not int or value < 0:
                raise ValueError(f"JEV row {key} must be a nonnegative integer")
            setattr(self, key, getattr(self, key) + value)
        self.requests += 1

    def as_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": self.prompt_tokens,
            "vision_tokens": self.vision_tokens,
            "cache_reuse_tokens": self.cache_reuse_tokens,
        }


class JevExecutor:
    """One executor may serve concurrent APIs; its row limit is shared by all.

    The callback must propagate cancellation to the engine RPC. Admission and
    HTTP disconnect handling remain in the caller's common request lifecycle.
    timeout_seconds, when supplied, covers all strategy rounds and queued rows.
    """

    def __init__(
        self,
        spec: JevHeadSpec,
        run_row: Callable[[JevRow], Awaitable[JevRowResult]],
        *,
        max_concurrency: int = 4,
        timeout_seconds: float | None = None,
    ):
        if type(max_concurrency) is not int or max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")
        if (
            timeout_seconds is not None
            and _finite(timeout_seconds, "timeout_seconds") <= 0
        ):
            raise ValueError("timeout_seconds must be positive")
        self.spec, self.run_row = spec, run_row
        self.max_concurrency, self.timeout_seconds = max_concurrency, timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrency)

    def info(self) -> dict:
        return {
            "protocol": PROTOCOL,
            "model": self.spec.model,
            "model_type": self.spec.model_type,
            "max_options": self.spec.max_options,
            "native_choice_options": 16,
            "temperatures": {
                kind: item.temperature for kind, item in self.spec.kinds.items()
            },
            "defaults": {
                "strategy": self.spec.default_strategy,
                "thinking": "off",
                "threshold": None,
                "mix": None,
                "think_budget": None,
            },
            "reasoning_controls": {
                "enable_thinking": False,
                "reasoning_effort": None,
                "think_budget": None,
            },
            "capabilities": {
                "systems": [1],
                "kinds": list(KINDS),
                "strategies": list(STRATEGIES),
                "images": True,
                "video": False,
                "audio": False,
                "max_images": self.spec.max_images,
                "max_questions": self.spec.max_questions,
                "max_concurrent_rows": self.max_concurrency,
            },
            "accounting": {
                "completion_tokens": 0,
                "num_model_requests": "actual prefill rows; no logprob-token chunking",
                "prompt_tokens": "sum of input tokens across actual prefill rows",
            },
        }

    async def _pass(
        self, request: JevRequest, options: Sequence[str], usage: _Usage
    ) -> list[float]:
        async with self._semaphore:
            result = await self.run_row(build_row(request, self.spec, options))
        values = probabilities(result, request.kind, len(options), self.spec)
        usage.add(result)
        return values

    async def _distribution(self, request: JevRequest, usage: _Usage) -> list[float]:
        if request.deterministic:
            if len(request.options) != 1:
                raise ValueError(
                    "deterministic compatibility requests require one option"
                )
            return [1.0]
        options, count = request.options, len(request.options)
        if request.kind != "choice" or count <= 16 or request.strategy == "single":
            return await self._pass(request, options, usage)

        async def run(order: Sequence[int]) -> list[float]:
            return await self._pass(request, [options[i] for i in order], usage)

        if request.strategy == "permute":
            rng = random.Random(zlib.crc32(request.question.encode()))
            orders = [list(range(count))] + [
                rng.sample(range(count), count) for _ in range(3)
            ]
            rows = await _bounded_map(run, orders, self.max_concurrency)
            result = [0.0] * count
            for order, row in zip(orders, rows):
                for index, value in zip(order, row):
                    result[index] += value / len(orders)
            return result

        partitions = groups(count)
        rows = await _bounded_map(run, partitions, self.max_concurrency)
        within = {
            index: value
            for group, row in zip(partitions, rows)
            for index, value in zip(group, row)
        }
        chosen = [
            max(group, key=lambda index: (within[index], -index))
            for group in partitions
        ]
        chosen_set = set(chosen)
        rest = sorted(
            (index for index in range(count) if index not in chosen_set),
            key=lambda index: (-within[index], index),
        )
        finalists = sorted(chosen + rest[: max(0, 16 - len(chosen))])
        final = dict(zip(finalists, await run(finalists)))
        group_of = {
            index: group_index
            for group_index, group in enumerate(partitions)
            for index in group
        }
        share, cap = [0.0] * len(partitions), [0.0] * len(partitions)
        for finalist in finalists:
            share[group_of[finalist]] += final[finalist]
            cap[group_of[finalist]] += within[finalist]
        among = sum(a * b for a, b in zip(share, cap))
        result = [
            (
                final[index] * among
                if index in final
                else share[group_of[index]] * within[index]
            )
            for index in range(count)
        ]
        total = sum(result)
        return [value / total for value in result]

    async def _timed(self, coroutine: Awaitable[R]) -> R:
        if self.timeout_seconds is None:
            return await coroutine
        return await asyncio.wait_for(coroutine, self.timeout_seconds)

    async def decide(self, request: JevRequest | dict) -> dict:
        parsed = (
            request
            if isinstance(request, JevRequest)
            else parse_decide_request(request, self.spec)
        )
        started, usage = time.monotonic(), _Usage()
        values = await self._timed(self._distribution(parsed, usage))
        winner = max(range(len(values)), key=values.__getitem__)
        return {
            "kind": parsed.kind,
            "effective_kind": parsed.kind,
            "options": list(parsed.options),
            "probabilities": values,
            "choice_index": winner,
            "choice": parsed.options[winner],
            "adaptation": (
                "native"
                if parsed.kind != "choice" or len(parsed.options) <= 16
                else parsed.strategy
            ),
            "protocol": PROTOCOL,
            "model": self.spec.model,
            "usage": usage.as_dict(),
            "elapsed_seconds": time.monotonic() - started,
            "num_model_requests": usage.requests,
        }

    async def systemone(self, request: dict) -> dict:
        model, records, metadata = parse_systemone_request(request, self.spec)
        started, usage = time.monotonic(), _Usage()

        async def run(record: JevRequest) -> list[float]:
            return await self._distribution(record, usage)

        rows = await self._timed(_bounded_map(run, records, self.max_concurrency))
        adaptations = {}
        for record, meta in zip(records, metadata):
            adaptations[meta["id"]] = {
                "effective_kind": meta["effective_kind"],
                "adaptation": meta["adaptation"],
                "strategy": record.strategy if len(record.options) > 16 else "single",
            }
        elapsed_seconds = time.monotonic() - started
        return {
            "model": model,
            "answers": systemone.to_answers(rows, metadata),
            "usage": {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": 0,
                **usage.as_dict(),
            },
            "elapsed_seconds": elapsed_seconds,
            "latency_ms": round(elapsed_seconds * 1000, 3),
            "num_model_requests": usage.requests,
            "jev": {"protocol": PROTOCOL, "questions": adaptations},
        }
