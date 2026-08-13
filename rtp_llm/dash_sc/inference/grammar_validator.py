# Copyright (c) Alibaba, Inc. and its affiliates.
"""Dash-SC inference admission-time validation of structured-output grammars.

In a PD deployment a grammar that fails to compile is only discovered at runtime
inside the prefill EngineCore (a separate process) and can never reach the client
over the PD protocol -- the request just hangs and 504s. We turn a bad grammar into
a clean HTTP 4xx at admission, before dispatch.

Every request runs cheap shape checks first, then trial-compiles in a spawned worker
pool. An uncatchable SIGSEGV/OOM therefore only kills the worker, never DashLLM.

The trial compile only needs a classified answer, so the compiled grammar is never
shipped. Repeat specs with deterministic answers are served from an LRU; transient
pool failures are raised and never cached.
"""

from __future__ import annotations

import functools
import json
import logging
import multiprocessing
import os
import queue
import resource
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from enum import Enum
from typing import Any, NamedTuple

from jsonschema import Draft7Validator

from rtp_llm.config.py_config_modules import GrammarAdmissionConfig
from rtp_llm.ops import GrammarConfig

logger = logging.getLogger(__name__)

_CRASH_CONFIRMATION_ATTEMPTS = 2
_MAX_COMPILE_ERROR_MESSAGE_LENGTH = 4096
_JSON_OBJECT_RESPONSE_SCHEMA = {"anyOf": [{"type": "object"}, {"type": "array"}]}


class _WorkerStatus(Enum):
    VALID = "valid"
    INVALID = "invalid"
    UNAVAILABLE = "unavailable"


class _GrammarCheckResult(NamedTuple):
    ok: bool
    compile_error: str = ""


# Per-thread request id for log correlation (dashserving is thread-per-request): validate_*
# stashes it here, the logging calls read it. Kept off the method args so it never becomes
# part of the lru_cache key.
_req_ctx = threading.local()


def _rid() -> str:
    return getattr(_req_ctx, "rid", "")


def _with_request_id(message: str) -> str:
    request_id = _rid()
    return f"{message}, request_id={request_id}" if request_id else message


def _is_resource_exhaustion(error: BaseException) -> bool:
    return isinstance(error, MemoryError) or "bad_alloc" in str(error).lower()


def _compile_exception_reply(error: Exception) -> tuple[_WorkerStatus, bool, str]:
    """Classify every catchable exception raised by an xgrammar compile call.

    xgrammar may expose its registered InvalidJSON/InvalidStructuralTag errors,
    built-in TypeError/ValueError/RuntimeError through TVM FFI, or JSON decoding
    errors from the structural-tag adapter.  They are all deterministic input
    rejections.  Resource exhaustion means this grammar exceeded the sandbox's
    per-request memory budget: reject and cache it as a 400, but retire the
    contaminated worker before serving another request.
    """
    if _is_resource_exhaustion(error):
        message = (str(error) or type(error).__name__)[
            :_MAX_COMPILE_ERROR_MESSAGE_LENGTH
        ]
        return _WorkerStatus.INVALID, True, message
    message = (str(error) or type(error).__name__)[:_MAX_COMPILE_ERROR_MESSAGE_LENGTH]
    return _WorkerStatus.INVALID, False, message


class GrammarCheckUnavailable(RuntimeError):
    """The grammar check could not get an answer for transient infrastructure reasons.
    Callers may reject the request, but this result must not be cached as invalid."""


class GrammarCompilationError(ValueError):
    """Any catchable xgrammar compiler error returned by a sandbox worker."""


class GrammarValidator:
    def __init__(
        self,
        tokenizer_info_json: str,
        grammar_config: GrammarConfig,
        admission_config: GrammarAdmissionConfig,
    ) -> None:
        self._result_cache_max_entries = int(
            admission_config.result_cache_max_entries
        )
        if self._result_cache_max_entries < 0:
            raise ValueError(
                "grammar admission result_cache_max_entries must be non-negative"
            )
        self._result_cache_lock = threading.Lock()
        self._result_cache: OrderedDict[
            tuple[str, str], _GrammarCheckResult
        ] = OrderedDict()
        self._initialize_compiler(
            tokenizer_info_json, grammar_config, admission_config
        )
        self._worker_tokenizer_info_json = tokenizer_info_json
        self._worker_grammar_config = grammar_config
        self._worker_admission_config = admission_config
        # Per sandbox-worker memory headroom beyond the spawned process's initialized
        # Python/xgrammar/compiler baseline. Applied only in the worker: a pathological
        # grammar kills/rejects that worker while the DashLLM parent replaces it.
        self._worker_memory_limit_bytes = (
            max(0, int(admission_config.sandbox_process_memory_limit_mb)) * 1024 * 1024
        )
        self._queue_timeout_s = float(admission_config.queue_timeout_s)
        self._compile_timeout_s = float(admission_config.compile_timeout_s)
        if self._queue_timeout_s <= 0:
            raise ValueError("grammar admission queue_timeout_s must be greater than 0")
        if self._compile_timeout_s <= 0:
            raise ValueError(
                "grammar admission compile_timeout_s must be greater than 0"
            )

        configured_pool_size = int(admission_config.sandbox_pool_size)
        if configured_pool_size < 0:
            raise ValueError("grammar admission sandbox_pool_size must be non-negative")
        if configured_pool_size > 0:
            self._pool_target = configured_pool_size
        else:
            self._pool_target = max(
                1, (os.cpu_count() or 8) // (2 * self._compile_threads)
            )

        # Persistent pool of N spawned workers, each compiling one spec at a time, so
        # concurrent validate_* calls compile in parallel (up to N). A worker that crashes
        # mid-compile is retired and replaced in the background. Idle workers block on a
        # pipe recv (~0 CPU), so the standing cost is memory, not CPU.
        #
        self._mp: Any = None
        self._idle: Any = None  # queue.Queue of idle (proc, conn)
        self._pool_lock = threading.Lock()  # guards _live / _spawning
        # functools.lru_cache protects its own state but allows concurrent cache misses
        # for the same key to execute more than once. Keep one in-flight Future per exact
        # grammar so duplicate requests share the leader's compile result.
        self._inflight_lock = threading.Lock()
        self._inflight: dict[
            tuple[str, str], Future[_GrammarCheckResult]
        ] = {}
        self._live = 0
        self._spawning = 0
        self._coordinator_running = False
        self._mp = multiprocessing.get_context("spawn")
        self._idle = queue.Queue()
        self._ensure_pool()  # warm N workers in the background; never blocks init

        worker_limit_mb = max(0, self._worker_memory_limit_bytes // 1024 // 1024)
        msg = f"GrammarValidator mode=sandbox compile_backend=on queue_timeout_s={self._queue_timeout_s:g} compile_timeout_s={self._compile_timeout_s:g} compiler_threads={self._compile_threads} compiler_cache_bytes={self._cache_limit_bytes} result_cache_max_entries={self._result_cache_max_entries} pool={self._pool_target} worker_memory_limit_mb={worker_limit_mb}"
        logger.debug(msg)

    # -- public entry points (shape checks first, then maybe compile) ------- #

    def validate_json(self, schema: str | dict, request_id: str = "") -> bool:
        _req_ctx.rid = request_id
        return self._check_grammar("json", schema)

    def validate_structural_tag(
        self, payload: str | dict, request_id: str = ""
    ) -> bool:
        _req_ctx.rid = request_id
        return self._check_grammar("structural_tag", payload)

    def validate_response_format(
        self, response_format: str | dict, request_id: str = ""
    ) -> bool:
        """Validate and trial-compile one OpenAI-style response_format envelope."""
        _req_ctx.rid = request_id
        payload = self._as_nonempty_dict(response_format)
        if payload is None:
            return False

        fmt_type = payload.get("type")
        if fmt_type == "text":
            return True
        if fmt_type == "json_object":
            return self._check_grammar("json", _JSON_OBJECT_RESPONSE_SCHEMA)
        if fmt_type == "json_schema":
            schema = payload.get("json_schema")
            if isinstance(schema, dict) and "schema" in schema:
                schema = schema["schema"]
            return self._check_grammar("json", schema)
        if fmt_type == "regex":
            pattern = payload.get("pattern")
            if not isinstance(pattern, str) or not pattern.strip():
                return False
            return self._check_grammar("regex", pattern)
        if fmt_type == "ebnf":
            grammar = payload.get("grammar")
            if not isinstance(grammar, str) or not grammar.strip():
                return False
            return self._check_grammar("ebnf", grammar)
        if fmt_type == "structural_tag":
            return self._check_grammar("structural_tag", payload.get("structural_tag"))
        return False

    # -- grammar admission check ------------------------------------------- #

    def _check_grammar(self, kind: str, spec: Any) -> bool:
        """Memoized full admission result for ``spec``. Once grammar validation is enabled, an
        unavailable check rejects the request instead of letting a risky grammar reach the engine.
        """
        logger.debug(
            _with_request_id(
                f"GrammarValidator: start sandbox check grammar kind={kind}"
            )
        )
        try:
            # no sort_keys: validate/cache the exact string xgrammar will compile.
            spec_str = spec if isinstance(spec, str) else json.dumps(spec)
        except Exception as e:
            detail = (str(e) or type(e).__name__)[
                :_MAX_COMPILE_ERROR_MESSAGE_LENGTH
            ]
            logger.warning(
                _with_request_id(
                    f"GrammarValidator: cannot serialize grammar spec ({detail}); rejecting it"
                )
            )
            raise GrammarCompilationError(
                f"cannot serialize grammar spec: {detail}"
            ) from e
        try:
            result = self._check_grammar_singleflight(kind, spec_str)
            if result.compile_error:
                raise GrammarCompilationError(result.compile_error)
            return result.ok
        except GrammarCompilationError:
            raise
        except GrammarCheckUnavailable as e:
            logger.warning(
                _with_request_id(
                    f"GrammarValidator: grammar check unavailable ({e}); request may be retried"
                )
            )
            raise
        except Exception as e:
            detail = (str(e) or type(e).__name__)[
                :_MAX_COMPILE_ERROR_MESSAGE_LENGTH
            ]
            logger.warning(
                _with_request_id(
                    f"GrammarValidator: unexpected grammar check failure ({detail}); request may be retried"
                )
            )
            raise GrammarCheckUnavailable(
                f"unexpected grammar validation failure: {detail}"
            ) from e

    def _check_grammar_singleflight(
        self, kind: str, spec_str: str
    ) -> _GrammarCheckResult:
        """Collapse concurrent checks for one exact grammar into a single compile."""
        key = (kind, spec_str)
        cached_result = self._get_cached_result(key)
        if cached_result is not None:
            return cached_result

        with self._inflight_lock:
            future = self._inflight.get(key)
            if future is None:
                future = Future()
                self._inflight[key] = future
                is_leader = True
            else:
                is_leader = False

        if not is_leader:
            logger.debug(
                _with_request_id(
                    f"GrammarValidator: join in-flight grammar check kind={kind}"
                )
            )
            return future.result()

        try:
            # Close the race where a previous leader populated the result cache
            # between this request's first lookup and its in-flight registration.
            result = self._get_cached_result(key)
            if result is None:
                result = self._check_grammar_uncached(kind, spec_str)
                self._cache_result(key, result)
        except BaseException as e:
            # Followers receive the same transient/deterministic failure. Exceptions
            # are not stored, so a later request can become a fresh leader.
            future.set_exception(e)
            raise
        else:
            future.set_result(result)
            return result
        finally:
            with self._inflight_lock:
                if self._inflight.get(key) is future:
                    del self._inflight[key]

    def _get_cached_result(
        self, key: tuple[str, str]
    ) -> _GrammarCheckResult | None:
        if self._result_cache_max_entries == 0:
            return None
        with self._result_cache_lock:
            result = self._result_cache.get(key)
            if result is not None:
                self._result_cache.move_to_end(key)
            return result

    def _cache_result(
        self, key: tuple[str, str], result: _GrammarCheckResult
    ) -> None:
        if self._result_cache_max_entries == 0:
            return
        with self._result_cache_lock:
            self._result_cache[key] = result
            self._result_cache.move_to_end(key)
            while len(self._result_cache) > self._result_cache_max_entries:
                self._result_cache.popitem(last=False)

    def _check_grammar_uncached(self, kind: str, spec_str: str) -> _GrammarCheckResult:
        """Full admission result for one exact grammar string.
        The result contains whether shape/support checks and compilation passed, plus
        xgrammar's bounded diagnostic for catchable compiler rejections.
        Raises GrammarCheckUnavailable for transient worker-pool failures.
        """
        t0 = time.perf_counter()
        result: _GrammarCheckResult | None = None
        try:
            if kind == "json":
                if not self.validate_json_schema(spec_str):
                    result = _GrammarCheckResult(False)
                    return result
            elif kind == "structural_tag":
                if not self.check_structural_tag(spec_str):
                    result = _GrammarCheckResult(False)
                    return result
            elif kind in ("regex", "ebnf"):
                if not spec_str.strip():
                    result = _GrammarCheckResult(False)
                    return result
            else:
                raise ValueError(f"unsupported grammar kind {kind!r}")

            try:
                result = _GrammarCheckResult(self._compile_in_worker(kind, spec_str))
            except GrammarCompilationError as e:
                # Cache deterministic compiler rejections together with xgrammar's
                # diagnostic so cache hits return the same client-visible detail.
                result = _GrammarCheckResult(False, str(e))
            return result
        finally:
            if result is None:
                result_name = "unavailable"
            elif result.ok:
                result_name = "ok"
            else:
                result_name = "fail"
            logger.debug(
                _with_request_id(
                    f"GrammarValidator grammar check: kind={kind} mode=sandbox result={result_name} took={(time.perf_counter() - t0) * 1000.0:.1f}ms"
                )
            )

    def _initialize_compiler(
        self,
        tokenizer_info_json: str,
        grammar_config: GrammarConfig,
        admission_config: GrammarAdmissionConfig,
    ) -> None:
        """Build the compiler state shared by the parent compatibility check and workers."""
        self._disable_any_whitespace = bool(
            grammar_config.constrained_json_disable_any_whitespace
        )
        self._compile_threads = max(1, int(grammar_config.num_workers))
        config_cache_bytes = int(admission_config.compiler_cache_bytes)
        self._cache_limit_bytes = config_cache_bytes if config_cache_bytes > 0 else -1
        self._tokenizer_info_json = str(tokenizer_info_json)
        # Fail service startup if the configured xgrammar/tokenizer pair is unusable.
        # Spawned workers independently build the same backend before announcing ready.
        self._backend = self._build_backend()

    def _compile(self, kind: str, spec_str: str) -> None:
        """Trial-compile directly with xgrammar. Raises on a catchable error; may SIGSEGV
        the process on an uncatchable one (0.1.29), which is why sandbox mode exists.
        Catchable error text is returned to the parent for the client-facing 400."""
        xgr = self._xgrammar()
        any_whitespace = not self._disable_any_whitespace
        if kind == "json":
            self._backend.compile_json_schema(
                spec_str,
                any_whitespace=any_whitespace,
                indent=None,
                separators=None,
                strict_mode=True,
            )
        elif kind == "structural_tag":
            s_tag = json.loads(spec_str)
            if "structures" in s_tag:  # deprecated legacy form
                tags = [
                    xgr.StructuralTagItem(
                        begin=s["begin"], schema=json.dumps(s["schema"]), end=s["end"]
                    )
                    for s in s_tag["structures"]
                ]
                self._backend.compile_structural_tag(tags, s_tag["triggers"])
            else:
                self._backend.compile_structural_tag(spec_str)
        elif kind == "regex":
            self._backend.compile_regex(spec_str)
        elif kind == "ebnf":
            self._backend.compile_grammar(spec_str)
        else:
            raise ValueError(f"unsupported compile kind {kind!r}")

    def _build_backend(self) -> Any:
        """Build a compiler from the same serialized TokenizerInfo as EngineCore.

        Raises on failure so compile/sandbox cannot silently become shape-only.
        """
        xgr = self._xgrammar()
        if xgr is None:
            raise RuntimeError(
                "GrammarValidator requires xgrammar for compile/sandbox validation"
            )
        try:
            tokenizer_info = xgr.TokenizerInfo.deserialize_json(
                self._tokenizer_info_json
            )
            return xgr.GrammarCompiler(
                tokenizer_info,
                max_threads=self._compile_threads,
                cache_enabled=True,
                cache_limit_bytes=self._cache_limit_bytes,
            )
        except Exception as e:
            raise RuntimeError(
                f"GrammarValidator failed to initialize GrammarCompiler: {e}"
            ) from e

    # -- persistent sandbox worker pool ------------------------------------- #

    def _compile_in_worker(self, kind: str, spec_str: str) -> bool:
        """Compile with independent checkout and execution budgets.

        A checked-out worker is always either returned healthy or retired. Queue/compile
        timeouts and transport failures are transient unavailable outcomes. Catchable
        resource-exhaustion exceptions are cacheable input rejections, but their worker is
        retired after replying.
        """
        checkout_deadline = time.monotonic() + self._queue_timeout_s
        queue_wait_s = 0.0
        last_error: Exception | None = None
        crash_attempts = 0
        owned_worker: tuple[Any, Any] | None = None
        self._ensure_pool()

        def retire_owned() -> None:
            nonlocal owned_worker
            worker = owned_worker
            owned_worker = None
            if worker is not None:
                self._retire(*worker)

        def release_owned() -> None:
            nonlocal owned_worker
            worker = owned_worker
            owned_worker = None
            if worker is None:
                return
            proc, conn = worker
            try:
                healthy = proc is not None and proc.is_alive()
            except Exception:
                healthy = False
            if healthy:
                self._idle.put(worker)
            else:
                self._retire(proc, conn)

        try:
            while True:
                remaining = checkout_deadline - time.monotonic()
                if remaining <= 0:
                    detail = (
                        f" after worker error: {last_error}"
                        if last_error is not None
                        else ""
                    )
                    raise GrammarCheckUnavailable(
                        f"no idle sandbox worker within {self._queue_timeout_s:g}s{detail}"
                    )
                try:
                    checkout_started = time.monotonic()
                    owned_worker = self._idle.get(timeout=remaining)
                    queue_wait_s += time.monotonic() - checkout_started
                except queue.Empty as e:
                    queue_wait_s += time.monotonic() - checkout_started
                    raise GrammarCheckUnavailable(
                        f"no idle sandbox worker within {self._queue_timeout_s:g}s"
                    ) from e

                proc, conn = owned_worker
                try:
                    alive = proc is not None and proc.is_alive()
                except Exception:
                    alive = False
                if not alive:
                    retire_owned()
                    continue

                # Queue time is no longer relevant after checkout: this worker gets a full
                # compile budget even if checkout completed on the queue-timeout boundary.
                compile_started = time.monotonic()
                try:
                    conn.send((kind, spec_str))
                except (EOFError, OSError, BrokenPipeError, ValueError) as e:
                    last_error = e
                    retire_owned()  # the spec never landed, so this is infrastructure-only
                    continue

                if not conn.poll(self._compile_timeout_s):
                    logger.warning(
                        _with_request_id(
                            f"GrammarValidator: sandbox compile timed out after {self._compile_timeout_s:g}s; retiring the worker"
                        )
                    )
                    retire_owned()
                    raise GrammarCheckUnavailable("sandbox grammar compile timed out")

                try:
                    reply = conn.recv()
                except (EOFError, OSError, BrokenPipeError, ValueError) as e:
                    crash_attempts += 1
                    retire_owned()
                    if crash_attempts >= _CRASH_CONFIRMATION_ATTEMPTS:
                        logger.warning(
                            _with_request_id(
                                "GrammarValidator: independent sandbox workers reproducibly "
                                f"died compiling this spec ({e}); rejecting it"
                            )
                        )
                        return False
                    logger.warning(
                        _with_request_id(
                            "GrammarValidator: sandbox worker died compiling this spec; "
                            "retrying once in an independent worker"
                        )
                    )
                    checkout_deadline = time.monotonic() + self._queue_timeout_s
                    continue

                if (
                    not isinstance(reply, tuple)
                    or len(reply) != 3
                    or not isinstance(reply[0], _WorkerStatus)
                    or not isinstance(reply[1], bool)
                    or not isinstance(reply[2], str)
                ):
                    retire_owned()
                    raise GrammarCheckUnavailable(
                        f"invalid sandbox worker reply: {reply!r}"
                    )

                status, retire_after_reply, compile_error = reply
                compile_ms = (time.monotonic() - compile_started) * 1000.0
                error_log = f" error={compile_error}" if compile_error else ""
                logger.debug(
                    _with_request_id(
                        "GrammarValidator sandbox compile finished "
                        f"kind={kind} result={status.value} "
                        f"queue_ms={queue_wait_s * 1000.0:.3f} "
                        f"compile_ms={compile_ms:.3f}{error_log}"
                    )
                )
                if status is _WorkerStatus.UNAVAILABLE:
                    retire_owned()
                    raise GrammarCheckUnavailable(
                        "sandbox grammar compiler exhausted worker resources"
                    )
                if retire_after_reply:
                    retire_owned()
                else:
                    release_owned()
                if status is _WorkerStatus.INVALID and compile_error:
                    raise GrammarCompilationError(compile_error)
                return status is _WorkerStatus.VALID
        finally:
            # Covers future early returns/exceptions: checkout ownership must never vanish.
            if owned_worker is not None:
                retire_owned()

    def _retire(self, proc: Any, conn: Any) -> None:
        """Kill a dead/bad worker, drop it from the live count, and schedule a background
        replacement so the pool self-heals to its target."""
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        if proc is not None:
            try:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=1.0)
                    if proc.is_alive():
                        proc.kill()
                proc.join(timeout=1.0)  # a segfaulted worker joins instantly
            except Exception:
                pass
        with self._pool_lock:
            self._live = max(0, self._live - 1)
        self._ensure_pool()

    def _ensure_pool(self) -> None:
        """Top the pool up through one background coordinator.

        The coordinator starts workers sequentially. Combined with the ``spawn`` context,
        this avoids concurrent process creation from arbitrary request threads.
        """
        with self._pool_lock:
            if self._coordinator_running:
                return
            deficit = self._pool_target - self._live - self._spawning
            if deficit <= 0:
                return
            self._spawning += deficit
            self._coordinator_running = True
        threading.Thread(
            target=self._spawn_many,
            args=(deficit,),
            name="grammar-sandbox-coordinator",
            daemon=True,
        ).start()

    def _spawn_many(self, count: int) -> None:
        try:
            for _ in range(count):
                self._spawn_one()
        finally:
            with self._pool_lock:
                self._coordinator_running = False

    def _spawn_one(self) -> None:
        """Spawn one worker, wait for its readiness handshake, add it to the idle pool. Off
        the request path; on failure the pool stays smaller until the next _ensure_pool.
        """
        proc = None
        parent_conn = None
        try:
            parent_conn, child_conn = self._mp.Pipe()
            proc = self._mp.Process(
                target=_spawned_sandbox_worker,
                args=(
                    child_conn,
                    self._worker_tokenizer_info_json,
                    self._worker_grammar_config,
                    self._worker_admission_config,
                    self._worker_memory_limit_bytes,
                ),
                name="grammar-sandbox-worker",
                daemon=True,
            )
            proc.start()
            child_conn.close()  # parent keeps only its end -> sees EOF when worker dies
            if not parent_conn.poll(self._compile_timeout_s):
                raise TimeoutError("worker readiness handshake timed out")
            handshake = parent_conn.recv()
            if not isinstance(handshake, tuple) or len(handshake) not in (2, 3):
                raise RuntimeError(f"unexpected handshake {handshake!r}")
            tag, ok = handshake[:2]
            if tag != "__ready__":
                raise RuntimeError(f"unexpected handshake tag {tag!r}")
            if not ok:
                detail = f": {handshake[2]}" if len(handshake) == 3 else ""
                raise RuntimeError(
                    f"sandbox worker failed to initialize its compiler backend{detail}"
                )
            with self._pool_lock:
                self._live += 1
                self._spawning -= 1
            self._idle.put((proc, parent_conn))
            proc = parent_conn = None  # owned by the pool now
            logger.debug("GrammarValidator sandbox worker ready (backend=on)")
        except Exception as e:
            with self._pool_lock:
                self._spawning -= 1
            logger.warning(
                f"GrammarValidator: sandbox worker spawn failed ({e}); pool smaller until retry"
            )
            if parent_conn is not None:
                try:
                    parent_conn.close()
                except Exception:
                    pass
            if proc is not None:
                try:
                    proc.terminate()
                except Exception:
                    pass

    def _worker_loop(self, conn: Any) -> None:
        """Spawned worker: announce its local compiler, then serve compile requests.

        A bad-case compile may SIGSEGV this process; the parent confirms such crashes
        independently before caching rejection.
        """
        if self._worker_memory_limit_bytes > 0:
            # RLIMIT_AS is absolute, so add the configured headroom to the spawned
            # worker's initialized Python/xgrammar/compiler address-space baseline.
            try:
                cap = self._current_vsz_bytes() + self._worker_memory_limit_bytes
                resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
            except Exception as e:
                logger.warning(
                    f"GrammarValidator: failed to set sandbox worker memory limit ({e}); continuing without this cap"
                )
        backend_ready = self._backend is not None
        try:
            conn.send(("__ready__", backend_ready))
        except (EOFError, OSError):
            return
        if not backend_ready:
            return
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                return
            try:
                kind, spec_str = msg
            except Exception:
                continue
            status = _WorkerStatus.INVALID
            exit_after_reply = False
            compile_error = ""
            try:
                self._compile(kind, spec_str)
                status = _WorkerStatus.VALID
            except Exception as e:
                status, exit_after_reply, compile_error = _compile_exception_reply(e)
            try:
                conn.send((status, exit_after_reply, compile_error))
            except (EOFError, OSError):
                return
            if exit_after_reply:
                return

    # -- helpers ------------------------------------------------------------ #

    @staticmethod
    def _current_vsz_bytes() -> int:
        """Return this process's current virtual address space (VmSize)."""
        with open("/proc/self/statm") as f:
            return int(f.read().split()[0]) * resource.getpagesize()

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _xgrammar() -> Any:
        """The xgrammar module if importable, else None (memoized)."""
        try:
            import xgrammar as xgr

            return xgr
        except ImportError:
            return None

    @staticmethod
    def _as_nonempty_dict(payload: str | dict) -> dict | None:
        """Parse a JSON string (or accept a dict) to a non-empty dict, else None."""
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                return None
        if not isinstance(payload, dict) or len(payload) == 0:
            return None
        return payload

    # -- shape checks (pure; never build/compile a grammar) ----------------- #

    @staticmethod
    def _has_oversized_min_length(node: Any) -> bool:
        """Return whether any nested JSON value requests ``minLength > 128``.

        Large minimum string lengths currently make xgrammar bitmask generation
        pathologically slow. Keep this admission guard until the upstream issue is
        fixed: https://github.com/mlc-ai/xgrammar/issues/805.
        """
        if isinstance(node, dict):
            min_length = node.get("minLength")
            if (
                isinstance(min_length, (int, float))
                and not isinstance(min_length, bool)
                and min_length > 128
            ):
                return True
            return any(
                GrammarValidator._has_oversized_min_length(value)
                for value in node.values()
            )
        if isinstance(node, list):
            return any(
                GrammarValidator._has_oversized_min_length(value) for value in node
            )
        return False

    @staticmethod
    def validate_json_schema(json_schema: str | dict) -> bool:
        """Return whether the schema parses and is valid Draft 7 JSON Schema.

        Feature support is checked by the native xgrammar trial compile at the
        ``compile`` and ``sandbox`` levels.
        """
        schema = GrammarValidator._as_nonempty_dict(json_schema)
        if schema is None:
            return False
        try:
            Draft7Validator.check_schema(schema)
        except Exception:
            return False
        if GrammarValidator._has_oversized_min_length(schema):
            return False
        return True

    @staticmethod
    def check_structural_tag(structural_tag: str | dict) -> bool:
        """Validate a structural_tag payload without compiling. Legacy structures/triggers
        form gets a hand-rolled check; the new format form is validated against xgrammar's
        own pydantic model (falls back to _check_structural_format without xgrammar)."""
        structural_tag = GrammarValidator._as_nonempty_dict(structural_tag)
        if structural_tag is None:
            return False
        if GrammarValidator._has_oversized_min_length(structural_tag):
            return False

        if structural_tag.get("structures") is not None:  # legacy form
            structures = structural_tag.get("structures")
            triggers = structural_tag.get("triggers")
            if not isinstance(structures, list) or len(structures) == 0:
                return False
            if not isinstance(triggers, list) or len(triggers) == 0:
                return False
            for structure in structures:
                if not isinstance(structure, dict):
                    return False
                if not isinstance(structure.get("begin"), str) or not isinstance(
                    structure.get("end"), str
                ):
                    return False
                schema = structure.get("schema")
                if schema is not None and not GrammarValidator.validate_json_schema(
                    schema
                ):
                    return False
            return True

        # New form: prefer xgrammar's pydantic validator; fall back without xgrammar.
        xgr = GrammarValidator._xgrammar()
        if xgr is not None:
            try:
                xgr.StructuralTag.model_validate(structural_tag)
                return True
            except Exception:
                return False

        fmt = structural_tag.get("format")
        if not isinstance(fmt, dict):
            return False
        return GrammarValidator._check_structural_format(fmt)

    @staticmethod
    def _check_structural_format(fmt: Any) -> bool:
        """Recursively validate a structural-tag format dict (fallback when xgrammar is
        unavailable). Mirrors xgrammar's format types; inner schemas go through validate_json_schema.
        """
        if not isinstance(fmt, dict):
            return False
        fmt_type = fmt.get("type")
        if not isinstance(fmt_type, str):
            return False

        if fmt_type in {"json_schema", "qwen_xml_parameter"}:
            schema = fmt.get("json_schema")
            return schema is None or GrammarValidator.validate_json_schema(schema)
        if fmt_type in {
            "const_string",
            "any_text",
            "token",
            "exclude_token",
            "any_tokens",
            "grammar",
            "regex",
        }:
            return True
        if fmt_type in {"tag", "optional", "plus", "star", "repeat"}:
            content = fmt.get("content")
            return content is None or GrammarValidator._check_structural_format(content)
        if fmt_type in {"sequence", "or"}:
            elements = fmt.get("elements")
            if not isinstance(elements, list) or len(elements) == 0:
                return False
            return all(GrammarValidator._check_structural_format(e) for e in elements)
        if fmt_type in {
            "triggered_tags",
            "token_triggered_tags",
            "tags_with_separator",
        }:
            tags = fmt.get("tags")
            if not isinstance(tags, list) or len(tags) == 0:
                return False
            return all(GrammarValidator._check_structural_format(t) for t in tags)
        if fmt_type in {"dispatch", "token_dispatch"}:
            return True  # version-specific layout; accept structurally in this fallback
        return False  # unknown -> reject


def _spawned_sandbox_worker(
    conn: Any,
    tokenizer_info_json: str,
    grammar_config: GrammarConfig,
    admission_config: GrammarAdmissionConfig,
    worker_memory_limit_bytes: int,
) -> None:
    """Spawn entry point: construct all Python/xgrammar state inside the child."""
    try:
        # Do not call GrammarValidator.__init__ here: the public validator always owns a
        # sandbox pool, while a worker only needs its local compiler and request loop.
        validator = GrammarValidator.__new__(GrammarValidator)
        validator._initialize_compiler(
            tokenizer_info_json, grammar_config, admission_config
        )
        validator._worker_memory_limit_bytes = worker_memory_limit_bytes
    except BaseException as e:
        try:
            detail = f"{type(e).__name__}: {e}"[:512]
            conn.send(("__ready__", False, detail))
        except (EOFError, OSError):
            pass
        finally:
            conn.close()
        return

    try:
        validator._worker_loop(conn)
    finally:
        conn.close()
