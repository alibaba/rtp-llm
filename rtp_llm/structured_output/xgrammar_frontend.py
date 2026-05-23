import hashlib
import importlib.metadata
import json
import logging
import os
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from glob import glob
from typing import Any, Dict, Optional, Tuple


XGRAMMAR_VERSION_HASH = "xgrammar-0.2.1"
GPU_LOWERING_VERSION = "rtp-xgrammar-gpu-lowering-v1"
DEFAULT_XGRAMMAR_COMPILE_CACHE_SIZE = 1024
EXPECTED_XGRAMMAR_VERSION = "0.2.1"
BACKEND_PAYLOAD_VERSION = "rtp-xgrammar-backend-payload-v1"


def _prepend_env_path(env_name: str, path: str) -> None:
    if not path or not os.path.exists(path):
        return
    entries = [entry for entry in os.environ.get(env_name, "").split(os.pathsep) if entry]
    if path not in entries:
        os.environ[env_name] = os.pathsep.join([path] + entries)


def _ensure_xgrammar_runfiles_paths() -> None:
    """Make Bazel pip runfiles visible to xgrammar and its native loader."""
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workspace_root = os.path.dirname(package_root)
    runfiles_roots = [workspace_root]

    for env_name in ("RUNFILES_DIR", "TEST_SRCDIR"):
        runfiles_dir = os.environ.get(env_name)
        if runfiles_dir:
            runfiles_roots.extend(
                [
                    os.path.join(runfiles_dir, "rtp_llm"),
                    runfiles_dir,
                ]
            )

    seen = set()
    for root in runfiles_roots:
        if not root or root in seen:
            continue
        seen.add(root)
        site_packages = []
        for pattern in (
            os.path.join(root, "external", "pip*_xgrammar", "site-packages"),
            os.path.join(root, "external", "pip*_apache_tvm_ffi", "site-packages"),
            os.path.join(os.path.dirname(root), "pip*_xgrammar", "site-packages"),
            os.path.join(os.path.dirname(root), "pip*_apache_tvm_ffi", "site-packages"),
        ):
            site_packages.extend(glob(pattern))

        for site_package in site_packages:
            if site_package not in sys.path:
                sys.path.insert(0, site_package)

            xgrammar_lib_dir = os.path.join(site_package, "xgrammar")
            tvm_ffi_lib_dir = os.path.join(site_package, "tvm_ffi", "lib")
            for lib_dir in (xgrammar_lib_dir, tvm_ffi_lib_dir):
                _prepend_env_path("LD_LIBRARY_PATH", lib_dir)
                _prepend_env_path("PATH", lib_dir)


def verify_xgrammar_runtime(required: bool = False) -> None:
    _ensure_xgrammar_runfiles_paths()
    try:
        import xgrammar  # type: ignore
    except Exception as e:
        if required:
            raise RuntimeError(
                f"xgrammar is required for response_format structured output: {e}"
            ) from e
        return
    version = getattr(xgrammar, "__version__", "")
    if not version:
        try:
            version = importlib.metadata.version("xgrammar")
        except importlib.metadata.PackageNotFoundError:
            version = ""
    if version != EXPECTED_XGRAMMAR_VERSION:
        raise RuntimeError(
            f"xgrammar version mismatch: expected {EXPECTED_XGRAMMAR_VERSION}, got {version}"
        )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unwrap_tokenizer(tokenizer: Any) -> Any:
    get_real_tokenizer = getattr(tokenizer, "get_real_tokenizer", None)
    if callable(get_real_tokenizer):
        try:
            real_tokenizer = get_real_tokenizer()
            if real_tokenizer is not None:
                return real_tokenizer
        except Exception as e:
            logging.warning("xgrammar get_real_tokenizer failed: %s", e)
    return tokenizer


def canonicalize_schema(schema: Dict[str, Any]) -> str:
    """Return a stable JSON representation for cache keys."""
    return json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _tokenizer_vocab_hash(tokenizer: Any) -> str:
    tokenizer = _unwrap_tokenizer(tokenizer)
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        return ""
    try:
        vocab = get_vocab()
    except Exception as e:
        logging.warning("xgrammar tokenizer get_vocab failed: %s", e)
        return ""
    try:
        return _sha256(canonicalize_schema(vocab))
    except Exception:
        return _sha256(str(sorted(vocab.items())))


def fingerprint_tokenizer(tokenizer: Any) -> str:
    """Create a stable tokenizer fingerprint without depending on tokenizer internals."""
    tokenizer = _unwrap_tokenizer(tokenizer)
    info = {
        "class": tokenizer.__class__.__name__ if tokenizer is not None else "None",
        "path": getattr(tokenizer, "path", ""),
        "name_or_path": getattr(tokenizer, "name_or_path", ""),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "vocab_size": len(tokenizer) if tokenizer is not None and hasattr(tokenizer, "__len__") else None,
        "vocab_hash": _tokenizer_vocab_hash(tokenizer),
    }
    return _sha256(canonicalize_schema(info))


def _normalize_response_format(response_format: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    if not isinstance(response_format, dict):
        raise ValueError("response_format must be an object")
    grammar_kind = response_format.get("type")
    if grammar_kind == "json_object":
        schema = {"type": "object"}
        metadata = {"type": "json_object", "strict": False, "name": ""}
    elif grammar_kind == "json_schema":
        json_schema = response_format.get("json_schema")
        if not isinstance(json_schema, dict):
            raise ValueError("response_format.json_schema must be an object")
        schema = json_schema.get("schema")
        if not isinstance(schema, dict):
            raise ValueError("response_format.json_schema.schema must be an object")
        metadata = {
            "type": "json_schema",
            "strict": bool(json_schema.get("strict", False)),
            "name": str(json_schema.get("name", "")),
        }
    else:
        raise ValueError("response_format.type must be json_object or json_schema")
    return grammar_kind, schema, metadata


@dataclass(frozen=True)
class XGrammarCompileResult:
    cache_key: str
    tokenizer_fp: str
    grammar_kind: str
    canonical_schema: str
    canonical_schema_sha256: str
    xgrammar_version_hash: str
    gpu_lowering_version: str
    compiled_blob: bytes
    gpu_lowering_blob: bytes
    backend_tokenizer_payload: str
    metadata: Dict[str, Any]
    cache_hit: bool = False


class XGrammarFrontendCompiler:
    """Process-local LRU and singleflight wrapper for structured output compilation."""

    def __init__(
        self, capacity: int = DEFAULT_XGRAMMAR_COMPILE_CACHE_SIZE, thread_num: int = 4
    ):
        verify_xgrammar_runtime(required=True)
        self.capacity = max(0, int(capacity))
        self._thread_num = max(1, int(thread_num))
        self._cache: "OrderedDict[str, XGrammarCompileResult]" = OrderedDict()
        self._inflight: Dict[str, Future] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=self._thread_num, thread_name_prefix="xgrammar-compile")
        self.hit_total = 0
        self.miss_total = 0
        self.eviction_total = 0
        self.compile_latency_ms = 0.0

    @property
    def entries(self) -> int:
        with self._lock:
            return len(self._cache)

    def stats(self) -> Dict[str, Any]:
        return {
            "xgrammar_compile_cache_hit_total": self.hit_total,
            "xgrammar_compile_cache_miss_total": self.miss_total,
            "xgrammar_compile_cache_eviction_total": self.eviction_total,
            "xgrammar_compile_cache_entries": self.entries,
            "xgrammar_compile_latency_ms": self.compile_latency_ms,
        }

    def compile_response_format(
        self, response_format: Optional[Dict[str, Any]], tokenizer: Any
    ) -> Optional[XGrammarCompileResult]:
        if not response_format:
            return None
        verify_xgrammar_runtime(required=True)
        grammar_kind, schema, metadata = _normalize_response_format(response_format)
        tokenizer_fp = fingerprint_tokenizer(tokenizer)
        canonical_schema = canonicalize_schema(schema)
        schema_sha256 = _sha256(canonical_schema)
        cache_key = "|".join(
            [
                tokenizer_fp,
                grammar_kind,
                schema_sha256,
                XGRAMMAR_VERSION_HASH,
                GPU_LOWERING_VERSION,
            ]
        )
        return self.get_or_compile(cache_key, tokenizer_fp, grammar_kind, canonical_schema, schema_sha256, metadata, tokenizer)

    def get_or_compile(
        self,
        cache_key: str,
        tokenizer_fp: str,
        grammar_kind: str,
        canonical_schema: str,
        schema_sha256: str,
        metadata: Dict[str, Any],
        tokenizer: Any,
    ) -> XGrammarCompileResult:
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                self.hit_total += 1
                return self._with_cache_hit(cached, True)
            future = self._inflight.get(cache_key)
            if future is None:
                self.miss_total += 1
                future = self._executor.submit(
                    self._compile,
                    cache_key,
                    tokenizer_fp,
                    grammar_kind,
                    canonical_schema,
                    schema_sha256,
                    metadata,
                    tokenizer,
                )
                self._inflight[cache_key] = future

        try:
            result = future.result()
        finally:
            with self._lock:
                if self._inflight.get(cache_key) is future:
                    self._inflight.pop(cache_key, None)

        with self._lock:
            if self.capacity > 0 and cache_key not in self._cache:
                self._cache[cache_key] = result
                self._cache.move_to_end(cache_key)
                while len(self._cache) > self.capacity:
                    self._cache.popitem(last=False)
                    self.eviction_total += 1
            elif self.capacity == 0:
                return result
            return self._with_cache_hit(self._cache.get(cache_key, result), False)

    def _compile(
        self,
        cache_key: str,
        tokenizer_fp: str,
        grammar_kind: str,
        canonical_schema: str,
        schema_sha256: str,
        metadata: Dict[str, Any],
        tokenizer: Any,
    ) -> XGrammarCompileResult:
        start = time.perf_counter()
        compiled_blob, tokenizer_info_json = self._compile_with_xgrammar(
            grammar_kind, canonical_schema, metadata, tokenizer
        )
        backend_tokenizer_payload = self._make_backend_payload(
            tokenizer_info_json, compiled_blob.decode("utf-8")
        )
        payload = {
            "grammar_kind": grammar_kind,
            "canonical_schema": canonical_schema,
            "canonical_schema_sha256": schema_sha256,
            "xgrammar_version_hash": XGRAMMAR_VERSION_HASH,
            "gpu_lowering_version": GPU_LOWERING_VERSION,
            "metadata": metadata,
        }
        gpu_lowering_blob = canonicalize_schema(
            {
                **payload,
                "compiled_grammar_json": compiled_blob.decode("utf-8"),
                "tokenizer_info_json": tokenizer_info_json,
            }
        ).encode("utf-8")
        self.compile_latency_ms = (time.perf_counter() - start) * 1000.0
        return XGrammarCompileResult(
            cache_key=cache_key,
            tokenizer_fp=tokenizer_fp,
            grammar_kind=grammar_kind,
            canonical_schema=canonical_schema,
            canonical_schema_sha256=schema_sha256,
            xgrammar_version_hash=XGRAMMAR_VERSION_HASH,
            gpu_lowering_version=GPU_LOWERING_VERSION,
            compiled_blob=compiled_blob,
            gpu_lowering_blob=gpu_lowering_blob,
            backend_tokenizer_payload=backend_tokenizer_payload,
            metadata=metadata,
            cache_hit=False,
        )

    def _compile_with_xgrammar(
        self, grammar_kind: str, canonical_schema: str, metadata: Dict[str, Any], tokenizer: Any
    ) -> Tuple[bytes, str]:
        import xgrammar  # type: ignore

        tokenizer_info = self._get_or_create_tokenizer_info(tokenizer)
        compiler = xgrammar.GrammarCompiler(
            tokenizer_info,
            max_threads=max(1, getattr(self, "_thread_num", 1)),
            cache_enabled=True,
        )
        if grammar_kind == "json_object":
            compiled = compiler.compile_builtin_json_grammar()
        else:
            compiled = compiler.compile_json_schema(
                canonical_schema,
                strict_mode=bool(metadata.get("strict", False)),
            )
        return compiled.serialize_json().encode("utf-8"), tokenizer_info.serialize_json()

    def _get_or_create_tokenizer_info(self, tokenizer: Any) -> Any:
        import xgrammar  # type: ignore

        tokenizer = _unwrap_tokenizer(tokenizer)
        if tokenizer is None:
            raise RuntimeError("xgrammar tokenizer is not set for compilation")

        from_huggingface = getattr(getattr(xgrammar, "TokenizerInfo"), "from_huggingface", None)
        if callable(from_huggingface):
            try:
                return from_huggingface(
                    tokenizer,
                    vocab_size=len(tokenizer) if hasattr(tokenizer, "__len__") else None,
                    stop_token_ids=getattr(tokenizer, "eos_token_id", None),
                )
            except Exception as e:
                logging.warning("xgrammar TokenizerInfo.from_huggingface failed: %s", e)

        get_vocab = getattr(tokenizer, "get_vocab", None)
        if not callable(get_vocab):
            raise RuntimeError(f"xgrammar tokenizer {type(tokenizer)} does not expose get_vocab")
        vocab = get_vocab()
        vocab_size = len(tokenizer) if hasattr(tokenizer, "__len__") else max(vocab.values()) + 1
        encoded_vocab = [""] * vocab_size
        for token, idx in vocab.items():
            if 0 <= idx < vocab_size:
                encoded_vocab[idx] = token
        return xgrammar.TokenizerInfo(
            encoded_vocab,
            vocab_size=vocab_size,
            stop_token_ids=getattr(tokenizer, "eos_token_id", None),
        )

    @staticmethod
    def _make_backend_payload(tokenizer_info_json: str, compiled_grammar_json: str) -> str:
        tokenizer_info_json_bytes = tokenizer_info_json.encode("utf-8")
        return (
            f"{BACKEND_PAYLOAD_VERSION}:"
            f"{len(tokenizer_info_json_bytes)}:"
            f"{tokenizer_info_json}"
            f"{compiled_grammar_json}"
        )

    @staticmethod
    def _with_cache_hit(result: XGrammarCompileResult, cache_hit: bool) -> XGrammarCompileResult:
        return XGrammarCompileResult(
            cache_key=result.cache_key,
            tokenizer_fp=result.tokenizer_fp,
            grammar_kind=result.grammar_kind,
            canonical_schema=result.canonical_schema,
            canonical_schema_sha256=result.canonical_schema_sha256,
            xgrammar_version_hash=result.xgrammar_version_hash,
            gpu_lowering_version=result.gpu_lowering_version,
            compiled_blob=result.compiled_blob,
            gpu_lowering_blob=result.gpu_lowering_blob,
            backend_tokenizer_payload=result.backend_tokenizer_payload,
            metadata=result.metadata,
            cache_hit=cache_hit,
        )
