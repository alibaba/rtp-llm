from __future__ import annotations

import ctypes
import importlib
import importlib.util
import json
import os
import resource
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from rtp_llm.dash_sc.inference.core_dump_control import (
    _XGRAMMAR_SANDBOX_CORE_DUMP_ENV,
    _configure_xgrammar_sandbox_core_dump_for_current_process,
)
from rtp_llm.dash_sc.inference.grammar_validator import (
    _compile_exception_reply,
    _WorkerStatus,
)

_OOM_PROBE_ENV = "RTP_LLM_XGRAMMAR_SANDBOX_OOM_PROBE"
_OOM_PROBE_CWD_ENV = "RTP_LLM_XGRAMMAR_SANDBOX_OOM_PROBE_CWD"
_OOM_PROBE_RESULT_ENV = "RTP_LLM_XGRAMMAR_SANDBOX_OOM_PROBE_RESULT"
_OOM_HEADROOM_MB = 1
_OOM_SCHEMA_ENUM_ENTRIES = 100_000
_OOM_SCHEMA_LITERAL_PADDING = 64
_PR_SET_DUMPABLE = 4


def _load_xgrammar() -> Any:
    module_spec = importlib.util.find_spec("xgrammar")
    if module_spec is not None and module_spec.origin is not None:
        package_dir = str(Path(module_spec.origin).parent)
        search_path = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            path for path in (package_dir, search_path) if path
        )
    return importlib.import_module("xgrammar")


xgr = _load_xgrammar()


def _enable_core_dump_for_probe() -> None:
    _, hard_limit = resource.getrlimit(resource.RLIMIT_CORE)
    if hard_limit == 0:
        raise RuntimeError("RLIMIT_CORE hard limit is zero")
    resource.setrlimit(resource.RLIMIT_CORE, (hard_limit, hard_limit))
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_DUMPABLE, 1, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def _build_tokenizer_info_json() -> str:
    vocab = [bytes([token_id]) for token_id in range(256)]
    vocab.append(b"<eos>")
    tokenizer_info = xgr.TokenizerInfo(
        vocab,
        xgr.VocabType.RAW,
        vocab_size=len(vocab),
        stop_token_ids=[len(vocab) - 1],
    )
    return tokenizer_info.serialize_json()


def _build_oom_schema() -> str:
    padding = "x" * _OOM_SCHEMA_LITERAL_PADDING
    return json.dumps(
        {
            "type": "string",
            "enum": [
                f"value_{index:08d}_{padding}"
                for index in range(_OOM_SCHEMA_ENUM_ENTRIES)
            ],
        }
    )


def _current_vsz_bytes() -> int:
    with open("/proc/self/statm") as statm:
        return int(statm.read().split()[0]) * resource.getpagesize()


def _write_reply(reply: tuple[Any, ...]) -> None:
    serialized = [
        item.value if isinstance(item, _WorkerStatus) else item for item in reply
    ]
    result_path = Path(os.environ[_OOM_PROBE_RESULT_ENV])
    with result_path.open("a", encoding="utf-8") as result_file:
        result_file.write(json.dumps(serialized) + "\n")


def _run_real_xgrammar_oom_probe() -> None:
    os.chdir(os.environ[_OOM_PROBE_CWD_ENV])
    _enable_core_dump_for_probe()
    _configure_xgrammar_sandbox_core_dump_for_current_process()
    schema = _build_oom_schema()
    tokenizer_info = xgr.TokenizerInfo.deserialize_json(
        _build_tokenizer_info_json()
    )
    compiler = xgr.GrammarCompiler(
        tokenizer_info,
        max_threads=1,
        cache_enabled=True,
        cache_limit_bytes=1,
    )
    _write_reply(("__ready__", True))
    memory_limit = _current_vsz_bytes() + _OOM_HEADROOM_MB * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
    try:
        compiler.compile_json_schema(schema)
    except Exception as error:
        _write_reply(_compile_exception_reply(error))
        return
    _write_reply((_WorkerStatus.VALID, False, ""))


if os.getenv(_OOM_PROBE_ENV) == "1":
    _run_real_xgrammar_oom_probe()
    raise SystemExit(0)


class GrammarValidatorCoreDumpTest(unittest.TestCase):
    def _oom_status(self) -> tuple[int, list[list[Any]]]:
        env = os.environ.copy()
        env[_XGRAMMAR_SANDBOX_CORE_DUMP_ENV] = "0"
        env[_OOM_PROBE_ENV] = "1"
        env["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
        with tempfile.TemporaryDirectory() as probe_dir:
            result_path = Path(probe_dir) / "worker_replies.json"
            env[_OOM_PROBE_CWD_ENV] = probe_dir
            env[_OOM_PROBE_RESULT_ENV] = str(result_path)
            pid = os.posix_spawn(
                sys.executable,
                [sys.executable, os.path.abspath(__file__)],
                env,
            )
            _, status = os.waitpid(pid, 0)
            replies = (
                [
                    json.loads(line)
                    for line in result_path.read_text(encoding="utf-8").splitlines()
                ]
                if result_path.exists()
                else []
            )
        return status, replies

    def test_real_xgrammar_oom_exits_without_core_when_disabled(self) -> None:
        _, hard_limit = resource.getrlimit(resource.RLIMIT_CORE)
        if hard_limit == 0:
            self.skipTest("RLIMIT_CORE hard limit prevents the probe setup")

        status, replies = self._oom_status()

        self.assertFalse(os.WCOREDUMP(status))
        self.assertTrue(os.WIFEXITED(status))
        self.assertEqual(os.WEXITSTATUS(status), 0)
        self.assertGreaterEqual(len(replies), 2)
        self.assertEqual(replies[0][:2], ["__ready__", True])
        self.assertEqual(replies[-1][0], _WorkerStatus.INVALID.value)
        self.assertTrue(replies[-1][1])
        self.assertTrue(replies[-1][2])


if __name__ == "__main__":
    unittest.main()
