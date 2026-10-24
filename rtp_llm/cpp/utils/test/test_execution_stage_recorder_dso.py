"""Validate recorder/Scope behavior across DSOs with one implementation owner.

This CPU-only Linux test compiles tiny fixtures, not the production BUILD graph.
It therefore verifies the single-implementation linkage contract, not whether
the production shared libraries currently obey that contract.
"""

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROBE_HEADER = r"""
#pragma once
#include <cstdint>
struct ProbeResult {
    uint64_t outer_id;
    uint64_t child_id;
    uint64_t child_parent_id;
    uintptr_t outer_recorder;
    uintptr_t child_recorder;
    uint64_t child_active_count;
};
using ChildCallback = void (*)(ProbeResult*);
extern "C" void a_child(ProbeResult* result);
"""

PROBE_A = r"""
#include "probe.h"
#include "rtp_llm/cpp/utils/ExecutionStageRecorder.h"
extern "C" void a_child(ProbeResult* result) {
    auto& recorder = rtp_llm::executionStageRecorder();
    rtp_llm::ExecutionStageScope child("dso.a.child");
    result->child_id = child.id();
    result->child_parent_id = child.parentId();
    result->child_recorder = reinterpret_cast<uintptr_t>(&recorder);
    result->child_active_count = recorder.snapshot().active.size();
}
extern "C" void a_outer_callback(ChildCallback callback, ProbeResult* result) {
    auto& recorder = rtp_llm::executionStageRecorder();
    rtp_llm::ExecutionStageScope outer("dso.a.outer");
    result->outer_id = outer.id();
    result->outer_recorder = reinterpret_cast<uintptr_t>(&recorder);
    callback(result);
}
"""

PROBE_B = r"""
#include "probe.h"
#include "rtp_llm/cpp/utils/ExecutionStageRecorder.h"
extern "C" void b_outer(ProbeResult* result) {
    auto& recorder = rtp_llm::executionStageRecorder();
    rtp_llm::ExecutionStageScope outer("dso.b.outer");
    result->outer_id = outer.id();
    result->outer_recorder = reinterpret_cast<uintptr_t>(&recorder);
    a_child(result);
}
extern "C" void b_child(ProbeResult* result) {
    auto& recorder = rtp_llm::executionStageRecorder();
    rtp_llm::ExecutionStageScope child("dso.b.child");
    result->child_id = child.id();
    result->child_parent_id = child.parentId();
    result->child_recorder = reinterpret_cast<uintptr_t>(&recorder);
    result->child_active_count = recorder.snapshot().active.size();
}
"""

# Each loader order runs in a fresh interpreter so the preceding test cannot
# promote symbols or leave an already-loaded library in the dynamic linker.
LOADER = r"""
import ctypes
import json
import os
from pathlib import Path
import sys

directory = Path(sys.argv[1])
order = sys.argv[2]
os.environ["RTP_LLM_GPU_STAGE_TRACE"] = "1"
os.environ["RTP_LLM_GPU_STAGE_TRACE_CAPACITY"] = "128"
os.environ["RTP_LLM_GPU_STAGE_TRACE_STALL_MS"] = "60000"
os.environ["RTP_LLM_GPU_STAGE_TRACE_FILE"] = str(directory / "trace")

class ProbeResult(ctypes.Structure):
    _fields_ = [
        ("outer_id", ctypes.c_uint64),
        ("child_id", ctypes.c_uint64),
        ("child_parent_id", ctypes.c_uint64),
        ("outer_recorder", ctypes.c_size_t),
        ("child_recorder", ctypes.c_size_t),
        ("child_active_count", ctypes.c_uint64),
    ]

libraries = {}
for name in order:
    libraries[name] = ctypes.CDLL(str(directory / ("lib" + name + ".so")))
a, b = libraries["A"], libraries["B"]
b.b_outer.argtypes = [ctypes.POINTER(ProbeResult)]
b.b_outer.restype = None
callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(ProbeResult))
a.a_outer_callback.argtypes = [callback_type, ctypes.POINTER(ProbeResult)]
a.a_outer_callback.restype = None
results = []
for direction in ("B_to_A", "A_to_B"):
    value = ProbeResult()
    if direction == "B_to_A":
        b.b_outer(ctypes.byref(value))
    else:
        callback = ctypes.cast(b.b_child, callback_type)
        a.a_outer_callback(callback, ctypes.byref(value))
    result = {name: getattr(value, name) for name, _ in ProbeResult._fields_}
    result["direction"] = direction
    result["passed"] = (
        value.outer_id != 0
        and value.child_id != 0
        and value.child_id != value.outer_id
        and value.child_parent_id == value.outer_id
        and value.outer_recorder == value.child_recorder
        and value.child_active_count == 2
    )
    results.append(result)
print(json.dumps({
    "load_order": order,
    "ctypes_default_mode": ctypes.DEFAULT_MODE,
    "rtld_local": os.RTLD_LOCAL,
    "results": results,
}), flush=True)
"""


class ExecutionStageRecorderDsoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform != "linux":
            raise unittest.SkipTest("requires Linux ELF shared-library loading")
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        if not compiler or shutil.which(compiler[0]) is None:
            raise unittest.SkipTest("C++ compiler unavailable: set CXX or install c++")
        cls.workspace = Path(__file__).resolve().parents[4]
        source = cls.workspace / "rtp_llm/cpp/utils/ExecutionStageRecorder.cc"
        header = source.with_suffix(".h")
        cls.temporary_directory = tempfile.TemporaryDirectory(prefix="rtp-stage-dso-")
        cls.addClassCleanup(cls.temporary_directory.cleanup)
        cls.directory = Path(cls.temporary_directory.name)
        for name, contents in (
            ("probe.h", PROBE_HEADER),
            ("a.cc", PROBE_A),
            ("b.cc", PROBE_B),
        ):
            (cls.directory / name).write_text(contents)
        common = compiler + [
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fPIC",
            "-shared",
            "-pthread",
            "-I" + str(cls.workspace),
        ]
        commands = [
            common
            + [
                "a.cc",
                str(source),
                "-Wl,-soname,libA.so",
                "-o",
                "libA.so",
            ],
            # B uses only the header; all recorder/Scope symbols come from A.
            common
            + [
                "b.cc",
                "-L.",
                "-lA",
                "-Wl,-rpath,$ORIGIN",
                "-Wl,-soname,libB.so",
                "-o",
                "libB.so",
            ],
        ]
        print(
            json.dumps(
                {
                    "build_commands": commands,
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "header_sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
                }
            ),
            flush=True,
        )
        for command in commands:
            result = subprocess.run(
                command, cwd=cls.directory, capture_output=True, text=True, timeout=60
            )
            if result.returncode != 0:
                raise AssertionError(
                    "DSO fixture build failed: "
                    + shlex.join(command)
                    + "\n"
                    + result.stdout
                    + result.stderr
                )

    def check_load_order(self, order):
        result = subprocess.run(
            [sys.executable, "-c", LOADER, str(self.directory), order],
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        print(json.dumps(report), flush=True)
        self.assertEqual(report["ctypes_default_mode"], report["rtld_local"])
        self.assertEqual(len(report["results"]), 2)
        for direction in report["results"]:
            with self.subTest(load_order=order, direction=direction["direction"]):
                self.assertTrue(direction["passed"], json.dumps(direction))

    def test_a_then_b(self):
        self.check_load_order("AB")

    def test_b_then_a(self):
        self.check_load_order("BA")


if __name__ == "__main__":
    unittest.main(verbosity=2)
