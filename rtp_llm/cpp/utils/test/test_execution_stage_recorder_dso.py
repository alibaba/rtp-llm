"""Validate recorder/Scope behavior with an independent CPU shared-library owner.

This CPU-only Linux test compiles tiny fixtures, not the production BUILD graph.
Use check_execution_stage_recorder_elf.py against built production artifacts to
verify that their ELF dependencies also obey this contract.
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
    int64_t child_request_id;
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
    const auto snapshot = recorder.snapshot();
    result->child_active_count = snapshot.active.size();
    for (const auto& event : snapshot.active) {
        if (event.id == child.id()) {
            result->child_request_id = event.request_id;
        }
    }
}
extern "C" void a_outer_callback(ChildCallback callback, ProbeResult* result) {
    auto& recorder = rtp_llm::executionStageRecorder();
    rtp_llm::ExecutionStageScope outer("dso.a.outer", {}, 3608368576110075969LL);
    result->outer_id = outer.id();
    result->outer_recorder = reinterpret_cast<uintptr_t>(&recorder);
    callback(result);
}
extern "C" void a_dump() {
    rtp_llm::executionStageRecorder().dump("dso_test_manual");
}
"""

PROBE_B = r"""
#include "probe.h"
#include "rtp_llm/cpp/utils/ExecutionStageRecorder.h"
extern "C" void b_outer(ProbeResult* result) {
    auto& recorder = rtp_llm::executionStageRecorder();
    rtp_llm::ExecutionStageScope outer("dso.b.outer", {}, 3608368576110075969LL);
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
    const auto snapshot = recorder.snapshot();
    result->child_active_count = snapshot.active.size();
    for (const auto& event : snapshot.active) {
        if (event.id == child.id()) {
            result->child_request_id = event.request_id;
        }
    }
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
setting = sys.argv[3]
if setting == "unset":
    os.environ.pop("RTP_LLM_GPU_STAGE_TRACE", None)
else:
    os.environ["RTP_LLM_GPU_STAGE_TRACE"] = setting
enabled = setting == "1"
os.environ["RTP_LLM_GPU_STAGE_TRACE_CAPACITY"] = "128"
os.environ["RTP_LLM_GPU_STAGE_TRACE_STALL_MS"] = "60000"
prefix = directory / ("trace_" + order + "_" + setting)
os.environ["RTP_LLM_GPU_STAGE_TRACE_FILE"] = str(prefix)

class ProbeResult(ctypes.Structure):
    _fields_ = [
        ("outer_id", ctypes.c_uint64),
        ("child_id", ctypes.c_uint64),
        ("child_parent_id", ctypes.c_uint64),
        ("outer_recorder", ctypes.c_size_t),
        ("child_recorder", ctypes.c_size_t),
        ("child_active_count", ctypes.c_uint64),
        ("child_request_id", ctypes.c_int64),
    ]

libraries = {}
for name in order:
    libraries[name] = ctypes.CDLL(str(directory / ("lib" + name + ".so")))
a, b = libraries["A"], libraries.get("B")
threads_before = len(list(Path("/proc/self/task").iterdir()))
if b is not None:
    b.b_outer.argtypes = [ctypes.POINTER(ProbeResult)]
    b.b_outer.restype = None
callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(ProbeResult))
a.a_outer_callback.argtypes = [callback_type, ctypes.POINTER(ProbeResult)]
a.a_outer_callback.restype = None
results = []
for direction in (("B_to_A", "A_to_B") if b is not None else ("A_to_A",)):
    value = ProbeResult()
    if direction == "B_to_A":
        b.b_outer(ctypes.byref(value))
    elif direction == "A_to_B":
        callback = ctypes.cast(b.b_child, callback_type)
        a.a_outer_callback(callback, ctypes.byref(value))
    else:
        callback = ctypes.cast(a.a_child, callback_type)
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
        and value.child_request_id == 3608368576110075969
    ) if enabled else (
        value.outer_id == 0
        and value.child_id == 0
        and value.child_parent_id == 0
        and value.outer_recorder == value.child_recorder
        and value.child_active_count == 0
        and value.child_request_id == 0
    )
    results.append(result)
a.a_dump.argtypes = []
a.a_dump.restype = None
a.a_dump()
threads_after = len(list(Path("/proc/self/task").iterdir()))
trace_files = [str(path) for path in directory.glob(prefix.name + ".*.jsonl")]
identity = {}
if enabled:
    # Cross-check emitted identities against procfs independently from the C++
    # parser. All fixture scopes run on this interpreter's original thread.
    process_ticks = int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])
    thread_ticks = int(Path("/proc/thread-self/stat").read_text().rsplit(")", 1)[1].split()[19])
    documents = [
        json.loads(line)
        for filename in trace_files
        for line in Path(filename).read_text().splitlines()
    ]
    events = [event for document in documents for event in document["history"]]
    identity = {
        "pid": os.getpid(),
        "process_start_ticks": process_ticks,
        "thread_start_ticks": thread_ticks,
        "documents": len(documents),
        "events": len(events),
        "passed": (
            bool(documents)
            and bool(events)
            and all(
                document["pid"] == os.getpid()
                and document["process_start_ticks"] == process_ticks
                for document in documents
            )
            and all(
                event["tid"] == os.getpid()
                and event["thread_start_ticks"] == thread_ticks
                and event["request_id"] == 3608368576110075969
                and event["request_id_str"] == "3608368576110075969"
                for event in events
            )
        ),
    }
print(json.dumps({
    "load_order": order,
    "setting": setting,
    "threads_before": threads_before,
    "threads_after": threads_after,
    "trace_files": trace_files,
    "identity": identity,
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
                str(source),
                "-Wl,-soname,librtp_execution_stage_recorder.so",
                "-o",
                "librtp_execution_stage_recorder.so",
            ],
            common
            + [
                "a.cc",
                "-L.",
                "-lrtp_execution_stage_recorder",
                "-Wl,-z,defs",
                "-Wl,-rpath,$ORIGIN",
                "-Wl,-soname,libA.so",
                "-o",
                "libA.so",
            ],
            # Both consumers must directly name the CPU owner. In particular A
            # cannot depend on another GPU DSO being loaded first.
            common
            + [
                "b.cc",
                "-L.",
                "-lA",
                "-lrtp_execution_stage_recorder",
                "-Wl,-z,defs",
                "-Wl,-rpath,$ORIGIN",
                "-Wl,-soname,libB.so",
                "-o",
                "libB.so",
            ],
            # This deliberately reproduces the rejected old linkage model:
            # implementation embedded into a consumer, with no owner NEEDED.
            common
            + [
                "a.cc",
                str(source),
                "-Wl,-soname,libLegacy.so",
                "-o",
                "libLegacy.so",
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

    def check_load_order(self, order, setting="1"):
        result = subprocess.run(
            [sys.executable, "-c", LOADER, str(self.directory), order, setting],
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        print(json.dumps(report), flush=True)
        self.assertEqual(report["ctypes_default_mode"], report["rtld_local"])
        self.assertEqual(len(report["results"]), 1 if order == "A" else 2)
        if setting != "1":
            self.assertEqual(report["threads_before"], report["threads_after"])
            self.assertEqual(report["trace_files"], [])
        else:
            self.assertTrue(
                report["identity"]["passed"], json.dumps(report["identity"])
            )
        for direction in report["results"]:
            with self.subTest(load_order=order, direction=direction["direction"]):
                self.assertTrue(direction["passed"], json.dumps(direction))

    def test_a_then_b(self):
        self.check_load_order("AB")

    def test_b_then_a(self):
        self.check_load_order("BA")

    def test_consumer_without_compute_library_loaded(self):
        self.check_load_order("A")

    def test_elf_contract_and_independent_loader(self):
        if shutil.which("readelf") is None:
            self.skipTest("readelf unavailable")
        checker = Path(__file__).with_name("check_execution_stage_recorder_elf.py")
        result = subprocess.run(
            [
                sys.executable,
                str(checker),
                "--lib-dir",
                str(self.directory),
                "--consumer",
                "libA.so",
                "--consumer",
                "libB.so",
                "--load-library",
                "libA.so",
                "--load-library",
                "libB.so",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["passed"], result.stdout)
        self.assertFalse(report["loader"]["result"]["torch_imported"])
        self.assertEqual(report["loader"]["result"]["trace_setting"], "0")

    def test_elf_contract_rejects_embedded_legacy_owner(self):
        if shutil.which("readelf") is None:
            self.skipTest("readelf unavailable")
        checker = Path(__file__).with_name("check_execution_stage_recorder_elf.py")
        result = subprocess.run(
            [
                sys.executable,
                str(checker),
                "--lib-dir",
                str(self.directory),
                "--consumer",
                "libLegacy.so",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report["passed"])
        self.assertTrue(
            any("no direct DT_NEEDED" in error for error in report["errors"])
        )
        self.assertTrue(
            any(
                "another recorder implementation" in error for error in report["errors"]
            )
        )

    def test_disabled_does_not_start_monitor_or_write_files(self):
        for setting in ("unset", "0", "false", ""):
            for order in ("AB", "BA"):
                with self.subTest(setting=setting, load_order=order):
                    self.check_load_order(order, setting)


if __name__ == "__main__":
    unittest.main(verbosity=2)
