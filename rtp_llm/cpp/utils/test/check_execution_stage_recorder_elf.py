"""Validate recorder linkage in actual production ELF artifacts.

The default mode only invokes readelf; it never loads native libraries, imports
torch, or initializes CUDA. Pass --load-library explicitly, in the desired
order, to run RTLD_NOW/RTLD_LOCAL loading in a fresh child process on a suitable
runtime host. This is a loader check, not an inference or CUDA execution test.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_OWNER = "librtp_execution_stage_recorder.so"
DEFAULT_CONSUMERS = (
    "librtp_compute_ops.so",
    "libmm_rdma_exporter.so",
    "libth_transformer.so",
)
DIAGNOSTIC_NAMES = (
    "ExecutionStageScope",
    "ExecutionStageRecorder",
    "executionStageRecorder",
    "GpuStageTrace",
    "gpuStageTraceEnabled",
)
CPU_DIAGNOSTIC_NAMES = DIAGNOSTIC_NAMES[:3]
GPU_DEPENDENCY = re.compile(
    r"cuda|torch|c10|hip|rocm|rtp_compute|th_transformer|mm_rdma", re.I
)


def run_readelf(readelf, path, option):
    result = subprocess.run(
        [readelf, option, "--wide", str(path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode:
        raise RuntimeError("readelf failed for " + str(path) + ": " + result.stderr)
    return result.stdout


def inspect_elf(readelf, path):
    dynamic = run_readelf(readelf, path, "--dynamic")
    symbols = run_readelf(readelf, path, "--dyn-syms")
    defined, undefined, strong_defined = set(), set(), set()
    for line in symbols.splitlines():
        fields = line.split(None, 7)
        if len(fields) != 8 or not fields[0].rstrip(":").isdigit():
            continue
        if fields[4] not in ("GLOBAL", "WEAK", "UNIQUE"):
            continue
        symbol = fields[7].split("@", 1)[0].split()[0]
        if fields[6] == "UND":
            undefined.add(symbol)
        elif fields[5] in ("DEFAULT", "PROTECTED"):
            defined.add(symbol)
            if fields[4] == "GLOBAL":
                strong_defined.add(symbol)
    sonames = re.findall(r"\(SONAME\).*?\[([^]]+)\]", dynamic)
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "soname": sonames[0] if len(sonames) == 1 else None,
        "needed": re.findall(r"\(NEEDED\).*?\[([^]]+)\]", dynamic),
        "rpath": re.findall(r"\((?:RUNPATH|RPATH)\).*?\[([^]]*)\]", dynamic),
        "defined": defined,
        "undefined": undefined,
        "strong_defined": strong_defined,
    }


def matches(symbol, names=DIAGNOSTIC_NAMES):
    return symbol.startswith(("_ZN7rtp_llm", "_ZNK7rtp_llm")) and any(
        name in symbol for name in names
    )


def check_artifacts(directory, owner_name, consumers, readelf):
    errors = []
    artifacts = {}

    def obtain(name):
        if name in artifacts:
            return artifacts[name]
        path = directory / name
        if not path.is_file():
            errors.append("missing artifact: " + str(path))
            return None
        try:
            artifacts[name] = inspect_elf(readelf, path)
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            errors.append(str(error))
            return None
        return artifacts[name]

    owner = obtain(owner_name)
    if owner is not None:
        if owner["soname"] != owner_name:
            errors.append("owner must publish SONAME " + owner_name)
        for name in CPU_DIAGNOSTIC_NAMES:
            if not any(name in symbol for symbol in owner["defined"]):
                errors.append("owner does not export " + name)
        gpu_needed = [name for name in owner["needed"] if GPU_DEPENDENCY.search(name)]
        if gpu_needed:
            errors.append(
                "CPU owner has GPU/runtime dependencies: " + ", ".join(gpu_needed)
            )

    for name in consumers:
        consumer = obtain(name)
        if consumer is None:
            continue
        if owner_name not in consumer["needed"]:
            errors.append(name + " has no direct DT_NEEDED " + owner_name)
        if not any(
            part in ("$ORIGIN", "${ORIGIN}")
            for search_path in consumer["rpath"]
            for part in search_path.split(":")
        ):
            errors.append(
                name
                + " cannot locate its colocated owner through $ORIGIN RPATH/RUNPATH"
            )
        duplicate = sorted(
            symbol
            for symbol in consumer["strong_defined"]
            if matches(symbol, CPU_DIAGNOSTIC_NAMES)
        )
        if duplicate:
            errors.append(
                name
                + " defines another recorder implementation: "
                + ", ".join(duplicate)
            )
        required = sorted(symbol for symbol in consumer["undefined"] if matches(symbol))
        # Follow local DT_NEEDED edges to resolve diagnostic symbols. System
        # libraries need no inspection; all RTP diagnostic providers must be in
        # the selected artifact directory, and external preloads do not count.
        closure = {}
        pending = list(consumer["needed"])
        while pending:
            dependency = pending.pop()
            if dependency in closure or not (directory / dependency).is_file():
                continue
            target = obtain(dependency)
            if target is not None:
                closure[dependency] = target
                pending.extend(target["needed"])
        providers = {
            symbol: sorted(
                dependency
                for dependency, target in closure.items()
                if symbol in target["defined"]
            )
            for symbol in required
        }
        consumer["diagnostic_providers"] = providers
        for symbol, found in providers.items():
            if not found:
                errors.append(
                    name
                    + " has unresolved diagnostic symbol in its DT_NEEDED closure: "
                    + symbol
                )

    report = {
        "validation": "static_elf",
        "lib_dir": str(directory),
        "owner": owner_name,
        "errors": errors,
        "artifacts": {},
    }
    for name, artifact in sorted(artifacts.items()):
        item = {
            key: value
            for key, value in artifact.items()
            if key not in ("defined", "undefined", "strong_defined")
        }
        item["diagnostic_exports"] = sorted(
            symbol for symbol in artifact["defined"] if matches(symbol)
        )
        item["diagnostic_undefined"] = sorted(
            symbol for symbol in artifact["undefined"] if matches(symbol)
        )
        report["artifacts"][name] = item
    report["passed"] = not errors
    return report


LOADER = r"""
import ctypes
import json
import os
from pathlib import Path
import sys

directory = Path(sys.argv[1])
owner = sys.argv[2]
loaded = []
handles = []
for name in sys.argv[3:]:
    handles.append(ctypes.CDLL(str(directory / name), mode=os.RTLD_NOW | os.RTLD_LOCAL))
    loaded.append(name)
owner_paths = sorted({
    os.path.realpath(line.split(None, 5)[5])
    for line in Path("/proc/self/maps").read_text().splitlines()
    if len(line.split(None, 5)) == 6
    and Path(line.split(None, 5)[5]).name == owner
})
print("RTP_RECORDER_ELF_RESULT=" + json.dumps({
    "load_order": loaded,
    "owner_paths": owner_paths,
    "torch_imported": "torch" in sys.modules,
    "trace_setting": os.environ.get("RTP_LLM_GPU_STAGE_TRACE"),
    "passed": owner_paths == [str((directory / owner).resolve())],
}), flush=True)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib-dir", required=True, type=Path)
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument(
        "--consumer",
        action="append",
        help="override defaults; repeat for each required consumer",
    )
    parser.add_argument("--readelf", default="readelf")
    parser.add_argument(
        "--load-library",
        action="append",
        default=[],
        help="explicit fresh-child dlopen order",
    )
    parser.add_argument("--load-timeout", type=float, default=30)
    args = parser.parse_args()
    directory = args.lib_dir.resolve()
    consumers = args.consumer or list(DEFAULT_CONSUMERS)
    for name in [args.owner] + consumers + args.load_library:
        if Path(name).name != name or name in (".", ".."):
            parser.error("library arguments must be basenames inside --lib-dir")
    if not directory.is_dir():
        parser.error("--lib-dir is not an existing directory")
    readelf = shutil.which(args.readelf)
    if readelf is None:
        parser.error("readelf unavailable")
    report = check_artifacts(directory, args.owner, consumers, readelf)
    report["loader_requested"] = bool(args.load_library)
    if args.load_library and report["passed"]:
        environment = os.environ.copy()
        environment["RTP_LLM_GPU_STAGE_TRACE"] = "0"
        environment["LD_LIBRARY_PATH"] = str(directory) + (
            os.pathsep + environment["LD_LIBRARY_PATH"]
            if environment.get("LD_LIBRARY_PATH")
            else ""
        )
        try:
            loaded = subprocess.run(
                [sys.executable, "-c", LOADER, str(directory), args.owner]
                + args.load_library,
                env=environment,
                capture_output=True,
                text=True,
                timeout=args.load_timeout,
            )
            report["loader"] = {
                "returncode": loaded.returncode,
                "stdout": loaded.stdout,
                "stderr": loaded.stderr,
            }
            if loaded.returncode == 0:
                results = [
                    line.removeprefix("RTP_RECORDER_ELF_RESULT=")
                    for line in loaded.stdout.splitlines()
                    if line.startswith("RTP_RECORDER_ELF_RESULT=")
                ]
                if len(results) != 1:
                    raise ValueError("loader produced no unique result marker")
                report["loader"]["result"] = json.loads(results[0])
                report["passed"] = report["loader"]["result"]["passed"]
            else:
                report["passed"] = False
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            report["loader"] = {"error": str(error)}
            report["passed"] = False
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
