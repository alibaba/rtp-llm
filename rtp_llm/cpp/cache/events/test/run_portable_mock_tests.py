#!/usr/bin/env python3
"""Run real cache-tree/event code with test-only pool and receiver doubles.

Requires Python 3, git, a C++17 compiler and GitHub access unless dependency
source directories are supplied.
No GPU, CUDA, Torch, Bazel, running KVCM, or production configuration is needed.
"""

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[5]
TEST = ROOT / "rtp_llm/cpp/cache/events/test"


def run(command, **kwargs):
    return subprocess.run([str(arg) for arg in command], check=True, **kwargs)


def dependency(work, name, supplied, repository, revision):
    if supplied:
        return supplied.resolve()
    path = work / name
    print(f"Preparing {name} at {revision}", flush=True)
    run(["git", "init", "--quiet", path])
    run(["git", "-C", path, "fetch", "--quiet", "--depth=1", repository, revision], timeout=180)
    run(["git", "-C", path, "checkout", "--quiet", "--detach", "FETCH_HEAD"])
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "clang++"))
    parser.add_argument("--googletest-dir", type=Path, help="Use an existing GoogleTest source tree")
    parser.add_argument("--rapidjson-dir", type=Path, help="Use an existing RapidJSON source tree")
    parser.add_argument("--sanitize", action="store_true", help="Enable AddressSanitizer")
    parser.add_argument("--check-mutations", action="store_true", help="Prove tests reject five intentional regressions")
    args = parser.parse_args()
    if not shutil.which(args.cxx):
        parser.error(f"C++ compiler not found: {args.cxx}")

    tmp_root = ROOT / ".tmp"
    tmp_root.mkdir(exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="reportevent-mock-", dir=tmp_root) as directory:
            work = Path(directory)
            # Exact pins from deps/git.bzl; never alter the user's installed dependencies.
            gtest = dependency(work, "googletest", args.googletest_dir,
                               "https://github.com/google/googletest.git", "f8d7d77c06936315286eb55f8de22cd23c188571")
            rapidjson = dependency(work, "rapidjson", args.rapidjson_dir,
                                   "https://github.com/Tencent/rapidjson.git", "f54b0e47a08782a6131cc3d60f94d038fa6e0a51")
            if not args.rapidjson_dir:
                run(["git", "-C", rapidjson, "apply", "-p0", ROOT / "3rdparty/rapidjson/0001-document_h.patch"])
            tree_source = ROOT / "rtp_llm/cpp/cache/block_tree_cache/BlockTree.cc"
            report_source = ROOT / "rtp_llm/cpp/cache/events/KVCMEventReport.cc"
            sources = [
                tree_source,
                ROOT / "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSetResource.cc",
                report_source,
                ROOT / "rtp_llm/cpp/cache/events/KVCacheEventQueue.cc",
                TEST / "portable/BlockTreeReportEventTest.cc",
                TEST / "KVCMEventReportTest.cc",
                TEST / "KVCacheEventStateTest.cc",
                TEST / "KVCacheEventQueueTest.cc",
                gtest / "googletest/src/gtest-all.cc",
                gtest / "googletest/src/gtest_main.cc",
            ]
            flags = [
                args.cxx, "-std=c++17", "-pthread", "-Wall", "-Wextra", "-Werror",
                "-DRAPIDJSON_HAS_STDSTRING=1", "-DRAPIDJSON_WRITE_DEFAULT_FLAGS=2",
                "-I", TEST / "portable/include", "-I", ROOT,
                "-isystem", rapidjson / "include", "-isystem", gtest / "googletest/include",
                "-I", gtest / "googletest",
            ]
            if args.sanitize:
                flags += ["-fsanitize=address", "-fno-omit-frame-pointer", "-g"]
            binary = work / "reportevent_mock_test"
            print("Compiling real BlockTree/resources/state/queue/JSON with mocked physical pools and receiver", flush=True)
            run(flags + sources + ["-o", binary], timeout=180)
            run([binary], timeout=60)

            if args.check_mutations:
                mutants = [
                    ("missing-host", tree_source,
                     "for (const auto medium : publication_mediums_) {",
                     "for (const auto medium : publication_mediums_) { if (medium == KVCacheEventMedium::HOST) continue;"),
                    ("publish-in-flight", tree_source,
                     "!resource.transfer_detached && resource.transfer_state == GroupSetTransferState::IDLE",
                     "!resource.transfer_detached"),
                    ("missing-insert-hook", tree_source,
                     "refreshPublishedState(current);", "/* intentionally omitted publication hook */"),
                    ("conflate-media", report_source,
                     "std::make_tuple(event.block_key, event.medium, event.group_index)",
                     "std::make_tuple(event.block_key, KVCacheEventMedium::HBM, event.group_index)"),
                    ("omit-host-snapshot", report_source,
                     "for (const auto medium : kKVCacheEventMediums) {",
                     "for (const auto medium : kKVCacheEventMediums) { if (medium == KVCacheEventMedium::HOST) continue;"),
                ]
                for name, source, before, after in mutants:
                    original = source.read_text()
                    if original.count(before) != 1:
                        raise RuntimeError(f"Mutation anchor changed for {name}; update the test runner")
                    mutant = work / f"{name}.cc"
                    mutant.write_text(original.replace(before, after))
                    mutant_binary = work / name
                    print(f"Checking intentional regression: {name}", flush=True)
                    run(flags + [mutant if path == source else path for path in sources]
                        + ["-o", mutant_binary], timeout=180)
                    result = subprocess.run([str(mutant_binary), "--gtest_filter=BlockTreeReportEventTest.*"],
                                            capture_output=True, text=True, timeout=60)
                    failures = re.findall(r"\[  FAILED  \] (BlockTreeReportEventTest\.[^\n]+)", result.stdout)
                    if result.returncode == 0 or not failures:
                        print(result.stdout + result.stderr)
                        raise RuntimeError(f"Mutation {name} did not fail a behavioral assertion")
                    print(f"Caught {name}: {failures[0]}", flush=True)
                print("All five intentional regressions were caught; production sources were never modified.", flush=True)
    finally:
        try:
            tmp_root.rmdir()  # Remove only if empty; preserve other sessions' files.
        except OSError:
            pass


if __name__ == "__main__":
    main()
