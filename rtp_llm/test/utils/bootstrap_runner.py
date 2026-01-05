#!/usr/bin/env python3
"""
Bootstrap runner for Bazel tests with custom sys.path priority.

This script is executed as a child process to ensure cached packages
(flashinfer, torch, deepgemm) are imported with highest priority,
overriding Bazel's default sys.path injection.
"""

import os
import runpy
import sys
import traceback


def setup_sys_path():
    """Setup sys.path with cache paths at highest priority."""
    cache_paths_str = os.environ.get("_JIT_CACHE_PATHS", "")
    cache_paths = []

    if cache_paths_str:
        cache_paths = cache_paths_str.split(":")

        print("=" * 80)
        print("[Bootstrap] Adjusting sys.path priority for cached packages")
        print(f"[Bootstrap] Cache paths to prioritize: {cache_paths}")

        # Remove cached paths from sys.path (wherever they are)
        for path in cache_paths:
            while path in sys.path:
                sys.path.remove(path)

        # Insert at the very beginning
        for path in reversed(cache_paths):
            sys.path.insert(0, path)

        print(f"[Bootstrap] First 5 sys.path entries after adjustment:")
        for i, path in enumerate(sys.path[:5]):
            print(f"[Bootstrap]   [{i}] {path}")
        print("=" * 80)

        # Clear any pre-imported torch/flashinfer modules
        keys_to_remove = [
            key
            for key in sys.modules.keys()
            if "flashinfer" in key.lower()
            or (key.startswith("torch") and "vision" not in key.lower())
        ]

        for key in keys_to_remove:
            del sys.modules[key]

        if keys_to_remove:
            print(f"[Bootstrap] Cleared {len(keys_to_remove)} pre-imported modules")

    return cache_paths


class SysPathGuard:
    """Guard to maintain priority paths at the top of sys.path."""

    def __init__(self, priority_paths):
        self.priority_paths = priority_paths

    def fix_path(self):
        """Re-insert priority paths at the beginning."""
        for path in self.priority_paths:
            while path in sys.path:
                sys.path.remove(path)
        for path in reversed(self.priority_paths):
            sys.path.insert(0, path)


class ProtectedPath(list):
    """Protected sys.path that maintains priority order."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._guard = None

    def insert(self, index, value):
        super().insert(index, value)
        if self._guard and value not in (self._guard.priority_paths or []):
            self._guard.fix_path()

    def __setitem__(self, index, value):
        super().__setitem__(index, value)
        if (
            self._guard
            and index == 0
            and value not in (self._guard.priority_paths or [])
        ):
            self._guard.fix_path()


def protect_sys_path(priority_paths):
    """Replace sys.path with protected version."""
    corrected_sys_path = sys.path.copy()
    path_guard = SysPathGuard(priority_paths)

    sys.path = ProtectedPath(corrected_sys_path)
    sys.path._guard = path_guard


def find_test_file(test_binary, original_cwd):
    """Find the actual .py test file in Bazel runfiles."""
    # Path 1: Standard py_test - test_binary.py
    test_py_file = os.path.join(original_cwd, test_binary + ".py")

    if not os.path.exists(test_py_file):
        # Path 2: smoke_test macro - test_dir/entry.py
        test_dir = test_binary.rsplit("/", 1)[0]
        test_py_file = os.path.join(original_cwd, test_dir + "/entry.py")

    return test_py_file if os.path.exists(test_py_file) else None


def setup_test_environment(test_py_file):
    """Setup working directory and sys.path for test execution."""
    test_py_abspath = os.path.abspath(test_py_file)

    if ".runfiles/rtp_llm/" not in test_py_abspath:
        print(f"[Bootstrap] Warning: Could not find .runfiles/rtp_llm/ in path")
        print(f"[Bootstrap] Using current directory: {os.getcwd()}")
        return

    # Extract runfiles/rtp_llm/ path
    runfiles_rtp_llm = (
        test_py_abspath.split(".runfiles/rtp_llm/")[0] + ".runfiles/rtp_llm"
    )
    print(f"[Bootstrap] Runfiles rtp_llm dir: {runfiles_rtp_llm}")

    # Change to runfiles/rtp_llm/ directory (for relative file paths)
    os.chdir(runfiles_rtp_llm)
    print(f"[Bootstrap] Changed CWD to: {os.getcwd()}")

    # For module imports like "from smoke.xxx import", we need to add
    # the parent directory of the module to sys.path
    test_file_dir = os.path.dirname(test_py_abspath)
    test_file_parent = os.path.dirname(test_file_dir)

    print(f"[Bootstrap] Test file dir: {test_file_dir}")
    print(f"[Bootstrap] Test file parent: {test_file_parent}")

    # Add parent directory to sys.path for module imports
    if test_file_parent not in sys.path:
        sys.path.append(test_file_parent)
        print(f"[Bootstrap] Added to sys.path: {test_file_parent}")
        # Ensure priority paths are still at the beginning
        if hasattr(sys.path, "_guard") and sys.path._guard:
            sys.path._guard.fix_path()


def main():
    """Main entry point for bootstrap runner."""
    # Setup sys.path with cached packages
    priority_paths = setup_sys_path()

    # Protect sys.path from being modified
    protect_sys_path(priority_paths)

    # Get test binary path
    if len(sys.argv) < 2:
        print("[Bootstrap] ERROR: No test binary specified")
        sys.exit(1)

    test_binary = sys.argv[1]
    original_cwd = os.environ.get("_JIT_ORIGINAL_CWD", os.getcwd())

    print(f"[Bootstrap] Test binary: {test_binary}")
    print(f"[Bootstrap] Original CWD: {original_cwd}")
    print(f"[Bootstrap] Current CWD: {os.getcwd()}")

    # Find the actual test file
    test_py_file = find_test_file(test_binary, original_cwd)

    if not test_py_file:
        print(f"[Bootstrap] ERROR: Test file not found")
        print(f"[Bootstrap] Tried:")
        print(f"[Bootstrap]   1. {os.path.join(original_cwd, test_binary + '.py')}")
        test_dir = test_binary.rsplit("/", 1)[0] if "/" in test_binary else test_binary
        print(f"[Bootstrap]   2. {os.path.join(original_cwd, test_dir + '/entry.py')}")
        sys.exit(1)

    print(f"[Bootstrap] Executing: {test_py_file}")

    # Setup test environment (CWD and sys.path)
    setup_test_environment(test_py_file)

    # Verify sys.path before execution
    print(f"[Bootstrap] sys.path before execution (first 5):")
    for i, path in enumerate(sys.path[:5]):
        print(f"[Bootstrap]   [{i}] {path}")
    print("=" * 80)

    # Set sys.argv for the test
    sys.argv = sys.argv[1:]

    # Execute the test
    try:
        runpy.run_path(test_py_file, run_name="__main__")
        sys.exit(0)
    except SystemExit as e:
        sys.exit(e.code if e.code is not None else 0)
    except Exception as e:
        print(f"[Bootstrap] Test execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
