"""Shared coredump cleanup helper for smoke and perf tests.

When a crash leaves multi-GB core files in TEST_UNDECLARED_OUTPUTS_DIR, bazel
will spend a long time (or hang) packing them into outputs.zip. This helper
records a one-line summary per core file (process name, signal, size from
`file`) and then deletes them.
"""

import glob
import logging
import os
import subprocess


def summarize_and_cleanup_coredumps(output_dir: str) -> None:
    """Scan output_dir for core dump files, write a summary log, then delete them."""
    if not output_dir or not os.path.isdir(output_dir):
        return
    core_files = glob.glob(os.path.join(output_dir, "core-*")) + glob.glob(
        os.path.join(output_dir, "core.*")
    )
    if not core_files:
        return

    summary_path = os.path.join(output_dir, "coredump_summary.log")
    total_size = 0
    lines = [f"[COREDUMP_SUMMARY] Found {len(core_files)} core dump(s)\n"]
    for path in sorted(core_files):
        name = os.path.basename(path)
        try:
            size = os.path.getsize(path)
        except OSError:
            size = -1
        total_size += max(size, 0)
        size_mb = size / (1024 * 1024) if size >= 0 else -1

        # Use `file` command to extract process name / signal info
        file_info = ""
        try:
            result = subprocess.run(
                ["file", path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                file_info = result.stdout.strip()
        except Exception:
            file_info = "(file command failed)"

        line = f"  {name}  size={size_mb:.1f}MB  info={file_info}"
        lines.append(line)
        logging.info(f"[COREDUMP_SUMMARY] {line}")

    lines.append(
        f"\n[COREDUMP_SUMMARY] Total core dump size: {total_size / (1024*1024):.1f}MB"
    )
    lines.append(
        "[COREDUMP_SUMMARY] Core dump files deleted to reduce artifact size.\n"
    )

    try:
        with open(summary_path, "w") as f:
            f.write("\n".join(lines))
        logging.info(f"[COREDUMP_SUMMARY] Summary written to {summary_path}")
    except OSError as e:
        logging.warning(f"[COREDUMP_SUMMARY] Failed to write summary: {e}")

    for path in core_files:
        try:
            os.remove(path)
            logging.info(f"[COREDUMP_SUMMARY] Deleted {os.path.basename(path)}")
        except OSError as e:
            logging.warning(f"[COREDUMP_SUMMARY] Failed to delete {path}: {e}")
