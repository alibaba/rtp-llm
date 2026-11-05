"""Temporary-directory setup that can run before importing TileLang/TVM."""

import os
import tempfile


def ensure_tvm_tmpdir_writable() -> None:
    """Route TVM debug tempdirs away from a root-owned /tmp parent."""
    default_parent = os.path.join(tempfile.gettempdir(), "tvm-debug-mode-tempdirs")
    try:
        # Another TP rank may create this directory concurrently. Two separate
        # exists() checks can observe absent -> present, incorrectly take the
        # fallback, and change TMPDIR on only one rank. Symmetric-memory FD
        # exchange then looks for peers' sockets in different directories.
        os.makedirs(default_parent, exist_ok=True)
        if os.access(default_parent, os.W_OK):
            return
    except OSError:
        pass

    fallback = os.path.join("/tmp", f"rtp_tvm_tmp_{os.getuid()}")
    try:
        os.makedirs(fallback, exist_ok=True)
    except OSError:
        return
    os.environ["TMPDIR"] = fallback
    tempfile.tempdir = None
