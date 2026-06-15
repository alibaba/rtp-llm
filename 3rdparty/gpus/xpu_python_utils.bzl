"""Shared Python utilities for XPU repository rules."""

def resolve_venv_python(repository_ctx, python_bin):
    """Resolve a possibly-symlinked python to the actual venv python.

    When PYTHON_BIN_PATH points to a symlink (e.g. /opt/conda310/bin/python3 ->
    /opt/venv/bin/python3), Python doesn't activate the venv because pyvenv.cfg
    isn't found relative to the invoked path. This walks the symlink chain to
    find the first python whose parent directory contains pyvenv.cfg.
    Falls back to the original path if no venv is found.
    """
    result = repository_ctx.execute([
        python_bin, "-c",
        "import os, sys\npath = sys.executable\nfor _ in range(10):\n    if os.path.islink(path):\n        t = os.readlink(path)\n        if not os.path.isabs(t): t = os.path.join(os.path.dirname(path), t)\n        path = os.path.normpath(t)\n        p = os.path.dirname(os.path.dirname(path))\n        if os.path.isfile(os.path.join(p, 'pyvenv.cfg')):\n            print(path); raise SystemExit\n    else: break\nprint(sys.executable)",
    ])
    if result.return_code == 0 and result.stdout.strip():
        return result.stdout.strip()
    return python_bin
