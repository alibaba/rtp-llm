import os as _os

# Make `rtp_llm.models_py` a regular package (not a PEP 420 namespace package)
# so pytest's pkg_root walk for `models_py/{kernels,modules,standalone}/__init__.py`
# CONTINUES UP past `models_py/` to `rtp_llm/` and then to the project root.
#
# Without this `__init__.py`, pytest stops at `models_py/` (first init-less dir)
# when computing pkg_root for inner packages and does
# `sys.path.insert(0, '.../rtp_llm/models_py')`. That makes `modules.X.Y` importable
# as a TOP-LEVEL bare dotted name in addition to its canonical
# `rtp_llm.models_py.modules.X.Y` path. The same .py file then gets loaded under
# TWO sys.modules keys, each producing a fresh class object whose
# `LinearFactory.register` call bypasses the (module, name) dedup because the two
# `__module__` strings differ.
#
# Concrete symptom: `Multiple Linear strategies found:
# ['RocmF16LinearWithSwizzle', 'RocmF16LinearWithSwizzle']` — 504 sub-test
# failures in rocm_linear_test under py_ut_amd. Bazel py_test never exposed it
# because each test ran in its own subprocess with a fresh registry.
#
# Mirror of the namespace-merge pattern in rtp_llm/__init__.py: extend `__path__`
# with the sibling internal_source/rtp_llm/models_py tree so kernels/cuda code in
# internal_source/ remains reachable as `rtp_llm.models_py.kernels.cuda.X`.
_internal = _os.path.normpath(
    _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)),
        "..",
        "..",
        "internal_source",
        "rtp_llm",
        "models_py",
    )
)
if _os.path.isdir(_internal) and _internal not in __path__:
    __path__.append(_internal)
del _os, _internal
