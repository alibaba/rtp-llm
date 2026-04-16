# Squash Plan for wt-gho-squash-537

When squashing this branch for merge, split into these logical commits:

## Commit A: `build: remove Bazel Python infra, add setup.py + pyproject.toml`
- All BUILD file deletions (rtp_llm/, open_source/, bazel/, tools/)
- open_source/deps/ deletion (requirements_*.txt, pip.bzl, etc.)
- open_source/pyproject.toml (new)
- setup.py (new, ~1237 lines)
- _build/platform.py (new, platform detection extraction)
- conftest.py (new, pytest test framework)
- Dockerfile updates (remove pip install lines)
- ops/__init__.py rewrite (SO-loading, _preload_nvidia_deps — NOT the FMHAType import removal)
- __init__.py pytest compat (try/except around `from .ops import *`)
- server_args.py pytest compat (args parameter)
- All test files: pytest marker additions (@pytest.mark.gpu, conftest.py files)
- rtp_llm/test/remote_tests/ (REAPI pytest plugin, CAS client, etc.)
- rtp_llm/test/ci_profile_plugin.py
- rtp_llm/test/utils/ changes (crash_diag.py, device_resource.py, platform_skip.py)

## Commit B: `refactor: string-based attention backend selection (replace FMHAType enum)`
- rtp_llm/cpp/config/ConfigModules.h — FMHAType enum deletion + string fields
- rtp_llm/cpp/config/ConfigModules.cc — to_string update
- rtp_llm/cpp/pybind/ConfigInit.cc — FMHAType binding removal + string field bindings + pickle update
- rtp_llm/cpp/core/DeviceData.h, ExecOps.cc — remove use_aiter_pa/use_asm_pa from ExecInitParams
- rtp_llm/cpp/cuda/cuda_fmha_utils.h — deletion (dead code)
- rtp_llm/ops/__init__.py — FMHAType import removal (1 line)
- rtp_llm/server/server_args/fmha_group_args.py — new string args + deprecated flag stubs
- rtp_llm/models_py/modules/factory/attention/attn_factory.py — string-based dispatch rewrite
- All attention impl files — add NAME attribute, remove FMHAType imports

## Key constraint
ops/__init__.py has changes from BOTH concerns. The FMHAType import removal (1 line in the `from libth_transformer_config import (...)` block) goes to Commit B; all other changes (SO-loading rewrite) go to Commit A.
