"""Task-scoped stdlib TOML API backport for the qualified Python3.10 runtime.

Do not alter installed rLLM or generic interpreter behavior. Python3.11+ uses
its real stdlib module; only this explicit Qwen3.8 world72 job gets the already
installed tomli implementation of the same public parsing API.
"""
from __future__ import annotations
import importlib
import os
import sys


def install_python310_tomllib():
    target=(os.environ.get('Q38_ENABLE_BF16_STATE_RUNTIME')=='1'
            and os.environ.get('RTP_LLM_RL_WORLD72')=='1')
    if not target or sys.version_info >= (3,11):
        return dict(active=False,interpreter_or_task_unchanged=True)
    if 'tomllib' in sys.modules:
        return dict(active=True,already_available=True,backend=sys.modules['tomllib'].__name__)
    try:
        module=importlib.import_module('tomllib')
    except ModuleNotFoundError as error:
        if error.name!='tomllib':raise
        module=importlib.import_module('tomli')
        if not all(hasattr(module,name) for name in ('load','loads','TOMLDecodeError')):
            raise RuntimeError('The Python3.10 TOML backport lacks the required stdlib parsing API')
        sys.modules['tomllib']=module
    return dict(active=True,already_available=False,backend=module.__name__,backend_path=module.__file__)
