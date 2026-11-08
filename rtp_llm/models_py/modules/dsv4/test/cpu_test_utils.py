"""Scoped runtime isolation for source-loaded CPU unit tests."""

import os
import sys
import unittest
from contextlib import ExitStack
from unittest.mock import patch


def isolate_cpu_test_module() -> ExitStack:
    """Restore runtime modules, interpreter mode and CUDA guards after a suite.

    Called from setUpModule so discovery does not modify process state. Fresh
    runtime modules prevent the suite from reusing another test's stubs. Keep
    discovered test modules registered for unittest and dataclass introspection.
    """
    import torch

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU tests require CUDA_VISIBLE_DEVICES to be empty")

    stack = ExitStack()
    unittest.addModuleCleanup(stack.close)
    modules = {
        name: module
        for name, module in sys.modules.copy().items()
        if name == "rtp_llm" or name.startswith("rtp_llm.")
    }

    def restore_modules():
        for name in list(sys.modules):
            if name == "rtp_llm" or name.startswith("rtp_llm."):
                del sys.modules[name]
        sys.modules.update(modules)

    stack.callback(restore_modules)
    for name in modules:
        if ".test." not in name and not name.endswith(".test"):
            del sys.modules[name]
    stack.enter_context(patch.dict(os.environ, {"TRITON_INTERPRET": "1"}))

    def forbid_cuda():
        raise AssertionError("CUDA initialization is forbidden in CPU tests")

    stack.enter_context(patch.object(torch.cuda, "_lazy_init", forbid_cuda))
    return stack
