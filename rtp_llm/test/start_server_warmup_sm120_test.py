"""SM120 cases split out of start_server_warmup_test.py.
These cases live in this sibling file, leaving the original test file
untouched."""

import os
import subprocess
import sys

_ENTRY_MODULES = (
    "rtp_llm.start_server",
    "rtp_llm.start_backend_server",
    "rtp_llm.start_frontend_server",
)

_CUTLASS_PATH_PROBE = """
import importlib
import os
import sys

needle = os.path.join("nvidia_cutlass_dsl", "python_packages")


def cutlass_entries():
    return {entry for entry in sys.path if needle in entry}


before = cutlass_entries()
importlib.import_module(sys.argv[1])
added = sorted(cutlass_entries() - before)
print(repr(added))
raise SystemExit(1 if added else 0)
"""
import unittest


class StartServerImportTest(unittest.TestCase):
    def test_entry_imports_do_not_setup_cutlass_import_path(self):
        for module_name in _ENTRY_MODULES:
            with self.subTest(module=module_name):
                result = subprocess.run(
                    [sys.executable, "-c", _CUTLASS_PATH_PROBE, module_name],
                    env=os.environ.copy(),
                    capture_output=True,
                    text=True,
                    timeout=90,
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    f"importing {module_name} changed the CUTLASS sys.path "
                    f"delta to {result.stdout.strip()}:\n{result.stderr}",
                )


def _load_warmup_predicate():
    """Compile the REAL ``_should_run_startup_real_warmup`` chain out of
    ``rtp_llm/start_server.py`` source — no module import, no backend deps.
    Expected results are asserted independently below, never derived from a
    copied predicate."""
    import ast
    import logging as _logging
    import pathlib
    import types

    src = (pathlib.Path(__file__).resolve().parents[1] / "start_server.py").read_text()
    tree = ast.parse(src)
    names = {
        "_should_run_startup_real_warmup",
        "_role_is_prefill",
        "_is_startup_real_warmup_entry_rank",
    }
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in funcs} == names
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future] + funcs, type_ignores=[])
    )
    role_stub = types.SimpleNamespace(
        PREFILL="PREFILL", DECODE="DECODE", PDFUSION="PDFUSION"
    )
    env = {"logging": _logging, "RoleType": role_stub}
    exec(compile(module, "start_server.py", "exec"), env)
    return env["_should_run_startup_real_warmup"], role_stub


def _warmup_configs(
    role_type,
    *,
    warm_up=True,
    model_warm_up=True,
    world_rank=0,
    world_size=4,
    tp_size=4,
    model_type="deepseek_v4",
    **extra,
):
    from types import SimpleNamespace

    return SimpleNamespace(
        runtime_config=SimpleNamespace(warm_up=warm_up, model_warm_up=model_warm_up),
        role_config=SimpleNamespace(role_type=role_type),
        parallelism_config=SimpleNamespace(
            world_rank=world_rank, world_size=world_size, tp_size=tp_size
        ),
        model_args=SimpleNamespace(model_type=model_type),
        **extra,
    )


class StartupRealWarmupTestSm120(unittest.TestCase):
    """Independent matrix over the restored main warmup predicate."""

    def test_main_warmup_predicate_matrix(self):
        predicate, RoleType = _load_warmup_predicate()

        # Both warmup booleans must be set.
        for warm_up, model_warm_up in ((False, True), (True, False), (False, False)):
            cfg = _warmup_configs(
                RoleType.PREFILL, warm_up=warm_up, model_warm_up=model_warm_up
            )
            self.assertIs(predicate(cfg), False, (warm_up, model_warm_up))

        # Role must be PREFILL.
        for role in (RoleType.DECODE, RoleType.PDFUSION):
            self.assertIs(predicate(_warmup_configs(role)), False, role)

        # Only the DSv4 model identity qualifies.
        for mt in ("deepseek_v3", "deepseek_v2", "", "qwen3"):
            cfg = _warmup_configs(RoleType.PREFILL, model_type=mt)
            self.assertIs(predicate(cfg), False, mt)

        # Entry rank: world_rank % tp_size == 0 when world_size > 1.
        for rank in (1, 5):
            cfg = _warmup_configs(RoleType.PREFILL, world_rank=rank)
            self.assertIs(predicate(cfg), False, rank)
        for rank, ws in ((0, 4), (4, 8), (8, 16)):
            cfg = _warmup_configs(RoleType.PREFILL, world_rank=rank, world_size=ws)
            self.assertIs(predicate(cfg), True, (rank, ws))
        # Single-process world has no entry-rank restriction.
        self.assertIs(
            predicate(_warmup_configs(RoleType.PREFILL, world_size=1, world_rank=0)),
            True,
        )
        # The happy path: every condition satisfied.
        self.assertIs(predicate(_warmup_configs(RoleType.PREFILL)), True)

        # Ambient retired-option state must not influence the predicate:
        # neither an env var nor a leftover config field changes the result.
        for cfg in (
            _warmup_configs(RoleType.PREFILL, dsv4_startup_real_warmup=False),
            _warmup_configs(RoleType.PREFILL, dsv4_startup_real_warmup=True),
        ):
            self.assertIs(predicate(cfg), True)
        from unittest.mock import patch

        with patch.dict(os.environ, {"DSV4_STARTUP_REAL_WARMUP": "0"}):
            self.assertIs(predicate(_warmup_configs(RoleType.PREFILL)), True)
            self.assertIs(
                predicate(_warmup_configs(RoleType.PREFILL, warm_up=False)),
                False,
            )

        # The retired CLI option is not registered: neither the args module
        # nor the entrypoint source may carry it.
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1]
        for rel in (
            "server/server_args/misc_group_args.py",
            "start_server.py",
        ):
            self.assertNotIn(
                "dsv4_startup_real_warmup",
                (root / rel).read_text(),
                rel,
            )


if __name__ == "__main__":
    unittest.main()
