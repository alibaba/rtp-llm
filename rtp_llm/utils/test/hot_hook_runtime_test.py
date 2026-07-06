import importlib
import inspect
import json
import os
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

from rtp_llm.utils import hot_hook_runtime
from rtp_llm.utils.test import hot_hook_target


def _target(name: str) -> str:
    return f"rtp_llm.utils.test.hot_hook_target.{name}"


class HotHookRuntimeTest(unittest.TestCase):
    def setUp(self):
        hot_hook_runtime.reset_for_test()
        self._old_env = dict(os.environ)
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.hook_file = self.root / "hooks.py"
        self.config_file = self.root / "config.json"
        os.environ["RTP_HOT_HOOK"] = "1"
        os.environ["RTP_HOT_HOOK_FILE"] = str(self.hook_file)
        os.environ["RTP_HOT_HOOK_CONFIG"] = str(self.config_file)
        os.environ["RTP_HOT_HOOK_DUMP_DIR"] = str(self.root / "dumps")
        importlib.reload(hot_hook_target)

    def tearDown(self):
        hot_hook_runtime.reset_for_test()
        os.environ.clear()
        os.environ.update(self._old_env)
        self.tmpdir.cleanup()
        importlib.reload(hot_hook_target)

    def _write_hooks(self, source: str) -> None:
        self.hook_file.write_text(textwrap.dedent(source))
        time.sleep(0.02)

    def _write_config(self, config: dict) -> None:
        self.config_file.write_text(json.dumps(config, indent=2))
        time.sleep(0.02)

    def _install(self, config: dict, hooks: str) -> None:
        self._write_hooks(hooks)
        self._write_config(config)
        self.assertTrue(hot_hook_runtime.install_if_enabled())

    def test_disabled_env_does_not_patch(self):
        os.environ.pop("RTP_HOT_HOOK", None)
        self._write_hooks("def replace(ctx): return 99\n")
        self._write_config(
            {
                "function_hooks": [
                    {"target": _target("add"), "replace": "replace"},
                ]
            }
        )
        self.assertFalse(hot_hook_runtime.install_if_enabled())
        self.assertEqual(3, hot_hook_target.add(1, b=2))

    def test_function_hooks_cover_before_after_replace_exception_and_methods(self):
        self._install(
            {
                "case": "function",
                "function_hooks": [
                    {
                        "target": _target("add"),
                        "before": "before",
                        "after": "after",
                    },
                    {"target": _target("raises"), "exception": "on_exception"},
                    {"target": _target("Sample.method"), "replace": "replace_method"},
                    {"target": _target("Sample.static"), "replace": "replace_static"},
                ],
            },
            """
            def before(ctx):
                ctx.note("before", {"args": ctx.args, "kwargs": ctx.kwargs})
                ctx.args[0] = ctx.args[0] + 10

            def after(ctx):
                ctx.note("after", ctx.result)
                return ctx.result * 2

            def on_exception(ctx):
                ctx.note("exception", str(ctx.exception))
                return "handled"

            def replace_method(ctx):
                return ctx.args[1] * 10

            def replace_static(ctx):
                return ctx.args[0] + 7
            """,
        )

        self.assertEqual(26, hot_hook_target.add(1, b=2))
        self.assertEqual("handled", hot_hook_target.raises())
        self.assertEqual(50, hot_hook_target.Sample().method(5))
        self.assertEqual(12, hot_hook_target.Sample.static(5))
        notes = (self.root / "dumps" / "function" / "notes.jsonl").read_text()
        self.assertIn('"name": "before"', notes)
        self.assertIn('"name": "after"', notes)
        self.assertIn('"name": "exception"', notes)

    def test_reload_new_hook_and_syntax_error_keeps_previous_version(self):
        self._install(
            {
                "function_hooks": [
                    {"target": _target("add"), "replace": "replace"},
                ]
            },
            "def replace(ctx):\n    return 10\n",
        )
        self.assertEqual(10, hot_hook_target.add(1, b=2))

        self._write_hooks("def replace(ctx):\n    return 20\n")
        self.assertEqual(20, hot_hook_target.add(1, b=2))

        self._write_hooks("def replace(ctx):\n    return (\n")
        self.assertEqual(20, hot_hook_target.add(1, b=2))

        self._write_config({"enabled": False})
        self.assertEqual(3, hot_hook_target.add(1, b=2))

    def test_line_hook_reads_locals_and_dumps_tensor(self):
        line_no = hot_hook_target.line_target.__code__.co_firstlineno + 2
        self._install(
            {
                "case": "line",
                "line_hooks": [
                    {
                        "file": inspect.getsourcefile(hot_hook_target.line_target),
                        "line": line_no,
                        "hook": "on_line",
                    }
                ],
            },
            """
            import torch

            def on_line(ctx):
                ctx.note("locals", dict(ctx.locals))
                ctx.dump("tensor", torch.tensor([ctx.locals["tmp"]]))
            """,
        )
        self.assertEqual(11, hot_hook_target.line_target(3))
        notes = (self.root / "dumps" / "line" / "notes.jsonl").read_text()
        self.assertIn('"tmp": 4', notes)
        dumps = list((self.root / "dumps" / "line").glob("*.pt"))
        self.assertEqual(1, len(dumps))
        self.assertEqual(15, hot_hook_target.other_line_target(5))

    def test_dump_path_bounds_long_utf8_components(self):
        runtime = hot_hook_runtime.HotHookRuntime()
        runtime.dump_dir = str(self.root / "long_path_dumps")
        runtime.case = "case/" + "长" * 200
        context = hot_hook_runtime.HookContext(
            runtime=runtime,
            kind="line",
            target="/remote/execution/root/" + "长路径/" * 100 + "target.py:11",
            event="line",
            hook_config={},
        )

        path = Path(context.dump("tensor" * 100, {"value": 1}))

        self.assertTrue(path.is_file())
        self.assertLessEqual(len(path.parent.name.encode("utf-8")), 100)
        self.assertLessEqual(len(path.name.encode("utf-8")), 255)

    def test_line_hook_file_suffix_matches_loaded_module_path(self):
        line_no = hot_hook_target.line_target.__code__.co_firstlineno + 2
        self._install(
            {
                "case": "line_suffix",
                "line_hooks": [
                    {
                        "file_suffix": "rtp_llm/utils/test/hot_hook_target.py",
                        "line": line_no,
                        "hook": "on_line",
                    }
                ],
            },
            """
            def on_line(ctx):
                ctx.note("suffix_locals", dict(ctx.locals))
            """,
        )
        self.assertEqual(11, hot_hook_target.line_target(3))
        notes = (self.root / "dumps" / "line_suffix" / "notes.jsonl").read_text()
        self.assertIn('"tmp": 4', notes)

    def test_line_hook_config_reload_matches_new_file_location(self):
        line_no = hot_hook_target.line_target.__code__.co_firstlineno + 2
        other_line_no = hot_hook_target.other_line_target.__code__.co_firstlineno + 1
        hooks = """
        def on_line(ctx):
            ctx.note("line_hit", ctx.target)
        """
        self._install(
            {
                "case": "line_reload",
                "line_hooks": [
                    {
                        "file": inspect.getsourcefile(hot_hook_target.line_target),
                        "line": line_no,
                        "hook": "on_line",
                    }
                ],
            },
            hooks,
        )
        self.assertEqual(11, hot_hook_target.line_target(3))

        self._write_config(
            {
                "case": "line_reload",
                "line_hooks": [
                    {
                        "file": inspect.getsourcefile(
                            hot_hook_target.other_line_target
                        ),
                        "line": other_line_no,
                        "hook": "on_line",
                    }
                ],
            }
        )
        self.assertEqual(15, hot_hook_target.other_line_target(5))
        notes = (self.root / "dumps" / "line_reload" / "notes.jsonl").read_text()
        self.assertIn(f":{other_line_no}", notes)

    def test_line_hook_local_mutation_requires_explicit_env(self):
        line_no = hot_hook_target.line_target.__code__.co_firstlineno + 2
        config = {
            "line_hooks": [
                {
                    "file": inspect.getsourcefile(hot_hook_target.line_target),
                    "line": line_no,
                    "hook": "mutate",
                }
            ]
        }
        hooks = """
        def mutate(ctx):
            ctx.set_local("tmp", 10)
        """
        self._install(config, hooks)
        self.assertEqual(11, hot_hook_target.line_target(3))

        os.environ["RTP_HOT_HOOK_ALLOW_LOCAL_MUTATION"] = "1"
        self.assertEqual(23, hot_hook_target.line_target(3))


if __name__ == "__main__":
    unittest.main()
