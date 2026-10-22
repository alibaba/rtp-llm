"""Repository-context contracts; these fixtures do not build or import Native Torch."""

import ast
import shlex
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_PYTHON = "/opt/conda310/bin/python3"


class _RepositoryFailure(Exception):
    pass


def _fail(message):
    raise _RepositoryFailure(message)


def _implementation():
    tree = ast.parse((PROJECT_ROOT / "def.bzl").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_torch_repo_impl"
    )
    scope = {"fail": _fail}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "def.bzl", "exec"), scope
    )
    return scope[function.name]


class _RepositoryPath:
    def __init__(self, path, extra_existing):
        self.path = Path(path)
        self.extra_existing = extra_existing

    @property
    def exists(self):
        return str(self.path) in self.extra_existing or self.path.exists()

    @property
    def dirname(self):
        return _RepositoryPath(self.path.parent, self.extra_existing)

    def __str__(self):
        return str(self.path)


class _Context:
    """A recording repository API fixture, not an installed Torch provider."""

    def __init__(self, root, environ=None, image_python=True, result=None):
        self.os = SimpleNamespace(environ=environ or {})
        self.attr = SimpleNamespace(build_file=PROJECT_ROOT / "BUILD.pytorch.native")
        self.extra_existing = {IMAGE_PYTHON} if image_python else set()
        self.active_python = Path(root) / "system" / "bin" / "python3"
        self.result = result or SimpleNamespace(
            return_code=0, stdout=str(Path(root) / "torch") + "\n", stderr=""
        )
        self.commands = []
        self.files = []
        self.links = []

    def execute(self, argv):
        self.commands.append(argv)
        return self.result

    def which(self, name):
        assert name == "python3"
        return self.path(self.active_python)

    def path(self, path):
        return _RepositoryPath(str(path), self.extra_existing)

    def read(self, path):
        return Path(path).read_text()

    def file(self, path, content):
        self.files.append((path, content))

    def symlink(self, source, destination):
        self.links.append((str(source), destination))


class TorchRepositoryDiscoveryTest(TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "torch").mkdir()
        self.implementation = _implementation()

    def assertInterpreter(self, context, expected):
        self.implementation(context)
        self.assertEqual(
            context.commands,
            [[expected, "-c", "import torch; print(torch.__path__[0])"]],
        )
        self.assertEqual(context.links[0], (str(self.root / "torch"), "torch"))

    def test_explicit_torch_root_preserves_template_and_all_adjacent_links(self):
        for name in ["nvidia", "torch.libs"]:
            (self.root / name).mkdir()
        context = _Context(
            self.root,
            {
                "TORCH_ROOT": str(self.root / "torch"),
                "PYTHON_BIN_PATH": "/custom/python",
            },
        )
        self.implementation(context)
        self.assertEqual(context.commands, [])
        self.assertEqual(
            context.files,
            [("BUILD", (PROJECT_ROOT / "BUILD.pytorch.native").read_text())],
        )
        self.assertEqual(
            context.links,
            [
                (str(self.root / name), name)
                for name in ["torch", "nvidia", "torch.libs"]
            ],
        )

    def test_explicit_python_overrides_active_venv_and_image_default(self):
        context = _Context(
            self.root,
            {"PYTHON_BIN_PATH": "/custom/python", "VIRTUAL_ENV": "/active/venv"},
        )
        self.assertInterpreter(context, "/custom/python")

    def test_active_virtual_env_is_not_replaced_with_image_torch(self):
        context = _Context(self.root, {"VIRTUAL_ENV": "/active/venv"})
        self.assertInterpreter(context, "/active/venv/bin/python3")

    def test_path_venv_without_activation_is_preserved(self):
        context = _Context(self.root)
        context.active_python = self.root / "venv" / "bin" / "python3"
        context.active_python.parent.mkdir(parents=True)
        (context.active_python.parent.parent / "pyvenv.cfg").write_text("home=/base\n")
        self.assertInterpreter(context, str(context.active_python))

    def test_plain_ci_uses_existing_action_configured_image_python(self):
        context = _Context(self.root)
        self.assertInterpreter(context, IMAGE_PYTHON)

    def test_host_without_image_interpreter_keeps_legacy_python3(self):
        context = _Context(self.root, image_python=False)
        self.assertInterpreter(context, "python3")

    def test_explicit_import_failure_retains_complete_stderr_without_fallback(self):
        failure = SimpleNamespace(
            return_code=23,
            stdout="original stdout\n",
            stderr="Traceback (most recent call last):\nImportError: original-soname.so\n",
        )
        context = _Context(
            self.root, {"PYTHON_BIN_PATH": "/custom/python"}, result=failure
        )
        with self.assertRaises(_RepositoryFailure) as caught:
            self.implementation(context)
        message = str(caught.exception)
        self.assertIn("/custom/python", message)
        self.assertIn("Interpreter exit code: 23", message)
        self.assertIn("stdout:\n" + failure.stdout, message)
        self.assertIn("stderr:\n" + failure.stderr, message)
        self.assertEqual(len(context.commands), 1)
        self.assertEqual(context.files, [])
        self.assertEqual(context.links, [])

    def test_active_venv_failure_never_falls_back_to_image_torch(self):
        context = _Context(
            self.root,
            {"VIRTUAL_ENV": "/active/venv"},
            result=SimpleNamespace(
                return_code=1, stdout="", stderr="venv import failed\n"
            ),
        )
        with self.assertRaisesRegex(_RepositoryFailure, "venv import failed"):
            self.implementation(context)
        self.assertEqual(context.commands[0][0], "/active/venv/bin/python3")
        self.assertEqual(len(context.commands), 1)

    def test_success_with_empty_path_fails_before_creating_repository(self):
        context = _Context(
            self.root,
            result=SimpleNamespace(return_code=0, stdout="\n", stderr="note\n"),
        )
        with self.assertRaisesRegex(_RepositoryFailure, "Interpreter exit code: 0"):
            self.implementation(context)
        self.assertEqual(context.files, [])
        self.assertEqual(context.links, [])

    def test_explicit_nonexistent_torch_root_fails_without_running_python(self):
        context = _Context(self.root, {"TORCH_ROOT": str(self.root / "missing")})
        with self.assertRaisesRegex(
            _RepositoryFailure, "TORCH_ROOT path does not exist"
        ):
            self.implementation(context)
        self.assertEqual(context.commands, [])
        self.assertEqual(context.links, [])

    def test_configuration_inherits_overrides_and_tracks_discovery_inputs(self):
        tree = ast.parse((PROJECT_ROOT / "def.bzl").read_text())
        rule = next(
            statement.value
            for statement in tree.body
            if isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "torch_local_repository"
                for target in statement.targets
            )
        )
        environ = ast.literal_eval(
            next(keyword.value for keyword in rule.keywords if keyword.arg == "environ")
        )
        self.assertEqual(
            environ,
            [
                "TORCH_ROOT",
                "PYTHON_BIN_PATH",
                "VIRTUAL_ENV",
                "PATH",
                "RTP_TORCH_PROVIDER",
            ],
        )
        arguments = [
            shlex.split(line, comments=True)
            for line in (PROJECT_ROOT / ".bazelrc").read_text().splitlines()
        ]
        self.assertIn(["build", "--repo_env=PYTHON_BIN_PATH"], arguments)
        self.assertIn(["build", "--repo_env=VIRTUAL_ENV"], arguments)
        self.assertFalse(
            any(
                arg.startswith(
                    ("--repo_env=PYTHON_BIN_PATH=", "--repo_env=VIRTUAL_ENV=")
                )
                for command in arguments
                for arg in command
            )
        )
        action_python = [
            arg.split("=", 1)[1]
            for command in arguments
            if command and command[0].startswith("build")
            for arg in command
            if arg.startswith("PYTHON_BIN_PATH=")
        ]
        self.assertEqual(action_python, [IMAGE_PYTHON])
