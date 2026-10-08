import tempfile
import unittest
import zipfile
from importlib import metadata
from pathlib import Path

from bazel.merge_cutlass_wheels import CUTLASS_DISTRIBUTIONS, merge_wheels


class MergeCutlassWheelsTest(unittest.TestCase):
    def _wheels(self, root, changes=None, versions=None):
        changes = changes or {}
        versions = versions or {}
        payloads = {
            "nvidia-cutlass-dsl": {"cutlass.pth": "nvidia_cutlass_dsl/dsl_packages\n"},
            "nvidia-cutlass-dsl-libs-core": {
                "nvidia_cutlass_dsl/dsl_packages/cutlass/__init__.py": "VERSION = '4.7.1'\n"
            },
            "nvidia-cutlass-dsl-libs-base": {
                "nvidia_cutlass_dsl/dsl_packages/cutlass/_mlir/__init__.py": ""
            },
            "nvidia-cutlass-dsl-libs-cu12": {
                "nvidia_cutlass_dsl/lib/libcu12.so": b"\x00\x01"
            },
            "nvidia-cutlass-dsl-libs-cu13": {
                "nvidia_cutlass_dsl/lib/libcu13.so": b"\x00\x02"
            },
        }
        result = []
        for name in sorted(CUTLASS_DISTRIBUTIONS):
            wheel = root / f"{name}.whl"
            version = versions.get(name, "4.7.1")
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr(
                    f"{name.replace('-', '_')}-{version}.dist-info/METADATA",
                    f"Name: {name}\nVersion: {version}\n",
                )
                archive.writestr(
                    f"{name.replace('-', '_')}-{version}.dist-info/RECORD", ""
                )
                for path, content in {
                    **payloads[name],
                    **changes.get(name, {}),
                }.items():
                    archive.writestr(path, content)
            result.append(wheel)
        return result

    def test_merges_original_payload_and_distribution_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "site"
            merge_wheels(self._wheels(root), output, "4.7.1")
            self.assertEqual(
                (output / "nvidia_cutlass_dsl/lib/libcu13.so").read_bytes(), b"\x00\x02"
            )
            self.assertTrue(
                (output / "nvidia_cutlass_dsl/dsl_packages/cutlass/_mlir").is_dir()
            )
            installed = {
                dist.metadata["Name"]: dist.version
                for dist in metadata.distributions(path=[str(output)])
            }
            self.assertEqual(installed, dict.fromkeys(CUTLASS_DISTRIBUTIONS, "4.7.1"))
            self.assertFalse((output / "nvidia_cutlass_dsl/__init__.py").exists())

    def test_accepts_identical_shared_files_and_rejects_conflicts(self):
        path = "nvidia_cutlass_dsl/shared.txt"
        for second, succeeds in (("same", True), ("different", False)):
            with self.subTest(
                second=second
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                wheels = self._wheels(
                    root,
                    changes={
                        "nvidia-cutlass-dsl-libs-core": {path: "same"},
                        "nvidia-cutlass-dsl-libs-base": {path: second},
                    },
                )
                if succeeds:
                    merge_wheels(wheels, root / "site", "4.7.1")
                else:
                    with self.assertRaisesRegex(ValueError, "Conflicting"):
                        merge_wheels(wheels, root / "site", "4.7.1")

    def test_rejects_missing_and_mismatched_components_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheels = self._wheels(root)
            with self.assertRaisesRegex(ValueError, "Missing"):
                merge_wheels(wheels[:-1], root / "site", "4.7.1")
            self.assertFalse((root / "site").exists())
            wheels = self._wheels(
                root, versions={"nvidia-cutlass-dsl-libs-base": "4.4.2"}
            )
            with self.assertRaisesRegex(ValueError, "expected 4.7.1"):
                merge_wheels(wheels, root / "site", "4.7.1")

    def test_rejects_members_outside_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheels = self._wheels(
                root, changes={"nvidia-cutlass-dsl-libs-core": {"../outside": "bad"}}
            )
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                merge_wheels(wheels, root / "site", "4.7.1")
            self.assertFalse((root / "outside").exists())


if __name__ == "__main__":
    unittest.main()
