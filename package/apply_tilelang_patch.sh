#!/usr/bin/env bash
# Share the TVM compatibility fix with Bazel's checksum-pinned TileLang wheels.
set -euo pipefail
if [[ "${EXPECTED_CUDA_MAJOR:-}" != "13" ]]; then
    exit 0
fi
python_bin="${PYTHON_BIN:-/opt/conda310/bin/python3}"
patch_dir="$(realpath "${1:?dependency patch directory is required}")"
attrs_file="$("$python_bin" - <<'PY'
from importlib.metadata import distribution

dist = distribution("tilelang")
if dist.version not in ("0.1.9", "0.1.9+cuda.git441c3b06"):
    raise SystemExit(f"Unreviewed CUDA13 TileLang version: {dist.version}")
print(dist.locate_file("tilelang/3rdparty/tvm/python/tvm/ir/attrs.py"))
PY
)"
site_dir="$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$attrs_file")")")")")")")"
patch_file="$patch_dir/0018-tilelang-tvm-ffi-compat.patch"
tensor_file="$site_dir/tilelang/3rdparty/tvm/python/tvm/runtime/_tensor.py"
compat_state() {
"$python_bin" - "$attrs_file" "$tensor_file" <<'PY'
import ast
import pathlib
import sys

tree = ast.parse(pathlib.Path(sys.argv[1]).read_text())
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DictAttrs")
properties = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__dict__"]
attrs_patched = bool(properties)
if attrs_patched:
    expected = ast.parse("@property\ndef __dict__(self):\n    return dict(self._dict())").body[0]
    actual = properties[0]
    if ast.get_docstring(actual):
        actual.body = actual.body[1:]
    if len(properties) != 1 or ast.dump(actual) != ast.dump(expected):
        raise SystemExit("Unexpected DictAttrs.__dict__ implementation")
tensor = next(node for node in ast.parse(pathlib.Path(sys.argv[2]).read_text()).body
              if isinstance(node, ast.ClassDef) and node.name == "Tensor")
slots = [node for node in tensor.body if isinstance(node, ast.Assign)
         and any(isinstance(target, ast.Name) and target.id == "__slots__" for target in node.targets)]
if slots and (len(slots) != 1 or not isinstance(slots[0].value, ast.Tuple) or slots[0].value.elts):
    raise SystemExit("Unexpected TVM Tensor instance layout")
if attrs_patched != bool(slots):
    raise SystemExit("Partially patched TVM FFI compatibility; refuse mixed source")
print("patched" if attrs_patched else "old")
PY
}
state="$(compat_state)"
if [[ "$state" == "old" ]]; then
    patch --dry-run --batch --forward --fuzz=0 -p1 -d "$site_dir" -i "$patch_file"
    patch --batch --forward --fuzz=0 --no-backup-if-mismatch -p1 -d "$site_dir" -i "$patch_file"
fi
[[ "$(compat_state)" == "patched" ]]
echo "Verified TileLang DictAttrs and Tensor FFI compatibility patch: $site_dir"
