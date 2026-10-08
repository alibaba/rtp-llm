#!/usr/bin/env bash
# Keep the installed Python dependency consistent with deps/pip.bzl whl_patches.
set -euo pipefail
python_bin="${PYTHON_BIN:-/opt/conda310/bin/python3}"
patch_dir="$(realpath "$1")"
patch_target="$("$python_bin" - <<'PY'
from importlib.metadata import PackageNotFoundError, distribution
import sys

try:
    dist = distribution("flashinfer-python")
except PackageNotFoundError:
    print("FlashInfer absent; no decode patch needed", file=sys.stderr)
else:
    if dist.version == "0.6.9" or dist.version.startswith("0.6.9+"):
        print(dist.locate_file("flashinfer/decode.py"))
        print("0016-python-decode-max-kv-len.patch")
        print(2)
    elif dist.version == "0.6.18.post1":
        print(dist.locate_file("flashinfer/decode.py"))
        print("0017-python-decode-max-kv-len-0.6.18.patch")
        print(3)
    else:
        print(f"FlashInfer {dist.version} is outside the supported patch versions", file=sys.stderr)
PY
)"
if [[ -z "$patch_target" ]]; then
    exit 0
fi
mapfile -t target_fields <<< "$patch_target"
decode_file="${target_fields[0]}"
patch_file="$patch_dir/${target_fields[1]}"
expected_count="${target_fields[2]}"
site_dir="$(dirname "$(dirname "$decode_file")")"
state="$("$python_bin" - "$decode_file" "$expected_count" <<'PY'
import ast
import pathlib
import sys

text = pathlib.Path(sys.argv[1]).read_text()
ast.parse(text)
old = text.count("max(kv_lens_arr_host).item()")
new = text.count("kv_lens_arr_host.max().item()")
count = int(sys.argv[2])
if (old, new) not in [(count, 0), (0, count)]:
    raise SystemExit(f"Unexpected FlashInfer max-length sites: old={old}, patched={new}")
print("old" if old else "patched")
PY
)"
if [[ "$state" == "old" ]]; then
    patch --dry-run --batch --forward --fuzz=0 -p1 -d "$site_dir" -i "$patch_file"
    patch --batch --forward --fuzz=0 --no-backup-if-mismatch -p1 -d "$site_dir" -i "$patch_file"
fi
"$python_bin" - "$decode_file" "$expected_count" <<'PY'
import ast
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()
ast.parse(text)
count = int(sys.argv[2])
if "max(kv_lens_arr_host).item()" in text or text.count("kv_lens_arr_host.max().item()") != count:
    raise SystemExit(f"FlashInfer max-length patch verification failed: {path}")
print(f"Verified {count} vectorized FlashInfer decode max-length sites: {path}")
PY
