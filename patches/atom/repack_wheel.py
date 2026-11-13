#!/usr/bin/env python3
"""Apply reviewed patches to a pinned wheel and rebuild its RECORD."""
import base64
import csv
import hashlib
import io
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

source = Path(sys.argv[1])
output = Path(sys.argv[2])
patches = [Path(value) for value in sys.argv[3:]]
if not patches:
    raise RuntimeError("At least one ATOM patch is required")
with tempfile.TemporaryDirectory() as temp_dir:
    root = Path(temp_dir)
    with zipfile.ZipFile(source) as archive:
        records = [name for name in archive.namelist() if name.endswith(".dist-info/RECORD")]
        if len(records) != 1:
            raise RuntimeError("Expected one wheel RECORD")
        record_name = records[0]
        archive.extractall(root)
        infos = archive.infolist()
    for patch in patches:
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=root, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=root, check=True)
    original_names = {info.filename for info in infos}
    added = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in original_names
    )
    entries = []
    for info in infos:
        if info.filename.endswith("/"):
            continue
        path = root / info.filename
        if not path.is_file():
            raise RuntimeError("Patch removed a wheel member: " + info.filename)
        entries.append((info, path))
    for name in added:
        if not name.startswith("atom/"):
            raise RuntimeError("Patch added an unexpected wheel member: " + name)
        info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        entries.append((info, root / name))
    rows = []
    for info, path in entries:
        name = info.filename
        if name == record_name:
            rows.append((name, "", ""))
        else:
            data = path.read_bytes()
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            rows.append((name, "sha256=" + digest, str(len(data))))
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    record_data = buffer.getvalue().encode()
    with zipfile.ZipFile(output, "w") as archive:
        for info, path in entries:
            archive.writestr(info, record_data if info.filename == record_name else path.read_bytes())
with zipfile.ZipFile(output) as archive:
    if archive.testzip() is not None:
        raise RuntimeError("Patched wheel ZIP integrity check failed")
print(hashlib.sha256(output.read_bytes()).hexdigest(), output)
