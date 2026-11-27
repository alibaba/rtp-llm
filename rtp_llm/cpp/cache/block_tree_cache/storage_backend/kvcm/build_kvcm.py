# SPDX-License-Identifier: Apache-2.0
"""Build the optional BTC KVCM Host library with the deployment SDK.

Provide the matching KVCM SDK headers under <include>/kvcm_client, plus
libcurl, nlohmann/json and xxHash headers. The SDK shared library is supplied
by the deployment. The CPU tree build and its dependencies are independent.
"""

from __future__ import annotations

import argparse
import ast
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include", type=Path, action="append", required=True)
    parser.add_argument("--sdk-library", type=Path, required=True)
    parser.add_argument("--curl-library", type=Path)
    parser.add_argument("--runpath", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("BTC KVCM builds against the Linux deployment SDK")

    here = Path(__file__).resolve().parent
    manifest = ast.parse((here / "standalone_sources.bzl").read_text(encoding="utf-8"))
    files = next(ast.literal_eval(item.value) for item in manifest.body
                 if isinstance(item, ast.Assign) and item.targets[0].id == "KVCM_STANDALONE_FILES")
    sources = [here / path for path in files if path.endswith(".cc") and "/" not in path]
    includes = [path.resolve(strict=True) for path in args.include]
    sdk = args.sdk_library.resolve(strict=True)
    curl = (
        str(args.curl_library.resolve(strict=True))
        if args.curl_library is not None
        else "-lcurl"
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Use a relative runtime lookup for the standard installed package layout,
    # and allow the deployment to add alternative locations explicitly.
    runpath = [
        "$ORIGIN",
        "$ORIGIN/../kv_cache_manager/client/pybind",
        *args.runpath,
    ]
    with TemporaryDirectory(dir=output.parent) as temporary:
        staging = Path(temporary)
        destination = staging / output.name
        compat = here.parents[5] / "3rdparty/remote_kv_cache_manager/include/TransferClientCompat.h"
        include_compat = staging / "include"
        dest = include_compat / "kvcm_client" / compat.name
        dest.parent.mkdir(parents=True)
        shutil.copyfile(compat, dest)
        command = [
            *shlex.split(os.environ.get("CXX", "c++")),
            "-std=c++17",
            "-O2",
            "-fPIC",
            "-fvisibility=hidden",
            "-shared",
            "-pthread",
            f"-I{include_compat}",
            *(f"-I{path}" for path in includes),
            *(str(path) for path in sources),
            f"-L{sdk.parent}",
            f"-l:{sdk.name}",
            curl,
            "-Wl,--as-needed,--no-undefined",
            f"-Wl,-rpath,{':'.join(runpath)}",
            "-o",
            str(destination),
        ]
        subprocess.run(command, check=True)
        destination.replace(output)


if __name__ == "__main__":
    main()
