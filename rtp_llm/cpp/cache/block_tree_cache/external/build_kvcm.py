# SPDX-License-Identifier: Apache-2.0
"""Build the optional BTC KVCM Host library with the deployment SDK.

Provide the matching KVCM SDK headers under <include>/kvcm_client, plus
libcurl, nlohmann/json and xxHash headers. The SDK shared library is supplied
by the deployment. The CPU tree build and its dependencies are independent.
"""

from __future__ import annotations

import argparse
import os
import shlex
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
    root = here.parents[4]
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
        destination = Path(temporary) / output.name
        command = [
            *shlex.split(os.environ.get("CXX", "c++")),
            "-std=c++17",
            "-O2",
            "-fPIC",
            "-fvisibility=hidden",
            "-shared",
            "-pthread",
            f"-I{root}",
            *(f"-I{path}" for path in includes),
            str(here / "BtcKvcmBackend.cc"),
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
