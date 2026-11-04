#!/usr/bin/env bash
# Install the missing vendor dependency and pinned build launcher as root.
set -euo pipefail
if [[ $(id -u) != 0 ]]; then
    echo 'Run this installer as container root.' >&2
    exit 1
fi
v41_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
v41_bazelisk_file="${1:-/tmp/v41-bazelisk-v1.26.0-linux-amd64}"
v41_bazelisk_sha=6539c12842ad76966f3d493e8f80d67caa84ec4a000e220d5459833c967c12bc
v41_requirement=$(sed -n '/^fast-hadamard-transform /p' "$v41_repo_root/deps/requirements_torch_ppu.txt")
[[ -n "$v41_requirement" ]] || { echo 'Vendor Hadamard requirement missing' >&2; exit 1; }
/opt/conda312/bin/python -m pip install --no-deps --require-hashes "$v41_requirement"
if [[ ! -f "$v41_bazelisk_file" ]]; then
    curl --fail --location --connect-timeout 15 --max-time 180 \
        https://github.com/bazelbuild/bazelisk/releases/download/v1.26.0/bazelisk-linux-amd64 \
        -o "$v41_bazelisk_file"
fi
printf '%s  %s\n' "$v41_bazelisk_sha" "$v41_bazelisk_file" | sha256sum --check -
install -m 0755 "$v41_bazelisk_file" /usr/local/bin/bazelisk
if [[ ! -e /usr/local/bin/bazel ]]; then
    ln -s bazelisk /usr/local/bin/bazel
fi
cd "$v41_repo_root"
USE_BAZEL_VERSION=6.4.0 /usr/local/bin/bazelisk --version
echo 'Persistent container prerequisites: --shm-size=1g --ulimit memlock=-1:-1 --device=/dev/infiniband:/dev/infiniband'
