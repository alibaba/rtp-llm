#!/bin/sh

set -eu

if [ "$#" -ne 11 ] && [ "$#" -ne 9 ]; then
    echo "usage: $0 PYTHON WHEEL SOURCE_INFO_DIR INSTALL_INFO_DIR NAVI_COMMIT WHEEL_VERSION VLLM_BRANCH VLLM_COMMIT TARGET_ARCH NAVI_SOURCE_BRANCH RTP_INTERNAL_COMMIT" >&2
    exit 1
fi

python_bin=$1
wheel_path=$2
source_info_dir=$3
install_info_dir=$4
expected_navi_commit=$5
expected_wheel_version=$6
vllm_branch=$7
vllm_commit=$8
target_arch=$9
expected_source_branch=${10:-navi_sched}
expected_internal_commit=${11:-}
if [ -n "$expected_internal_commit" ]; then
    printf '%s\n' "$expected_internal_commit" | grep -Eq '^[0-9a-f]{40}$'
fi

printf '%s\n' "$expected_navi_commit" | grep -Eq '^[0-9a-f]{40}$'
printf '%s\n' "$vllm_commit" | grep -Eq '^[0-9a-f]{40}$'
case "$target_arch" in
    amd64)
        expected_wheel_suffix='-linux_x86_64.whl'
        ;;
    arm64)
        expected_wheel_suffix='-linux_aarch64.whl'
        ;;
    *)
        echo "unsupported target architecture: $target_arch" >&2
        exit 1
        ;;
esac
case "${wheel_path##*/}" in
    *"$expected_wheel_suffix") ;;
    *)
        echo "wheel architecture does not match target $target_arch: $wheel_path" >&2
        exit 1
        ;;
esac
[ "$(cat "$source_info_dir/NAVI_SCHED_SOURCE_BRANCH")" = "$expected_source_branch" ]
if [ -n "$expected_internal_commit" ]; then
    [ "$(cat "$source_info_dir/NAVI_SCHED_SOURCE_REPOSITORY")" = foundation_models/RTP-LLM ]
    [ "$(cat "$source_info_dir/NAVI_SCHED_RTP_INTERNAL_COMMIT")" = "$expected_internal_commit" ]
fi
[ "$(cat "$source_info_dir/NAVI_SCHED_COMMIT")" = "$expected_navi_commit" ]
[ "$(cat "$source_info_dir/NAVI_SCHED_WHEEL_VERSION")" = "$expected_wheel_version" ]
[ "$(cat "$source_info_dir/NAVI_SCHED_WHEEL_ARCHITECTURE")" = "$target_arch" ]

actual_sha256=$(sha256sum "$wheel_path" | awk '{print $1}')
[ "$actual_sha256" = "$(cat "$source_info_dir/NAVI_SCHED_WHEEL_SHA256")" ]

if [ "${NAVI_SCHED_VLLM_PLATFORM:-}" = ppu ]; then
    python_tag=$("$python_bin" -c 'import sys; print("cp%s%s" % sys.version_info[:2])')
    [ "$python_tag" = cp312 ]
    [ "$(cat "$source_info_dir/NAVI_SCHED_WHEEL_PYTHON_TAG")" = "$python_tag" ]
    case "${wheel_path##*/}" in
        *-"$python_tag"-"$python_tag"-linux_x86_64.whl) ;;
        *) echo "wheel does not match PPU Python ABI" >&2; exit 1 ;;
    esac
fi

"$python_bin" -m pip install --no-cache-dir --no-deps --force-reinstall "$wheel_path"
installed_version=$(
    "$python_bin" -c \
        'from importlib.metadata import version; print(version("navi-sched"))'
)
[ "$installed_version" = "$expected_wheel_version" ]

mkdir -p "$install_info_dir"
cp "$source_info_dir"/NAVI_SCHED_* "$install_info_dir"/
{
    printf 'NAVI_SCHED_SOURCE_BRANCH=%s\n' "$expected_source_branch"
    printf 'NAVI_SCHED_RTP_INTERNAL_COMMIT=%s\n' "$expected_internal_commit"
    printf 'NAVI_SCHED_SOURCE_COMMIT=%s\n' "$expected_navi_commit"
    printf 'NAVI_SCHED_WHEEL_VERSION=%s\n' "$installed_version"
    printf 'NAVI_SCHED_WHEEL_SHA256=%s\n' "$actual_sha256"
    printf 'NAVI_SCHED_WHEEL_ARCHITECTURE=%s\n' "$target_arch"
    printf 'VLLM_SOURCE_BRANCH=%s\n' "$vllm_branch"
    printf 'VLLM_SOURCE_COMMIT=%s\n' "$vllm_commit"
    printf 'IMAGE_ARCHITECTURE=%s\n' "$target_arch"
} > "$install_info_dir/build-info.env"

printf 'Verified NaviSched source commit: %s\n' "$expected_navi_commit"
printf 'Verified NaviSched wheel version: %s\n' "$installed_version"
printf 'Verified NaviSched wheel sha256: %s\n' "$actual_sha256"
printf 'Verified vLLM source branch: %s\n' "$vllm_branch"
printf 'Verified vLLM source commit: %s\n' "$vllm_commit"
printf 'Verified image architecture: %s\n' "$target_arch"

rm -f "$wheel_path"
