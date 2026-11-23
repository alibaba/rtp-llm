#!/bin/sh
# Install the inputs built from pinned sources by this vLLM CI run.
set -eu
python_bin=${1:?}; vllm_branch=${2:?}; vllm_commit=${3:?}
architecture=${4:?}; public_commit=${5:?}; internal_commit=${6:?}
source_branch=${7:?}
if [ "${NAVI_SCHED_INSTALL:-1}" = 0 ]; then
    "$python_bin" -m pip uninstall -y navi-sched
    exit 0
fi
scripts=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
bundle=$(dirname "$scripts")
set -- "$bundle"/navi_sched-*.whl
[ "$#" -eq 1 ]
sh "$scripts/install_navi_sched_wheel.sh" "$python_bin" "$1" "$bundle" \
    /opt/navi_sched/build-info "$public_commit" 0.1.0 "$vllm_branch" \
    "$vllm_commit" "$architecture" "$source_branch" "$internal_commit"
mkdir -p /opt/navi_sched/integration
cp -R "$scripts"/. /opt/navi_sched/integration/
