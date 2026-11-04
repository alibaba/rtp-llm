#!/bin/bash
# Vendor upstream FlashInfer as the private package ``fi_up`` (only its PCIe IPC all-reduce / reduce-scatter /
# all-gather are used, by DSV4_PCIE_AR_BACKEND=fi and DSV4_TP_RSAG=1) and pre-build its two JIT extensions, so the
# 16 engine ranks do not race on the first build. The engine's own ``flashinfer`` package is untouched.
# Idempotent; run once per host (or once on shared storage). Pass the output directory to the launchers as FI_PCIE.
# usage: [CUDA_HOME=/usr/local/cuda] [PYTHON=python3] [GPU=0] bash setup_fi_up.sh <out_dir> [commit=79f0637]
set -e
O=${1:?usage: setup_fi_up.sh <out_dir> [commit]}
C=${2:-79f0637}
CUDA=${CUDA_HOME:-/usr/local/cuda}
PY=${PYTHON:-python3}
S=$O.src
if [ ! -d "$S/.git" ]; then
  git clone --filter=blob:none https://github.com/flashinfer-ai/flashinfer.git "$S"
fi
(cd "$S" && (git checkout -q "$C" 2>/dev/null || (git fetch -q origin main && git checkout -q "$C")))
rm -rf "$O/fi_up" && mkdir -p "$O"
cp -r "$S/flashinfer" "$O/fi_up"
rm -rf "$O/fi_up/data" && mkdir -p "$O/fi_up/data/csrc"
cp -r "$S/include" "$O/fi_up/data/include"
cp "$S"/csrc/pcie_ipc_*.cu "$S/csrc/tvm_ffi_utils.h" "$O/fi_up/data/csrc/"
echo '"""Upstream FlashInfer loaded under a private name for the PCIe IPC collectives only."""' > "$O/fi_up/__init__.py"
: > "$O/fi_up/comm/__init__.py"
(cd "$O" && find fi_up -name "*.py" -print0 | xargs -0 sed -i -E \
  's/(^|[^A-Za-z0-9_.])from flashinfer([. ])/\1from fi_up\2/g; s/(^|[^A-Za-z0-9_.])import flashinfer\b/\1import fi_up/g; s/"flashinfer\.([a-z_]+)"/"fi_up.\1"/g')
echo "commit $(cd "$S" && git rev-parse --short HEAD)" > "$O/VERSION"
FLASHINFER_WORKSPACE_BASE=$O FLASHINFER_CUDA_ARCH_LIST=${FI_ARCH:-12.0f} CUDA_HOME=$CUDA PATH=$CUDA/bin:$PATH \
  CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONPATH=$O $PY -c "
from fi_up.jit.comm import gen_pcie_ipc_comm_module, gen_pcie_ipc_ag_rs_module
gen_pcie_ipc_comm_module().build_and_load(); gen_pcie_ipc_ag_rs_module().build_and_load(); print('fi_up built')
"
cat "$O/VERSION"
