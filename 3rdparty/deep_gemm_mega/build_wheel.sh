#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
build_dir=${DEEP_GEMM_MEGA_BUILD_DIR:-"${TMPDIR:-/tmp}/deep-gemm-mega-build"}
python_bin=${PYTHON_BIN:-python}
upstream_url=$(sed -n 's/^repository = //p' "${root_dir}/UPSTREAM")
upstream_commit=$(sed -n 's/^commit = //p' "${root_dir}/UPSTREAM")

rm -rf "${build_dir}"
mkdir -p "${build_dir}"
git clone --quiet --filter=blob:none "${upstream_url}" "${build_dir}/src"
git -C "${build_dir}/src" checkout --quiet "${upstream_commit}"
# The public upstream carries actlize_v1.0.0 and fmt as submodules.  The PPU
# Megamoe extension also needs the public actlize v0.5.0 headers; pin it here
# because that path is intentionally not part of the upstream submodule list.
git -C "${build_dir}/src" submodule update --init --depth 1
mkdir -p "${build_dir}/src/third-party/actlize_v0.5.0"
git clone --quiet --filter=blob:none https://github.com/t-head/actlize.git "${build_dir}/src/third-party/actlize_v0.5.0"
git -C "${build_dir}/src/third-party/actlize_v0.5.0" checkout --quiet 425b8101e3d849c1a7125f4ac0eef39b93c61836
git -C "${build_dir}/src" apply "${root_dir}/patches/0001-megamoe-ppu.patch"
git -C "${build_dir}/src" apply "${root_dir}/patches/0002-register-megamoe-apis.patch"
git -C "${build_dir}/src" apply "${root_dir}/patches/0003-largeep-local-experts.patch"

# The Megamoe fork uses DG_HGGC_CHECK for driver calls, while the public
# upstream names that helper DG_HGGC_DRIVER_CHECK. Keep this compatibility
# alias inside the isolated wheel source only.
cat >> "${build_dir}/src/csrc/utils/exception.hpp" <<'EOF'

#ifndef DG_HGGC_CHECK
#define DG_HGGC_CHECK(cmd) DG_HGGC_DRIVER_CHECK(cmd)
#endif
EOF

"${python_bin}" - "${build_dir}/src" <<'PY'
import pathlib
import re
import shutil
import sys

root = pathlib.Path(sys.argv[1])
src = root / "deep_gemm"
dst = root / "deep_gemm_mega"
shutil.move(src, dst)

for path in dst.rglob("*.py"):
    text = path.read_text()
    text = re.sub(r"\bfrom deep_gemm\b", "from deep_gemm_mega", text)
    text = re.sub(r"\bimport deep_gemm\b", "import deep_gemm_mega as deep_gemm", text)
    text = text.replace("deep_gemm_cpp", "deep_gemm_mega_cpp")
    path.write_text(text)

setup = root / "setup.py"
text = setup.read_text()
text = text.replace("deep_gemm", "deep_gemm_mega")
text = text.replace("include/deep_gemm_mega/", "include/deep_gemm/")
text = text.replace("deep_gemm_mega_tuner", "deep_gemm_tuner")
text = text.replace("name='deep_gemm_mega'", "name='deep-gemm-mega'")
text = text.replace("'include/deep_gemm/**/*',", "'include/deep_gemm/**/*', 'configs/**/*.lut', 'lib/*.so',")
setup.write_text(text)
shutil.copytree(root / "configs", dst / "configs")

# Keep the official wheel's JIT cache and LUT selection independent.
for path in list(dst.rglob("*.py")) + list((root / "csrc").rglob("*.hpp")) + [setup]:
    text = path.read_text().replace("DG_JIT_CACHE_DIR", "DG_MEGA_JIT_CACHE_DIR")
    text = text.replace('".deep_gemm"', '".deep_gemm_mega"')
    text = text.replace("SAIL_DEEPGEMM_LUT_PATH", "RTP_PPU_MEGA_LUT_PATH")
    path.write_text(text)

# Both wheels expose this enum; its registration must stay local to each module.
activation = root / "csrc/apis/activation.hpp"
text = activation.read_text()
text = text.replace('(m, "ActivationQuantMode")', '(m, "ActivationQuantMode", py::module_local())')
activation.write_text(text)

mega_mhc = root / "csrc/jit_kernels/impls/mega_mhc.hpp"
text = mega_mhc.read_text()
text = re.sub(
    r"dg_prof_params\.set_mega_mhc_params\(.*?current_stream\(\)\);",
    "dg_prof_params.set_params(GemmType::DenseGemm, false, \"bf16\", 1, "
    "args.kernel_args.tokens, args.config.hidden, 1, 0, nullptr, current_stream());\n"
    "        dg_prof_params.add_params(\"mode\", \"shifted\");\n"
    "        dg_prof_params.add_params(\"sinkhorn_iters\", args.config.iterations);",
    text,
    flags=re.S,
)
mega_mhc.write_text(text)

device_runtime = root / "csrc/jit/device_runtime.hpp"
text = device_runtime.read_text()
cuda_include = "#include <ATen/cuda/CUDAContext.h>"
cuda_guarded = "#if !defined(DG_TORCH_CUDA_ABI)\n" + cuda_include + "\n#endif"
if text.count(cuda_include) != 1:
    raise SystemExit("unexpected CUDAContext include count")
text = text.replace(cuda_include, cuda_guarded)
macro = "#define PYTORCH_SUPPORTS_GET_ACBLASLT_HANDLE (TORCH_VERSION_MAJOR > 2 or (TORCH_VERSION_MAJOR == 2 and TORCH_VERSION_MINOR >= 3))"
guarded_macro = "#if defined(DG_TORCH_CUDA_ABI)\n#define PYTORCH_SUPPORTS_GET_ACBLASLT_HANDLE 0\n#else\n" + macro + "\n#endif"
if text.count(macro) != 1:
    raise SystemExit("unexpected PyTorch handle macro count")
device_runtime.write_text(text.replace(macro, guarded_macro))

utils = root / "csrc/utils/utils.hpp"
text = utils.read_text()
if '#include "../jit/acarch.h"' not in text:
    text = text.replace('#include "../jit/device_runtime.hpp"', '#include "../jit/acarch.h"\n#include "../jit/device_runtime.hpp"')
marker = "bool is_ppu1v5_device() {"
start = text.index(marker)
end = text.index("\n}\n", start) + 3
if "static acArch_t get_ppu_arch()" not in text:
    text = text[:end] + "\nstatic acArch_t get_ppu_arch() {\n    const auto [major, minor] = device_runtime->get_arch_pair();\n    if (major == 8 and minor == 9) return AC_PPU0015;\n    if (major == 8 and minor == 0) return AC_PPU0010;\n    return AC_PPU_MAX_ARCH;\n}\n" + text[end:]
utils.write_text(text)

mega_mhc = root / "csrc/apis/mega_mhc.hpp"
text = mega_mhc.read_text()
old = "const auto* prop = at::cuda::getDeviceProperties(x.get_device());"
new = "hggcDeviceProp device_prop{};\n    DG_HGGC_RUNTIME_CHECK(hggcGetDeviceProperties(&device_prop, x.get_device()));\n    const auto* prop = &device_prop;"
if text.count(old) != 1:
    raise SystemExit("unexpected mega-mHC device property call count")
mega_mhc.write_text(text.replace(old, new))

for path in (root / "csrc").rglob("*.hpp"):
    text = path.read_text()
    text = text.replace(", ActlizeLib::kV050)", ")")
    path.write_text(text)
PY

"${python_bin}" "${root_dir}/test_transport_initialization.py"
bash "${root_dir}/build_transport.sh" "${build_dir}/src/deep_gemm_mega/lib"
"${python_bin}" "${root_dir}/test_package.py" "${build_dir}/src"
"${python_bin}" -m pip wheel --no-deps --no-build-isolation --wheel-dir "${root_dir}/dist" "${build_dir}/src"
