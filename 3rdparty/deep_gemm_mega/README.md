# DeepGEMM MegaMoE for PPU

This directory builds the PPU-only `deep-gemm-mega` distribution. It starts
from the public Alibaba/T-Head DeepGEMM-for-sail source pinned in `UPSTREAM`
and applies the local MegaMoE patch.

The resulting Python package is `deep_gemm_mega`; the regular `deep_gemm`
package remains the official PPU wheel used by HC and other PPU kernels.

The patch is intentionally applied at source-build time. A prebuilt
`deep_gemm` wheel cannot gain the MegaMoE native extension by changing its
metadata or replacing a header file.

Build requirements are the PPU SDK, the matching CUDA-ABI PyTorch, git, and
the Python build frontend available in the target PPU image.

Build in the target PPU environment (including `setuptools-scm`, `wheel`,
`ninja`, and `packaging`):

```bash
PYTHON_BIN=python3 DG_TORCH_CUDA_ABI=1 bash 3rdparty/deep_gemm_mega/build_wheel.sh
python3 -m pip install --no-deps --force-reinstall 3rdparty/deep_gemm_mega/dist/deep_gemm_mega-*.whl
python3 3rdparty/deep_gemm_mega/check_install.py
```

The build runs source/package contracts before native compilation. Install the
wheel alongside the official PPU `deep_gemm` wheel in PPU images only. The install
check verifies separate file ownership, both import orders, MegaMoE entry points,
all native interfaces used by the Python wrapper, a host-only buffer-size call,
and the packaged LUT. It does not validate M890P kernel numerics or performance.

The upstream pin and MegaMoE patch are taken from PR #1561 at
`53bf97072085d124879554e19512647c14825f56`. Build-time adjustments keep Python
imports and package-data paths consistent after renaming, and preserve separate
JIT caches (`DG_MEGA_JIT_CACHE_DIR`) and LUT selection (`RTP_PPU_MEGA_LUT_PATH`).
The follow-up `0002` patch registers the MegaMoE APIs in the native extension;
the donor patch includes their implementation but omits the module registration.

## LargeEP Decode

`0003-largeep-local-experts.patch` extends the validated M890P Flash local
kernel layout to 8/16 experts per rank for hierarchical EP32/EP16. The EP8
kernel conditions remain supported. The wheel also builds and packages
`lib/libsparse_gin.so` and `lib/libsignal_gin.so`; this requires the PPU SDK's
PCCL device/GPU-initiated RDMA headers. RTP-LLM loads these libraries from the
isolated wheel, not from the official `deep_gemm` package.

The retained runtime configurations all use TP1:

| Decode | Transport | Additional optimization settings |
| --- | --- | --- |
| D8 | Existing local MegaMoE path | Keep the PR's baseline; rejected D8 experiments are omitted |
| D16 | `DSV4_PPU_MEGAMOE_TRANSPORT=collective` | `DSV4_PPU_EP16_EXACT_CAPACITY=1`, `DSV4_PPU_HCA_SPLIT=1`, `DSV4_PPU_WOA_SMALL_TILE=1` |
| D32 | `DSV4_PPU_MEGAMOE_TRANSPORT=sparse` | `DSV4_PPU_MEGAMOE_SIGNAL_ONLY=1`, `DSV4_PPU_HCA_SPLIT=1`, `DSV4_PPU_WOA_SMALL_TILE=1` |

Use `DSV4_PPU_DECODE_MOE_BACKEND=megamoe`. Preserve the existing kernel shape
guards. Admission capacity must cover CUDA Graph's padded batch size: for
example, three requests per rank require capacity at least four. The exact
EP16 transport capacity option does not remove this graph-buffer requirement.

The optimization evidence was collected on the earlier `50ac8f3b0` runtime.
Porting it onto this upstream-based build does not establish new EP16/EP32
native numerical or performance qualification; rerun those checks after
building the resulting wheel.
