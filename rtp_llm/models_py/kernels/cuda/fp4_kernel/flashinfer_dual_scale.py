"""RTP's dual-scale variant of the pinned FlashInfer CUTLASS NVFP4 GEMM.

Generate a separate extension from FlashInfer's bundled sources. Its upstream
tile dispatch, autotuning and workspace sizing stay in use, while its
epilogue reads both scale pointers. The installed wheel is read-only.
"""

import re
from dataclasses import replace
from functools import lru_cache

import torch

_HEADERS = (
    "fp4_gemm_cutlass.h",
    "fp4_gemm_cutlass_template.h",
    "fp4_gemm_cutlass_template_sm103.h",
    "fp4_gemm_template_sm100.h",
    "fp4_gemm_template_sm103.h",
)
_SCALE_ARGUMENTS = """
struct RtpNvfp4Scales {
  float const* activation;
  float const* weight = nullptr;
  // The upstream workspace query passes nullptr in place of the scales.
  RtpNvfp4Scales(float const* activation, float const* weight = nullptr)
      : activation(activation), weight(weight) {}
};

template <typename Output>
using RtpNvfp4DualScaleEpilogue = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, Output, float,
                                         cutlass::FloatRoundStyle::round_to_nearest>,
    cutlass::epilogue::fusion::Sm90ScalarBroadcast<float, cute::Stride<cute::_0, cute::_0, cute::_0>, 2>,
    cutlass::epilogue::fusion::Sm90AccFetch>;
"""
_ROW_SCALE_EPILOGUE = """
template <typename Output, typename TileShape>
using RtpNvfp4RowScaleEpilogue = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, Output, float,
                                         cutlass::FloatRoundStyle::round_to_nearest>,
    cutlass::epilogue::fusion::Sm90EVT<
        cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, float, float,
                                             cutlass::FloatRoundStyle::round_to_nearest>,
        cutlass::epilogue::fusion::Sm90ColBroadcast<0, TileShape, float, float,
                                                  cute::Stride<cute::_1, cute::_0, cute::_0>>,
        cutlass::epilogue::fusion::Sm90ScalarBroadcast<float>>,
    cutlass::epilogue::fusion::Sm90AccFetch>;
"""


def _replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"Unsupported FlashInfer NVFP4 source: expected one {old!r}")
    return text.replace(old, new, 1)


def _dual_scale_source(name, source, row_scale):
    # Keep the generated extension's types, headers and entry point separate
    # from FlashInfer's own scalar-alpha module in the same process.
    for header in _HEADERS:
        source = source.replace(f'"flashinfer/gemm/{header}"', f'"rtp_{header}"')
        source = source.replace(f'"{header}"', f'"rtp_{header}"')
    variant = "RowScale" if row_scale else "DualScale"
    source = source.replace("CutlassFp4GemmRunner", f"RtpNvfp4{variant}GemmRunner")
    source = source.replace("DeviceGemmFp4Gemm", f"DeviceGemmNvfp4{variant}")
    source = source.replace("float const* global_sf", "RtpNvfp4Scales global_sf")

    if name == "fp4_gemm_cutlass.h":
        arguments = _SCALE_ARGUMENTS + (_ROW_SCALE_EPILOGUE if row_scale else "")
        source = _replace_once(
            source, "namespace gemm {", "namespace gemm {\n" + arguments
        )
        source = _replace_once(
            source,
            '#include "flashinfer/gemm/cutlass_gemm_configs.h"',
            '#include "flashinfer/gemm/cutlass_gemm_configs.h"\n#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"',
        )
    if name in ("fp4_gemm_template_sm100.h", "fp4_gemm_template_sm103.h"):
        epilogue = (
            "RtpNvfp4RowScaleEpilogue<OutElementType, MmaTileShape>"
            if row_scale
            else "RtpNvfp4DualScaleEpilogue<OutElementType>"
        )
        fusion_args = (
            "fusion_args = {{{global_sf.activation, 1.0f, {}}, "
            "{{1.0f}, {global_sf.weight}, {}}, {}}, {}, {}};"
            if row_scale
            else "fusion_args = {{{1.0f, 1.0f}, {global_sf.activation, global_sf.weight}, {}}, {}, {}};"
        )
        source, replacements = re.subn(
            r"cutlass::epilogue::fusion::LinearCombination<OutElementType, float, void,\s*\\\n\s*float>",
            epilogue,
            source,
        )
        if replacements != 1:
            raise RuntimeError(f"Unsupported FlashInfer epilogue in {name}")
        source = _replace_once(
            source,
            "fusion_args.alpha_ptr = static_cast<ElementCompute const*>(global_sf);",
            fusion_args,
        )
    if name == "fp4_gemm_template_sm103.h":
        # Reserve the aligned SM103 kernel storage beyond the epilogue's
        # reported footprint when selecting the Ultra mainloop stage count.
        source = _replace_once(
            source,
            "sizeof(typename CollectiveEpilogue::SharedStorage))>",
            "sizeof(typename CollectiveEpilogue::SharedStorage) + 1024)>",
        )
    if name in ("fp4_gemm_cutlass.cu", "fp4_gemm_cutlass_sm103.cu"):
        if source.count("TensorView globalScale,") != 3:
            raise RuntimeError(f"Unsupported FlashInfer entry point in {name}")
        source = source.replace(
            "TensorView globalScale,",
            "TensorView globalScale, TensorView weightGlobalScale,",
        )
        source = source.replace(
            "globalScale, m, n, k,", "globalScale, weightGlobalScale, m, n, k,"
        )
        source = _replace_once(
            source,
            "globalScale, out, workspace_buffer, tactic);",
            "globalScale, weightGlobalScale, out, workspace_buffer, tactic);",
        )
        source = _replace_once(
            source,
            "static_cast<float*>(globalScale.data_ptr()),",
            "flashinfer::gemm::RtpNvfp4Scales{static_cast<float const*>(globalScale.data_ptr()), "
            "static_cast<float const*>(weightGlobalScale.data_ptr())},",
        )
        scale_rows = "mat1.size(0)" if row_scale else "1"
        source = _replace_once(
            source,
            "CHECK_INPUT_AND_TYPE(globalScale, dl_float32);",
            "CHECK_INPUT_AND_TYPE(globalScale, dl_float32);\n"
            "  CHECK_INPUT_AND_TYPE(weightGlobalScale, dl_float32);\n"
            f"  TVM_FFI_ICHECK_EQ(globalScale.size(0), {scale_rows});\n"
            "  TVM_FFI_ICHECK_EQ(weightGlobalScale.size(0), 1);",
        )
        source = re.sub(r"\bfp4_gemm\b", "nvfp4_dual_scale_gemm", source)
    return source


@lru_cache(maxsize=None)
def _module(capability, row_scale):
    from flashinfer.jit import env
    from flashinfer.jit.gemm.core import (
        gen_gemm_sm100_module_cutlass_fp4,
        gen_gemm_sm103_module_cutlass_fp4,
    )
    from flashinfer.jit.utils import write_if_different

    if capability not in ((10, 0), (10, 3)):
        raise ValueError(f"Dual-scale NVFP4 requires SM100 or SM103, got {capability}")
    from filelock import FileLock

    scale_kind = "row" if row_scale else "dual"
    name = f"rtp_nvfp4_{scale_kind}_scale_{capability[0]}{capability[1]}"
    generated = env.FLASHINFER_GEN_SRC_DIR / name
    generated.mkdir(parents=True, exist_ok=True)
    # All ranks share a JIT cache on a host. Publish sources once, then let
    # FlashInfer manage its own compilation lock.
    with FileLock(str(generated / "source.lock")):
        upstream = (
            gen_gemm_sm103_module_cutlass_fp4()
            if capability == (10, 3)
            else gen_gemm_sm100_module_cutlass_fp4()
        )
        for header in _HEADERS:
            path = env.FLASHINFER_INCLUDE_DIR / "flashinfer/gemm" / header
            write_if_different(
                generated / f"rtp_{header}",
                _dual_scale_source(header, path.read_text(), row_scale),
            )
        sources = []
        for path in upstream.sources:
            target = generated / path.name
            write_if_different(
                target, _dual_scale_source(path.name, path.read_text(), row_scale)
            )
            sources.append(target)
        return replace(
            upstream,
            name=name,
            sources=sources,
            extra_include_dirs=[generated, *(upstream.extra_include_dirs or [])],
        ).build_and_load()


@lru_cache(maxsize=None)
def _runner(capability, row_scale):
    from flashinfer.gemm import gemm_base

    module = _module(capability, row_scale)

    class DualScaleNvfp4Runner(gemm_base.TunableRunner):
        def get_valid_tactics(self, inputs, profile):
            return list(range(module.fp4_gemm_tactic_num()))

        def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
            a, b, a_scales, b_scales, s_a, _, out, _, _, workspace, s_w = inputs
            module.nvfp4_dual_scale_gemm(
                a,
                b,
                a_scales.view(torch.uint8),
                b_scales.view(torch.uint8),
                s_a,
                s_w,
                out,
                workspace,
                tactic,
            )
            return out

    return DualScaleNvfp4Runner()


def select_runner(inputs, *, row_scale=False):
    from flashinfer.gemm import gemm_base

    runner = _runner(torch.cuda.get_device_capability(inputs[0].device), row_scale)
    return gemm_base.AutoTuner.get().choose_one(
        "rtp_nvfp4_row_scale" if row_scale else "rtp_nvfp4_dual_scale",
        [runner],
        gemm_base._MM_FP4_TUNING_CONFIG_128x4,
        inputs,
    )
