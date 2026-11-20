// Private binding for the grouped GEMM and its DeepJIT runtime.
#include <pybind11/stl.h>
#include "deep_gemm_sm120/adapter/apis/gemm.hpp"

namespace py = pybind11;

static void init_grouped_jit(const std::string& root_path, const std::string& cutlass_path) {
    const auto root   = std::filesystem::absolute(root_path);
    const auto config = deep_jit::Config(root,
                                         "RTP_DG120",
                                         "deepgemm-522942a-cutlass-" + std::to_string(CUTLASS_VERSION),
                                         {root / "deep_gemm_sm120/shim/include",
                                          root / "deep_gemm_sm120/vendor/deep_gemm/include",
                                          root / "deep_gemm/include",
                                          std::filesystem::absolute(cutlass_path)},
                                         {"deep_gemm/"});
    deep_gemm::jit.init(config);
}

PYBIND11_MODULE(rtp_dg120_grouped, m) {
    m.def("warmup_device", [](int index) {
        const c10::cuda::CUDAGuard device_guard(index);
        (void)deep_gemm::jit.operator->();
    });
    m.def("init", &init_grouped_jit, py::arg("native_root"), py::arg("cutlass_include"));
    m.def("m_grouped_fp8_fp4_gemm_nt_contiguous",
          &deep_gemm::sm120::m_grouped_fp8_fp4_gemm_nt_contiguous,
          py::arg("a"),
          py::arg("b"),
          py::arg("d"),
          py::arg("grouped_layout"),
          py::arg("recipe")                     = std::nullopt,
          py::arg("recipe_a")                   = std::nullopt,
          py::arg("recipe_b")                   = std::nullopt,
          py::arg("compiled_dims")              = "nk",
          py::arg("disable_ue8m0_cast")         = false,
          py::arg("use_psum_layout")            = false,
          py::arg("ensure_zero_padding")        = true,
          py::arg("expected_m_for_psum_layout") = std::nullopt,
          py::arg("epilogue")                   = std::nullopt);
}
