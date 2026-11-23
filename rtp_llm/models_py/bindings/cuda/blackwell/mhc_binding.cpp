#include <pybind11/pybind11.h>
#include <torch/python.h>
#include "csrc/apis/config.hpp"
#include "mhc_runtime.h"
PYBIND11_MODULE(mhc_isolated, m) {
    deep_gemm::config::register_apis(m);
    deep_gemm::mega_mhc::register_apis(m);
    m.def("prepare_mega_mhc_stream",
          &rtp_kernel::blackwell::prepare_mhc_stream,
          "Initialize split barriers on the current CUDA stream before graph capture.");
}
