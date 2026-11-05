#include <pybind11/pybind11.h>
#include <torch/python.h>
#include "csrc/apis/config.hpp"
#include "csrc/apis/mega_mhc.hpp"
PYBIND11_MODULE(mhc_isolated, m) {
  deep_gemm::config::register_apis(m);
  deep_gemm::mega_mhc::register_apis(m);
}
