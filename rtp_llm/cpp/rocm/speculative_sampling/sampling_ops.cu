#include <sampling.cuh>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("chain_speculative_sampling", &chain_speculative_sampling, "chain_speculative_sampling");
}
