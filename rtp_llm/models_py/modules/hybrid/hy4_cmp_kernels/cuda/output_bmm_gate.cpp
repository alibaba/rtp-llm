#include "output_bmm_gate.cuh"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { bind_output_bmm_gate(m); }
