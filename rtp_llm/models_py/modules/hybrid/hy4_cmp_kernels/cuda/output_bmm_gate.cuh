#pragma once
#include <torch/extension.h>
void bind_output_bmm_gate(pybind11::module_& m);
