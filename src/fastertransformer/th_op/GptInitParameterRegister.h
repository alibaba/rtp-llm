#pragma once

#include <torch/custom_class.h>
#include <torch/script.h>
#include <torch/extension.h>

namespace fastertransformer {

void registerGptInitParameter(py::module m);

}
