#pragma once
#include <torch/custom_class.h>
#include <torch/script.h>
#include "src/fastertransformer/utils/logger.h"

namespace torch_ext {

void setDebugLogLevel(bool debug);
void setDebugPrintLevel(bool debug);

}  // namespace torch_ext
