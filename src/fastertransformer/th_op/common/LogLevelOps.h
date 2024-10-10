#pragma once
#include <torch/custom_class.h>
#include <torch/script.h>
#include "src/fastertransformer/utils/logger.h"

namespace torch_ext {

bool setLogLevel(const std::string& log_level_str);

}  // namespace torch_ext
