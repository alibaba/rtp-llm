#pragma once
#include <torch/custom_class.h>
#include <torch/script.h>
#include "maga_transformer/cpp/utils/Logger.h"

namespace torch_ext {

bool setLogLevel(const std::string& log_level_str);

}  // namespace torch_ext
