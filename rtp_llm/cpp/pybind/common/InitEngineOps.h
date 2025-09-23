#pragma once
#include <torch/extension.h>
#include "rtp_llm/cpp/utils/Logger.h"

namespace torch_ext {

void initEngine(std::string py_ft_alog_file_path);

}  // namespace torch_ext
