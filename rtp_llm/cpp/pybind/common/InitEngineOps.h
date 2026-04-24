#pragma once
#include <torch/extension.h>
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

void initEngine(std::string py_ft_alog_file_path);

}  // namespace rtp_llm
