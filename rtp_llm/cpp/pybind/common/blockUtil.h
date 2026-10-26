#pragma once

#include "rtp_llm/cpp/pybind/PyUtils.h"
#include <vector>

std::vector<int64_t> getBlockCacheKey(const std::vector<std::vector<int64_t>>& token_ids_list,
                                      bool                                     mtp_lookahead = false);

void registerCommon(py::module& m);
