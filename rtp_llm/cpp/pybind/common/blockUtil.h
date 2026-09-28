#pragma once

#include "rtp_llm/cpp/pybind/PyUtils.h"
#include <vector>

std::vector<int64_t> getBlockCacheKey(const std::vector<std::vector<int64_t>>& token_ids_list,
                                    int64_t initial_hash = 0);

void registerCommon(py::module& m);
