#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace navi_sched {

class SummaryUtil {
public:
    static int32_t getSummaryLevel(const std::string &query);
    static size_t getSummaryLimit(int32_t level, size_t size);
};

} // namespace navi_sched
