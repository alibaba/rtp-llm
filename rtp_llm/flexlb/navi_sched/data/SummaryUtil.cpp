#include "rtp_llm/flexlb/navi_sched/data/SummaryUtil.h"
#include <algorithm>
#include <cstdlib>

namespace navi_sched {

int32_t SummaryUtil::getSummaryLevel(const std::string &query)
{
    if (query.empty()) {
        return 1;
    }
    char *end = nullptr;
    auto level = std::strtol(query.c_str(), &end, 10);
    if (!end || *end != '\0') {
        return 1;
    }
    return std::max<int32_t>(0, level);
}

size_t SummaryUtil::getSummaryLimit(int32_t level, size_t size)
{
    return level >= 3 ? size : std::min<size_t>(size, 64);
}

} // namespace navi_sched
