#pragma once

namespace rtp_llm {

enum class EplbMode {
    NONE,
    STATS,  // stats, only
    EPLB,   // load balance, only
    ALL     // stats + load balance
};

}  // namespace rtp_llm