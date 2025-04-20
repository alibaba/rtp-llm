#pragma once

namespace fastertransformer {

enum class EplbMode {
    NONE,
    STATS,  // stats, only
    EPLB,   // load balance, only
    ALL     // stats + load balance
};

}  // namespace fastertransformer