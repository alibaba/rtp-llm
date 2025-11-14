#include "rtp_llm/cpp/config/EplbConfig.h"
#include <sstream>

namespace rtp_llm {

// Static conversion methods for EplbMode
EplbMode EPLBConfig::from_string(const std::string& str) {
    if (str.empty() || str == "NONE" || str == "none") {
        return EplbMode::NONE;
    } else if (str == "STATS" || str == "stats") {
        return EplbMode::STATS;
    } else if (str == "EPLB" || str == "eplb") {
        return EplbMode::EPLB;
    } else if (str == "ALL" || str == "all") {
        return EplbMode::ALL;
    } else {
        return EplbMode::NONE;  // Default to NONE for unknown values
    }
}

std::string EPLBConfig::to_string(EplbMode mode) {
    switch (mode) {
        case EplbMode::NONE:
            return "NONE";
        case EplbMode::STATS:
            return "STATS";
        case EplbMode::EPLB:
            return "EPLB";
        case EplbMode::ALL:
            return "ALL";
        default:
            return "NONE";
    }
}

std::string EPLBConfig::to_string() const {
    std::ostringstream oss;
    oss << "  enable_eplb: " << enable_eplb() << "\n"
        << "  eplb_update_time: " << eplb_update_time << "\n"
        << "  eplb_mode: " << (int)eplb_mode << "\n"
        << "  redundant_expert: " << redundant_expert << "\n"
        << "  balance_method: " << balance_method << "\n"
        << "  eplb_force_repack: " << eplb_force_repack << "\n"
        << "  eplb_stats_window_size: " << eplb_stats_window_size << "\n"
        << "  eplb_control_step: " << eplb_control_step << "\n"
        << "  eplb_test_mode: " << eplb_test_mode << "\n"
        << "  eplb_balance_layer_per_step: " << eplb_balance_layer_per_step;
    return oss.str();
}

}  // namespace rtp_llm

