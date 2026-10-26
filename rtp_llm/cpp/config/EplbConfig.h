#pragma once
#include <cstdint>
#include <string>

namespace rtp_llm {

enum class EplbMode {
    NONE,
    STATS,  // Sample only; allow online switching to EPLB or ALL.
    EPLB,   // Sample and rearrange experts.
    ALL     // Same runtime behavior as EPLB.
};

struct EPLBConfig {
    // Zero disables rearrangement; positive intervals require 1 <= window <= interval.
    int64_t  eplb_update_time             = 0;
    EplbMode eplb_mode                    = EplbMode::NONE;
    int64_t  redundant_expert             = 0;
    int64_t  eplb_stats_window_size       = 100;
    int      eplb_control_step            = 100;
    bool     eplb_rearrange_after_prefill = false;
    int      eplb_rearrange_p2p_groups    = 1;
    // Independent of eplb_mode. Fixed before graph capture; 0 disables collection.
    int eplb_metrics_report_interval    = 0;
    int eplb_metrics_snapshot_max_files = 0;
    template<typename... CheckModes>
    bool checkEplbMode(EplbMode mode, CheckModes... modes) {
        return ((mode == modes) || ...);
    }

    // STATS also requires a supported runtime model and full EP=WORLD experts.
    // Independent monitoring uses NONE with eplb_metrics_report_interval > 0.
    bool enable_eplb() const {
        return eplb_mode != EplbMode::NONE;
    }
    
    int64_t phy_exp_num(int64_t expert_num) const {
        return redundant_expert + expert_num;
    }

    // Static conversion methods for EplbMode
    static EplbMode from_string(const std::string& str);
    static std::string to_string(EplbMode mode);
    std::string to_string() const;
};

}  // namespace rtp_llm




