#pragma once
#include <cstdint>
#include <string>

namespace rtp_llm {

enum class EplbMode {
    NONE,
    STATS,  // Sample only; prebuild the same resources as EPLB for online switching.
    EPLB,   // load balance, only
    ALL     // stats + load balance
};

struct EPLBConfig {
    int64_t          eplb_update_time = 5000;
    EplbMode         eplb_mode        = EplbMode::NONE;
    int64_t          redundant_expert = 0;
    std::string      balance_method = "mix";
    int64_t          eplb_force_repack = 0;
    int64_t          eplb_stats_window_size = 10;
    int              eplb_control_step = 100;
    bool             eplb_test_mode = false;
    int              eplb_balance_layer_per_step = 1;
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




