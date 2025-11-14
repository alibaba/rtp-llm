#pragma once
#include <string>
#include <vector>

namespace rtp_llm {

enum class EplbMode {
    NONE,
    STATS,  // stats, only
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
    template<typename... CheckModes>
    bool checkEplbMode(EplbMode mode, CheckModes... modes) {
        return ((mode == modes) || ...);
    }

    std::vector<int> toList() const {
        std::vector<int> list;
        list.push_back((int)eplb_mode);
        list.push_back((int)eplb_update_time);
        return list;
    }

    static EPLBConfig fromList(const int* list) {
        EPLBConfig data;
        data.eplb_mode = (EplbMode)list[0];
        data.eplb_update_time = list[1];
        return data;
    }

    // Getter methods
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




