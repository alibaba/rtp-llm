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

template<typename... CheckModes>
bool checkEplbMode(EplbMode mode, CheckModes... modes) {
    return ((mode == modes) || ...);
}

struct EplbConfig {
    EplbMode mode;
    int      update_time;

    std::vector<int> toList() const {
        std::vector<int> list;
        list.push_back((int)mode);
        list.push_back(update_time);
        return list;
    }

    static EplbConfig fromList(const int* list) {
        EplbConfig data;
        data.mode        = (EplbMode)list[0];
        data.update_time = list[1];
        return data;
    }

    bool operator==(const EplbConfig& other) const {
        return mode == other.mode && update_time == other.update_time;
    }

    bool operator!=(const EplbConfig& other) const {
        return !(*this == other);
    }

    std::string toString() const {
        return "EplbControlData{mode=" + std::to_string((int)mode) + ", update_time=" + std::to_string(update_time)
               + "}";
    }
};

}  // namespace rtp_llm
