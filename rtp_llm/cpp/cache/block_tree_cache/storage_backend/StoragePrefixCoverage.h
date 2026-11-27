#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_map>

namespace rtp_llm {

// Required pages form a suffix of each group's own page rows. A missing
// earlier page matters only while the group's latest gap is inside that suffix.
class StoragePrefixCoverage {
public:
    void observe(size_t row, const std::string& group, bool available) {
        if (!available) {
            latest_missing_[group] = row;
        }
    }

    template<typename Required>
    bool covers(size_t end, Required&& required) const {
        return std::none_of(latest_missing_.begin(), latest_missing_.end(), [&](const auto& gap) {
            return required(gap.second, end, gap.first);
        });
    }

private:
    std::unordered_map<std::string, size_t> latest_missing_;
};

}  // namespace rtp_llm
