#pragma once

#include <array>
#include <cmath>
#include <cstddef>

namespace navi_sched {
// Predicted workload moments only: count, context, remaining, remaining^2,
// context*remaining. Counts/tokens in observed workload remain integer facts.
struct NaviSchedExpectedKVWorkload {
    std::array<double, 5> running{};
    std::array<double, 5> waiting{};
    double precedingRequests = 0.0;
    double batchCount = 0.0;
    bool valid() const {
        const auto ok=[](double v){return std::isfinite(v) && v >= 0.0;};
        for (auto v:running) if (!ok(v)) return false;
        for (auto v:waiting) if (!ok(v)) return false;
        return ok(precedingRequests) && ok(batchCount);
    }
    bool empty() const {
        for (auto v:running) if (v != 0.0) return false;
        for (auto v:waiting) if (v != 0.0) return false;
        return precedingRequests == 0.0 && batchCount == 0.0;
    }
    double hitTokens() const { return running[1]+waiting[1]; }
    double missTokens() const { return running[2]+waiting[2]; }
    bool append(const NaviSchedExpectedKVWorkload &other) {
        auto result=*this;
        for(size_t i=0;i<5;++i){result.running[i]+=other.running[i];result.waiting[i]+=other.waiting[i];}
        result.precedingRequests+=other.precedingRequests;result.batchCount+=other.batchCount;
        if(!result.valid()) return false;
        *this=result;return true;
    }
    bool subtract(const NaviSchedExpectedKVWorkload &other) {
        auto result=*this;
        for(size_t i=0;i<5;++i){result.running[i]-=other.running[i];result.waiting[i]-=other.waiting[i];}
        result.precedingRequests-=other.precedingRequests;result.batchCount-=other.batchCount;
        if(!result.valid()) return false;
        *this=result;return true;
    }
};
} // namespace navi_sched
