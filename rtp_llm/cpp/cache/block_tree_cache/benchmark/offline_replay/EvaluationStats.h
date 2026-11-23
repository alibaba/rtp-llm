#pragma once
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <map>
#include <sstream>
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/offline_replay/StateIO.h"
namespace rtp_llm::benchmark {
struct LengthDistribution {
    uint64_t                     count{0}, sum{0};
    std::map<uint64_t, uint64_t> histogram;
    void                         add(uint64_t value) {
        ++count;
        sum += value;
        ++histogram[value];
    }
    uint64_t percentile(unsigned percentage) const {
        const uint64_t rank  = (count * percentage + 99) / 100;
        uint64_t       total = 0;
        for (const auto& [value, n] : histogram) {
            total += n;
            if (total >= rank)
                return value;
        }
        return 0;
    }
    std::string json() const {
        std::ostringstream out;
        out << std::setprecision(17);
        out << "{\"count\":" << count << ",\"sum\":" << sum << ",\"mean\":" << (count ? double(sum) / count : 0)
            << ",\"max\":" << (histogram.empty() ? 0 : histogram.rbegin()->first) << ",\"p50\":" << percentile(50)
            << ",\"p95\":" << percentile(95) << ",\"p99\":" << percentile(99) << '}';
        return out.str();
    }
};
struct MatchWindow {
    uint64_t                        requests{0}, input_tokens{0}, zero_full{0};
    double                          ratio_sum{0};
    LengthDistribution              full, joint, gap;
    std::map<std::string, uint64_t> evictions;
    void add(uint64_t tokens, uint64_t f, uint64_t j, const std::map<std::string, uint64_t>& events) {
        stateRequire(j <= f && f <= tokens, "match lengths outside request");
        ++requests;
        input_tokens += tokens;
        zero_full += f == 0;
        if (f)
            ratio_sum += double(j) / f;
        full.add(f);
        joint.add(j);
        gap.add(f - j);
        for (const auto& [name, n] : events)
            evictions[name] += n;
    }
    std::string json() const {
        std::ostringstream out;
        out << std::setprecision(17);
        out << "{\"requests\":" << requests << ",\"input_tokens\":" << input_tokens
            << ",\"zero_full_requests\":" << zero_full << ",\"full_tokens\":" << full.json()
            << ",\"joint_tokens\":" << joint.json() << ",\"gap_tokens\":" << gap.json()
            << ",\"joint_full_ratio_mean\":";
        if (requests > zero_full)
            out << ratio_sum / (requests - zero_full);
        else
            out << "null";
        out << ",\"joint_full_ratio_weighted\":";
        if (full.sum)
            out << double(joint.sum) / full.sum;
        else
            out << "null";
        out << ",\"hit_rate\":" << (input_tokens ? double(joint.sum) / input_tokens : 0) << ",\"evictions\":{";
        bool comma = false;
        for (const auto& [name, n] : evictions) {
            if (comma)
                out << ',';
            comma = true;
            out << '"' << name << "\":" << n;
        }
        out << "}}";
        return out.str();
    }
};
struct EvaluationStats {
    uint64_t                        window_ns{300000000000ULL};
    MatchWindow                     total;
    std::map<uint64_t, MatchWindow> timeline;
    void add(uint64_t ts, uint64_t tokens, uint64_t f, uint64_t j, const std::map<std::string, uint64_t>& events) {
        total.add(tokens, f, j, events);
        timeline[ts / window_ns * window_ns].add(tokens, f, j, events);
    }
    std::string json() const {
        std::ostringstream out;
        out << "{\"window_ns\":" << window_ns << ",\"total\":" << total.json() << ",\"timeline\":[";
        bool comma = false;
        for (const auto& [time, window] : timeline) {
            if (comma)
                out << ',';
            comma = true;
            out << "{\"start_ns\":" << time << ",\"metrics\":" << window.json() << '}';
        }
        out << "]}";
        return out.str();
    }
};
}  // namespace rtp_llm::benchmark
