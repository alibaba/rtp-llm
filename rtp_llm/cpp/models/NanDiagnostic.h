#pragma once

#include <cstdint>
#include <functional>
#include <iterator>
#include <string>
#include <vector>

namespace rtp_llm {

struct NanDiagnosticEvent {
    int32_t              request_index = -1;
    std::string          trace_id;
    std::string          phase;
    std::string          model_role;
    std::string          stage;
    int32_t              layer_id        = -1;
    int64_t              iteration       = 0;
    int64_t              first_bad_index = -1;
    int64_t              n_nan           = 0;
    int64_t              n_inf           = 0;
    bool                 cuda_graph      = false;
    std::string          dtype;
    std::vector<int64_t> shape;
};

using NanDiagnostics       = std::vector<NanDiagnosticEvent>;
using NanDiagnosticLoader  = std::function<NanDiagnostics()>;
using NanDiagnosticLoaders = std::vector<NanDiagnosticLoader>;

inline NanDiagnostics loadNanDiagnostics(const NanDiagnosticLoaders& loaders) {
    NanDiagnostics result;
    for (const auto& load : loaders) {
        auto events = load();
        result.insert(result.end(), std::make_move_iterator(events.begin()), std::make_move_iterator(events.end()));
    }
    return result;
}

inline void appendNanDiagnosticLoaders(NanDiagnosticLoaders& target, NanDiagnosticLoaders& source) {
    target.insert(target.end(), std::make_move_iterator(source.begin()), std::make_move_iterator(source.end()));
    source.clear();
}

inline NanDiagnostics nanDiagnosticsForRequest(const NanDiagnostics& events, int32_t begin, int32_t size) {
    NanDiagnostics result;
    for (const auto& event : events) {
        if (event.request_index < 0 || (event.request_index >= begin && event.request_index < begin + size)) {
            result.push_back(event);
        }
    }
    return result;
}

}  // namespace rtp_llm
