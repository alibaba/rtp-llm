#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace rtp_llm {

inline bool nativeMtpPrefillChunkTopologySupported(int64_t tp, int64_t ep, int64_t dp) {
    // Independent TP groups may need different chunk counts. Until rounds
    // are coordinated across DP, their expert collectives must stay local.
    return tp > 1 && dp == 1 && ep == tp;
}

inline bool nativeMtpPrefillChunkPdSupported(bool role_pd_marker, bool has_request_flags, bool any_request_pd) {
    // GptModelInputs::pd_separation identifies a PREFILL role, including
    // standalone local requests. Only the synchronized per-request mask
    // determines whether the batch actually publishes PD cache state.
    // Without that mask, retain the conservative legacy role fallback.
    return has_request_flags ? !any_request_pd : !role_pd_marker;
}

struct MtpPrefillSlice {
    int64_t request;
    int64_t source_offset;
    int64_t request_offset;
    int64_t length;
    int64_t prefix;
    bool terminal;
};

// A budget is shared by the packed batch, rather than multiplied by BS.
// Input tokens have already been shifted independently by the MTP executor.
inline std::vector<std::vector<MtpPrefillSlice>> planMtpPrefillChunks(
    const std::vector<int64_t>& lengths, const std::vector<int64_t>& prefixes,
    int64_t budget, int64_t alignment) {
    if (alignment <= 0 || budget < alignment || budget % alignment || lengths.empty()
        || lengths.size() != prefixes.size()) {
        throw std::invalid_argument("Invalid native MTP prefill chunk geometry");
    }
    std::vector<std::vector<MtpPrefillSlice>> rounds;
    std::vector<MtpPrefillSlice> round;
    int64_t remaining = budget, source = 0;
    for (size_t request = 0; request < lengths.size(); ++request) {
        if (lengths[request] <= 0 || prefixes[request] < 0)
            throw std::invalid_argument("MTP chunk requires positive Q and nonnegative prefix");
        int64_t offset = 0;
        while (offset < lengths[request]) {
            int64_t take = std::min(remaining, lengths[request] - offset);
            if (offset + take < lengths[request])
                take = ((offset + take) / alignment) * alignment - offset;
            if (take <= 0) {
                rounds.push_back(std::move(round));
                round = {};
                remaining = budget;
                continue;
            }
            round.push_back({static_cast<int64_t>(request), source + offset, offset, take,
                             prefixes[request] + offset, offset + take == lengths[request]});
            offset += take;
            remaining -= take;
            if (!remaining) {
                rounds.push_back(std::move(round));
                round = {};
                remaining = budget;
            }
        }
        source += lengths[request];
    }
    if (!round.empty()) rounds.push_back(std::move(round));
    return rounds;
}
}  // namespace rtp_llm
