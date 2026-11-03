#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <unordered_set>
#include <vector>

#include "kvcm_client/common.h"
#include "rtp_llm/cpp/utils/DecodeCacheOrigin.h"

namespace rtp_llm::kvcm {

struct DtopOriginMetadata {
    kv_cache_manager::Locations    locations;
    std::vector<DecodeCacheOrigin> origins;
};

// Call with the *required* spec/rank view after ordinary cache-shape checks.
// Unknown/legacy/mixed source affects attribution only, not cache usability.
template<typename Location>
std::optional<DecodeCacheOrigin> readDtopOriginMetadata(const Location& location, size_t key, size_t canonical_unit) {
    const auto max_end = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    if (location.empty() || canonical_unit == 0 || key >= max_end / canonical_unit)
        return std::nullopt;
    const auto                           begin = static_cast<int64_t>(key * canonical_unit);
    const auto                           end   = static_cast<int64_t>((key + 1) * canonical_unit);
    std::optional<DecodeCacheOrigin>     agreed;
    std::unordered_set<std::string_view> seen;
    for (const auto& spec : location) {
        if (spec.spec_name.empty() || !seen.emplace(spec.spec_name).second)
            return std::nullopt;
        auto source = decodeCacheOriginFromUri(spec.uri);
        if (!source || source->begin != begin || source->end != end || (agreed && !(*agreed == *source)))
            return std::nullopt;
        agreed = std::move(source);
    }
    return agreed;
}

// CPU control metadata only. The SDK payload transfer has already completed;
// do not alter its authoritative addresses, key order, spec set or payload.
// The Manager validates the same allocation/session atomically with Serving.
inline std::optional<DtopOriginMetadata> makeDtopOriginMetadata(const kv_cache_manager::Locations& locations,
                                                                const std::string&                 session,
                                                                const std::vector<size_t>&         key_indices,
                                                                size_t                             canonical_unit,
                                                                size_t                             published_end,
                                                                size_t                             computed_end,
                                                                int64_t                            handoff_end) {
    const auto max_end = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    if (locations.empty() || key_indices.size() != locations.size() || canonical_unit == 0 || handoff_end < 0
        || computed_end >= max_end || published_end > computed_end || static_cast<uint64_t>(handoff_end) > computed_end)
        return std::nullopt;
    DtopOriginMetadata result{locations, {}};
    result.origins.reserve(locations.size());
    for (size_t i = 0; i < key_indices.size(); ++i) {
        const auto key = key_indices[i];
        if ((i && key <= key_indices[i - 1]) || key >= max_end / canonical_unit || result.locations[i].empty())
            return std::nullopt;
        const auto begin = key * canonical_unit;
        const auto end   = (key + 1) * canonical_unit;
        if (end > published_end)
            return std::nullopt;
        DecodeCacheOrigin               origin{session,
                                 handoff_end,
                                 static_cast<int64_t>(begin),
                                 static_cast<int64_t>(end),
                                 static_cast<int64_t>(computed_end),
                                 static_cast<int64_t>(computed_end + 1)};
        std::unordered_set<std::string> seen;
        for (auto& spec : result.locations[i]) {
            if (spec.spec_name.empty() || !seen.insert(spec.spec_name).second)
                return std::nullopt;
            auto marked = attachDecodeCacheOrigin(spec.uri, origin);
            if (!marked)
                return std::nullopt;
            spec.uri = std::move(*marked);
        }
        result.origins.push_back(std::move(origin));
    }
    return result;
}

}  // namespace rtp_llm::kvcm
