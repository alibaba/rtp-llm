#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

// Canonical cross-boundary order for cache groups.
//
// The semantic tag is the only cache-group identity. Local C++ record order --
// CacheConfig group order, request-resource slot order, map iteration order --
// carries no business meaning and must never reach a positional payload.
//
// Whenever a tensor, bitmask or wire buffer needs one entry per cache group, the
// producing adapter and the consuming adapter each derive the entry order from
// sortedCacheGroupTags() and permute their own parallel payload in that same
// function. Because both sides sort independently, no ordering has to be
// transmitted and reordering the declaration records cannot change the binding.
//
// The resulting entry index is an adapter-local `group_index`: it is never
// stored, returned, bound to Python, or serialized as identity.
inline std::vector<std::string> sortedCacheGroupTags(const std::vector<std::string>& tags,
                                                     const char*                     what = "cache group") {
    std::vector<std::string> sorted(tags);
    std::sort(sorted.begin(), sorted.end());
    for (const auto& tag : sorted) {
        RTP_LLM_CHECK_WITH_INFO(!tag.empty(), "%s tag must not be empty at a positional boundary", what);
    }
    const auto duplicate = std::adjacent_find(sorted.begin(), sorted.end());
    if (duplicate != sorted.end()) {
        RTP_LLM_FAIL("duplicate %s tag=%s at a positional boundary", what, duplicate->c_str());
    }
    return sorted;
}

// Adapter-local entry index of `tag` inside a sorted boundary order. Callers use
// the result only to address the entry they pack or unpack in the same function.
inline size_t
groupIndexForTag(const std::vector<std::string>& sorted_tags, std::string_view tag, const char* what = "cache group") {
    const auto it = std::lower_bound(sorted_tags.begin(), sorted_tags.end(), tag);
    if (it == sorted_tags.end() || *it != tag) {
        RTP_LLM_FAIL("unknown %s tag=%s at a positional boundary", what, std::string(tag).c_str());
    }
    return static_cast<size_t>(std::distance(sorted_tags.begin(), it));
}

// PP column projection: maps each local cache group tag to the column carrying
// it in a payload produced by another stage. Resolution is by tag name, never by
// position, since the producer may own more groups than the consumer.
inline std::vector<size_t> projectCacheGroupColumns(const std::vector<std::string>& local_tags,
                                                    const std::vector<std::string>& input_tags,
                                                    const char*                     what = "cache group") {
    std::vector<size_t> columns;
    columns.reserve(local_tags.size());
    for (const auto& tag : local_tags) {
        const auto it = std::find(input_tags.begin(), input_tags.end(), tag);
        RTP_LLM_CHECK_WITH_INFO(it != input_tags.end(),
                                "%s tag=%s has no column in the incoming payload (which carries %zu columns)",
                                what,
                                tag.c_str(),
                                input_tags.size());
        columns.push_back(static_cast<size_t>(std::distance(input_tags.begin(), it)));
    }
    return columns;
}

}  // namespace rtp_llm
