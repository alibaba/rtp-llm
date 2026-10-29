#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/utils/DecodeCacheOrigin.h"

namespace rtp_llm {

// Metadata only; serialized by the owning physical block pool's existing
// mutex. A physical ID is not an identity without its allocation incarnation.
// Construct only for feature-on pools; no global cache-key/address registry.
class PhysicalDecodeOrigins {
public:
    using Origin = std::shared_ptr<const DecodeCacheOrigin>;
    explicit PhysicalDecodeOrigins(size_t capacity): incarnations_(capacity, 0) {}
    PhysicalDecodeOrigins(const PhysicalDecodeOrigins&)            = delete;
    PhysicalDecodeOrigins& operator=(const PhysicalDecodeOrigins&) = delete;

    uint64_t beginAllocation(size_t block) {
        auto& incarnation = incarnations_.at(block);
        origins_.erase(block);
        // Exhausting the metadata sequence must not wrap and accept a stale
        // callback. Leave attribution unavailable without changing KV data.
        incarnation = next_ == std::numeric_limits<uint64_t>::max() ? 0 : ++next_;
        return incarnation;
    }
    void release(size_t block) {
        incarnations_.at(block) = 0;
        origins_.erase(block);
    }
    uint64_t incarnation(size_t block) const {
        return incarnations_.at(block);
    }
    bool bind(size_t block, uint64_t expected, std::string_view region, Origin origin) {
        if (expected == 0 || incarnation(block) != expected || region.empty() || region.size() > 256)
            return false;
        if (origin) {
            try {
                origins_[block][std::string(region)] = std::move(origin);
            } catch (const std::bad_alloc&) {
                return false;  // Missing telemetry must not fail a completed read.
            }
        } else {
            const auto found = origins_.find(block);
            if (found != origins_.end()) {
                const auto tag = found->second.find(region);
                if (tag != found->second.end())
                    found->second.erase(tag);
                if (found->second.empty())
                    origins_.erase(found);
            }
        }
        return true;
    }
    Origin get(size_t block, std::string_view region) const {
        if (incarnation(block) == 0)
            return {};
        const auto found = origins_.find(block);
        if (found == origins_.end())
            return {};
        const auto tag = found->second.find(region);
        return tag == found->second.end() ? Origin{} : tag->second;
    }

private:
    uint64_t              next_{0};
    std::vector<uint64_t> incarnations_;
    // A packed physical page can hold several groups with different logical
    // positions. They share an incarnation but never overwrite each other's
    // provenance. Region identities come from validated model group tags.
    using Regions = std::map<std::string, Origin, std::less<>>;
    std::unordered_map<size_t, Regions> origins_;
};

}  // namespace rtp_llm
