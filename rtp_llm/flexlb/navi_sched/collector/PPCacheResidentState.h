#pragma once

#include "aios/navi_sched/collector/PPCacheCollectorState.h"
#include <string>

namespace navi_sched {

enum class PPCacheGroupRole { FULL, TAIL };
struct PPCacheResidentObject {
    std::string objectId;
    std::string identity;
};

// Adapter-owned metadata index, driven only by sealed-resident object events.
// A native object may contain several hashes; overlapping objects are counted.
// Calls are serialized by the Collector so publication preserves event order.
class PPCacheResidentState {
public:
    PPCacheResidentState(std::vector<PPCacheGroupRole> roles,
                         std::vector<uint32_t> blockScales);
    PPCacheShardUpdate resident(const PPCacheResidentObject &object);
    PPCacheShardUpdate removed(const std::string &objectId);
    PPCacheShardUpdate replace(const std::vector<PPCacheResidentObject> &objects);

private:
    struct Metadata {
        std::string identity;
        uint32_t group = 0;
        std::vector<uint64_t> hashes;
    };
    using Counts = std::unordered_map<uint64_t, size_t>;
    Metadata parse(const std::string &identity) const;
    bool ready(uint64_t hash, PPCacheGroupRole role) const;
    PPCacheShardUpdate change(const std::string &id, const Metadata *replacement);
    void add(const Metadata &metadata);
    void erase(const Metadata &metadata);
    std::vector<PPCacheGroupRole> _roles;
    std::vector<uint32_t> _blockScales;
    std::vector<Counts> _counts;
    std::unordered_map<std::string, Metadata> _objects;
};

}
