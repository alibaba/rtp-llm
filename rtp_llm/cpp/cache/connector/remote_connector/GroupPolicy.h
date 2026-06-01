#pragma once

#include <memory>
#include <vector>
#include <set>
#include <string>
#include <map>
#include <unordered_map>
#include "kvcm_client/common.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

class KVCacheAllocator;

namespace remote_connector {

struct LocationSpecUnitView {
    LocationSpecUnitView(const kv_cache_manager::LocationSpecUnit& unit): spec_name(unit.spec_name), uri(unit.uri) {}
    std::string_view spec_name;
    std::string_view uri;
};
using LocationView  = std::vector<LocationSpecUnitView>;
using LocationsView = std::vector<LocationView>;

class GroupPolicy {
public:
    struct Group {
        bool        is_full            = true;
        uint64_t    group_name_bithash = 0;
        std::string group_name;
    };
    using GroupIdMap = std::map<int32_t, Group>;
    struct SpecInfo {
        int32_t group_id;
        int32_t tp_rank;
    };
    using SpecInfoMap = std::map<std::string, SpecInfo, std::less<>>;

    GroupPolicy(std::shared_ptr<KVCacheAllocator> allocator,
                const std::vector<int32_t>&       full_group_ids,
                const std::vector<int32_t>&       other_group_ids):
        allocator_(allocator),
        full_group_ids_(full_group_ids.begin(), full_group_ids.end()),
        other_group_ids_(other_group_ids.begin(), other_group_ids.end()) {}
    virtual ~GroupPolicy() = default;

    virtual bool init() = 0;

    // Build the filtered Locations view used for remote loads. When `resource` is non-null,
    // spec units whose owning group has NULL_BLOCK_IDX at the corresponding cache_key index
    // are dropped (required for DSV4 + linear_step where SWA / state ring-buffer groups
    // carry NULL_BLOCK_IDX padding at non-step positions).
    virtual bool filterNeedLoadLocations(const kv_cache_manager::Locations&      locations,
                                         LocationsView&                          locations_view,
                                         const std::shared_ptr<KVCacheResource>& resource   = nullptr,
                                         kv_cache_manager::BlockMaskOffset       block_mask = 0) const = 0;

    virtual bool getNeedWriteGroups(const std::shared_ptr<KVCacheResource>& resource,
                                    std::vector<std::string>&               location_spec_group_names) const = 0;

    virtual bool genBlockBuffers(const std::vector<int32_t>&     group_ids,
                                 const std::vector<int32_t>&     block_ids,
                                 kv_cache_manager::BlockBuffers& block_buffers) const = 0;

    const GroupIdMap& groups() const {
        return groups_;
    }

    void addLocationSpecGroup(uint64_t bithash, const std::string& location_spec_group_name) {
        location_spec_group_map_[bithash] = location_spec_group_name;
    }
    virtual bool       addSpecInfo(const std::string& spec_name, int32_t group_id, int32_t tp_rank);
    const SpecInfoMap& spec_info_map() const {
        return spec_name_to_info_;
    }
    virtual std::string debugString() const;

protected:
    // Return true when `unit`'s owning group has NULL_BLOCK_IDX at cache_key index `key_idx`.
    // Used by filterNeedLoadLocations subclasses to skip spec units that have no local block
    // to load into. Returns false when `resource` is null (no local state to compare against).
    bool isUnitNullAtKey(const kv_cache_manager::LocationSpecUnit& unit,
                         size_t                                    key_idx,
                         const std::shared_ptr<KVCacheResource>&   resource) const;

protected:
    std::shared_ptr<KVCacheAllocator> allocator_;
    std::set<int32_t>                 full_group_ids_;
    std::set<int32_t>                 other_group_ids_;

    // group_id -> group
    GroupIdMap groups_;
    // max support 64 groups, contains all group combinations
    std::unordered_map<uint64_t, std::string> location_spec_group_map_;
    // spec_name -> spec_info
    SpecInfoMap spec_name_to_info_;
};

class DefaultLayerGroupPolicy: public GroupPolicy {
public:
    DefaultLayerGroupPolicy(std::shared_ptr<KVCacheAllocator> allocator,
                            const std::vector<int32_t>&       full_group_ids,
                            const std::vector<int32_t>&       other_group_ids):
        GroupPolicy(allocator, full_group_ids, other_group_ids) {}

    virtual bool init() override;

    virtual bool filterNeedLoadLocations(const kv_cache_manager::Locations&      locations,
                                         LocationsView&                          locations_view,
                                         const std::shared_ptr<KVCacheResource>& resource   = nullptr,
                                         kv_cache_manager::BlockMaskOffset       block_mask = 0) const override;

    virtual bool getNeedWriteGroups(const std::shared_ptr<KVCacheResource>& resource,
                                    std::vector<std::string>&               location_spec_group_names) const override;

    bool genBlockBuffers(const std::vector<int32_t>&     group_ids,
                         const std::vector<int32_t>&     block_ids,
                         kv_cache_manager::BlockBuffers& block_buffers) const override;

    std::string debugString() const override;

protected:
    virtual std::string GetOtherGroupPrefixName() const {
        return "G";
    }

    std::map<int32_t, std::vector<int>> group_to_layer_ids_;
    // DSV4: group_id -> KVCacheRegionName mapping for multi-pool buffer access
    std::map<int32_t, KVCacheRegionName> group_to_region_name_;
};

class FullLayerGroupPolicy: public DefaultLayerGroupPolicy {
public:
    FullLayerGroupPolicy(std::shared_ptr<KVCacheAllocator> allocator,
                         const std::vector<int32_t>&       full_group_ids,
                         const std::vector<int32_t>&       other_group_ids):
        DefaultLayerGroupPolicy(allocator, full_group_ids, other_group_ids) {}
    bool init() override;

    bool getNeedWriteGroups(const std::shared_ptr<KVCacheResource>& resource,
                            std::vector<std::string>&               location_spec_group_names) const override;
};

// Policy for hybrid full + linear (e.g. SWA / state ring-buffer) layouts. Each cache key
// is either "full only" or "full + every linear group" (full_linear); partial linear is
// an error. Writes are emitted unconditionally for whichever shape the upper layer
// provides (was previously gated by write_interval_, which was always 1 in production).
//
// filterNeedLoadLocations walks the locations backward and treats the LAST full_linear
// position as the anchor. The 4 per-position helpers (IsValidFullLocation /
// CheckInvalidFullLocationAndSetView / CheckInvalidFullLinearLocationAndSetView /
// SkipLinearSpecAndSetView) each drop spec units whose owning group has NULL_BLOCK_IDX
// at this cache_key (sparse SWA / state ring-buffer locally), and reset
// exist_linear_location on any such drop — this enforces the invariant that the last
// loaded location must be a clean full_linear (no null drops). CheckInvalidFullLinear
// additionally clears its view on drops so the position cannot serve as the anchor and
// is not partially loaded.
class FullLinearLayerGroupPolicy: public DefaultLayerGroupPolicy {
public:
    FullLinearLayerGroupPolicy(std::shared_ptr<KVCacheAllocator> allocator,
                               const std::vector<int32_t>&       full_group_ids,
                               const std::vector<int32_t>&       linear_group_ids):
        DefaultLayerGroupPolicy(allocator, full_group_ids, linear_group_ids) {}

    bool init() override;

    bool addSpecInfo(const std::string& spec_name, int32_t group_id, int32_t tp_rank) override;

    bool filterNeedLoadLocations(const kv_cache_manager::Locations&      locations,
                                 LocationsView&                          locations_view,
                                 const std::shared_ptr<KVCacheResource>& resource   = nullptr,
                                 kv_cache_manager::BlockMaskOffset       block_mask = 0) const override;

    bool getNeedWriteGroups(const std::shared_ptr<KVCacheResource>& resource,
                            std::vector<std::string>&               location_spec_group_names) const override;

    std::string debugString() const override;

protected:
    std::string GetOtherGroupPrefixName() const override {
        return "L";
    }

    bool IsValidFullLocation(const kv_cache_manager::Location&       location,
                             size_t                                  key_idx,
                             const std::shared_ptr<KVCacheResource>& resource,
                             bool&                                   exist_linear_location) const;
    bool CheckInvalidFullLocationAndSetView(const kv_cache_manager::Location&       location,
                                            size_t                                  key_idx,
                                            const std::shared_ptr<KVCacheResource>& resource,
                                            LocationView&                           location_view,
                                            bool&                                   exist_linear_location) const;
    bool CheckInvalidFullLinearLocationAndSetView(const kv_cache_manager::Location&       location,
                                                  size_t                                  key_idx,
                                                  const std::shared_ptr<KVCacheResource>& resource,
                                                  LocationView&                           location_view,
                                                  bool&                                   exist_linear_location) const;
    bool SkipLinearSpecAndSetView(const kv_cache_manager::Location&       location,
                                  size_t                                  key_idx,
                                  const std::shared_ptr<KVCacheResource>& resource,
                                  LocationView&                           location_view,
                                  bool&                                   exist_linear_location) const;

protected:
    uint64_t                        valid_full_bithash_        = 0;
    uint64_t                        valid_full_linear_bithash_ = 0;
    std::map<std::string, uint64_t> full_spec_name_bithash_;
    std::map<std::string, uint64_t> full_linear_spec_name_bithash_;
};

}  // namespace remote_connector
}  // namespace rtp_llm
