#pragma once

#include <vector>
#include <set>
#include <string>
#include <map>
#include <unordered_map>
#include "kvcm_client/common.h"
#include "rtp_llm/cpp/cache_new/BatchKVCacheResource.h"

namespace rtp_llm {

enum class RemoteConnectorGroupMode {
    RCGM_LAYER_DEFAULT,
    RCGM_ONLY_FULL_LAYER,
    RCGM_FULL_LINEAR_LAYER,
    RCGM_FULL_SW_LAYER
};

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

    virtual bool filterNeedLoadLocations(const kv_cache_manager::Locations& locations,
                                         LocationsView&                     locations_view,
                                         kv_cache_manager::BlockMaskOffset  block_mask = 0) const = 0;

    virtual bool getNeedWriteGroups(const std::shared_ptr<KVCacheResourceV1>& resource,
                                    std::vector<std::string>&                 location_spec_group_names) const = 0;

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

    virtual bool filterNeedLoadLocations(const kv_cache_manager::Locations& locations,
                                         LocationsView&                     locations_view,
                                         kv_cache_manager::BlockMaskOffset  block_mask = 0) const override;

    virtual bool getNeedWriteGroups(const std::shared_ptr<KVCacheResourceV1>& resource,
                                    std::vector<std::string>&                 location_spec_group_names) const override;

    bool genBlockBuffers(const std::vector<int32_t>&     group_ids,
                         const std::vector<int32_t>&     block_ids,
                         kv_cache_manager::BlockBuffers& block_buffers) const override;

    std::string debugString() const override;

protected:
    virtual std::string GetOtherGroupPrefixName() const {
        return "G";
    }

    std::map<int32_t, std::vector<int>> group_to_layer_ids_;
};

class FullLayerGroupPolicy: public DefaultLayerGroupPolicy {
public:
    FullLayerGroupPolicy(std::shared_ptr<KVCacheAllocator> allocator,
                         const std::vector<int32_t>&       full_group_ids,
                         const std::vector<int32_t>&       other_group_ids):
        DefaultLayerGroupPolicy(allocator, full_group_ids, other_group_ids) {}
    bool init() override;

    bool getNeedWriteGroups(const std::shared_ptr<KVCacheResourceV1>& resource,
                            std::vector<std::string>&                 location_spec_group_names) const override;
};

class FullOtherGroupPolicy: public DefaultLayerGroupPolicy {
public:
    bool init() override;

    bool addSpecInfo(const std::string& spec_name, int32_t group_id, int32_t tp_rank) override;

    bool getNeedWriteGroups(const std::shared_ptr<KVCacheResourceV1>& resource,
                            std::vector<std::string>&                 location_spec_group_names) const override;

    std::string debugString() const override;

protected:
    FullOtherGroupPolicy(std::shared_ptr<KVCacheAllocator> allocator,
                         const std::vector<int32_t>&       full_group_ids,
                         const std::vector<int32_t>&       other_group_ids,
                         uint32_t                          write_interval):
        DefaultLayerGroupPolicy(allocator, full_group_ids, other_group_ids), write_interval_(write_interval) {}
    bool IsValidFullLocation(const kv_cache_manager::Location& location) const;
    bool CheckInvalidFullLocationAndSetView(const kv_cache_manager::Location& location,
                                            LocationView&                     location_view) const;
    bool CheckInvalidFullOtherLocationAndSetView(const kv_cache_manager::Location& location,
                                                 LocationView&                     location_view) const;
    bool SkipOtherSpecAndSetView(const kv_cache_manager::Location& location, LocationView& location_view) const;

protected:
    uint64_t                        valid_full_bithash_       = 0;
    uint64_t                        valid_full_other_bithash_ = 0;
    std::map<std::string, uint64_t> full_spec_name_bithash_;
    std::map<std::string, uint64_t> full_other_spec_name_bithash_;
    /*
        interval == 0 :         only write last key's other attention
        interval == n (n > 0) : every n keys, write a other attention
    */
    uint32_t write_interval_ = 0;
};

class FullLinearLayerGroupPolicy: public FullOtherGroupPolicy {
public:
    FullLinearLayerGroupPolicy(std::shared_ptr<KVCacheAllocator> allocator,
                               const std::vector<int32_t>&       full_group_ids,
                               const std::vector<int32_t>&       other_group_ids,
                               uint32_t                          linear_attention_write_interval):
        FullOtherGroupPolicy(allocator, full_group_ids, other_group_ids, linear_attention_write_interval) {}

    bool filterNeedLoadLocations(const kv_cache_manager::Locations& locations,
                                 LocationsView&                     locations_view,
                                 kv_cache_manager::BlockMaskOffset  block_mask = 0) const override;

private:
    std::string GetOtherGroupPrefixName() const override {
        return "L";
    }
};

class FullSWLayerGroupPolicy: public FullOtherGroupPolicy {
public:
    FullSWLayerGroupPolicy(std::shared_ptr<KVCacheAllocator> allocator,
                           const std::vector<int32_t>&       full_group_ids,
                           const std::vector<int32_t>&       other_group_ids,
                           size_t                            sink_size,
                           size_t                            sw_size):
        FullOtherGroupPolicy(allocator, full_group_ids, other_group_ids, 1), sink_size_(sink_size), sw_size_(sw_size) {}

    bool filterNeedLoadLocations(const kv_cache_manager::Locations& locations,
                                 LocationsView&                     locations_view,
                                 kv_cache_manager::BlockMaskOffset  block_mask = 0) const override;

    std::string debugString() const override;

private:
    std::string GetOtherGroupPrefixName() const override {
        return "S";
    }

private:
    size_t sink_size_ = 0;
    size_t sw_size_   = 0;
};

}  // namespace remote_connector
}  // namespace rtp_llm