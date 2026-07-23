#pragma once

#include <vector>
#include <set>
#include <string>
#include <map>
#include <unordered_map>
#include <utility>
#include "kvcm_client/common.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"

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
        Group() = default;
        Group(bool        is_full,
              uint64_t    group_name_bithash,
              std::string group_name,
              std::string tag              = {},
              size_t      block_size_bytes = 0):
            is_full(is_full),
            group_name_bithash(group_name_bithash),
            group_name(std::move(group_name)),
            tag(std::move(tag)),
            block_size_bytes(block_size_bytes) {}

        bool        is_full            = true;
        uint64_t    group_name_bithash = 0;
        std::string group_name;
        std::string tag;
        size_t      block_size_bytes = 0;
    };
    using GroupMap = std::map<std::string, Group, std::less<>>;
    struct SpecInfo {
        int32_t     tp_rank;
        std::string tag;
    };
    using SpecInfoMap = std::map<std::string, SpecInfo, std::less<>>;

    GroupPolicy(std::shared_ptr<KVCacheAllocator> allocator,
                const std::vector<std::string>&   full_group_tags,
                const std::vector<std::string>&   other_group_tags):
        allocator_(allocator),
        full_group_tags_(full_group_tags.begin(), full_group_tags.end()),
        other_group_tags_(other_group_tags.begin(), other_group_tags.end()) {}
    virtual ~GroupPolicy() = default;

    virtual bool init() = 0;

    virtual bool filterNeedLoadLocations(const kv_cache_manager::Locations& locations,
                                         LocationsView&                     locations_view,
                                         kv_cache_manager::BlockMaskOffset  block_mask = 0) const = 0;

    virtual bool getNeedWriteGroups(const std::shared_ptr<KVCacheResource>& resource,
                                    std::vector<std::string>&               location_spec_group_names) const = 0;

    virtual bool genBlockBuffersByTag(const std::vector<std::string>& tags,
                                      const std::vector<int32_t>&     block_ids,
                                      kv_cache_manager::BlockBuffers& block_buffers) const = 0;

    const GroupMap& groups() const {
        return groups_;
    }

    // Aggregate group masks that getNeedWriteGroups() can actually emit.
    // Singleton specs are registered independently by RemoteConnector.
    virtual std::vector<uint64_t> reachableAggregateMasks() const {
        return {};
    }

    void addLocationSpecGroup(uint64_t bithash, const std::string& location_spec_group_name) {
        location_spec_group_map_[bithash] = location_spec_group_name;
    }
    virtual bool       addSpecInfo(const std::string& spec_name, std::string_view tag, int32_t tp_rank);
    const SpecInfoMap& spec_info_map() const {
        return spec_name_to_info_;
    }
    virtual std::string debugString() const;

protected:
    std::shared_ptr<KVCacheAllocator>  allocator_;
    std::set<std::string, std::less<>> full_group_tags_;
    std::set<std::string, std::less<>> other_group_tags_;

    // stable semantic tag -> group
    GroupMap groups_;
    // max support 64 groups, contains all group combinations
    std::unordered_map<uint64_t, std::string> location_spec_group_map_;
    // spec_name -> spec_info
    SpecInfoMap spec_name_to_info_;
};

class DefaultLayerGroupPolicy: public GroupPolicy {
public:
    DefaultLayerGroupPolicy(std::shared_ptr<KVCacheAllocator> allocator,
                            const std::vector<std::string>&   full_group_tags,
                            const std::vector<std::string>&   other_group_tags):
        GroupPolicy(allocator, full_group_tags, other_group_tags) {}

    virtual bool init() override;

    virtual bool filterNeedLoadLocations(const kv_cache_manager::Locations& locations,
                                         LocationsView&                     locations_view,
                                         kv_cache_manager::BlockMaskOffset  block_mask = 0) const override;

    virtual bool getNeedWriteGroups(const std::shared_ptr<KVCacheResource>& resource,
                                    std::vector<std::string>&               location_spec_group_names) const override;

    bool genBlockBuffersByTag(const std::vector<std::string>& tags,
                              const std::vector<int32_t>&     block_ids,
                              kv_cache_manager::BlockBuffers& block_buffers) const override;

    std::string debugString() const override;

protected:
    virtual std::string GetOtherGroupPrefixName() const {
        return "G";
    }

    std::map<std::string, std::vector<int>, std::less<>> tag_to_layer_ids_;
};

class FullLayerGroupPolicy: public DefaultLayerGroupPolicy {
public:
    FullLayerGroupPolicy(std::shared_ptr<KVCacheAllocator> allocator,
                         const std::vector<std::string>&   full_group_tags,
                         const std::vector<std::string>&   other_group_tags):
        DefaultLayerGroupPolicy(allocator, full_group_tags, other_group_tags) {}
    bool init() override;

    bool getNeedWriteGroups(const std::shared_ptr<KVCacheResource>& resource,
                            std::vector<std::string>&               location_spec_group_names) const override;

    std::vector<uint64_t> reachableAggregateMasks() const override;
};

class FullOtherGroupPolicy: public DefaultLayerGroupPolicy {
public:
    bool init() override;

    bool addSpecInfo(const std::string& spec_name, std::string_view tag, int32_t tp_rank) override;

    bool getNeedWriteGroups(const std::shared_ptr<KVCacheResource>& resource,
                            std::vector<std::string>&               location_spec_group_names) const override;

    std::string debugString() const override;

    std::vector<uint64_t> reachableAggregateMasks() const override;

protected:
    FullOtherGroupPolicy(std::shared_ptr<KVCacheAllocator> allocator,
                         const std::vector<std::string>&   full_group_tags,
                         const std::vector<std::string>&   other_group_tags,
                         uint32_t                          write_interval):
        DefaultLayerGroupPolicy(allocator, full_group_tags, other_group_tags), write_interval_(write_interval) {}
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
                               const std::vector<std::string>&   full_group_tags,
                               const std::vector<std::string>&   other_group_tags,
                               uint32_t                          linear_attention_write_interval):
        FullOtherGroupPolicy(allocator, full_group_tags, other_group_tags, linear_attention_write_interval) {}

    bool filterNeedLoadLocations(const kv_cache_manager::Locations& locations,
                                 LocationsView&                     locations_view,
                                 kv_cache_manager::BlockMaskOffset  block_mask = 0) const override;

private:
    std::string GetOtherGroupPrefixName() const override {
        return "L";
    }
};

}  // namespace remote_connector
}  // namespace rtp_llm
