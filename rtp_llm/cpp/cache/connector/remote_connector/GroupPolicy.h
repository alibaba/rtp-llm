#pragma once

#include <functional>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include "kvcm_client/common.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackend.h"

namespace rtp_llm {

namespace remote_connector {

std::string genLocationSpecName(int tp_rank, const std::string& group_name);

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
    using GroupIdMap = std::map<int32_t, Group>;
    struct SpecInfo {
        int32_t     group_id;
        int32_t     tp_rank;
        std::string tag;
    };
    using SpecInfoMap        = std::map<std::string, SpecInfo, std::less<>>;
    using LocationSpecGroups = std::map<std::string, std::vector<std::string>>;

    GroupPolicy(const CacheTopology&           topology,
                StorageBackend::BufferResolver buffer_resolver,
                const std::vector<int32_t>&    full_group_ids,
                const std::vector<int32_t>&    other_group_ids):
        topology_(topology),
        buffer_resolver_(std::move(buffer_resolver)),
        full_group_ids_(full_group_ids.begin(), full_group_ids.end()),
        other_group_ids_(other_group_ids.begin(), other_group_ids.end()) {}
    virtual ~GroupPolicy() = default;

    virtual bool init() = 0;

    virtual bool filterNeedLoadLocations(const kv_cache_manager::Locations& locations,
                                         LocationsView&                     locations_view,
                                         kv_cache_manager::BlockMaskOffset  block_mask = 0) const = 0;

    virtual bool getNeedWriteGroups(const StorageRequest&     request,
                                    size_t                    valid_keys_size,
                                    std::vector<std::string>& location_spec_group_names) const = 0;

    virtual bool genBlockBuffers(const std::vector<int32_t>&     group_ids,
                                 const std::vector<int32_t>&     block_ids,
                                 kv_cache_manager::BlockBuffers& block_buffers) const = 0;
    bool         genBlockBuffersByTag(const std::vector<std::string>& tags,
                                      const std::vector<int32_t>&     block_ids,
                                      kv_cache_manager::BlockBuffers& block_buffers) const;

    const GroupIdMap& groups() const {
        return groups_;
    }

    // Aggregate group masks that getNeedWriteGroups() can actually emit.
    // Singleton specs are registered independently by RemoteConnector.
    virtual std::vector<uint64_t> reachableAggregateMasks() const {
        return {};
    }

    // Build singleton and reachable aggregate location groups using the
    // canonical group-name order used by the legacy KVCM protocol.
    bool buildLocationSpecGroups(int tp_size, LocationSpecGroups& location_spec_groups);

    const SpecInfoMap& spec_info_map() const {
        return spec_name_to_info_;
    }
    virtual std::string debugString() const;

protected:
    virtual void rebuildDerivedSpecInfo() {}

    const CacheTopology&           topology_;
    StorageBackend::BufferResolver buffer_resolver_;
    std::set<int32_t>              full_group_ids_;
    std::set<int32_t>              other_group_ids_;

    // group_id -> group
    GroupIdMap groups_;
    // Stable semantic identity resolved once during policy initialization.
    std::unordered_map<std::string, int32_t> tag_to_group_id_;
    // max support 64 groups, contains all group combinations
    std::unordered_map<uint64_t, std::string> location_spec_group_map_;
    // spec_name -> spec_info
    SpecInfoMap spec_name_to_info_;
};

class DefaultLayerGroupPolicy: public GroupPolicy {
public:
    DefaultLayerGroupPolicy(const CacheTopology&           topology,
                            StorageBackend::BufferResolver buffer_resolver,
                            const std::vector<int32_t>&    full_group_ids,
                            const std::vector<int32_t>&    other_group_ids):
        GroupPolicy(topology, std::move(buffer_resolver), full_group_ids, other_group_ids) {}

    virtual bool init() override;

    virtual bool filterNeedLoadLocations(const kv_cache_manager::Locations& locations,
                                         LocationsView&                     locations_view,
                                         kv_cache_manager::BlockMaskOffset  block_mask = 0) const override;

    bool getNeedWriteGroups(const StorageRequest&     request,
                            size_t                    valid_keys_size,
                            std::vector<std::string>& location_spec_group_names) const override;

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
    FullLayerGroupPolicy(const CacheTopology&           topology,
                         StorageBackend::BufferResolver buffer_resolver,
                         const std::vector<int32_t>&    full_group_ids,
                         const std::vector<int32_t>&    other_group_ids):
        DefaultLayerGroupPolicy(topology, std::move(buffer_resolver), full_group_ids, other_group_ids) {}
    bool init() override;

    bool getNeedWriteGroups(const StorageRequest&     request,
                            size_t                    valid_keys_size,
                            std::vector<std::string>& location_spec_group_names) const override;

    std::vector<uint64_t> reachableAggregateMasks() const override;
};

class FullOtherGroupPolicy: public DefaultLayerGroupPolicy {
public:
    bool init() override;

    bool getNeedWriteGroups(const StorageRequest&     request,
                            size_t                    valid_keys_size,
                            std::vector<std::string>& location_spec_group_names) const override;

    std::string debugString() const override;

    std::vector<uint64_t> reachableAggregateMasks() const override;

protected:
    void rebuildDerivedSpecInfo() override;

    FullOtherGroupPolicy(const CacheTopology&           topology,
                         StorageBackend::BufferResolver buffer_resolver,
                         const std::vector<int32_t>&    full_group_ids,
                         const std::vector<int32_t>&    other_group_ids,
                         uint32_t                       write_interval):
        DefaultLayerGroupPolicy(topology, std::move(buffer_resolver), full_group_ids, other_group_ids),
        write_interval_(write_interval) {}
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
    FullLinearLayerGroupPolicy(const CacheTopology&           topology,
                               StorageBackend::BufferResolver buffer_resolver,
                               const std::vector<int32_t>&    full_group_ids,
                               const std::vector<int32_t>&    other_group_ids,
                               uint32_t                       linear_attention_write_interval):
        FullOtherGroupPolicy(
            topology, std::move(buffer_resolver), full_group_ids, other_group_ids, linear_attention_write_interval) {}

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
