#include <sstream>
#include <bitset>
#include <algorithm>
#include <typeinfo>
#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/cache/connector/remote_connector/GroupPolicy.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/CoordinatorKVCacheManager.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace remote_connector {

namespace {

std::string getBitHashStr(uint64_t bithash, size_t width = 64) {
    std::string res = std::bitset<64>(bithash).to_string().substr(64 - width);
    return "0b" + res;
}

bool validateResourceGroupBlocks(const KVCacheResource&       resource,
                                 const GroupPolicy::GroupMap& expected_groups,
                                 size_t                       required_block_count) {
    const auto& resource_groups = resource.groupBlocks();
    if (resource_groups.size() != expected_groups.size()) {
        RTP_LLM_LOG_WARNING(
            "group size not equal, expect [%zu], real [%zu]", expected_groups.size(), resource_groups.size());
        return false;
    }
    for (const auto& [tag, group] : expected_groups) {
        const auto resource_group = resource_groups.find(tag);
        if (resource_group == resource_groups.end() || resource_group->second == nullptr) {
            RTP_LLM_LOG_WARNING("resource missing cache group tag [%s]", tag.c_str());
            return false;
        }
        const size_t actual_block_count = resource_group->second->blocksNum();
        if (actual_block_count < required_block_count) {
            RTP_LLM_LOG_WARNING("cache group tag [%s] has insufficient blocks, require [%zu], real [%zu]",
                                tag.c_str(),
                                required_block_count,
                                actual_block_count);
            return false;
        }
    }
    return true;
}

}  // namespace

bool GroupPolicy::addSpecInfo(const std::string& spec_name, std::string_view tag, int32_t tp_rank) {
    if (spec_name_to_info_.find(spec_name) != spec_name_to_info_.end()) {
        RTP_LLM_LOG_ERROR("spec_name [%s] already exist", spec_name.c_str());
        return false;
    }
    const auto group_it = groups_.find(tag);
    if (group_it == groups_.end()) {
        RTP_LLM_LOG_ERROR("not find tag [%s], spec_name [%s]", std::string(tag).c_str(), spec_name.c_str());
        return false;
    }
    spec_name_to_info_[spec_name] = {tp_rank, group_it->second.tag};
    return true;
}

std::string GroupPolicy::debugString() const {
    size_t            gs = groups_.size();
    std::stringstream debug_ss;
    debug_ss << "groups (" << typeid(*this).name() << "):\n";
    for (const auto& entry : groups_) {
        debug_ss << '\t';
        const auto& group = entry.second;
        debug_ss << entry.first << ":[" << group.is_full << "|" << getBitHashStr(group.group_name_bithash, gs) << "|"
                 << group.group_name << "];\n";
    }

    debug_ss << "location_spec_group_map : \n\t";
    for (const auto& entry : location_spec_group_map_) {
        debug_ss << '[' << getBitHashStr(entry.first, gs) << ':' << entry.second << "]";
    }
    debug_ss << "\nspec_name_to_info :\n";
    for (const auto& entry : spec_name_to_info_) {
        const auto& spec_info = entry.second;
        debug_ss << '\t' << entry.first << ":[" << spec_info.tag << "|" << spec_info.tp_rank << "]\n";
    }
    return debug_ss.str();
}

bool DefaultLayerGroupPolicy::init() {
    std::vector<std::string> intersection;
    std::set_intersection(full_group_tags_.begin(),
                          full_group_tags_.end(),
                          other_group_tags_.begin(),
                          other_group_tags_.end(),
                          std::back_inserter(intersection));

    if (!intersection.empty()) {
        std::stringstream ss;
        for (const auto& tag : intersection) {
            ss << tag << "|";
        }
        RTP_LLM_LOG_ERROR("exist intersection between full and other [%s]", ss.str().c_str());
        return false;
    }
    const auto  layer_layout       = coordinator_cache_manager_->allLayerCacheBase();
    const auto& topology           = layer_layout.topology();
    uint64_t    group_name_bithash = 1;
    for (const auto& layer_config : topology.layers()) {
        const int layer = layer_config.layer_id;
        if (layer_config.group_tags.empty()) {
            RTP_LLM_LOG_ERROR("layer [%d] has no cache group tag", layer);
            return false;
        }
        for (const auto& cache_tag : layer_config.group_tags) {
            const bool is_full_group = full_group_tags_.find(cache_tag) != full_group_tags_.end();
            if (!is_full_group && other_group_tags_.find(cache_tag) == other_group_tags_.end()) {
                RTP_LLM_LOG_ERROR("not find valid group tag [%s]", cache_tag.c_str());
                return false;
            }
            if (groups_.count(cache_tag) == 0) {
                if (groups_.size() >= 64) {
                    RTP_LLM_LOG_ERROR("not support bigger than 64 groups");
                    return false;
                }
                const auto&       topology_group = topology.group(cache_tag);
                const std::string prefix         = is_full_group ? "F" : GetOtherGroupPrefixName();
                std::string       group_name     = prefix + cache_tag;
                const size_t      block_size_bytes =
                    topology_group.layer_ids.size()
                    * (topology_group.kv_block_stride_bytes + topology_group.kv_scale_stride_bytes);
                groups_[cache_tag] = Group{is_full_group, group_name_bithash, group_name, cache_tag, block_size_bytes};
                group_to_layer_ids_[cache_tag] = {};
                if (groups_.size() < 64) {
                    group_name_bithash <<= 1;
                }
            }
            group_to_layer_ids_.at(cache_tag).push_back(layer);
        }
    }
    return true;
}

bool DefaultLayerGroupPolicy::filterNeedLoadLocations(const kv_cache_manager::Locations& locations,
                                                      LocationsView&                     locations_view,
                                                      kv_cache_manager::BlockMaskOffset  block_mask) const {
    //  just copy
    locations_view.resize(locations.size(), {});
    for (size_t i = block_mask; i < locations.size(); i++) {
        locations_view[i].reserve(locations[i].size());
        for (const auto& unit : locations[i]) {
            locations_view[i].emplace_back(unit);
        }
    }
    return true;
}

bool DefaultLayerGroupPolicy::getNeedWriteGroups(const std::shared_ptr<KVCacheResource>& resource,
                                                 std::vector<std::string>& location_spec_group_names) const {
    const auto& cache_keys = resource->cacheKeys();
    RTP_LLM_CHECK(!cache_keys.empty());
    size_t valid_keys_size = cache_keys.size();
    if (!resource->lastBlockAligned()) {
        valid_keys_size--;
    }
    if (!validateResourceGroupBlocks(*resource, groups_, valid_keys_size)) {
        return false;
    }
    location_spec_group_names.reserve(valid_keys_size);
    for (size_t key_idx = 0; key_idx < valid_keys_size; key_idx++) {
        uint64_t groups_name_bithash = 0;
        for (const auto& [tag, group] : groups_) {
            const auto gpu_block_idx = resource->blocks(tag).at(key_idx);
            if (!isNullBlockIdx(gpu_block_idx)) {
                groups_name_bithash |= group.group_name_bithash;
            }
        }
        location_spec_group_names.push_back(location_spec_group_map_.at(groups_name_bithash));
    }
    return true;
}

#define CHECK_BLOCK_INFO_VALID(block_info, format, args...)                                                            \
    do {                                                                                                               \
        if (block_info.addr == nullptr || block_info.size_bytes == 0) {                                                \
            RTP_LLM_LOG_WARNING(format, ##args);                                                                       \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

bool DefaultLayerGroupPolicy::genBlockBuffers(const std::vector<std::string>& tags,
                                              const std::vector<int32_t>&     block_ids,
                                              kv_cache_manager::BlockBuffers& output_block_buffers) const {
    static auto push_iov = [](std::vector<kv_cache_manager::Iov>& iovs, const BlockInfo& block_info) {
        iovs.push_back({kv_cache_manager::MemoryType::GPU, block_info.addr, block_info.size_bytes, false});
    };
    if (tags.size() != block_ids.size()) {
        RTP_LLM_LOG_WARNING(
            "remote cache tag/block count mismatch: tags=%zu blocks=%zu", tags.size(), block_ids.size());
        return false;
    }
    for (const auto& tag : tags) {
        if (tag.empty() || groups_.find(tag) == groups_.end()) {
            RTP_LLM_LOG_WARNING("invalid remote cache tag [%s]", tag.c_str());
            return false;
        }
    }

    kv_cache_manager::BlockBuffers block_buffers;
    block_buffers.reserve(block_ids.size());
    for (size_t i = 0; i < block_ids.size(); ++i) {
        const auto& tag      = tags[i];
        const auto  block_id = block_ids[i];
        block_buffers.push_back({});
        const auto& layer_ids          = group_to_layer_ids_.at(tag);
        auto&       iovs               = block_buffers.back().iovs;
        size_t      actual_block_bytes = 0;
        iovs.reserve(layer_ids.size() * 2);
        for (size_t j = 0; j < layer_ids.size(); ++j) {
            // if support scale, block_infos: {kv_info, scale_info}
            const auto& block_infos = coordinator_cache_manager_->convertIndexToBuffer(layer_ids[j], tag, block_id);
            if (block_infos.empty()) {
                RTP_LLM_LOG_WARNING("convertIndexToBuffer returned empty for layer_id [%d] tag [%s] block_id[%d]",
                                    layer_ids[j],
                                    tag.c_str(),
                                    block_id);
            }
            for (size_t idx = 0; idx < block_infos.size(); ++idx) {
                CHECK_BLOCK_INFO_VALID(
                    block_infos[idx],
                    "convertIndexToBuffer failed layer_id [%d] tag [%s] block_id[%d], block_info.addr or block_info.size_bytes is invalid",
                    layer_ids[j],
                    tag.c_str(),
                    block_id);
                actual_block_bytes += block_infos[idx].size_bytes;
                push_iov(iovs, block_infos[idx]);
            }
        }
        const size_t expected_block_bytes = groups_.at(tag).block_size_bytes;
        if (actual_block_bytes != expected_block_bytes) {
            RTP_LLM_LOG_WARNING("remote cache block size mismatch tag [%s] block_id [%d], expected [%zu] actual [%zu]",
                                tag.c_str(),
                                block_id,
                                expected_block_bytes,
                                actual_block_bytes);
            return false;
        }
    }
    output_block_buffers = std::move(block_buffers);
    return true;
}

std::string DefaultLayerGroupPolicy::debugString() const {
    std::stringstream debug_ss;
    debug_ss << GroupPolicy::debugString();
    debug_ss << "group_to_layer_ids:\n";
    for (const auto& entry : group_to_layer_ids_) {
        debug_ss << '\t' << entry.first << " : ";
        for (int layer_id : entry.second) {
            debug_ss << layer_id << '|';
        }
        debug_ss << '\n';
    }
    return debug_ss.str();
}

bool FullLayerGroupPolicy::init() {
    if (full_group_tags_.empty()) {
        RTP_LLM_LOG_ERROR("FullLayerGroupPolicy requires at least one full group");
        return false;
    }
    if (!other_group_tags_.empty()) {
        RTP_LLM_LOG_ERROR("FullLayerGroupPolicy not support other groups");
        return false;
    }
    return DefaultLayerGroupPolicy::init();
}

bool FullLayerGroupPolicy::getNeedWriteGroups(const std::shared_ptr<KVCacheResource>& resource,
                                              std::vector<std::string>&               location_spec_group_names) const {
    if (groups_.size() == 1) {
        const auto& cache_keys = resource->cacheKeys();
        RTP_LLM_CHECK(!cache_keys.empty());
        size_t valid_keys_size = cache_keys.size();
        if (!resource->lastBlockAligned()) {
            valid_keys_size--;
        }
        if (!validateResourceGroupBlocks(*resource, groups_, valid_keys_size)) {
            return false;
        }
        const auto& tag    = groups_.begin()->first;
        const auto& blocks = resource->blocks(tag);
        for (size_t key_idx = 0; key_idx < valid_keys_size; ++key_idx) {
            if (isNullBlockIdx(blocks.at(key_idx))) {
                RTP_LLM_LOG_WARNING("cache group tag [%s] has invalid block at index [%zu]", tag.c_str(), key_idx);
                return false;
            }
        }
        return true;
    }
    return DefaultLayerGroupPolicy::getNeedWriteGroups(resource, location_spec_group_names);
}

std::vector<uint64_t> FullLayerGroupPolicy::reachableAggregateMasks() const {
    RTP_LLM_CHECK_WITH_INFO(!groups_.empty(), "FullLayerGroupPolicy must be initialized before reading masks");
    uint64_t all_full_mask = 0;
    for (const auto& [tag, group] : groups_) {
        all_full_mask |= group.group_name_bithash;
    }
    return {all_full_mask};
}

bool FullOtherGroupPolicy::init() {
    if (full_group_tags_.empty()) {
        RTP_LLM_LOG_ERROR("FullOtherLayerGroupPolicy: not support empty full groups");
        return false;
    }
    if (other_group_tags_.empty()) {
        RTP_LLM_LOG_ERROR("FullOtherLayerGroupPolicy: not support empty other groups");
        return false;
    }
    if (!DefaultLayerGroupPolicy::init()) {
        return false;
    }
    for (const auto& tag : full_group_tags_) {
        const auto it = groups_.find(tag);
        if (it == groups_.end()) {
            RTP_LLM_LOG_ERROR("not find full group tag [%s]", tag.c_str());
            return false;
        }
        valid_full_bithash_ |= it->second.group_name_bithash;
        valid_full_other_bithash_ |= it->second.group_name_bithash;
    }
    for (const auto& tag : other_group_tags_) {
        const auto it = groups_.find(tag);
        if (it == groups_.end()) {
            RTP_LLM_LOG_ERROR("not find other group tag [%s]", tag.c_str());
            return false;
        }
        valid_full_other_bithash_ |= it->second.group_name_bithash;
    }
    if (groups_.size() < 2) {
        RTP_LLM_LOG_ERROR("FullOtherLayerGroupPolicy: invalid group size [%lu]", groups_.size());
        return false;
    }
    return true;
}

bool FullOtherGroupPolicy::getNeedWriteGroups(const std::shared_ptr<KVCacheResource>& resource,
                                              std::vector<std::string>&               location_spec_group_names) const {
    const auto& cache_keys = resource->cacheKeys();
    RTP_LLM_CHECK(!cache_keys.empty());
    size_t valid_keys_size = cache_keys.size();
    if (!resource->lastBlockAligned()) {
        valid_keys_size--;
    }
    if (!validateResourceGroupBlocks(*resource, groups_, valid_keys_size)) {
        return false;
    }
    location_spec_group_names.resize(valid_keys_size, {});
    bool   exist_full_other  = false;
    size_t count             = write_interval_;
    bool   is_all_full_other = true;
    for (size_t key_idx = valid_keys_size; key_idx-- > 0;) {
        uint64_t groups_name_bithash = 0;
        for (const auto& [tag, group] : groups_) {
            const auto gpu_block_idx = resource->blocks(tag).at(key_idx);
            if (!rtp_llm::isNullBlockIdx(gpu_block_idx)) {
                groups_name_bithash |= group.group_name_bithash;
            }
        }
        if (groups_name_bithash != valid_full_bithash_ && groups_name_bithash != valid_full_other_bithash_) {
            RTP_LLM_LOG_WARNING("invalid groups_name_bithash [%lu]", groups_name_bithash);
            return false;
        }
        if (write_interval_ > 0) {
            count++;
            bool need_full_other = (groups_name_bithash == valid_full_other_bithash_) && (count >= write_interval_);
            if (need_full_other) {
                groups_name_bithash = valid_full_other_bithash_;
                count               = 0;
            } else {
                groups_name_bithash = valid_full_bithash_;
                is_all_full_other   = false;
            }
            location_spec_group_names[key_idx] = location_spec_group_map_.at(groups_name_bithash);
        } else {
            if (!exist_full_other && groups_name_bithash == valid_full_other_bithash_) {
                location_spec_group_names[key_idx] = location_spec_group_map_.at(valid_full_other_bithash_);
                exist_full_other                   = true;
            } else {
                location_spec_group_names[key_idx] = location_spec_group_map_.at(valid_full_bithash_);
                is_all_full_other                  = false;
            }
        }
    }
    if (is_all_full_other) {
        RTP_LLM_LOG_DEBUG("all full_other, not need spec group names");
        location_spec_group_names.clear();
    }
    return true;
}

bool FullOtherGroupPolicy::addSpecInfo(const std::string& spec_name, std::string_view tag, int32_t tp_rank) {
    if (!GroupPolicy::addSpecInfo(spec_name, tag, tp_rank)) {
        return false;
    }
    const auto& it = groups_.find(tag);
    if (it == groups_.end()) {
        RTP_LLM_LOG_ERROR("not find tag [%s], spec_name [%s]", std::string(tag).c_str(), spec_name.c_str());
        return false;
    }
    const auto& group = it->second;
    if (group.is_full) {
        full_spec_name_bithash_[spec_name] = group.group_name_bithash;
    }
    full_other_spec_name_bithash_[spec_name] = group.group_name_bithash;
    return true;
}

std::vector<uint64_t> FullOtherGroupPolicy::reachableAggregateMasks() const {
    RTP_LLM_CHECK_WITH_INFO(valid_full_bithash_ != 0 && valid_full_other_bithash_ != 0,
                            "FullOtherGroupPolicy must be initialized before reading masks");
    return {valid_full_bithash_, valid_full_other_bithash_};
}

std::string FullOtherGroupPolicy::debugString() const {
    size_t            gs = groups_.size();
    std::stringstream debug_ss;
    debug_ss << DefaultLayerGroupPolicy::debugString();
    debug_ss << "write_interval : " << write_interval_ << '\n';
    debug_ss << "valid_full_bithash : " << getBitHashStr(valid_full_bithash_, gs) << '\n';
    debug_ss << "valid_full_other_bithash : " << getBitHashStr(valid_full_other_bithash_, gs) << '\n';
    debug_ss << "full_spec_name_bithash : ";
    for (const auto& entry : full_spec_name_bithash_) {
        debug_ss << '[' << entry.first << ":" << getBitHashStr(entry.second, gs) << ']';
    }
    debug_ss << '\n';
    debug_ss << "full_other_spec_name_bithash : ";
    for (const auto& entry : full_other_spec_name_bithash_) {
        debug_ss << '[' << entry.first << ":" << getBitHashStr(entry.second, gs) << ']';
    }
    debug_ss << '\n';
    return debug_ss.str();
}

bool FullOtherGroupPolicy::IsValidFullLocation(const kv_cache_manager::Location& location) const {
    uint64_t full_bithash = 0;
    for (const auto& unit : location) {
        const auto iter = full_spec_name_bithash_.find(unit.spec_name);
        if (iter == full_spec_name_bithash_.end()) {
            RTP_LLM_LOG_WARNING("not find full spec name [%s]", unit.spec_name.c_str());
            return false;
        }
        full_bithash |= iter->second;
    }
    if (full_bithash != valid_full_bithash_) {
        RTP_LLM_LOG_WARNING("invalid full bithash [%lu], expect [%lu]", full_bithash, valid_full_bithash_);
        return false;
    }
    return true;
}

#define CEHCK_AND_SET_LOCATIONS_VIEW(attention_name)                                                                   \
    location_view.reserve(location.size());                                                                            \
    uint64_t attention_name##_bithash = 0;                                                                             \
    for (const auto& unit : location) {                                                                                \
        const auto iter = attention_name##_spec_name_bithash_.find(unit.spec_name);                                    \
        if (iter == attention_name##_spec_name_bithash_.end()) {                                                       \
            RTP_LLM_LOG_WARNING("not find " #attention_name " spec name [%s]", unit.spec_name.c_str());                \
            return false;                                                                                              \
        }                                                                                                              \
        attention_name##_bithash |= iter->second;                                                                      \
        location_view.emplace_back(unit);                                                                              \
    }                                                                                                                  \
    if (attention_name##_bithash != valid_##attention_name##_bithash_) {                                               \
        RTP_LLM_LOG_WARNING("invalid " #attention_name " bithash [%lu], expect [%lu]",                                 \
                            attention_name##_bithash,                                                                  \
                            valid_##attention_name##_bithash_);                                                        \
        return false;                                                                                                  \
    }

bool FullOtherGroupPolicy::CheckInvalidFullLocationAndSetView(const kv_cache_manager::Location& location,
                                                              LocationView&                     location_view) const {
    CEHCK_AND_SET_LOCATIONS_VIEW(full);
    return true;
}

bool FullOtherGroupPolicy::CheckInvalidFullOtherLocationAndSetView(const kv_cache_manager::Location& location,
                                                                   LocationView& location_view) const {
    CEHCK_AND_SET_LOCATIONS_VIEW(full_other);
    return true;
}

#undef CEHCK_AND_SET_LOCATIONS_VIEW

bool FullOtherGroupPolicy::SkipOtherSpecAndSetView(const kv_cache_manager::Location& location,
                                                   LocationView&                     location_view) const {
    location_view.reserve(full_spec_name_bithash_.size());
    uint64_t full_bithash = 0;
    for (const auto& unit : location) {
        const auto iter = full_spec_name_bithash_.find(unit.spec_name);
        if (iter == full_spec_name_bithash_.end()) {
            RTP_LLM_LOG_DEBUG("skip spec_name [%s]", unit.spec_name.c_str());
            continue;
        }
        full_bithash |= iter->second;
        location_view.emplace_back(unit);
    }
    if (full_bithash != valid_full_bithash_) {
        RTP_LLM_LOG_WARNING("invalid full bithash [%lu], expect [%lu]", full_bithash, valid_full_bithash_);
        return false;
    }
    return true;
}

bool FullLinearLayerGroupPolicy::filterNeedLoadLocations(const kv_cache_manager::Locations& locations,
                                                         LocationsView&                     locations_view,
                                                         kv_cache_manager::BlockMaskOffset  block_mask) const {
    bool exist_linear_location = false;
    for (size_t i = locations.size(); i-- > block_mask;) {
        const auto& location = locations[i];
        if (location.size() == full_spec_name_bithash_.size()) {
            if (exist_linear_location) {
                if (!CheckInvalidFullLocationAndSetView(location, locations_view[i])) {
                    return false;
                }
            } else {
                // only do check
                if (!IsValidFullLocation(location)) {
                    return false;
                }
            }
        } else if (location.size() == full_other_spec_name_bithash_.size()) {
            if (!exist_linear_location) {
                locations_view.resize(i + 1, {});
                if (!CheckInvalidFullOtherLocationAndSetView(location, locations_view[i])) {
                    return false;
                }
                exist_linear_location = true;
            } else {
                if (!SkipOtherSpecAndSetView(location, locations_view[i])) {
                    return false;
                }
            }
        } else {
            RTP_LLM_LOG_WARNING("invalid spec size, full [%lu], linear [%lu], real [%lu]",
                                full_spec_name_bithash_.size(),
                                full_other_spec_name_bithash_.size(),
                                location.size());
            return false;
        }
    }
    return true;
}

}  // namespace remote_connector
}  // namespace rtp_llm
