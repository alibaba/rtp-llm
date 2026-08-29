#include <sstream>
#include <bitset>
#include <algorithm>
#include <typeinfo>
#include "rtp_llm/cpp/cache/connector/remote_connector/GroupPolicy.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace remote_connector {

namespace {

std::string getBitHashStr(uint64_t bithash, size_t width = 64) {
    std::string res = std::bitset<64>(bithash).to_string().substr(64 - width);
    return "0b" + res;
}

}  // namespace

std::string genLocationSpecName(int tp_rank, const std::string& group_name) {
    return "tp" + std::to_string(tp_rank) + "_" + group_name;
}

bool GroupPolicy::genBlockBuffersByTag(const std::vector<std::string>& tags,
                                       const std::vector<int32_t>&     block_ids,
                                       kv_cache_manager::BlockBuffers& block_buffers) const {
    RTP_LLM_CHECK_WITH_INFO(tags.size() == block_ids.size(),
                            "remote cache tag/block count mismatch: tags=%zu blocks=%zu",
                            tags.size(),
                            block_ids.size());
    std::vector<int32_t> group_ids;
    group_ids.reserve(tags.size());
    for (const auto& tag : tags) {
        const auto group_it = tag_to_group_id_.find(tag);
        RTP_LLM_CHECK_WITH_INFO(group_it != tag_to_group_id_.end(), "remote cache policy missing tag=%s", tag.c_str());
        group_ids.push_back(group_it->second);
    }
    return genBlockBuffers(group_ids, block_ids, block_buffers);
}

bool GroupPolicy::buildLocationSpecGroups(int tp_size, LocationSpecGroups& location_spec_groups) {
    if (tp_size <= 0 || groups_.empty()) {
        RTP_LLM_LOG_ERROR("cannot build KVCM location groups: tp_size=%d groups=%zu", tp_size, groups_.size());
        return false;
    }
    LocationSpecGroups                        new_location_spec_groups;
    std::unordered_map<uint64_t, std::string> new_location_spec_group_map;
    SpecInfoMap                               new_spec_name_to_info;
    for (const auto& [group_id, group] : groups_) {
        auto [group_it, inserted] = new_location_spec_groups.emplace(group.group_name, std::vector<std::string>{});
        if (!inserted) {
            RTP_LLM_LOG_ERROR("duplicate KVCM singleton location group [%s]", group.group_name.c_str());
            return false;
        }
        new_location_spec_group_map[group.group_name_bithash] = group.group_name;
        for (int rank = 0; rank < tp_size; ++rank) {
            const std::string spec_name = genLocationSpecName(rank, group.group_name);
            group_it->second.push_back(spec_name);
            const auto [unused_it, spec_inserted] =
                new_spec_name_to_info.emplace(spec_name, SpecInfo{group_id, rank, group.tag});
            (void)unused_it;
            if (!spec_inserted) {
                RTP_LLM_LOG_ERROR("duplicate KVCM location spec [%s]", spec_name.c_str());
                return false;
            }
        }
    }

    for (uint64_t aggregate_mask : reachableAggregateMasks()) {
        uint64_t                                                      matched_mask = 0;
        std::vector<std::pair<std::string, std::vector<std::string>>> selected_groups;
        for (const auto& [group_id, group] : groups_) {
            (void)group_id;
            if ((aggregate_mask & group.group_name_bithash) == 0) {
                continue;
            }
            matched_mask |= group.group_name_bithash;
            selected_groups.emplace_back(group.group_name, new_location_spec_groups.at(group.group_name));
        }
        if (aggregate_mask == 0 || matched_mask != aggregate_mask) {
            RTP_LLM_LOG_ERROR("invalid KVCM aggregate mask [%lu], matched [%lu]", aggregate_mask, matched_mask);
            return false;
        }
        std::sort(selected_groups.begin(), selected_groups.end());
        std::string              aggregate_name;
        std::vector<std::string> aggregate_specs;
        for (const auto& [group_name, singleton_specs] : selected_groups) {
            aggregate_name += group_name;
            aggregate_specs.insert(aggregate_specs.end(), singleton_specs.begin(), singleton_specs.end());
        }
        new_location_spec_group_map[aggregate_mask] = aggregate_name;
        const auto [unused_it, inserted] =
            new_location_spec_groups.emplace(std::move(aggregate_name), std::move(aggregate_specs));
        (void)unused_it;
        // A one-group aggregate intentionally aliases its singleton.
        if (!inserted && selected_groups.size() != 1) {
            RTP_LLM_LOG_ERROR("duplicate KVCM aggregate location group");
            return false;
        }
    }
    location_spec_group_map_ = std::move(new_location_spec_group_map);
    spec_name_to_info_       = std::move(new_spec_name_to_info);
    rebuildDerivedSpecInfo();
    location_spec_groups = std::move(new_location_spec_groups);
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
        debug_ss << '\t' << entry.first << ":[" << spec_info.tag << "|" << spec_info.group_id << "|"
                 << spec_info.tp_rank << "]\n";
    }
    return debug_ss.str();
}

bool DefaultLayerGroupPolicy::init() {
    std::vector<int> intersection;
    std::set_intersection(full_group_ids_.begin(),
                          full_group_ids_.end(),
                          other_group_ids_.begin(),
                          other_group_ids_.end(),
                          std::back_inserter(intersection));

    if (!intersection.empty()) {
        std::stringstream ss;
        for (int group : intersection) {
            ss << group << "|";
        }
        RTP_LLM_LOG_ERROR("exist intersection between full and other [%s]", ss.str().c_str());
        return false;
    }
    uint64_t    group_name_bithash = 1;
    const auto& layer_group_ids    = topology_.layerGroupIdsSnapshot();
    for (int layer = 0; layer < static_cast<int>(layer_group_ids.size()); ++layer) {
        if (layer_group_ids.at(layer).empty()) {
            RTP_LLM_LOG_ERROR("layer [%d] has no cache group id", layer);
            return false;
        }
        for (const int group_idx : layer_group_ids.at(layer)) {
            bool is_full_group = false;
            if (full_group_ids_.find(group_idx) != full_group_ids_.end()) {
                is_full_group = true;
            }
            if (!is_full_group) {
                if (other_group_ids_.find(group_idx) == other_group_ids_.end()) {
                    RTP_LLM_LOG_ERROR("not find valid group id, [%d]", group_idx);
                    return false;
                }
            }
            if (groups_.count(group_idx) == 0) {
                if (groups_.size() >= 64) {
                    RTP_LLM_LOG_ERROR("not support bigger than 64 groups");
                    return false;
                }
                RTP_LLM_CHECK_WITH_INFO(group_idx >= 0, "invalid remote cache group id=%d", group_idx);
                const auto& topology_group    = topology_.groupById(static_cast<size_t>(group_idx));
                const auto& cache_tag         = topology_group.tag;
                const auto [tag_it, inserted] = tag_to_group_id_.emplace(cache_tag, group_idx);
                if (!inserted && tag_it->second != group_idx) {
                    RTP_LLM_LOG_ERROR("duplicate remote cache tag [%s] for group ids [%d] and [%d]",
                                      cache_tag.c_str(),
                                      tag_it->second,
                                      group_idx);
                    return false;
                }
                const std::string prefix     = is_full_group ? "F" : GetOtherGroupPrefixName();
                std::string       group_name = prefix + cache_tag;
                const size_t      block_size_bytes =
                    topology_group.layer_ids.size()
                    * (topology_group.kv_block_stride_bytes + topology_group.kv_scale_stride_bytes);
                groups_[group_idx] = Group{is_full_group, group_name_bithash, group_name, cache_tag, block_size_bytes};
                group_to_layer_ids_[group_idx] = {};
                if (groups_.size() < 64) {
                    group_name_bithash <<= 1;
                }
            }
            group_to_layer_ids_.at(group_idx).push_back(layer);
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

bool DefaultLayerGroupPolicy::getNeedWriteGroups(const StorageRequest&     request,
                                                 size_t                    valid_keys_size,
                                                 std::vector<std::string>& location_spec_group_names) const {
    RTP_LLM_CHECK(request.keys != nullptr);
    RTP_LLM_CHECK_WITH_INFO(valid_keys_size <= request.keys->size() && request.handles.size() == request.keys->size(),
                            "invalid storage write shape: valid=%zu keys=%zu handles=%zu",
                            valid_keys_size,
                            request.keys->size(),
                            request.handles.size());
    location_spec_group_names.reserve(valid_keys_size);
    for (size_t key_idx = 0; key_idx < valid_keys_size; ++key_idx) {
        uint64_t groups_name_bithash = 0;
        for (const auto& handle : request.handles[key_idx]) {
            if (isNullBlockIdx(handle.block)) {
                continue;
            }
            const auto group = groups_.find(static_cast<int32_t>(handle.group_id));
            if (group == groups_.end()) {
                RTP_LLM_LOG_WARNING("remote write references unknown group_id [%zu]", handle.group_id);
                return false;
            }
            groups_name_bithash |= group->second.group_name_bithash;
        }
        const auto group_name = location_spec_group_map_.find(groups_name_bithash);
        if (group_name == location_spec_group_map_.end()) {
            RTP_LLM_LOG_WARNING(
                "remote write has unsupported group mask [%lu] at key [%zu]", groups_name_bithash, key_idx);
            return false;
        }
        location_spec_group_names.push_back(group_name->second);
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

bool DefaultLayerGroupPolicy::genBlockBuffers(const std::vector<int32_t>&     group_ids,
                                              const std::vector<int32_t>&     block_ids,
                                              kv_cache_manager::BlockBuffers& block_buffers) const {
    static auto push_iov = [](std::vector<kv_cache_manager::Iov>& iovs, const BlockInfo& block_info) {
        iovs.push_back({kv_cache_manager::MemoryType::GPU, block_info.addr, block_info.size_bytes, false});
    };
    RTP_LLM_CHECK_WITH_INFO(group_ids.size() == block_ids.size(),
                            "remote cache group/block count mismatch: groups=%zu blocks=%zu",
                            group_ids.size(),
                            block_ids.size());
    block_buffers.reserve(block_ids.size());
    for (size_t i = 0; i < block_ids.size(); ++i) {
        RTP_LLM_CHECK_WITH_INFO(group_ids[i] >= 0, "invalid remote cache group id=%d", group_ids[i]);
        block_buffers.push_back({});
        const auto& layer_ids          = group_to_layer_ids_.at(group_ids[i]);
        const auto& tag                = groups_.at(group_ids[i]).tag;
        auto&       iovs               = block_buffers.back().iovs;
        size_t      actual_block_bytes = 0;
        iovs.reserve(layer_ids.size() * 2);
        for (size_t j = 0; j < layer_ids.size(); ++j) {
            // if support scale, block_infos: {kv_info, scale_info}
            const auto block_infos = buffer_resolver_(layer_ids[j], group_ids[i], block_ids[i]);
            if (block_infos.empty()) {
                RTP_LLM_LOG_WARNING("convertIndexToBuffer returned empty for layer_id [%d] group_id [%d] block_id[%d]",
                                    layer_ids[j],
                                    group_ids[i],
                                    block_ids[i]);
            }
            for (size_t idx = 0; idx < block_infos.size(); ++idx) {
                CHECK_BLOCK_INFO_VALID(
                    block_infos[idx],
                    "convertIndexToBuffer failed layer_id [%d] group_id [%d] block_id[%d], block_info.addr or block_info.size_bytes is invalid",
                    layer_ids[j],
                    group_ids[i],
                    block_ids[i]);
                actual_block_bytes += block_infos[idx].size_bytes;
                push_iov(iovs, block_infos[idx]);
            }
        }
        const size_t expected_block_bytes = groups_.at(group_ids[i]).block_size_bytes;
        if (actual_block_bytes != expected_block_bytes) {
            RTP_LLM_LOG_WARNING(
                "remote cache block size mismatch tag [%s] group_id [%d] block_id [%d], expected [%zu] actual [%zu]",
                tag.c_str(),
                group_ids[i],
                block_ids[i],
                expected_block_bytes,
                actual_block_bytes);
            block_buffers.pop_back();
            return false;
        }
    }
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
    if (full_group_ids_.empty()) {
        RTP_LLM_LOG_ERROR("FullLayerGroupPolicy requires at least one full group");
        return false;
    }
    if (!other_group_ids_.empty()) {
        RTP_LLM_LOG_ERROR("FullLayerGroupPolicy not support other groups");
        return false;
    }
    return DefaultLayerGroupPolicy::init();
}

bool FullLayerGroupPolicy::getNeedWriteGroups(const StorageRequest&     request,
                                              size_t                    valid_keys_size,
                                              std::vector<std::string>& location_spec_group_names) const {
    if (groups_.size() == 1) {
        return true;
    }
    return DefaultLayerGroupPolicy::getNeedWriteGroups(request, valid_keys_size, location_spec_group_names);
}

std::vector<uint64_t> FullLayerGroupPolicy::reachableAggregateMasks() const {
    RTP_LLM_CHECK_WITH_INFO(!groups_.empty(), "FullLayerGroupPolicy must be initialized before reading masks");
    uint64_t all_full_mask = 0;
    for (const auto& [group_id, group] : groups_) {
        (void)group_id;
        all_full_mask |= group.group_name_bithash;
    }
    return {all_full_mask};
}

bool FullOtherGroupPolicy::init() {
    if (full_group_ids_.empty()) {
        RTP_LLM_LOG_ERROR("FullOtherLayerGroupPolicy: not support empty full groups");
        return false;
    }
    if (other_group_ids_.empty()) {
        RTP_LLM_LOG_ERROR("FullOtherLayerGroupPolicy: not support empty other groups");
        return false;
    }
    if (!DefaultLayerGroupPolicy::init()) {
        return false;
    }
    for (int full_id : full_group_ids_) {
        const auto it = groups_.find(full_id);
        if (it == groups_.end()) {
            RTP_LLM_LOG_ERROR("not find full group id [%d]", full_id);
            return false;
        }
        valid_full_bithash_ |= it->second.group_name_bithash;
        valid_full_other_bithash_ |= it->second.group_name_bithash;
    }
    for (int other_id : other_group_ids_) {
        const auto it = groups_.find(other_id);
        if (it == groups_.end()) {
            RTP_LLM_LOG_ERROR("not find other group id [%d]", other_id);
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

bool FullOtherGroupPolicy::getNeedWriteGroups(const StorageRequest&     request,
                                              size_t                    valid_keys_size,
                                              std::vector<std::string>& location_spec_group_names) const {
    RTP_LLM_CHECK(request.keys != nullptr);
    RTP_LLM_CHECK_WITH_INFO(valid_keys_size <= request.keys->size() && request.handles.size() == request.keys->size(),
                            "invalid hybrid storage write shape: valid=%zu keys=%zu handles=%zu",
                            valid_keys_size,
                            request.keys->size(),
                            request.handles.size());
    location_spec_group_names.resize(valid_keys_size);
    bool   exist_full_other  = false;
    size_t count             = write_interval_;
    bool   is_all_full_other = true;
    for (size_t key_idx = valid_keys_size; key_idx-- > 0;) {
        uint64_t groups_name_bithash = 0;
        for (const auto& handle : request.handles[key_idx]) {
            if (isNullBlockIdx(handle.block)) {
                continue;
            }
            const auto group = groups_.find(static_cast<int32_t>(handle.group_id));
            if (group == groups_.end()) {
                RTP_LLM_LOG_WARNING("hybrid remote write references unknown group_id [%zu]", handle.group_id);
                return false;
            }
            groups_name_bithash |= group->second.group_name_bithash;
        }
        if (groups_name_bithash != valid_full_bithash_ && groups_name_bithash != valid_full_other_bithash_) {
            RTP_LLM_LOG_WARNING("invalid hybrid remote group mask [%lu]", groups_name_bithash);
            return false;
        }
        if (write_interval_ > 0) {
            ++count;
            const bool need_full_other = groups_name_bithash == valid_full_other_bithash_ && count >= write_interval_;
            if (need_full_other) {
                groups_name_bithash = valid_full_other_bithash_;
                count               = 0;
            } else {
                groups_name_bithash = valid_full_bithash_;
                is_all_full_other   = false;
            }
        } else if (!exist_full_other && groups_name_bithash == valid_full_other_bithash_) {
            exist_full_other = true;
        } else {
            groups_name_bithash = valid_full_bithash_;
            is_all_full_other   = false;
        }
        location_spec_group_names[key_idx] = location_spec_group_map_.at(groups_name_bithash);
    }
    if (is_all_full_other) {
        location_spec_group_names.clear();
    }
    return true;
}

void FullOtherGroupPolicy::rebuildDerivedSpecInfo() {
    full_spec_name_bithash_.clear();
    full_other_spec_name_bithash_.clear();
    for (const auto& [spec_name, spec_info] : spec_name_to_info_) {
        const auto group_it = groups_.find(spec_info.group_id);
        RTP_LLM_CHECK_WITH_INFO(group_it != groups_.end(),
                                "KVCM spec [%s] references unknown group [%d]",
                                spec_name.c_str(),
                                spec_info.group_id);
        if (group_it->second.is_full) {
            full_spec_name_bithash_[spec_name] = group_it->second.group_name_bithash;
        }
        full_other_spec_name_bithash_[spec_name] = group_it->second.group_name_bithash;
    }
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
