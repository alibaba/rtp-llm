#include <sstream>
#include <bitset>
#include <algorithm>
#include "rtp_llm/cpp/cache_new/remote_connector/GroupPolicy.h"
#include "rtp_llm/cpp/cache_new/types.h"
#include "rtp_llm/cpp/cache_new/KVCacheAllocator.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace remote_connector {

namespace {

std::string getBitHashStr(uint64_t bithash, size_t width = 64) {
    std::string res = std::bitset<64>(bithash).to_string().substr(64 - width);
    return "0b" + res;
}

}  // namespace

bool GroupPolicy::addSpecInfo(const std::string& spec_name, int32_t group_id, int32_t tp_rank) {
    if (spec_name_to_info_.find(spec_name) != spec_name_to_info_.end()) {
        RTP_LLM_LOG_ERROR("spec_name [%s] already exist", spec_name.c_str());
        return false;
    }
    spec_name_to_info_[spec_name] = {group_id, tp_rank};
    return true;
}

std::string GroupPolicy::debugString() const {
    size_t            gs = groups_.size();
    std::stringstream debug_ss;
    debug_ss << "groups :\n";
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
        debug_ss << '\t' << entry.first << ":[" << spec_info.group_id << "|" << spec_info.tp_rank << "]\n";
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
    const auto  layer_layout       = allocator_->layerCacheBase();
    uint64_t    group_name_bithash = 1;
    const auto& layer_to_groups    = layer_layout.layer_to_groups;
    for (int layer = 0; layer < static_cast<int>(layer_to_groups.size()); ++layer) {
        const int group_idx     = layer_to_groups.at(layer);
        bool      is_full_group = false;
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
            std::string group_name         = is_full_group ? ("F" + std::to_string(group_idx)) :
                                                             (GetOtherGroupPrefixName() + std::to_string(group_idx));
            groups_[group_idx]             = Group{is_full_group, group_name_bithash, group_name};
            group_to_layer_ids_[group_idx] = {};
            group_name_bithash <<= 1;
        }
        group_to_layer_ids_.at(group_idx).push_back(layer);
    }
    if (groups_.size() > 64) {
        RTP_LLM_LOG_ERROR("not support bigger than 64 groups");
        return false;
    }
    return true;
}

bool DefaultLayerGroupPolicy::filterNeedLoadLocations(const kv_cache_manager::Locations& locations,
                                                      LocationsView&                     locations_view) const {
    //  just copy
    locations_view.resize(locations.size(), {});
    for (size_t i = 0; i < locations.size(); i++) {
        locations_view[i].reserve(locations[i].size());
        for (const auto& unit : locations[i]) {
            locations_view[i].emplace_back(unit);
        }
    }
    return true;
}

bool DefaultLayerGroupPolicy::getNeedWriteGroups(const std::shared_ptr<KVCacheResourceV1>& resource,
                                                 std::vector<std::string>& location_spec_group_names) const {
    const auto& group_block_ids = resource->group_block_ids;
    const auto& cache_keys      = resource->cache_keys;
    location_spec_group_names.reserve(cache_keys.size());
    for (size_t key_idx = 0; key_idx < cache_keys.size(); key_idx++) {
        uint64_t groups_name_bithash = 0;
        for (const auto& [group_idx, group] : groups_) {
            const auto gpu_block_idx = group_block_ids.at(group_idx)->block_indices.at(key_idx);
            if (!isNullBlockIdx(gpu_block_idx)) {
                groups_name_bithash |= group.group_name_bithash;
            }
        }
        location_spec_group_names.push_back(location_spec_group_map_.at(groups_name_bithash));
    }
    return true;
}

bool DefaultLayerGroupPolicy::genBlockBuffers(const std::vector<int32_t>&     group_ids,
                                              const std::vector<int32_t>&     block_ids,
                                              kv_cache_manager::BlockBuffers& block_buffers) const {
    block_buffers.reserve(block_ids.size());
    for (size_t i = 0; i < block_ids.size(); ++i) {
        block_buffers.push_back({});
        const auto& layer_ids = group_to_layer_ids_.at(group_ids[i]);
        auto&       iovs      = block_buffers.back().iovs;
        iovs.reserve(layer_ids.size() * 2);
        for (size_t j = 0; j < layer_ids.size(); ++j) {
            const auto& buffer = allocator_->convertIndexToBuffer(layer_ids[j], block_ids[i]);
            if (buffer.k_addr == nullptr || buffer.v_addr == nullptr || buffer.k_addr->data() == nullptr
                || buffer.v_addr->data() == nullptr) {
                RTP_LLM_LOG_WARNING(
                    "convertIndexToBuffer failed layer_id [%d] block_id[%d]", layer_ids[j], block_ids[i]);
                return false;
            }
            iovs.push_back(
                {kv_cache_manager::MemoryType::GPU, buffer.k_addr->data(), buffer.k_addr->sizeBytes(), false});
            iovs.push_back(
                {kv_cache_manager::MemoryType::GPU, buffer.v_addr->data(), buffer.k_addr->sizeBytes(), false});
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
    if (full_group_ids_.size() != 1) {
        RTP_LLM_LOG_ERROR("FullLayerGroupPolicy only support one full group, real group size [%lu]",
                          full_group_ids_.size());
        return false;
    }
    if (!other_group_ids_.empty()) {
        RTP_LLM_LOG_ERROR("FullLayerGroupPolicy not support other groups");
        return false;
    }
    return DefaultLayerGroupPolicy::init();
}

bool FullLayerGroupPolicy::getNeedWriteGroups(const std::shared_ptr<KVCacheResourceV1>& resource,
                                              std::vector<std::string>& location_spec_group_names) const {

    // do nothing
    return true;
}

bool FullLinearLayerGroupPolicy::init() {
    if (full_group_ids_.empty()) {
        RTP_LLM_LOG_ERROR("FullLinearLayerGroupPolicy: not support empty full groups");
        return false;
    }
    if (other_group_ids_.empty()) {
        RTP_LLM_LOG_ERROR("FullLinearLayerGroupPolicy: not support empty linear groups");
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
        valid_full_linear_bithash_ |= it->second.group_name_bithash;
    }
    for (int linear_id : other_group_ids_) {
        const auto it = groups_.find(linear_id);
        if (it == groups_.end()) {
            RTP_LLM_LOG_ERROR("not find linear group id [%d]", linear_id);
            return false;
        }
        valid_full_linear_bithash_ |= it->second.group_name_bithash;
    }
    if (groups_.size() < 2) {
        RTP_LLM_LOG_ERROR("FullLinearLayerGroupPolicy: invalid group size [%lu]", groups_.size());
        return false;
    }
    return true;
}

bool FullLinearLayerGroupPolicy::addSpecInfo(const std::string& spec_name, int32_t group_id, int32_t tp_rank) {
    if (!GroupPolicy::addSpecInfo(spec_name, group_id, tp_rank)) {
        return false;
    }
    const auto& it = groups_.find(group_id);
    if (it == groups_.end()) {
        RTP_LLM_LOG_ERROR("not find group_id [%d], spec_name [%s]", group_id, spec_name.c_str());
        return false;
    }
    const auto& group = it->second;
    if (group.is_full) {
        full_spec_name_bithash_[spec_name] = group.group_name_bithash;
    }
    full_linear_spec_name_bithash_[spec_name] = group.group_name_bithash;
    return true;
}

bool FullLinearLayerGroupPolicy::filterNeedLoadLocations(const kv_cache_manager::Locations& locations,
                                                         LocationsView&                     locations_view) const {
#define SET_LOCATIONS_VIEW(attention_name)                                                                             \
    locations_view[i].reserve(location.size());                                                                        \
    uint64_t attention_name##_bithash = 0;                                                                             \
    for (const auto& unit : location) {                                                                                \
        const auto iter = attention_name##_spec_name_bithash_.find(unit.spec_name);                                    \
        if (iter == attention_name##_spec_name_bithash_.end()) {                                                       \
            RTP_LLM_LOG_WARNING("not find " #attention_name " spec name [%s]", unit.spec_name.c_str());                \
            return false;                                                                                              \
        }                                                                                                              \
        attention_name##_bithash |= iter->second;                                                                      \
        locations_view[i].emplace_back(unit);                                                                          \
    }                                                                                                                  \
    if (attention_name##_bithash != valid_##attention_name##_bithash_) {                                               \
        RTP_LLM_LOG_WARNING("invalid " #attention_name " bithash [%lu], expect [%lu]",                                 \
                            attention_name##_bithash,                                                                  \
                            valid_##attention_name##_bithash_);                                                        \
        return false;                                                                                                  \
    }

    bool exist_linear_location = false;
    for (size_t i = locations.size(); i-- > 0;) {
        const auto& location = locations[i];
        if (location.size() == full_spec_name_bithash_.size()) {
            if (exist_linear_location) {
                SET_LOCATIONS_VIEW(full);
            } else {
                // only do check
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
            }
        } else if (location.size() == full_linear_spec_name_bithash_.size()) {
            if (!exist_linear_location) {
                locations_view.resize(i + 1, {});
                SET_LOCATIONS_VIEW(full_linear);
                exist_linear_location = true;
            } else {
                locations_view[i].reserve(full_spec_name_bithash_.size());
                uint64_t full_bithash = 0;
                for (const auto& unit : location) {
                    const auto iter = full_spec_name_bithash_.find(unit.spec_name);
                    if (iter == full_spec_name_bithash_.end()) {
                        RTP_LLM_LOG_DEBUG("skip spec_name [%s]", unit.spec_name.c_str());
                        continue;
                    }
                    full_bithash |= iter->second;
                    locations_view[i].emplace_back(unit);
                }
                if (full_bithash != valid_full_bithash_) {
                    RTP_LLM_LOG_WARNING("invalid full bithash [%lu], expect [%lu]", full_bithash, valid_full_bithash_);
                    return false;
                }
            }
        } else {
            RTP_LLM_LOG_WARNING("invalid spec size, full [%lu], linear [%lu], real [%lu]",
                                full_spec_name_bithash_.size(),
                                full_linear_spec_name_bithash_.size(),
                                location.size());
            return false;
        }
    }
    return true;
#undef SET_LOCATIONS_VIEW
}

bool FullLinearLayerGroupPolicy::getNeedWriteGroups(const std::shared_ptr<KVCacheResourceV1>& resource,
                                                    std::vector<std::string>& location_spec_group_names) const {
    const auto& group_block_ids = resource->group_block_ids;
    if (group_block_ids.size() != groups_.size()) {
        RTP_LLM_LOG_WARNING("group size not equal, expect [%lu], real [%lu]", groups_.size(), group_block_ids.size());
        return false;
    }
    const auto& cache_keys = resource->cache_keys;
    location_spec_group_names.resize(cache_keys.size(), {});
    bool   exist_full_linear  = false;
    size_t count              = linear_attention_write_interval_;
    bool   is_all_full_linear = true;
    for (size_t key_idx = cache_keys.size(); key_idx-- > 0;) {
        uint64_t groups_name_bithash = 0;
        for (const auto& [group_idx, group] : groups_) {
            const auto gpu_block_idx = group_block_ids.at(group_idx)->block_indices.at(key_idx);
            if (!rtp_llm::isNullBlockIdx(gpu_block_idx)) {
                groups_name_bithash |= group.group_name_bithash;
            }
        }
        if (groups_name_bithash != valid_full_bithash_ && groups_name_bithash != valid_full_linear_bithash_) {
            RTP_LLM_LOG_WARNING("invalid groups_name_bithash [%lu]", groups_name_bithash);
            return false;
        }
        if (linear_attention_write_interval_ > 0) {
            count++;
            bool need_full_linear =
                (groups_name_bithash == valid_full_linear_bithash_) && (count >= linear_attention_write_interval_);
            if (need_full_linear) {
                groups_name_bithash = valid_full_linear_bithash_;
                count               = 0;
            } else {
                groups_name_bithash = valid_full_bithash_;
                is_all_full_linear  = false;
            }
            location_spec_group_names[key_idx] = location_spec_group_map_.at(groups_name_bithash);
        } else {
            if (!exist_full_linear && groups_name_bithash == valid_full_linear_bithash_) {
                location_spec_group_names[key_idx] = location_spec_group_map_.at(valid_full_linear_bithash_);
                exist_full_linear                  = true;
            } else {
                location_spec_group_names[key_idx] = location_spec_group_map_.at(valid_full_bithash_);
                is_all_full_linear                 = false;
            }
        }
    }
    if (is_all_full_linear) {
        location_spec_group_names.clear();
    }
    return true;
}

std::string FullLinearLayerGroupPolicy::debugString() const {
    size_t            gs = groups_.size();
    std::stringstream debug_ss;
    debug_ss << DefaultLayerGroupPolicy::debugString();
    debug_ss << "linear_attention_write_interval : " << linear_attention_write_interval_ << '\n';
    debug_ss << "valid_full_bithash : " << getBitHashStr(valid_full_bithash_, gs) << '\n';
    debug_ss << "valid_full_linear_bithash : " << getBitHashStr(valid_full_linear_bithash_, gs) << '\n';
    debug_ss << "full_spec_name_bithash : ";
    for (const auto& entry : full_spec_name_bithash_) {
        debug_ss << '[' << entry.first << ":" << getBitHashStr(entry.second, gs) << ']';
    }
    debug_ss << '\n';
    debug_ss << "full_linear_spec_name_bithash : ";
    for (const auto& entry : full_linear_spec_name_bithash_) {
        debug_ss << '[' << entry.first << ":" << getBitHashStr(entry.second, gs) << ']';
    }
    debug_ss << '\n';
    return debug_ss.str();
}

bool FullSWLayerGroupPolicy::init() {
    return false;
}

bool FullSWLayerGroupPolicy::filterNeedLoadLocations(const kv_cache_manager::Locations& locations,
                                                     LocationsView&                     locations_view) const {
    return false;
}

bool FullSWLayerGroupPolicy::getNeedWriteGroups(const std::shared_ptr<KVCacheResourceV1>& resource,
                                                std::vector<std::string>& location_spec_group_names) const {
    return false;
}

std::string FullSWLayerGroupPolicy::debugString() const {
    std::stringstream debug_ss;
    debug_ss << DefaultLayerGroupPolicy::debugString();
    debug_ss << "\tsink_size_ : " << sink_size_ << '\n';
    debug_ss << "\tsw_size_ : " << sw_size_ << '\n';
    return debug_ss.str();
}

}  // namespace remote_connector
}  // namespace rtp_llm