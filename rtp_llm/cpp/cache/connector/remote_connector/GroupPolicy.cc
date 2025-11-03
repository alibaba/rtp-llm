#include <sstream>
#include <bitset>
#include <algorithm>
#include <typeinfo>
#include "rtp_llm/cpp/cache/connector/remote_connector/GroupPolicy.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
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
    const auto  layer_layout       = allocator_->allLayerCacheBase();
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
    const auto& group_block_ids = resource->groupBlocks();
    const auto& cache_keys      = resource->cacheKeys();
    location_spec_group_names.reserve(cache_keys.size());
    for (size_t key_idx = 0; key_idx < cache_keys.size(); key_idx++) {
        uint64_t groups_name_bithash = 0;
        for (const auto& [group_idx, group] : groups_) {
            const auto gpu_block_idx = group_block_ids.at(group_idx)->blocks().at(key_idx);
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
            if (buffer.kv_addr == nullptr || buffer.kv_addr->data() == nullptr) {
                RTP_LLM_LOG_WARNING(
                    "convertIndexToBuffer failed layer_id [%d] block_id[%d]", layer_ids[j], block_ids[i]);
                return false;
            }
            iovs.push_back(
                {kv_cache_manager::MemoryType::GPU, buffer.kv_addr->data(), buffer.kv_addr->sizeBytes(), false});

            if (buffer.kv_scale_addr != nullptr && buffer.kv_scale_addr->data() != nullptr) {
                iovs.push_back({kv_cache_manager::MemoryType::GPU,
                                buffer.kv_scale_addr->data(),
                                buffer.kv_scale_addr->sizeBytes(),
                                false});
            }
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

bool FullLayerGroupPolicy::getNeedWriteGroups(const std::shared_ptr<KVCacheResource>& resource,
                                              std::vector<std::string>&               location_spec_group_names) const {

    // do nothing
    return true;
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

bool FullOtherGroupPolicy::getNeedWriteGroups(const std::shared_ptr<KVCacheResource>& resource,
                                              std::vector<std::string>&               location_spec_group_names) const {
    const auto& group_block_ids = resource->groupBlocks();
    if (group_block_ids.size() != groups_.size()) {
        RTP_LLM_LOG_WARNING("group size not equal, expect [%lu], real [%lu]", groups_.size(), group_block_ids.size());
        return false;
    }
    const auto& cache_keys = resource->cacheKeys();
    location_spec_group_names.resize(cache_keys.size(), {});
    bool   exist_full_other  = false;
    size_t count             = write_interval_;
    bool   is_all_full_other = true;
    for (size_t key_idx = cache_keys.size(); key_idx-- > 0;) {
        uint64_t groups_name_bithash = 0;
        for (const auto& [group_idx, group] : groups_) {
            const auto gpu_block_idx = group_block_ids.at(group_idx)->blocks().at(key_idx);
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

bool FullOtherGroupPolicy::addSpecInfo(const std::string& spec_name, int32_t group_id, int32_t tp_rank) {
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
    full_other_spec_name_bithash_[spec_name] = group.group_name_bithash;
    return true;
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

bool FullSWLayerGroupPolicy::filterNeedLoadLocations(const kv_cache_manager::Locations& locations,
                                                     LocationsView&                     locations_view,
                                                     kv_cache_manager::BlockMaskOffset  block_mask) const {
    locations_view.reserve(locations.size());
    size_t need_load_sink_size = 0;
    if (block_mask < sink_size_) {
        // load sink
        // TODO : 这里的min要加个ut
        need_load_sink_size = std::min(sink_size_ - block_mask, locations.size());
        locations_view.resize(need_load_sink_size);
        for (size_t i = 0; i < need_load_sink_size; i++) {
            if (!CheckInvalidFullLocationAndSetView(locations[i], locations_view[i])) {
                return false;
            }
        }
    }
    size_t load_sw_size = 0;
    for (size_t i = locations.size(); i-- > need_load_sink_size;) {
        // TODO : 现在信息不够, 还需要已经匹配的窗口长度
        if (load_sw_size < sw_size_) {}
    }
    RTP_LLM_LOG_WARNING("Not Implement");
    return false;
}

std::string FullSWLayerGroupPolicy::debugString() const {
    std::stringstream debug_ss;
    debug_ss << FullOtherGroupPolicy::debugString();
    debug_ss << "\tsink_size_ : " << sink_size_ << '\n';
    debug_ss << "\tsw_size_ : " << sw_size_ << '\n';
    return debug_ss.str();
}

}  // namespace remote_connector
}  // namespace rtp_llm