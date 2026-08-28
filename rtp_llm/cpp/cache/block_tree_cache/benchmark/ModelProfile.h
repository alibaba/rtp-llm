#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "rapidjson/document.h"

namespace rtp_llm::benchmark {

enum class CacheGroupType {
    FULL,
    SWA
};
struct GroupInfo {
    std::string    tag;
    size_t         layer_count{0};
    CacheGroupType type{CacheGroupType::FULL};
    size_t         sliding_window_size{0};
    size_t         layer_stride_bytes{0};
    size_t         group_payload_bytes{0};
};

struct GroupSetInfo {
    std::string              name;
    std::vector<std::string> member_tags;
    size_t                   payload_bytes{0};
    // Resolved group type: all members share the same policy type.
    // FULL members have sliding_window_size=0; SWA members record the resolved
    // sliding window.
    CacheGroupType group_type{CacheGroupType::FULL};
    size_t         sliding_window_size{0};
};

struct ModelProfile {
    std::string               profile_id;
    std::vector<GroupInfo>    groups;
    std::vector<GroupSetInfo> group_sets;
    std::string               sha256_hex;

    static ModelProfile load(const std::string& json_path);
    static ModelProfile fromString(const std::string& json_content);

    const GroupInfo*    findGroup(const std::string& tag) const;
    const GroupSetInfo* findGroupSet(const std::string& name) const;
    size_t              computeGroupSetPayloadBytes(const std::string& name) const;

private:
    static CacheGroupType parseGroupType(const std::string& type_str);
    static std::string    computeSha256(const std::string& content);
    static void           validateGroup(const rapidjson::Value& group, const std::string& profile_id);
    static void
    validateGroupSet(const rapidjson::Value& gs, const std::vector<GroupInfo>& groups, const std::string& profile_id);
};

}  // namespace rtp_llm::benchmark
