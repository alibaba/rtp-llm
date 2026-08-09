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
enum class TierPolicy {
    DEVICE_ONLY,
    DEVICE_HOST_DISK
};

struct GroupInfo {
    std::string    tag;
    size_t         layer_count{0};
    CacheGroupType type{CacheGroupType::FULL};
    size_t         entries_per_block{0};
    size_t         layer_stride_bytes{0};
    size_t         group_payload_bytes{0};
    TierPolicy     tier_policy{TierPolicy::DEVICE_HOST_DISK};
};

struct GroupSetInfo {
    std::string              name;
    std::vector<std::string> member_tags;
    size_t                   payload_bytes{0};
};

struct CapacityInfo {
    size_t device_block_count{0};
    size_t host_block_count{0};
    size_t disk_block_count{0};
};

struct ModelProfile {
    std::string               profile_id;
    std::string               model_name;
    std::string               dtype;
    size_t                    tp_size{1};
    size_t                    prefill_cp_size{1};
    size_t                    tokens_per_block{128};
    size_t                    kernel_tokens_block{128};
    std::vector<GroupInfo>    groups;
    std::vector<GroupSetInfo> group_sets;
    std::vector<std::string>  device_only_groups;
    CapacityInfo              default_capacity;
    std::string               sha256_hex;

    static ModelProfile load(const std::string& json_path);
    static ModelProfile fromString(const std::string& json_content);

    const GroupInfo*    findGroup(const std::string& tag) const;
    const GroupSetInfo* findGroupSet(const std::string& name) const;
    bool                isDeviceOnly(const std::string& tag) const;

    size_t computeGroupSetPayloadBytes(const std::string& name) const;

private:
    static CacheGroupType parseGroupType(const std::string& type_str);
    static TierPolicy     parseTierPolicy(const std::string& policy_str);
    static std::string    computeSha256(const std::string& content);
    static void           validateGroup(const rapidjson::Value& group, const std::string& profile_id);
    static void
    validateGroupSet(const rapidjson::Value& gs, const std::vector<GroupInfo>& groups, const std::string& profile_id);
};

}  // namespace rtp_llm::benchmark