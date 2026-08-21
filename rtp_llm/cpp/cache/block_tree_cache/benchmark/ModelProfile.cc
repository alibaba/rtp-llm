#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/ModelProfile.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <openssl/sha.h>

#include "rapidjson/error/en.h"

namespace rtp_llm::benchmark {

namespace {

std::string readFile(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    auto        size = ifs.tellg();
    std::string content(size, '\0');
    ifs.seekg(0);
    ifs.read(content.data(), size);
    return content;
}

}  // anonymous namespace

CacheGroupType ModelProfile::parseGroupType(const std::string& type_str) {
    if (type_str == "FULL")
        return CacheGroupType::FULL;
    if (type_str == "SWA")
        return CacheGroupType::SWA;
    throw std::runtime_error("Unknown group type: " + type_str);
}

std::string ModelProfile::computeSha256(const std::string& content) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(content.data()), content.size(), hash);
    char hex[SHA256_DIGEST_LENGTH * 2 + 1];
    for (size_t i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        std::snprintf(hex + i * 2, 3, "%02x", hash[i]);
    }
    return std::string(hex, SHA256_DIGEST_LENGTH * 2);
}

void ModelProfile::validateGroup(const rapidjson::Value& group, const std::string& profile_id) {
    if (!group.HasMember("tag") || !group["tag"].IsString()) {
        throw std::runtime_error("Profile " + profile_id + ": group missing 'tag'");
    }
    if (!group.HasMember("layer_count") || !group["layer_count"].IsUint64()) {
        throw std::runtime_error("Profile " + profile_id + ": group " + group["tag"].GetString()
                                 + " missing 'layer_count'");
    }
    if (!group.HasMember("type") || !group["type"].IsString()) {
        throw std::runtime_error("Profile " + profile_id + ": group " + group["tag"].GetString() + " missing 'type'");
    }
    if (!group.HasMember("layer_stride_bytes") || !group["layer_stride_bytes"].IsUint64()) {
        throw std::runtime_error("Profile " + profile_id + ": group " + group["tag"].GetString()
                                 + " missing 'layer_stride_bytes'");
    }
    if (!group.HasMember("group_payload_bytes") || !group["group_payload_bytes"].IsUint64()) {
        throw std::runtime_error("Profile " + profile_id + ": group " + group["tag"].GetString()
                                 + " missing 'group_payload_bytes'");
    }
    const std::string type = group["type"].GetString();
    if (type == "SWA"
        && (!group.HasMember("sliding_window_size") || !group["sliding_window_size"].IsUint64()
            || group["sliding_window_size"].GetUint64() == 0)) {
        throw std::runtime_error("Profile " + profile_id + ": SWA group " + group["tag"].GetString()
                                 + " missing positive 'sliding_window_size'");
    }
}

void ModelProfile::validateGroupSet(const rapidjson::Value&       gs,
                                    const std::vector<GroupInfo>& groups,
                                    const std::string&            profile_id) {
    if (!gs.HasMember("name") || !gs["name"].IsString()) {
        throw std::runtime_error("Profile " + profile_id + ": group_set missing 'name'");
    }
    if (!gs.HasMember("members") || !gs["members"].IsArray()) {
        throw std::runtime_error("Profile " + profile_id + ": group_set " + gs["name"].GetString()
                                 + " missing 'members'");
    }
    for (const auto& member : gs["members"].GetArray()) {
        if (!member.IsString()) {
            throw std::runtime_error("Profile " + profile_id + ": group_set " + gs["name"].GetString()
                                     + " member is not a string");
        }
        bool found = false;
        for (const auto& g : groups) {
            if (g.tag == member.GetString()) {
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("Profile " + profile_id + ": group_set " + gs["name"].GetString()
                                     + " references unknown group " + member.GetString());
        }
    }
}

ModelProfile ModelProfile::load(const std::string& json_path) {
    auto content = readFile(json_path);
    return fromString(content);
}

ModelProfile ModelProfile::fromString(const std::string& json_content) {
    ModelProfile profile;
    profile.sha256_hex = computeSha256(json_content);

    rapidjson::Document    doc;
    rapidjson::ParseResult ok = doc.Parse(json_content.data());
    if (!ok) {
        throw std::runtime_error("JSON parse error: " + std::string(rapidjson::GetParseError_En(ok.Code()))
                                 + " at offset " + std::to_string(ok.Offset()));
    }

    // Required fields
    if (!doc.HasMember("profile_id") || !doc["profile_id"].IsString()) {
        throw std::runtime_error("Profile missing 'profile_id'");
    }
    profile.profile_id = doc["profile_id"].GetString();

    if (!doc.HasMember("groups") || !doc["groups"].IsArray() || doc["groups"].GetArray().Empty()) {
        throw std::runtime_error("Profile " + profile.profile_id + ": missing or empty 'groups'");
    }
    if (!doc.HasMember("group_sets") || !doc["group_sets"].IsArray() || doc["group_sets"].GetArray().Empty()) {
        throw std::runtime_error("Profile " + profile.profile_id + ": missing or empty 'group_sets'");
    }

    // Parse groups
    for (const auto& group : doc["groups"].GetArray()) {
        validateGroup(group, profile.profile_id);
        GroupInfo info;
        info.tag         = group["tag"].GetString();
        info.layer_count = group["layer_count"].GetUint64();
        info.type        = parseGroupType(group["type"].GetString());
        info.sliding_window_size =
            group.HasMember("sliding_window_size") ? group["sliding_window_size"].GetUint64() : 0;
        info.layer_stride_bytes  = group["layer_stride_bytes"].GetUint64();
        info.group_payload_bytes = group["group_payload_bytes"].GetUint64();

        // Check for duplicate tags
        for (const auto& existing : profile.groups) {
            if (existing.tag == info.tag) {
                throw std::runtime_error("Profile " + profile.profile_id + ": duplicate group tag " + info.tag);
            }
        }
        profile.groups.push_back(info);
    }

    // Parse group_sets
    for (const auto& gs : doc["group_sets"].GetArray()) {
        validateGroupSet(gs, profile.groups, profile.profile_id);
        GroupSetInfo info;
        info.name = gs["name"].GetString();
        for (const auto& member : gs["members"].GetArray()) {
            info.member_tags.push_back(member.GetString());
        }
        info.payload_bytes = gs.HasMember("payload_bytes") ? gs["payload_bytes"].GetUint64() : 0;
        // Resolve one consistent type/window across every flattened member.
        if (!info.member_tags.empty()) {
            const auto* first_group = profile.findGroup(info.member_tags.front());
            if (first_group != nullptr) {
                info.group_type          = first_group->type;
                info.sliding_window_size = first_group->sliding_window_size;
                for (const auto& member_tag : info.member_tags) {
                    const auto* member = profile.findGroup(member_tag);
                    if (member == nullptr || member->type != info.group_type
                        || member->sliding_window_size != info.sliding_window_size) {
                        throw std::runtime_error("Profile " + profile.profile_id + ": group_set " + info.name
                                                 + " mixes cache type or sliding-window policy");
                    }
                }
            }
        }
        profile.group_sets.push_back(info);
    }

    return profile;
}

const GroupInfo* ModelProfile::findGroup(const std::string& tag) const {
    for (const auto& g : groups) {
        if (g.tag == tag)
            return &g;
    }
    return nullptr;
}

const GroupSetInfo* ModelProfile::findGroupSet(const std::string& name) const {
    for (const auto& gs : group_sets) {
        if (gs.name == name)
            return &gs;
    }
    return nullptr;
}

size_t ModelProfile::computeGroupSetPayloadBytes(const std::string& name) const {
    const auto* gs = findGroupSet(name);
    if (!gs)
        return 0;
    size_t total = 0;
    for (const auto& tag : gs->member_tags) {
        const auto* group = findGroup(tag);
        if (group)
            total += group->group_payload_bytes;
    }
    return total;
}

}  // namespace rtp_llm::benchmark
