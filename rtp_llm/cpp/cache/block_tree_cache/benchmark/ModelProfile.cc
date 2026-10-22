#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/ModelProfile.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <unordered_set>
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

size_t positiveSize(const rapidjson::Value& object, const char* field, const std::string& context) {
    if (!object.HasMember(field) || !object[field].IsUint64() || object[field].GetUint64() == 0
        || object[field].GetUint64() > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error(context + ": '" + field + "' must be a positive size integer");
    }
    return static_cast<size_t>(object[field].GetUint64());
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
    if (!group.IsObject() || !group.HasMember("tag") || !group["tag"].IsString()
        || group["tag"].GetStringLength() == 0) {
        throw std::runtime_error("Profile " + profile_id + ": group missing nonempty 'tag'");
    }
    const std::string context = "Profile " + profile_id + ": group " + group["tag"].GetString();
    if (!group.HasMember("type") || !group["type"].IsString()) {
        throw std::runtime_error(context + ": missing 'type'");
    }
    const auto   type    = parseGroupType(group["type"].GetString());
    const size_t layers  = positiveSize(group, "layer_count", context);
    const size_t stride  = positiveSize(group, "layer_stride_bytes", context);
    const size_t payload = positiveSize(group, "group_payload_bytes", context);
    if (layers > std::numeric_limits<size_t>::max() / stride || payload != layers * stride) {
        throw std::runtime_error(
            context + ": group_payload_bytes must equal layer_count * layer_stride_bytes without overflow");
    }
    if (group.HasMember("sliding_window_size")
        && (!group["sliding_window_size"].IsUint64()
            || group["sliding_window_size"].GetUint64() > std::numeric_limits<size_t>::max())) {
        throw std::runtime_error(context + ": invalid 'sliding_window_size'");
    }
    if (type == CacheGroupType::SWA) {
        if (positiveSize(group, "sliding_window_size", context)
            > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error(context + ": sliding_window_size exceeds the cache policy integer range");
        }
    } else if (group.HasMember("sliding_window_size") && group["sliding_window_size"].GetUint64() != 0) {
        throw std::runtime_error(context + ": FULL group must have zero sliding_window_size");
    }
}

void ModelProfile::validateGroupSet(const rapidjson::Value&       gs,
                                    const std::vector<GroupInfo>& groups,
                                    const std::string&            profile_id) {
    if (!gs.IsObject() || !gs.HasMember("name") || !gs["name"].IsString() || gs["name"].GetStringLength() == 0) {
        throw std::runtime_error("Profile " + profile_id + ": group_set missing 'name'");
    }
    if (!gs.HasMember("members") || !gs["members"].IsArray() || gs["members"].Empty()) {
        throw std::runtime_error("Profile " + profile_id + ": group_set " + gs["name"].GetString()
                                 + " missing 'members'");
    }
    std::unordered_set<std::string> seen_members;
    for (const auto& member : gs["members"].GetArray()) {
        if (!member.IsString()) {
            throw std::runtime_error("Profile " + profile_id + ": group_set " + gs["name"].GetString()
                                     + " member is not a string");
        }
        if (!seen_members.insert(member.GetString()).second) {
            throw std::runtime_error("Profile " + profile_id + ": duplicate group_set member " + member.GetString());
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
    rapidjson::ParseResult ok = doc.Parse(json_content.data(), json_content.size());
    if (!ok) {
        throw std::runtime_error("JSON parse error: " + std::string(rapidjson::GetParseError_En(ok.Code()))
                                 + " at offset " + std::to_string(ok.Offset()));
    }

    if (!doc.IsObject()) {
        throw std::runtime_error("Profile root must be an object");
    }

    // Required fields
    if (!doc.HasMember("profile_id") || !doc["profile_id"].IsString()) {
        throw std::runtime_error("Profile missing 'profile_id'");
    }
    profile.profile_id       = doc["profile_id"].GetString();
    profile.tokens_per_block = positiveSize(doc, "tokens_per_block", "Profile " + profile.profile_id);

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
        for (const auto& existing : profile.group_sets) {
            if (existing.name == info.name) {
                throw std::runtime_error("Profile " + profile.profile_id + ": duplicate group_set name " + info.name);
            }
        }
        for (const auto& tag : info.member_tags) {
            const size_t payload = profile.findGroup(tag)->group_payload_bytes;
            if (payload > std::numeric_limits<size_t>::max() - info.payload_bytes) {
                throw std::runtime_error("Profile " + profile.profile_id + ": group_set payload sum overflows");
            }
            info.payload_bytes += payload;
        }
        if (gs.HasMember("payload_bytes")
            && positiveSize(gs, "payload_bytes", "Profile " + profile.profile_id + ": group_set " + info.name)
                   != info.payload_bytes) {
            throw std::runtime_error("Profile " + profile.profile_id
                                     + ": group_set payload_bytes differs from member sum");
        }
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
