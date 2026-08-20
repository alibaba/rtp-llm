#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/DescriptorSizeProfile.h"

#include <array>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <utility>

#include <openssl/sha.h>
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"

namespace rtp_llm::benchmark {

namespace {

constexpr std::array<const char*, 2> kRequiredGroupSets = {"full_context", "swa"};

std::string readFile(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    const auto  size = input.tellg();
    std::string content(size, '\0');
    input.seekg(0);
    input.read(content.data(), size);
    return content;
}

std::string fileStem(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    const size_t begin = slash == std::string::npos ? 0 : slash + 1;
    const size_t dot   = path.find_last_of('.');
    const size_t end   = dot == std::string::npos || dot < begin ? path.size() : dot;
    return path.substr(begin, end - begin);
}

std::string computeSha256(const std::string& content) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(content.data()), content.size(), hash);
    char hex[SHA256_DIGEST_LENGTH * 2 + 1];
    for (size_t i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        std::snprintf(hex + i * 2, 3, "%02x", hash[i]);
    }
    return std::string(hex, SHA256_DIGEST_LENGTH * 2);
}

size_t requirePositiveSize(const rapidjson::Value& object, const char* name) {
    if (!object.HasMember(name) || !object[name].IsUint64() || object[name].GetUint64() == 0) {
        throw std::runtime_error(std::string("Descriptor size profile requires positive '") + name + "'");
    }
    return object[name].GetUint64();
}

}  // namespace

DescriptorSizeProfile DescriptorSizeProfile::load(const std::string& json_path) {
    auto profile       = fromString(readFile(json_path));
    profile.profile_id = fileStem(json_path);
    return profile;
}

DescriptorSizeProfile DescriptorSizeProfile::fromString(const std::string& json_content) {
    DescriptorSizeProfile profile;
    profile.sha256_hex = computeSha256(json_content);

    rapidjson::Document    doc;
    rapidjson::ParseResult ok = doc.Parse(json_content.data());
    if (!ok) {
        throw std::runtime_error("JSON parse error: " + std::string(rapidjson::GetParseError_En(ok.Code()))
                                 + " at offset " + std::to_string(ok.Offset()));
    }
    if (!doc.IsObject()) {
        throw std::runtime_error("Descriptor size profile must be a JSON object");
    }

    if (doc.HasMember("descriptor_size_bytes")) {
        const auto& sizes = doc["descriptor_size_bytes"];
        if (!sizes.IsObject()) {
            throw std::runtime_error("'descriptor_size_bytes' must be an object");
        }
        for (const char* name : kRequiredGroupSets) {
            profile.descriptor_size_bytes.emplace(name, requirePositiveSize(sizes, name));
        }
        return profile;
    }

    if (!doc.HasMember("group_sets") || !doc["group_sets"].IsArray()) {
        throw std::runtime_error("Descriptor size profile missing 'descriptor_size_bytes'");
    }
    for (const auto& group_set : doc["group_sets"].GetArray()) {
        if (!group_set.IsObject() || !group_set.HasMember("name") || !group_set["name"].IsString()
            || !group_set.HasMember("payload_bytes") || !group_set["payload_bytes"].IsUint64()
            || group_set["payload_bytes"].GetUint64() == 0) {
            continue;
        }
        profile.descriptor_size_bytes[group_set["name"].GetString()] = group_set["payload_bytes"].GetUint64();
    }
    for (const char* name : kRequiredGroupSets) {
        if (profile.descriptor_size_bytes.count(name) == 0) {
            throw std::runtime_error(std::string("Legacy model profile missing positive payload for '") + name + "'");
        }
    }
    return profile;
}

size_t DescriptorSizeProfile::descriptorSizeBytes(const std::string& group_set) const {
    const auto it = descriptor_size_bytes.find(group_set);
    if (it == descriptor_size_bytes.end()) {
        throw std::runtime_error("Unknown descriptor size group set: " + group_set);
    }
    return it->second;
}

ModelProfile DescriptorSizeProfile::toSyntheticModelProfile() const {
    ModelProfile profile;
    profile.profile_id = profile_id;
    profile.sha256_hex = sha256_hex;

    for (const char* name : kRequiredGroupSets) {
        const size_t         payload = descriptorSizeBytes(name);
        const CacheGroupType type = std::string(name) == "swa" ? CacheGroupType::SWA : CacheGroupType::FULL;
        const std::string    tag = std::string(name) + "_descriptor";

        GroupInfo group;
        group.tag                 = tag;
        group.layer_count         = 1;
        group.type                = type;
        group.sliding_window_size = type == CacheGroupType::SWA ? 1 : 0;
        group.layer_stride_bytes  = payload;
        group.group_payload_bytes = payload;
        profile.groups.push_back(std::move(group));

        GroupSetInfo group_set;
        group_set.name                = name;
        group_set.member_tags         = {tag};
        group_set.payload_bytes       = payload;
        group_set.group_type          = type;
        group_set.sliding_window_size = type == CacheGroupType::SWA ? 1 : 0;
        profile.group_sets.push_back(std::move(group_set));
    }
    return profile;
}

}  // namespace rtp_llm::benchmark
