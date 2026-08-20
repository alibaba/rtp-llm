#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>

#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/ModelProfile.h"

namespace rtp_llm::benchmark {

struct DescriptorSizeProfile {
    std::string                             profile_id{"descriptor_sizes"};
    std::unordered_map<std::string, size_t> descriptor_size_bytes;
    std::string                             sha256_hex;

    static DescriptorSizeProfile load(const std::string& json_path);
    static DescriptorSizeProfile fromString(const std::string& json_content);

    size_t       descriptorSizeBytes(const std::string& group_set) const;
    ModelProfile toSyntheticModelProfile() const;
};

}  // namespace rtp_llm::benchmark
