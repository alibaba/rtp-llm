#pragma once

#include <cstdint>
#include <string>

namespace rtp_llm {

inline std::string validateInterleavedMropeConfig(
    int index_factor, int rope_dim, int size_per_head, int mrope_dim1, int mrope_dim2, int mrope_dim3) {
    if (index_factor != 3) {
        return "requires index_factor=3, got " + std::to_string(index_factor);
    }
    if (rope_dim <= 0 || rope_dim % 2 != 0) {
        return "requires a positive even rope dim, got " + std::to_string(rope_dim);
    }
    if (rope_dim > size_per_head) {
        return "rope dim exceeds size_per_head: rope dim=" + std::to_string(rope_dim)
               + ", size_per_head=" + std::to_string(size_per_head);
    }
    if (mrope_dim1 < 0 || mrope_dim2 < 0 || mrope_dim3 < 0) {
        return "sections must be non-negative";
    }

    const int64_t section_sum =
        static_cast<int64_t>(mrope_dim1) + static_cast<int64_t>(mrope_dim2) + static_cast<int64_t>(mrope_dim3);
    const int rotary_pair_count = rope_dim / 2;
    if (section_sum != rotary_pair_count) {
        return "section sum must equal rope dim / 2: got " + std::to_string(section_sum) + ", expected "
               + std::to_string(rotary_pair_count);
    }

    const int max_height_pairs = (rotary_pair_count + 1) / 3;
    const int max_width_pairs  = rotary_pair_count / 3;
    if (mrope_dim2 > max_height_pairs || mrope_dim3 > max_width_pairs) {
        return "sections exceed interleaved H/W capacity: H=" + std::to_string(mrope_dim2) + " (max "
               + std::to_string(max_height_pairs) + "), W=" + std::to_string(mrope_dim3) + " (max "
               + std::to_string(max_width_pairs) + ")";
    }
    return {};
}

}  // namespace rtp_llm
