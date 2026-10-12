#include "rtp_llm/cpp/model_rpc/LinearTpCacheLayout.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

TEST(LinearTpCacheLayoutTest, ReassemblesStateAndEveryQkvHistoryRow) {
    for (size_t decode_tp : {1u, 2u}) {
        for (size_t prefill_tp : {2u, 4u, 8u}) {
            if (prefill_tp <= decode_tp) {
                continue;
            }
            constexpr size_t global_state      = 48 * 5 * 5 * sizeof(float);
            constexpr size_t global_q          = 16 * 5 * sizeof(uint16_t);
            constexpr size_t global_v          = 48 * 5 * sizeof(uint16_t);
            constexpr size_t rows              = 3;
            const size_t     destination_bytes = (global_state + rows * (2 * global_q + global_v)) / decode_tp;
            const size_t     source_bytes      = destination_bytes * decode_tp / prefill_tp;
            const size_t     parts             = prefill_tp / decode_tp;
            for (size_t decode_rank = 0; decode_rank < decode_tp; ++decode_rank) {
                std::vector<uint8_t> reference(destination_bytes);
                for (size_t j = 0; j < reference.size(); ++j) {
                    reference[j] = static_cast<uint8_t>((j * 47 + j / 97 + decode_rank * 53) % 251);
                }
                std::vector<uint8_t> destination(destination_bytes + 64, 0xfe);
                std::vector<uint8_t> naive;
                for (size_t part = 0; part < parts; ++part) {
                    // Build each prefill rank independently from the logical
                    // destination head/channel layout, not from the plan.
                    std::vector<uint8_t> source(source_bytes + 32, 0xfd);
                    const size_t         source_state = global_state / prefill_tp;
                    std::copy_n(reference.data() + part * source_state, source_state, source.data());
                    const size_t destination_state = global_state / decode_tp;
                    for (size_t row = 0; row < rows; ++row) {
                        size_t source_offset      = source_state + row * (2 * global_q + global_v) / prefill_tp;
                        size_t destination_offset = destination_state + row * (2 * global_q + global_v) / decode_tp;
                        for (size_t width : {global_q, global_q, global_v}) {
                            const size_t local_width = width / prefill_tp;
                            std::copy_n(reference.data() + destination_offset + part * local_width,
                                        local_width,
                                        source.data() + source_offset);
                            source_offset += local_width;
                            destination_offset += width / decode_tp;
                        }
                    }
                    naive.insert(naive.end(), source.begin(), source.begin() + source_bytes);
                    for (const auto& slice : linearTpCacheSlices(global_state / decode_tp,
                                                                 global_q / decode_tp,
                                                                 global_q / decode_tp,
                                                                 global_v / decode_tp,
                                                                 rows,
                                                                 parts,
                                                                 part)) {
                        ASSERT_LE(slice.source_offset + slice.bytes, source_bytes);
                        ASSERT_LE(slice.destination_offset + slice.bytes, destination_bytes);
                        std::memcpy(destination.data() + slice.destination_offset,
                                    source.data() + slice.source_offset,
                                    slice.bytes);
                    }
                    EXPECT_TRUE(std::all_of(
                        source.begin() + source_bytes, source.end(), [](auto byte) { return byte == 0xfd; }));
                }
                EXPECT_TRUE(std::equal(reference.begin(), reference.end(), destination.begin()));
                EXPECT_NE(naive, reference);
                EXPECT_TRUE(std::all_of(destination.begin() + destination_bytes, destination.end(), [](auto byte) {
                    return byte == 0xfe;
                }));
            }
        }
    }
}

TEST(LinearTpCacheLayoutTest, RejectsInvalidGeometryBeforeTransfer) {
    EXPECT_THROW(linearTpCacheSlices(64, 16, 16, 48, 3, 0, 0), std::invalid_argument);
    EXPECT_THROW(linearTpCacheSlices(64, 16, 16, 48, 3, 2, 2), std::invalid_argument);
    EXPECT_THROW(linearTpCacheSlices(64, 16, 16, 48, 0, 2, 0), std::invalid_argument);
    EXPECT_THROW(linearTpCacheSlices(65, 16, 16, 48, 3, 2, 0), std::invalid_argument);
    EXPECT_THROW(linearTpCacheSlices(64, 15, 16, 48, 3, 2, 0), std::invalid_argument);
    EXPECT_THROW(linearTpCacheSlices(64, 16, 0, 48, 3, 2, 0), std::invalid_argument);
}

TEST(LinearTpCacheLayoutTest, RejectsSizeOverflow) {
    const auto max = std::numeric_limits<size_t>::max();
    EXPECT_THROW(linearTpCacheSlices(64, max, 16, 48, 3, 1, 0), std::overflow_error);
    EXPECT_THROW(linearTpCacheSlices(64, 16, 16, 48, max, 1, 0), std::overflow_error);
}

}  // namespace
}  // namespace rtp_llm
