#include "rtp_llm/cpp/model_rpc/LinearTpCacheLayout.h"
#include "rtp_llm/cpp/model_rpc/MhaTpCacheLayout.h"
#include "rtp_llm/cpp/model_rpc/TpCacheTransferPlan.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

TEST(LinearTpCacheLayoutTest, CompletePlansRejectDuplicateMissingAndInvalidSlices) {
    auto first  = TpCacheTransferPlan::linear(32, 8, 8, 16, 3, 2, 0);
    auto second = TpCacheTransferPlan::linear(32, 8, 8, 16, 3, 2, 1);
    EXPECT_NO_THROW(validateCompleteTransfer({first, second}));
    EXPECT_THROW(validateCompleteTransfer({first}), std::invalid_argument);
    EXPECT_THROW(validateCompleteTransfer({first, first}), std::invalid_argument);
    second.slices.front().source_offset = second.source_bytes;
    EXPECT_THROW(validateCompleteTransfer({first, second}), std::invalid_argument);
}

TEST(LinearTpCacheLayoutTest, ReplicatedAndTwoHeadOwnersPreserveTheSupportedTopology) {
    for (size_t peers : {2u, 4u, 8u}) {
        TpCacheTransferTopology topology{1, peers, 1, 1};
        for (size_t peer = 0; peer < peers; ++peer) {
            EXPECT_EQ(replicatedTpCachePeerOwnsTag("full", true, 2, topology, peer), peer == 0 || peer == peers / 2);
            EXPECT_EQ(replicatedTpCachePeerOwnsTag("indexer_state", false, 0, topology, peer), peer == 0);
        }
        EXPECT_THROW(replicatedTpCachePeerOwnsTag("unknown", false, 0, topology, 0), std::invalid_argument);
    }
    TpCacheTransferTopology cp{1, 8, 2, 1};
    EXPECT_FALSE(cp.supportsTypedTransfer());
    TpCacheTransferTopology partitions{1, 8, 1, 2};
    EXPECT_FALSE(partitions.supportsTypedTransfer());
}

TEST(LinearTpCacheLayoutTest, RegistrationRoundingRejectsOverflow) {
    EXPECT_EQ(registeredCacheBytes(1), 65536u);
    EXPECT_EQ(registeredCacheBytes(65536), 65536u);
    EXPECT_EQ(registeredCacheBytes(65537), 131072u);
    EXPECT_THROW(registeredCacheBytes(0), std::overflow_error);
    EXPECT_THROW(registeredCacheBytes(std::numeric_limits<size_t>::max()), std::overflow_error);
}

TEST(LinearTpCacheLayoutTest, SelectsOnlyQwen4OpaqueAsymmetricTransfers) {
    for (const auto model : {"qwen4_exp", "qwen4_exp_mtp"}) {
        EXPECT_TRUE(useTypedTpCacheLayout(model, true, 8, 1));
        EXPECT_TRUE(useTypedTpCacheLayout(model, true, 1, 4));
        EXPECT_FALSE(useTypedTpCacheLayout(model, true, 1, 1));
        EXPECT_FALSE(useTypedTpCacheLayout(model, false, 8, 1));
    }
    for (const auto model : {"qwen3_next", "qwen3_moe", "deepseek_v3", ""}) {
        EXPECT_FALSE(useTypedTpCacheLayout(model, true, 8, 1));
        EXPECT_FALSE(useTypedTpCacheLayout(model, true, 1, 4));
    }
}

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

TEST(MhaTpCacheLayoutTest, ReassemblesTwoHeadsForDataAndFp8Scales) {
    for (size_t peers : {2u, 4u, 8u}) {
        for (size_t head_bytes : std::vector<size_t>{128u * 256u, 128u * 256u * 2u, 128u * sizeof(float)}) {
            const size_t         half = 2 * head_bytes;
            std::vector<uint8_t> reference(2 * half);
            for (size_t j = 0; j < reference.size(); ++j) {
                reference[j] = static_cast<uint8_t>((j * 29 + j / 113) % 251);
            }
            std::vector<uint8_t> destination(reference.size() + 32, 0xfe);
            std::vector<uint8_t> naive;
            size_t               owners = 0;
            for (size_t peer = 0; peer < peers; ++peer) {
                if (!mhaTpPeerOwnsHead(peers, 2, peer)) {
                    continue;
                }
                ++owners;
                const size_t         head = peer / (peers / 2);
                std::vector<uint8_t> source(2 * head_bytes + 32, 0xfd);
                std::copy_n(reference.data() + head * head_bytes, head_bytes, source.data());
                std::copy_n(reference.data() + half + head * head_bytes, head_bytes, source.data() + head_bytes);
                naive.insert(naive.end(), source.begin(), source.begin() + 2 * head_bytes);
                for (const auto& slice : mhaTpCacheSlices(half, half, 2, head)) {
                    ASSERT_LE(slice.source_offset + slice.bytes, 2 * head_bytes);
                    ASSERT_LE(slice.destination_offset + slice.bytes, reference.size());
                    std::memcpy(destination.data() + slice.destination_offset,
                                source.data() + slice.source_offset,
                                slice.bytes);
                }
                EXPECT_TRUE(std::all_of(
                    source.begin() + 2 * head_bytes, source.end(), [](auto value) { return value == 0xfd; }));
            }
            EXPECT_EQ(owners, 2);
            EXPECT_TRUE(std::equal(reference.begin(), reference.end(), destination.begin()));
            EXPECT_NE(naive, reference);
            EXPECT_TRUE(std::all_of(
                destination.begin() + reference.size(), destination.end(), [](auto value) { return value == 0xfe; }));
        }
    }
}

TEST(MhaTpCacheLayoutTest, RejectsInvalidGeometryAndOverflow) {
    EXPECT_THROW(mhaTpCacheSlices(64, 64, 0, 0), std::invalid_argument);
    EXPECT_THROW(mhaTpCacheSlices(65, 64, 2, 0), std::invalid_argument);
    EXPECT_THROW(mhaTpCacheSlices(64, 64, 2, 2), std::invalid_argument);
    EXPECT_THROW(mhaTpCacheSlices(0, 64, 2, 0), std::invalid_argument);
    EXPECT_THROW(mhaTpCacheSlices(std::numeric_limits<size_t>::max() - 1, 64, 2, 0), std::overflow_error);
    EXPECT_THROW(mhaTpPeerOwnsHead(3, 2, 0), std::invalid_argument);
    EXPECT_THROW(mhaTpPeerOwnsHead(4, 2, 4), std::invalid_argument);
    EXPECT_THROW(mhaTpPeerOwnsHead(0, 2, 0), std::invalid_argument);
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
