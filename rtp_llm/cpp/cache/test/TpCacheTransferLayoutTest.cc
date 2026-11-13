#include "rtp_llm/cpp/cache/LinearTpCacheLayout.h"
#include "rtp_llm/cpp/cache/MhaTpCacheLayout.h"
#include "rtp_llm/cpp/cache/TpCacheTransferPlan.h"

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
            EXPECT_EQ(tpCachePeerOwnsPayload(TpCacheTransferMode::MHA_HEADS, 2, topology, peer),
                      peer == 0 || peer == peers / 2);
            EXPECT_EQ(tpCachePeerOwnsPayload(TpCacheTransferMode::REPLICATED, 0, topology, peer), peer == 0);
        }
        EXPECT_THROW(tpCachePeerOwnsPayload(TpCacheTransferMode::LEGACY, 0, topology, 0), std::invalid_argument);
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

TEST(LinearTpCacheLayoutTest, UndeclaredLayoutsKeepLegacyTransport) {
    EXPECT_TRUE(useTypedTpCacheLayout(true, true, 8, 1));
    EXPECT_TRUE(useTypedTpCacheLayout(true, true, 1, 4));
    EXPECT_FALSE(useTypedTpCacheLayout(true, true, 1, 1));
    EXPECT_FALSE(useTypedTpCacheLayout(true, false, 8, 1));
    EXPECT_FALSE(useTypedTpCacheLayout(false, true, 8, 1));
}

TEST(LinearTpCacheLayoutTest, DeclaredGeometryBuildsPlansWithoutModelOrTag) {
    TpCacheTransferLayout layout;
    layout.mode        = TpCacheTransferMode::LINEAR_QKV;
    layout.q_row_bytes = layout.k_row_bytes = 8;
    layout.v_row_bytes                      = 16;
    layout.history_rows                     = 3;
    layout.key_heads                        = 2;
    layout.value_heads                      = 4;
    TpCacheTransferTopology topology{1, 2, 1, 1};
    auto                    first  = layout.plan(32, 96, 4, topology, 0);
    auto                    second = layout.plan(32, 96, 4, topology, 1);
    EXPECT_EQ(first.destination_bytes, 128u);
    EXPECT_NO_THROW(validateCompleteTransfer({first, second}));
    EXPECT_THROW(layout.plan(32, 95, 4, topology, 0), std::invalid_argument);
    layout.key_heads = 1;
    EXPECT_THROW(layout.plan(32, 96, 4, topology, 0), std::invalid_argument);
    layout.mode = TpCacheTransferMode::MHA_HEADS;
    EXPECT_THROW(layout.plan(32, 32, 2, TpCacheTransferTopology{1, 4, 1, 1}, 1), std::invalid_argument);
    EXPECT_THROW(layout.plan(32, 32, 4, topology, 0), std::invalid_argument);
    EXPECT_THROW(layout.plan(32, 32, 2, TpCacheTransferTopology{1, 2, 2, 1}, 0), std::invalid_argument);
}

TEST(TpCacheTransferRouteTest, ModesSharePeerSelectionAndExecutionPlanning) {
    const TpCacheTransferTopology topology{1, 4, 1, 1};
    const TpCacheTransferRoute    fallback{false, 2, 1, 1, 0};
    const TpCacheTransferRoute    heads{true, 1, 0, 4, 2};
    for (const auto mode : {TpCacheTransferMode::LEGACY,
                            TpCacheTransferMode::REPLICATED,
                            TpCacheTransferMode::MHA_HEADS,
                            TpCacheTransferMode::LINEAR_QKV}) {
        TpCacheTransferLayout layout;
        layout.mode       = mode;
        const auto direct = layout.route(2, topology, 0, false, fallback, heads);
        EXPECT_EQ(direct.selected, mode == TpCacheTransferMode::LINEAR_QKV);
        EXPECT_EQ(direct.segmented, mode == TpCacheTransferMode::LINEAR_QKV);
        EXPECT_FALSE(direct.reassemble);
        EXPECT_EQ(direct.local_partition_count, direct.segmented ? 1u : 2u);
        EXPECT_EQ(direct.remote_partition_count, direct.segmented ? 4u : 1u);
        EXPECT_EQ(direct.remote_partition_id, direct.segmented ? 2u : 0u);
        for (size_t peer = 0; peer < topology.peers; ++peer) {
            if (mode == TpCacheTransferMode::LEGACY) {
                EXPECT_THROW(layout.route(2, topology, peer, true, fallback, heads), std::invalid_argument);
                continue;
            }
            const auto typed = layout.route(2, topology, peer, true, fallback, heads);
            const bool owner = mode == TpCacheTransferMode::LINEAR_QKV
                               || (mode == TpCacheTransferMode::MHA_HEADS ? peer == 0 || peer == 2 : peer == 0);
            EXPECT_EQ(typed.selected, owner);
            EXPECT_EQ(typed.reassemble, owner && mode != TpCacheTransferMode::REPLICATED);
            EXPECT_FALSE(typed.segmented);
            EXPECT_EQ(typed.local_partition_count, fallback.local_partition_count);
            EXPECT_EQ(typed.local_partition_id, fallback.local_partition_id);
        }
    }
}

TEST(LinearTpCacheLayoutTest, PartitionedPlansPreserveStateAndHistoryAcrossAllTpRatios) {
    constexpr size_t          state = 48 * 5 * 5 * sizeof(float);
    constexpr size_t          q     = 16 * 5 * sizeof(uint16_t);
    constexpr size_t          v     = 48 * 5 * sizeof(uint16_t);
    constexpr size_t          rows  = 3;
    const std::vector<size_t> widths{state, q, q, v, q, q, v, q, q, v};
    std::vector<uint8_t>      global(state + rows * (2 * q + v));
    for (size_t i = 0; i < global.size(); ++i) {
        global[i] = static_cast<uint8_t>((i * 47 + i / 97) % 251);
    }
    // Serialize each rank independently from the global state/head layout.
    const auto serialize = [&](size_t tp, size_t rank) {
        std::vector<uint8_t> local;
        size_t               offset = 0;
        for (const auto width : widths) {
            const auto begin = global.begin() + offset + rank * (width / tp);
            local.insert(local.end(), begin, begin + width / tp);
            offset += width;
        }
        return local;
    };
    for (const size_t prefill_tp : {1u, 2u, 4u, 8u}) {
        for (const size_t decode_tp : {1u, 2u, 4u, 8u}) {
            TpCacheTransferLayout layout;
            layout.mode        = TpCacheTransferMode::LINEAR_QKV;
            layout.q_row_bytes = layout.k_row_bytes = q / decode_tp;
            layout.v_row_bytes                      = v / decode_tp;
            layout.history_rows                     = rows;
            for (size_t rank = 0; rank < decode_tp; ++rank) {
                const auto           reference = serialize(decode_tp, rank);
                std::vector<uint8_t> destination(reference.size(), 0xfe);
                const size_t         parts        = std::max<size_t>(1, prefill_tp / decode_tp);
                const size_t         source_parts = std::max<size_t>(1, decode_tp / prefill_tp);
                for (size_t part = 0; part < parts; ++part) {
                    const auto peer   = prefill_tp >= decode_tp ? rank * parts + part : rank / source_parts;
                    const auto source = serialize(prefill_tp, peer);
                    const TpCacheTransferRoute route{true, parts, part, source_parts, rank % source_parts};
                    const auto plan = layout.plan(state / decode_tp, rows * (2 * q + v) / decode_tp, route);
                    ASSERT_EQ(plan.source_bytes, source.size());
                    ASSERT_EQ(plan.destination_bytes, destination.size());
                    ASSERT_EQ(plan.slices.size(), widths.size());
                    ASSERT_NO_THROW(plan.validateSlices());
                    for (const auto& slice : plan.slices) {
                        std::memcpy(destination.data() + slice.destination_offset,
                                    source.data() + slice.source_offset,
                                    slice.bytes);
                    }
                }
                EXPECT_EQ(destination, reference) << prefill_tp << "->" << decode_tp << " rank=" << rank;
            }
        }
    }
}

TEST(LinearTpCacheLayoutTest, RejectsInvalidSourcePartitions) {
    EXPECT_THROW(linearTpCacheSlices(8, 4, 4, 4, 1, 1, 0, 0, 0), std::invalid_argument);
    EXPECT_THROW(linearTpCacheSlices(8, 4, 4, 4, 1, 1, 0, 2, 2), std::invalid_argument);
    EXPECT_THROW(linearTpCacheSlices(8, 4, 4, 4, 1, 1, 0, std::numeric_limits<size_t>::max(), 0), std::overflow_error);
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

TEST(MhaTpCacheLayoutTest, PreservesEveryKernelPageWithinPhysicalBlocks) {
    for (size_t pages : {1u, 2u, 4u}) {
        for (size_t k_size : {128u, 256u}) {
            for (size_t v_size : {128u, 512u}) {
                std::vector<uint8_t> expected;
                std::vector<uint8_t> sources[2];
                for (size_t page = 0; page < pages; ++page) {
                    for (size_t kind = 0; kind < 2; ++kind) {
                        for (size_t head = 0; head < 2; ++head) {
                            const auto value = static_cast<uint8_t>(1 + page * 4 + kind * 2 + head);
                            const auto size  = kind == 0 ? k_size : v_size;
                            expected.insert(expected.end(), size, value);
                            sources[head].insert(sources[head].end(), size, value);
                        }
                    }
                }
                std::vector<uint8_t>             actual(expected.size() + 16, 0xfe);
                std::vector<TpCacheTransferPlan> plans;
                for (size_t head = 0; head < 2; ++head) {
                    auto plan = TpCacheTransferPlan::mha(2 * k_size * pages, 2 * v_size * pages, 2, head, pages);
                    for (const auto& slice : plan.slices) {
                        ASSERT_LE(slice.source_offset + slice.bytes, sources[head].size());
                        std::memcpy(actual.data() + slice.destination_offset,
                                    sources[head].data() + slice.source_offset,
                                    slice.bytes);
                    }
                    plans.push_back(plan);
                }
                EXPECT_NO_THROW(validateCompleteTransfer(plans));
                EXPECT_TRUE(std::equal(expected.begin(), expected.end(), actual.begin()));
                EXPECT_TRUE(std::all_of(
                    actual.begin() + expected.size(), actual.end(), [](uint8_t value) { return value == 0xfe; }));
            }
        }
    }
    EXPECT_THROW(mhaTpCacheSlices(64, 64, 2, 0, 0), std::invalid_argument);
    EXPECT_THROW(mhaTpCacheSlices(64, 64, 2, 0, 3), std::invalid_argument);
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
