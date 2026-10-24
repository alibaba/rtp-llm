#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <memory>
#include <vector>

#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_target_verify_prepare.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/attention_input_metadata.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/cuda_graph_prepare.h"
#include "rtp_llm/models_py/bindings/common/kernels/fuse_copy_util.h"

namespace rtp_llm {
namespace {

class MtpRoundHeadPrepareTest: public ::testing::Test {
protected:
    void SetUp() override {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
            GTEST_SKIP() << "CUDA device required";
        }
        auto stream = c10::cuda::getStreamFromPool();
        stream_     = stream.stream();
        guard_      = std::make_unique<c10::cuda::CUDAStreamGuard>(stream);
    }

    void TearDown() override {
        if (guard_) {
            EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        }
        guard_.reset();
    }

    static torch::TensorOptions cudaInt32() {
        return torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    }

    cudaStream_t                                stream_ = nullptr;
    std::unique_ptr<c10::cuda::CUDAStreamGuard> guard_;
};

TEST_F(MtpRoundHeadPrepareTest, EngramHistoryMatchesTorchForPartialAcceptanceAndStrides) {
    for (auto token_dtype : {torch::kInt32, torch::kInt64}) {
        for (auto length_dtype : {torch::kInt32, torch::kInt64}) {
            for (int64_t batch : {0, 1, 7, 64, 65, 257}) {
                for (int64_t width : {0, 1, 3, 6, 9}) {
                    const auto options = cudaInt32().dtype(token_dtype);
                    auto       anchor  = torch::arange(batch * 8, options).reshape({batch, 8}).slice(1, 0, 8, 2);
                    auto       tokens =
                        torch::arange(batch * width * 2, options).reshape({batch, width * 2}).slice(1, 0, width * 2, 2);
                    auto lengths = (torch::arange(batch * 2, cudaInt32().dtype(length_dtype)) % (width + 1))
                                       .slice(0, 0, batch * 2, 2);
                    auto chronological = torch::cat({anchor.flip({1}), tokens}, 1);
                    auto indexes       = lengths.reshape({-1, 1}).to(torch::kLong)
                                   + torch::arange(3, -1, -1, cudaInt32().dtype(torch::kLong)).reshape({1, 4});
                    auto expected = chronological.gather(1, indexes);
                    auto actual   = tryMtpAdvanceEngramTokenWindows(anchor, tokens, lengths);
                    ASSERT_TRUE(actual.defined());
                    ASSERT_TRUE(torch::equal(expected, actual));
                }
            }
        }
    }
    EXPECT_FALSE(tryMtpAdvanceEngramTokenWindows(torch::zeros({1, 4}, torch::kInt32),
                                                 torch::zeros({1, 6}, torch::kInt32),
                                                 torch::ones({1}, torch::kInt32))
                     .defined());
}

TEST_F(MtpRoundHeadPrepareTest, DSparkVerifyMatchesTokensHistoryAndPositionsForVariableShapes) {
    for (int64_t batch : {0, 1, 7, 64, 65, 257}) {
        for (int64_t drafts : {1, 3, 5, 8}) {
            auto anchors          = torch::randint(1000, {batch, 2}, cudaInt32()).select(1, 0);
            auto proposals        = torch::randint(1000, {batch, drafts * 2}, cudaInt32()).slice(1, 0, drafts * 2, 2);
            auto history          = torch::randint(1000, {batch, 8}, cudaInt32()).slice(1, 0, 8, 2);
            auto tokens           = torch::cat({anchors.reshape({batch, 1}), proposals}, 1);
            auto chronological    = torch::cat({history.narrow(1, 1, 3).flip({1}), tokens}, 1);
            auto expected_history = chronological.unfold(1, 4, 1).flip({2}).contiguous().reshape({tokens.numel(), 4});
            for (int64_t factor : {0, 1, 3}) {
                auto bases = factor ?
                                 torch::randint(1000, {batch, factor * 2}, cudaInt32()).slice(1, 0, factor * 2, 2) :
                                 torch::Tensor();
                for (bool with_history : {false, true}) {
                    auto out = tryMtpDSparkVerifyPrepare(
                        anchors, proposals, with_history ? history : torch::Tensor(), bases, stream_);
                    ASSERT_TRUE(torch::equal(out.tokens, tokens.reshape({-1})));
                    ASSERT_TRUE(torch::equal(out.input_lengths, torch::full({batch}, drafts + 1, cudaInt32())));
                    ASSERT_TRUE(torch::equal(out.lm_indexes, torch::arange(tokens.numel(), cudaInt32())));
                    if (with_history) {
                        ASSERT_TRUE(torch::equal(out.history, expected_history));
                    } else {
                        ASSERT_FALSE(out.history.defined());
                    }
                    if (factor && batch) {
                        auto expected =
                            bases.unsqueeze(1) + torch::arange(drafts + 1, cudaInt32()).reshape({1, drafts + 1, 1});
                        ASSERT_TRUE(torch::equal(out.positions, expected.reshape({-1})));
                    } else {
                        ASSERT_FALSE(out.positions.defined());
                    }
                }
            }
            auto saved = proposals.clone();
            auto owned = tryMtpDSparkVerifyPrepare(anchors, proposals, history, {}, stream_, true).proposals;
            ASSERT_TRUE(owned.is_contiguous());
            proposals.fill_(-1);  // Simulate the next Markov graph overwriting its reusable output.
            ASSERT_TRUE(torch::equal(owned, saved));
            ASSERT_FALSE(
                tryMtpDSparkVerifyPrepare(anchors.to(torch::kLong), proposals, history, {}, stream_).tokens.defined());
            ASSERT_FALSE(tryMtpDSparkVerifyPrepare(anchors, proposals.cpu(), history, {}, stream_).tokens.defined());
            ASSERT_FALSE(
                tryMtpDSparkVerifyPrepare(anchors, proposals, history.narrow(1, 0, 3), {}, stream_).tokens.defined());
        }
    }
}

TEST_F(MtpRoundHeadPrepareTest, PageSnapshotsMixRaggedDeviceRowsAndPinnedHostSuffixes) {
    const auto pinned = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);
    for (int64_t batch : {0, 1, 7, 64, 65, 129, 257}) {
        for (int64_t width : {0, 3, 1537}) {
            torch::Tensor              hosts[2], expected[2], output[2];
            std::vector<torch::Tensor> snapshots[2];
            for (int table = 0; table < 2; ++table) {
                const int64_t groups  = table + 2;
                const int64_t columns = width ? width + table * 7 : 0;
                auto          backing =
                    torch::arange(groups * batch * 2 * (columns + 3), pinned).reshape({groups, batch * 2, columns + 3});
                hosts[table]    = backing.slice(1, 0, batch * 2, 2).slice(2, 1, columns + 1);
                expected[table] = hosts[table].clone();
                snapshots[table].resize(batch);
                for (int64_t row = 0; row < batch; ++row) {
                    if (row % 4 == 0) {
                        continue;  // Newly admitted host-only rows.
                    }
                    const int64_t live   = row % 3 == 0 ? columns : columns / (row % 3 + 1);
                    auto          source = (torch::arange(groups * 2 * (live + 2), torch::kInt32) + 100000 + row * 71)
                                      .reshape({groups, 2, live + 2});
                    snapshots[table][row] = source.to(torch::kCUDA).slice(1, 1, 2).slice(2, 1, live + 1);
                    expected[table]
                        .slice(1, row, row + 1)
                        .slice(2, 0, live)
                        .copy_(source.slice(1, 1, 2).slice(2, 1, live + 1));
                }
            }
            ASSERT_TRUE(tryMtpGatherPageSnapshots(
                hosts[0], hosts[1], snapshots[0], snapshots[1], output[0], output[1], stream_));
            if (output[0].numel() && output[1].numel()) {
                ASSERT_EQ(output[0].storage().data_ptr().get(), output[1].storage().data_ptr().get());
                ASSERT_EQ(output[0].data_ptr<int32_t>() + output[0].numel(), output[1].data_ptr<int32_t>());
            }
            for (int table = 0; table < 2; ++table) {
                ASSERT_TRUE(output[table].is_contiguous());
                ASSERT_TRUE(torch::equal(output[table].cpu(), expected[table]));
            }
        }
    }
}

TEST_F(MtpRoundHeadPrepareTest, PageSnapshotsRejectUnsupportedInputsBeforeWriting) {
    const auto                 pinned = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);
    auto                       host   = torch::zeros({2, 1, 7}, pinned);
    std::vector<torch::Tensor> rows(1);
    torch::Tensor              physical, kernel;
    EXPECT_FALSE(
        tryMtpGatherPageSnapshots(torch::zeros({2, 1, 7}, torch::kInt32), host, rows, rows, physical, kernel, stream_));
    rows[0] = torch::zeros({2, 1, 14}, cudaInt32()).slice(2, 0, 14, 2);
    EXPECT_FALSE(tryMtpGatherPageSnapshots(host, host, rows, rows, physical, kernel, stream_));
    rows[0] = torch::zeros({2, 1, 8}, cudaInt32());
    EXPECT_FALSE(tryMtpGatherPageSnapshots(host, host, rows, rows, physical, kernel, stream_));
    rows[0] = torch::zeros({2, 1, 7}, torch::kInt32);
    EXPECT_FALSE(tryMtpGatherPageSnapshots(host, host, rows, rows, physical, kernel, stream_));
    EXPECT_FALSE(physical.defined());
    EXPECT_FALSE(kernel.defined());
}

TEST_F(MtpRoundHeadPrepareTest, DispatchAndEngramHistoryShareOneLaunchForPartialAcceptance) {
    for (int64_t batch : {0, 1, 7, 64, 65, 257}) {
        for (int64_t width : {0, 1, 3, 6, 9}) {
            auto anchor = torch::arange(batch * 8, cudaInt32()).reshape({batch, 8}).slice(1, 0, 8, 2);
            auto tokens =
                torch::arange(batch * width * 2, cudaInt32()).reshape({batch, width * 2}).slice(1, 0, width * 2, 2);
            auto                        accepted      = torch::arange(batch, cudaInt32()) % (width + 1);
            auto                        source        = torch::arange(batch * 2, cudaInt32());
            auto                        expected_prev = torch::empty({batch}, torch::kInt32);
            std::vector<MtpSequenceRow> rows;
            for (int64_t row = 0; row < batch; ++row) {
                const auto value = static_cast<int32_t>(row % 2 ? 2 * (batch - 1 - row) : row + 100);
                expected_prev.data_ptr<int32_t>()[row] = value;
                rows.push_back({row % 2 ? source.data_ptr<int32_t>() + 2 * (batch - 1 - row) : nullptr, value});
            }
            auto prev    = torch::empty({batch}, cudaInt32());
            auto next    = torch::empty_like(prev);
            auto hidden  = torch::empty({batch}, cudaInt32().dtype(torch::kInt64));
            auto history = torch::empty({batch, 4}, cudaInt32());
            invokeMtpDispatchStatePrepareRows(rows, accepted, prev, next, hidden, stream_, anchor, tokens, history);
            auto chronological = torch::cat({anchor.flip({1}), tokens}, 1);
            auto indexes       = accepted.to(torch::kLong).reshape({-1, 1})
                           + torch::arange(3, -1, -1, cudaInt32().dtype(torch::kLong)).reshape({1, 4});
            ASSERT_TRUE(torch::equal(history, chronological.gather(1, indexes)));
            ASSERT_TRUE(torch::equal(prev.cpu(), expected_prev));
            ASSERT_TRUE(torch::equal(next, expected_prev.to(torch::kCUDA) + accepted));
            ASSERT_TRUE(torch::equal(hidden, accepted.to(torch::kLong) - 1));
        }
    }
}

TEST_F(MtpRoundHeadPrepareTest, AttentionMetadataFusesPrefixPlusOneWithoutChangingScans) {
    for (int64_t batch : {0, 1, 7, 64, 65, 257, 1025}) {
        auto          lengths_cpu = torch::arange(batch, torch::kInt32) % 6 + 1;
        auto          patterns    = torch::tensor({0, 32768, INT32_MAX, INT32_MIN}, torch::kInt32);
        auto          prefix_cpu  = patterns.index_select(0, torch::arange(batch, torch::kLong) % 4);
        const int64_t total       = lengths_cpu.sum().item<int64_t>();
        auto          lengths     = lengths_cpu.to(torch::kCUDA);
        auto          prefix      = prefix_cpu.to(torch::kCUDA);
        auto          cu          = torch::empty({batch + 1}, cudaInt32());
        auto          kv          = torch::empty_like(cu);
        auto          padding     = torch::empty({total}, cudaInt32());
        auto          plus_one    = torch::empty({batch}, cudaInt32());
        invokeBuildAttentionInputMetadata(lengths, prefix, cu, kv, padding, stream_, plus_one);
        auto expected_cu = torch::cat({torch::zeros({1}, torch::kLong), lengths_cpu.to(torch::kLong).cumsum(0)});
        auto expected_kv = torch::cat(
            {torch::zeros({1}, torch::kLong), (lengths_cpu.to(torch::kLong) + prefix_cpu.to(torch::kLong)).cumsum(0)});
        ASSERT_TRUE(torch::equal(cu.cpu(), expected_cu.to(torch::kInt32)));
        ASSERT_TRUE(torch::equal(kv.cpu(), expected_kv.to(torch::kInt32)));
        ASSERT_TRUE(torch::equal(plus_one.cpu(), prefix_cpu + 1));
        auto old_padding = torch::empty_like(padding);
        invokeBuildAttentionInputMetadata(lengths, prefix, cu, kv, old_padding, stream_);
        ASSERT_TRUE(torch::equal(padding, old_padding));
    }
}

TEST_F(MtpRoundHeadPrepareTest, GraphPrepareCopiesLivePagesAndClearsPaddingAcrossReplays) {
    for (int64_t batch : {1, 7, 64, 65, 257}) {
        for (int64_t width : {1, 3, 64, 2049}) {
            const int64_t graph_batch = batch + 3;
            auto          storage     = torch::arange(batch * (width + 3), cudaInt32()).reshape({batch, width + 3});
            auto          source = storage.slice(1, 1, width + 1);  // Unaligned rows with an independent source stride.
            auto          pages  = torch::full({graph_batch, width + 5}, -1, cudaInt32());
            auto          values = torch::arange(batch + 1, cudaInt32());
            auto          metadata    = torch::full({graph_batch + 1}, -1, cudaInt32());
            auto          byte_source = torch::arange(133, cudaInt32().dtype(torch::kUInt8));
            auto          byte_output = torch::empty_like(byte_source);
            CudaGraphPrepareFillParams fills;
            fills.regions[fills.region_count++] = {pages.data_ptr<int32_t>(), nullptr, pages.numel(), 0, 0, 0};
            fills.regions[fills.region_count++] = {
                metadata.data_ptr<int32_t>() + batch + 1, values.data_ptr<int32_t>() + batch, 3, 0, 6, 6};
            FusedD2DCopyParams copies;
            copies.add(values.data_ptr(), metadata.data_ptr(), (batch + 1) * sizeof(int32_t));
            copies.add(byte_source.data_ptr<uint8_t>() + 1, byte_output.data_ptr<uint8_t>() + 1, 131);
            FusedStridedCopyParams strided;
            strided.add(source.data_ptr(),
                        pages.data_ptr(),
                        batch,
                        width * sizeof(int32_t),
                        source.stride(0) * sizeof(int32_t),
                        pages.stride(0) * sizeof(int32_t));
            ASSERT_TRUE(tryInvokeCudaGraphPrepare(fills, copies, strided, stream_));
            cudaGraph_t     graph;
            cudaGraphExec_t executable;
            ASSERT_EQ(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal), cudaSuccess);
            ASSERT_TRUE(tryInvokeCudaGraphPrepare(fills, copies, strided, stream_));
            ASSERT_EQ(cudaStreamEndCapture(stream_, &graph), cudaSuccess);
            ASSERT_EQ(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), cudaSuccess);
            for (int replay = 0; replay < 2; ++replay) {
                storage.add_(7);
                values.add_(11);
                pages.fill_(-99);
                ASSERT_EQ(cudaGraphLaunch(executable, stream_), cudaSuccess);
                auto expected = torch::zeros_like(pages);
                expected.slice(0, 0, batch).slice(1, 0, width).copy_(source);
                ASSERT_TRUE(torch::equal(pages, expected));
                ASSERT_TRUE(torch::equal(metadata.slice(0, 0, batch + 1), values));
                auto expected_tail = values.select(0, batch) + torch::tensor({6, 12, 18}, cudaInt32());
                ASSERT_TRUE(torch::equal(metadata.slice(0, batch + 1), expected_tail));
                ASSERT_TRUE(torch::equal(byte_output.slice(0, 1, 132), byte_source.slice(0, 1, 132)));
            }
            ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
            EXPECT_EQ(cudaGraphExecDestroy(executable), cudaSuccess);
            EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
        }
    }
}

TEST_F(MtpRoundHeadPrepareTest, GraphPrepareRejectsAliasesAndExcessCapacityWithoutWriting) {
    auto                       buffer = torch::full({256}, -77, cudaInt32());
    auto                       source = torch::arange(256, cudaInt32());
    FusedD2DCopyParams         copies;
    FusedStridedCopyParams     strided;
    CudaGraphPrepareFillParams fills;
    fills.regions[fills.region_count++] = {buffer.data_ptr<int32_t>(), nullptr, 64, 0, 0, 0};
    copies.add(buffer.data_ptr<int32_t>(), buffer.data_ptr<int32_t>() + 128, 64 * sizeof(int32_t));
    EXPECT_FALSE(tryInvokeCudaGraphPrepare(fills, copies, strided, stream_));  // Read-after-write.
    copies.clear();
    copies.add(source.data_ptr(), buffer.data_ptr<int32_t>() + 1, 64 * sizeof(int32_t));
    EXPECT_FALSE(tryInvokeCudaGraphPrepare(fills, copies, strided, stream_));  // Partial destination overlap.
    copies.clear();
    fills.regions[0].value_ptr = buffer.data_ptr<int32_t>();
    EXPECT_FALSE(tryInvokeCudaGraphPrepare(fills, copies, strided, stream_));  // Fill value would be overwritten.
    fills.region_count = 0;
    for (int i = 0; i < 33; ++i) {
        copies.add(source.data_ptr<int32_t>() + i, buffer.data_ptr<int32_t>() + i, sizeof(int32_t));
    }
    EXPECT_FALSE(tryInvokeCudaGraphPrepare(fills, copies, strided, stream_));
    ASSERT_TRUE(torch::equal(buffer, torch::full_like(buffer, -77)));
    copies.num_copies = 32;
    EXPECT_TRUE(tryInvokeCudaGraphPrepare(fills, copies, strided, stream_));
    ASSERT_TRUE(torch::equal(buffer.slice(0, 0, 32), source.slice(0, 0, 32)));
}

TEST_F(MtpRoundHeadPrepareTest, GraphInputNodesFollowLivePointersBatchChangesAndFallback) {
    const int capacity    = 257;
    auto      captured_cu = torch::full({capacity + 1}, -1, cudaInt32());
    auto      observed    = torch::empty_like(captured_cu);
    observed.copy_(captured_cu);  // Warm the consumer before native capture.
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    cudaGraph_t graph = nullptr;
    ASSERT_EQ(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal), cudaSuccess);
    observed.copy_(captured_cu);
    ASSERT_EQ(cudaStreamEndCapture(stream_, &graph), cudaSuccess);
    CudaGraphPrepareReplay copies, metadata;
    copies.node   = addCudaGraphPrepareNode(graph);
    metadata.node = addAttentionInputMetadataNode(graph, copies.node);
#if CUDART_VERSION >= 12000
    ASSERT_EQ(cudaGraphInstantiate(&copies.exec, graph, 0), cudaSuccess);
#else
    ASSERT_EQ(cudaGraphInstantiate(&copies.exec, graph, nullptr, nullptr, 0), cudaSuccess);
#endif
    metadata.exec = copies.exec;

    for (int batch : {1, 64, 7, 257, 65, 1}) {
        auto                       lengths  = torch::arange(batch, cudaInt32()) % 6 + 1;
        auto                       prefix   = torch::arange(batch, cudaInt32()) * 13 + batch;
        auto                       cu       = torch::empty({batch + 1}, cudaInt32());
        auto                       kv       = torch::empty_like(cu);
        auto                       plus_one = torch::empty_like(lengths);
        auto                       padding  = torch::empty({lengths.sum().item<int>()}, cudaInt32());
        auto                       params   = prepareAttentionInputMetadata(lengths, prefix, cu, kv, padding, plus_one);
        CudaGraphPrepareFillParams fills;
        fills.regions[fills.region_count++] = {
            captured_cu.data_ptr<int32_t>() + batch + 1, nullptr, capacity - batch, -1};
        FusedD2DCopyParams plan;
        plan.add(cu.data_ptr(), captured_cu.data_ptr(), cu.nbytes());
        FusedStridedCopyParams strided;
        disableCudaGraphPrepareReplay(copies);
        disableCudaGraphPrepareReplay(metadata);
        ASSERT_TRUE(tryInvokeCudaGraphPrepare(fills, plan, strided, stream_, copies));
        invokeBuildAttentionInputMetadata(params, stream_, metadata);
        ASSERT_EQ(cudaGraphLaunch(copies.exec, stream_), cudaSuccess);

        auto expected = torch::cat({torch::zeros({1}, cudaInt32()), lengths.cumsum(0, torch::kInt32)});
        EXPECT_TRUE(torch::equal(observed.narrow(0, 0, batch + 1), expected));
        EXPECT_TRUE(observed.narrow(0, batch + 1, capacity - batch).eq(-1).all().item<bool>());
        EXPECT_TRUE(torch::equal(plus_one, prefix + 1));
        auto expected_kv = torch::cat({torch::zeros({1}, cudaInt32()), (lengths + prefix).cumsum(0, torch::kInt32)});
        EXPECT_TRUE(torch::equal(kv, expected_kv));
    }
    // A rejected/async plan must not retain the last replay's live pointers.
    disableCudaGraphPrepareReplay(copies);
    disableCudaGraphPrepareReplay(metadata);
    FusedD2DCopyParams alias;
    alias.add(captured_cu.data_ptr(), captured_cu.data_ptr(), captured_cu.nbytes());
    CudaGraphPrepareFillParams fills;
    FusedStridedCopyParams     strided;
    EXPECT_FALSE(tryInvokeCudaGraphPrepare(fills, alias, strided, stream_, copies));
    captured_cu.fill_(123);
    ASSERT_EQ(cudaGraphLaunch(copies.exec, stream_), cudaSuccess);
    EXPECT_TRUE(observed.eq(123).all().item<bool>());
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    EXPECT_EQ(cudaGraphExecDestroy(copies.exec), cudaSuccess);
    EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
}

TEST_F(MtpRoundHeadPrepareTest, MixedReorderedStridedRowsAndChunkBoundaries) {
    constexpr int32_t mask          = -17;
    constexpr int64_t storage_width = 24;
    for (const int64_t batch : {0, 1, 16, 32, 64, 65, 129}) {
        SCOPED_TRACE(batch);
        const int64_t source_rows = batch + 3;
        auto          tokens_cpu =
            torch::arange(source_rows * storage_width, torch::kInt32).reshape({source_rows, storage_width});
        auto accepts_cpu   = torch::zeros({source_rows * 3}, torch::kInt32);
        auto sequences_cpu = torch::arange(source_rows * 2, torch::kInt32) + 100;
        for (int64_t src = 0; src < source_rows; ++src) {
            const int32_t count                          = 3 + src % 7;
            accepts_cpu.data_ptr<int32_t>()[src * 3 + 1] = src % 3 == 0 ? count : 1 + src % count;
        }
        auto tokens    = tokens_cpu.to(torch::kCUDA);
        auto accepts   = accepts_cpu.to(torch::kCUDA);
        auto sequences = sequences_cpu.to(torch::kCUDA);
        for (int mode = 0; mode < 3; ++mode) {
            SCOPED_TRACE(mode);
            std::vector<MtpRoundHeadRow> rows(batch);
            auto                         expected_anchors = torch::empty({batch}, torch::kInt32);
            auto                         expected_ends    = torch::empty({batch}, torch::kInt32);
            for (int64_t row = 0; row < batch; ++row) {
                const int64_t src             = source_rows - 1 - row;
                const bool    device_anchor   = mode == 1 || (mode == 0 && row % 4 < 2);
                const bool    device_sequence = mode == 1 || (mode == 0 && row % 2 == 0);
                auto&         descriptor      = rows[row];
                descriptor.token_stride       = 2;
                descriptor.token_count        = 3 + src % 7;
                descriptor.anchor             = -1000 - row;
                descriptor.sequence_value     = -2000 - row;
                descriptor.sequence_bias      = row % 2 == 0 ? -1 : 0;
                int32_t expected_anchor       = descriptor.anchor;
                int32_t expected_end          = descriptor.sequence_value;
                if (device_anchor) {
                    descriptor.tokens      = tokens.data_ptr<int32_t>() + src * storage_width + 1;
                    descriptor.accept_len  = accepts.data_ptr<int32_t>() + src * 3 + 1;
                    const int32_t accepted = accepts_cpu.data_ptr<int32_t>()[src * 3 + 1];
                    expected_anchor = tokens_cpu.data_ptr<int32_t>()[src * storage_width + 1 + (accepted - 1) * 2];
                }
                if (device_sequence) {
                    descriptor.sequence = sequences.data_ptr<int32_t>() + src * 2 + 1;
                    expected_end        = sequences_cpu.data_ptr<int32_t>()[src * 2 + 1] + descriptor.sequence_bias;
                }
                expected_anchors.data_ptr<int32_t>()[row] = expected_anchor;
                expected_ends.data_ptr<int32_t>()[row]    = expected_end;
            }
            for (const int32_t width : {1, 6, 11}) {
                SCOPED_TRACE(width);
                auto anchors = torch::empty({batch}, cudaInt32());
                auto ends    = torch::empty({batch}, cudaInt32());
                auto combo   = torch::full({batch, width}, 123456, cudaInt32());
                invokeMtpDSparkRoundHeadPrepare(rows, anchors, ends, combo, width, mask, stream_);
                auto expected_combo = torch::full({batch, width}, mask, torch::kInt32);
                expected_combo.select(1, 0).copy_(expected_anchors);
                EXPECT_TRUE(torch::equal(anchors.cpu(), expected_anchors));
                EXPECT_TRUE(torch::equal(ends.cpu(), expected_ends));
                EXPECT_TRUE(torch::equal(combo.cpu(), expected_combo));
            }
        }
    }
}

TEST_F(MtpRoundHeadPrepareTest, BroadcastTokenStrideAndFlatCombo) {
    auto                         tokens   = torch::tensor({73}, cudaInt32());
    auto                         accepted = torch::tensor({7}, cudaInt32());
    std::vector<MtpRoundHeadRow> rows(1);
    rows[0].tokens         = tokens.data_ptr<int32_t>();
    rows[0].accept_len     = accepted.data_ptr<int32_t>();
    rows[0].token_stride   = 0;
    rows[0].token_count    = 7;
    rows[0].sequence_value = 19;
    rows[0].sequence_bias  = -1;
    auto anchors           = torch::empty({1}, cudaInt32());
    auto ends              = torch::empty({1}, cudaInt32());
    auto combo             = torch::empty({3}, cudaInt32());
    invokeMtpDSparkRoundHeadPrepare(rows, anchors, ends, combo, 3, 0, stream_);
    EXPECT_TRUE(torch::equal(anchors.cpu(), torch::tensor({73}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(ends.cpu(), torch::tensor({19}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(combo.cpu(), torch::tensor({73, 0, 0}, torch::kInt32)));
}

struct CapturedGraph {
    cudaGraph_t     graph      = nullptr;
    cudaGraphExec_t executable = nullptr;
    ~CapturedGraph() {
        if (executable != nullptr) {
            cudaGraphExecDestroy(executable);
        }
        if (graph != nullptr) {
            cudaGraphDestroy(graph);
        }
    }
};

TEST_F(MtpRoundHeadPrepareTest, CaptureOwnsDescriptorsAndReplayReadsCurrentDeviceState) {
    for (const int64_t batch : {32, 65}) {
        SCOPED_TRACE(batch);
        auto          tokens    = torch::arange(batch * 6, cudaInt32()).reshape({batch, 6});
        auto          accepts   = torch::full({batch}, 6, cudaInt32());
        auto          sequences = torch::arange(batch, cudaInt32()) + 100;
        auto          anchors   = torch::empty({batch}, cudaInt32());
        auto          ends      = torch::empty({batch}, cudaInt32());
        auto          combo     = torch::empty({batch, 6}, cudaInt32());
        CapturedGraph graph;
        {
            std::vector<MtpRoundHeadRow> rows(batch);
            for (int64_t row = 0; row < batch; ++row) {
                if (row % 2 == 0) {
                    rows[row].tokens        = tokens.data_ptr<int32_t>() + row * 6;
                    rows[row].accept_len    = accepts.data_ptr<int32_t>() + row;
                    rows[row].token_count   = 6;
                    rows[row].sequence      = sequences.data_ptr<int32_t>() + row;
                    rows[row].sequence_bias = -1;
                } else {
                    rows[row].anchor         = 71;
                    rows[row].sequence_value = 81;
                }
            }
            // Load the AOT module before capture and before any timing.
            invokeMtpDSparkRoundHeadPrepare(rows, anchors, ends, combo, 6, -1, stream_);
            ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
            ASSERT_EQ(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal), cudaSuccess);
            invokeMtpDSparkRoundHeadPrepare(rows, anchors, ends, combo, 6, -1, stream_);
            ASSERT_EQ(cudaStreamEndCapture(stream_, &graph.graph), cudaSuccess);
        }
        size_t node_count = 0;
        ASSERT_EQ(cudaGraphGetNodes(graph.graph, nullptr, &node_count), cudaSuccess);
        EXPECT_EQ(node_count, (batch + 63) / 64);
        ASSERT_EQ(cudaGraphInstantiate(&graph.executable, graph.graph, nullptr, nullptr, 0), cudaSuccess);
        tokens.add_(1000);
        sequences.add_(10);
        accepts.fill_(1);
        ASSERT_EQ(cudaGraphLaunch(graph.executable, stream_), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        auto anchors_cpu = anchors.cpu();
        auto ends_cpu    = ends.cpu();
        auto combo_cpu   = combo.cpu();
        for (int64_t row = 0; row < batch; ++row) {
            const int32_t expected_anchor = row % 2 == 0 ? row * 6 + 1000 : 71;
            const int32_t expected_end    = row % 2 == 0 ? row + 109 : 81;
            EXPECT_EQ(anchors_cpu.data_ptr<int32_t>()[row], expected_anchor);
            EXPECT_EQ(ends_cpu.data_ptr<int32_t>()[row], expected_end);
            EXPECT_EQ(combo_cpu.data_ptr<int32_t>()[row * 6], expected_anchor);
            for (int column = 1; column < 6; ++column) {
                EXPECT_EQ(combo_cpu.data_ptr<int32_t>()[row * 6 + column], -1);
            }
        }
    }
}

TEST_F(MtpRoundHeadPrepareTest, PositionGatherAndExpandRespectOrderAndRowStride) {
    for (const int64_t batch : {0, 1, 16, 32, 64, 65, 129}) {
        for (const int64_t factor : {1, 3}) {
            SCOPED_TRACE(batch);
            SCOPED_TRACE(factor);
            auto source_cpu = torch::arange((batch + 2) * (factor + 2), torch::kInt32).reshape({batch + 2, factor + 2});
            auto source     = source_cpu.to(torch::kCUDA);
            auto bases      = torch::empty({batch, factor}, cudaInt32());
            auto expected_bases = torch::empty({batch, factor}, torch::kInt32);
            std::vector<const int32_t*> rows(batch);
            for (int64_t row = 0; row < batch; ++row) {
                const int64_t src = batch + 1 - row;
                rows[row]         = source.data_ptr<int32_t>() + src * (factor + 2) + 1;
                expected_bases[row].copy_(source_cpu[src].narrow(0, 1, factor));
            }
            invokeMtpPositionGather(rows, bases, stream_);
            EXPECT_TRUE(torch::equal(bases.cpu(), expected_bases));
            auto strided_bases = torch::empty({batch, factor + 4}, cudaInt32()).narrow(1, 2, factor);
            strided_bases.copy_(bases);
            for (const int64_t width : {1, 6}) {
                auto expanded = torch::empty({batch * width * factor}, cudaInt32());
                invokeMtpPositionExpand(strided_bases, expanded, width, stream_);
                auto expected =
                    (expected_bases.unsqueeze(1) + torch::arange(width, torch::kInt32).reshape({1, width, 1}))
                        .flatten();
                EXPECT_TRUE(torch::equal(expanded.cpu(), expected));
            }
        }
    }
}

TEST_F(MtpRoundHeadPrepareTest, RejectsInvalidHostMetadataBeforeLaunching) {
    std::vector<MtpRoundHeadRow> rows(2);
    auto                         anchors         = torch::empty({2}, cudaInt32());
    auto                         ends            = torch::empty({2}, cudaInt32());
    auto                         combo           = torch::empty({2, 6}, cudaInt32());
    auto                         cpu_anchors     = torch::empty({2}, torch::kInt32);
    auto                         wrong_dtype     = torch::empty({2}, cudaInt32().dtype(torch::kInt64));
    auto                         strided_anchors = torch::empty({4}, cudaInt32()).slice(0, 0, 4, 2);
    auto                         wrong_combo     = torch::empty({6, 2}, cudaInt32());
    EXPECT_ANY_THROW(invokeMtpDSparkRoundHeadPrepare(rows, cpu_anchors, ends, combo, 6, 0, stream_));
    EXPECT_ANY_THROW(invokeMtpDSparkRoundHeadPrepare(rows, wrong_dtype, ends, combo, 6, 0, stream_));
    EXPECT_ANY_THROW(invokeMtpDSparkRoundHeadPrepare(rows, strided_anchors, ends, combo, 6, 0, stream_));
    EXPECT_ANY_THROW(invokeMtpDSparkRoundHeadPrepare(rows, anchors, ends, wrong_combo, 6, 0, stream_));
    EXPECT_ANY_THROW(invokeMtpDSparkRoundHeadPrepare(rows, anchors, ends, combo, 0, 0, stream_));
    rows[0].tokens = combo.data_ptr<int32_t>();
    EXPECT_ANY_THROW(invokeMtpDSparkRoundHeadPrepare(rows, anchors, ends, combo, 6, 0, stream_));
    rows[0].accept_len = anchors.data_ptr<int32_t>();
    EXPECT_ANY_THROW(invokeMtpDSparkRoundHeadPrepare(rows, anchors, ends, combo, 6, 0, stream_));
    rows[0].token_count  = 6;
    rows[0].token_stride = -1;
    EXPECT_ANY_THROW(invokeMtpDSparkRoundHeadPrepare(rows, anchors, ends, combo, 6, 0, stream_));
}

}  // namespace
}  // namespace rtp_llm
