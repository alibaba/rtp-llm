#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/record_function.h>
#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/csrc/profiler/api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <iostream>
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

// Compatibility oracle: the unchanged dense preparation plus the original
// Torch planner and cumsum/searchsorted/gather packing chain. It deliberately
// does not reuse the new kernel's sort keys or prefix-product implementation.
MtpDSparkBudgetedVerifyInputs oldBudgetedVerify(const torch::Tensor&        rates,
                                                const std::vector<int64_t>& stable_order,
                                                const torch::Tensor&        anchors,
                                                const torch::Tensor&        proposals,
                                                const torch::Tensor&        history,
                                                const torch::Tensor&        bases,
                                                const torch::Tensor&        prefixes,
                                                int64_t                     budget,
                                                cudaStream_t                stream,
                                                bool                        copy_proposals = true,
                                                const torch::Tensor&        prebuilt_order = {}) {
    const auto                    batch = rates.size(0), gamma = rates.size(1), width = gamma + 1;
    const auto                    i32 = anchors.options(), i64 = i32.dtype(torch::kInt64);
    MtpDSparkBudgetedVerifyInputs out;
    if (budget == 0) {
        out.valid_draft_lengths = torch::zeros({batch}, i32);
    } else {
        auto order = prebuilt_order;
        if (!order.defined()) {
            auto order_cpu = torch::tensor(stable_order, torch::kInt64).pin_memory();
            order          = order_cpu.to(rates.device(), true);
        }
        auto valid          = torch::isfinite(rates).logical_and(rates.ge(0.0)).logical_and(rates.le(1.0));
        auto clean          = torch::where(valid, rates, torch::zeros_like(rates));
        auto scores         = std::get<0>(clean.cumprod(1).cummin(1)).index_select(0, order);
        auto selected_order = torch::argsort(scores.reshape({-1}), true, 0, true);
        auto selected       = torch::zeros({rates.numel()}, i32);
        selected.index_fill_(0, selected_order.narrow(0, 0, budget), 1);
        auto sorted_lengths     = selected.reshape({batch, gamma}).sum(1, false, torch::kInt32);
        out.valid_draft_lengths = torch::empty({batch}, i32);
        out.valid_draft_lengths.index_copy_(0, order, sorted_lengths);
    }
    auto dense               = tryMtpDSparkVerifyPrepare(anchors, proposals, history, bases, stream, copy_proposals);
    out.dense_tokens         = dense.tokens.reshape({batch, width});
    out.dense_positions      = dense.positions;
    out.packed.input_lengths = out.valid_draft_lengths + 1;
    auto ends                = out.packed.input_lengths.cumsum(0, torch::kInt64);
    auto starts              = ends - out.packed.input_lengths;
    auto rows                = torch::arange(batch + budget, i64);
    auto owners              = at::searchsorted(ends, rows, false, true).clamp(0, batch - 1);
    auto positions           = (rows - starts.index_select(0, owners)).clamp(0, gamma);
    out.packed_to_dense      = owners * width + positions;
    out.packed.tokens        = dense.tokens.index_select(0, out.packed_to_dense);
    if (dense.positions.defined()) {
        out.packed.positions =
            dense.positions.reshape({batch * width, -1}).index_select(0, out.packed_to_dense).reshape({-1});
    }
    if (dense.history.defined()) {
        out.packed.history = dense.history.index_select(0, out.packed_to_dense);
    }
    out.packed.proposals        = dense.proposals;
    out.packed.lm_indexes       = rows.to(torch::kInt32);
    out.sequence_lengths_plus_1 = prefixes + 1;
    return out;
}

void expectBudgetedVerifyEqual(const MtpDSparkBudgetedVerifyInputs& actual,
                               const MtpDSparkBudgetedVerifyInputs& expected) {
    const std::vector<std::pair<torch::Tensor, torch::Tensor>> fields{
        {actual.valid_draft_lengths, expected.valid_draft_lengths},
        {actual.packed_to_dense, expected.packed_to_dense},
        {actual.dense_tokens, expected.dense_tokens},
        {actual.dense_positions, expected.dense_positions},
        {actual.sequence_lengths_plus_1, expected.sequence_lengths_plus_1},
        {actual.packed.tokens, expected.packed.tokens},
        {actual.packed.input_lengths, expected.packed.input_lengths},
        {actual.packed.lm_indexes, expected.packed.lm_indexes},
        {actual.packed.history, expected.packed.history},
        {actual.packed.positions, expected.packed.positions},
        {actual.packed.proposals, expected.packed.proposals}};
    for (size_t field = 0; field < fields.size(); ++field) {
        SCOPED_TRACE(field);
        const auto& left  = fields[field].first;
        const auto& right = fields[field].second;
        ASSERT_EQ(left.defined(), right.defined());
        if (right.defined()) {
            EXPECT_EQ(left.sizes(), right.sizes());
            EXPECT_EQ(left.scalar_type(), right.scalar_type());
            EXPECT_EQ(left.device(), right.device());
            EXPECT_TRUE(left.is_contiguous());
            EXPECT_TRUE(torch::equal(left, right));
        }
    }
}

std::vector<int32_t> independentBudgetFrontier(const std::vector<float>&   rates,
                                               const std::vector<int64_t>& order,
                                               int64_t                     gamma,
                                               int64_t                     budget) {
    std::vector<int32_t> lengths(order.size(), 0);
    std::vector<double>  scores(order.size(), 1.0);
    auto                 clean = [](float value) {
        return std::isfinite(value) && value >= 0.0f && value <= 1.0f ? static_cast<double>(value) : 0.0;
    };
    for (size_t row = 0; row < order.size(); ++row) {
        scores[row] = clean(rates[row * gamma]);
    }
    for (int64_t step = 0; step < budget; ++step) {
        int64_t winner = -1;
        for (auto row : order) {
            if (lengths[row] < gamma && (winner < 0 || scores[row] > scores[winner])) {
                winner = row;
            }
        }
        if (++lengths[winner] < gamma) {
            scores[winner] *= clean(rates[winner * gamma + lengths[winner]]);
        }
    }
    return lengths;
}

TEST_F(MtpRoundHeadPrepareTest, BudgetedVerifyMatchesIndependentDyadicFrontier) {
    const auto                 i32 = cudaInt32();
    const std::vector<int64_t> order{1, 0};
    auto                       anchors   = torch::tensor({11, 12}, i32);
    auto                       proposals = torch::tensor({{21, 22}, {31, 32}}, i32);
    auto                       prefixes  = torch::tensor({7, 8}, i32);
    for (int pattern = 0; pattern < 81; ++pattern) {
        std::vector<float> values(4);
        int                remaining = pattern;
        for (auto& value : values) {
            value = static_cast<float>(remaining % 3) * 0.5f;
            remaining /= 3;
        }
        auto rates = torch::tensor(values, i32.dtype(torch::kFloat32)).reshape({2, 2});
        for (int budget = 0; budget < 4; ++budget) {
            SCOPED_TRACE(::testing::Message() << "pattern=" << pattern << " budget=" << budget);
            auto actual = tryMtpDSparkBudgetedVerifyPrepare(
                rates, order, anchors, proposals, {}, {}, prefixes, budget, stream_, true);
            auto expected = torch::tensor(independentBudgetFrontier(values, order, 2, budget), torch::kInt32);
            ASSERT_TRUE(actual.valid_draft_lengths.defined());
            EXPECT_TRUE(torch::equal(actual.valid_draft_lengths.cpu(), expected));
        }
    }
}

TEST_F(MtpRoundHeadPrepareTest, BudgetedVerifyMatchesOldPlannerAndPackingForShapesStridesAndOptions) {
    const auto i32 = cudaInt32(), fp32 = i32.dtype(torch::kFloat32);
    for (int64_t batch : {2, 3, 31, 32, 47, 48, 49, 64, 128}) {
        for (int64_t gamma : {1, 5, 8}) {
            std::vector<int64_t> order(batch);
            std::iota(order.rbegin(), order.rend(), 0);
            auto rates   = torch::rand({batch, gamma}, fp32);
            auto anchors = torch::arange(batch * 2, i32).slice(0, 0, batch * 2, 2);
            auto proposals =
                torch::arange(batch * gamma * 2, i32).reshape({batch, gamma * 2}).slice(1, 0, gamma * 2, 2);
            auto history  = torch::arange(batch * 8, i32).reshape({batch, 8}).slice(1, 0, 8, 2);
            auto prefixes = (torch::arange(batch * 2, i32) + 37).slice(0, 0, batch * 2, 2);
            for (int64_t factor : {0, 1, 3}) {
                auto bases =
                    factor ?
                        torch::arange(batch * factor * 2, i32).reshape({batch, factor * 2}).slice(1, 0, factor * 2, 2) :
                        torch::Tensor();
                auto live_history = factor == 1 ? torch::Tensor() : history;
                for (int64_t budget : {int64_t{0}, int64_t{1}, batch * gamma / 2, batch * gamma - 1}) {
                    SCOPED_TRACE(::testing::Message()
                                 << "B=" << batch << " G=" << gamma << " F=" << factor << " budget=" << budget);
                    auto expected = oldBudgetedVerify(
                        rates, order, anchors, proposals, live_history, bases, prefixes, budget, stream_);
                    auto actual = tryMtpDSparkBudgetedVerifyPrepare(
                        rates, order, anchors, proposals, live_history, bases, prefixes, budget, stream_, true);
                    expectBudgetedVerifyEqual(actual, expected);
                    EXPECT_EQ(actual.valid_draft_lengths.sum().item<int64_t>(), budget);
                }
            }
        }
    }
}

TEST_F(MtpRoundHeadPrepareTest, BudgetedVerifyPreservesFloatBoundarySubnormalAndStableTieDecisions) {
    const float                           nan    = std::numeric_limits<float>::quiet_NaN();
    const float                           inf    = std::numeric_limits<float>::infinity();
    const float                           tiny   = std::numeric_limits<float>::denorm_min();
    const float                           normal = std::numeric_limits<float>::min();
    const std::vector<std::vector<float>> patterns{
        {0.01f, 0.01f, 0.15f, 1.0f, 1.0f, 0.01f, 0.01f, 0.15f, 1.0f, 1.0f},
        {normal, 0.5f, 0.5f, 0.5f, 1.0f, normal, 0.5f, 0.5f, 0.5f, std::nextafter(1.0f, 0.0f)},
        {tiny, 1.0f, 1.0f, 1.0f, 1.0f, -tiny, 1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, nan, inf, -inf, -0.0f, std::nextafter(1.0f, inf), 0.0f, -0.1f, 1.1f, 1.0f},
        {0.5f, 1.0f, 0.5f, 1.0f, 0.5f, 0.5f, 1.0f, 0.5f, 1.0f, 0.5f},
        {1.0f, std::nextafter(0.5f, 1.0f), 0.5f, 0.25f, 0.5f, 1.0f, 0.5f, std::nextafter(0.5f, 0.0f), 0.25f, 0.5f}};
    auto anchors   = torch::tensor({5, 7}, cudaInt32());
    auto proposals = torch::arange(10, cudaInt32()).reshape({2, 5});
    auto prefixes  = torch::tensor({std::numeric_limits<int32_t>::max(), 37}, cudaInt32());
    auto bases     = torch::tensor({{std::numeric_limits<int32_t>::max()}, {-17}}, cudaInt32());
    for (const auto& values : patterns) {
        auto rates      = torch::tensor(values, cudaInt32().dtype(torch::kFloat32)).reshape({2, 5});
        auto saved_bits = rates.view(torch::kInt32).clone();
        for (const std::vector<int64_t>& order : {std::vector<int64_t>{0, 1}, std::vector<int64_t>{1, 0}}) {
            for (int64_t budget = 0; budget < 10; ++budget) {
                auto expected =
                    oldBudgetedVerify(rates, order, anchors, proposals, {}, bases, prefixes, budget, stream_);
                auto actual = tryMtpDSparkBudgetedVerifyPrepare(
                    rates, order, anchors, proposals, {}, bases, prefixes, budget, stream_, true);
                expectBudgetedVerifyEqual(actual, expected);
            }
        }
        EXPECT_TRUE(torch::equal(rates.view(torch::kInt32), saved_bits));
    }
}

TEST_F(MtpRoundHeadPrepareTest, BudgetedVerifyRejectsUnsupportedLayoutsAndMalformedIdentityOrder) {
    for (const auto& shape : {std::make_pair(1, 5), std::make_pair(129, 5), std::make_pair(48, 9)}) {
        const auto           batch = shape.first, gamma = shape.second;
        std::vector<int64_t> order(batch);
        std::iota(order.begin(), order.end(), 0);
        auto out = tryMtpDSparkBudgetedVerifyPrepare(torch::ones({batch, gamma}, cudaInt32().dtype(torch::kFloat32)),
                                                     order,
                                                     torch::zeros({batch}, cudaInt32()),
                                                     torch::zeros({batch, gamma}, cudaInt32()),
                                                     {},
                                                     {},
                                                     torch::zeros({batch}, cudaInt32()),
                                                     0,
                                                     stream_);
        EXPECT_FALSE(out.packed.tokens.defined());
    }
    auto rates     = torch::ones({2, 5}, cudaInt32().dtype(torch::kFloat32));
    auto anchors   = torch::zeros({2}, cudaInt32());
    auto proposals = torch::zeros({2, 5}, cudaInt32());
    auto prefixes  = torch::zeros({2}, cudaInt32());
    auto invoke    = [&](const std::vector<int64_t>& order) {
        return tryMtpDSparkBudgetedVerifyPrepare(rates, order, anchors, proposals, {}, {}, prefixes, 3, stream_);
    };
    EXPECT_ANY_THROW(invoke({0}));
    EXPECT_ANY_THROW(invoke({0, 0}));
    EXPECT_ANY_THROW(invoke({-1, 0}));
    EXPECT_ANY_THROW(invoke({0, 2}));
    EXPECT_FALSE(tryMtpDSparkBudgetedVerifyPrepare(rates, {0, 1}, anchors, proposals, {}, {}, prefixes, 10, stream_)
                     .packed.tokens.defined());
    EXPECT_FALSE(
        tryMtpDSparkBudgetedVerifyPrepare(rates.cpu(), {0, 1}, anchors, proposals, {}, {}, prefixes, 3, stream_)
            .packed.tokens.defined());
    EXPECT_FALSE(tryMtpDSparkBudgetedVerifyPrepare(
                     rates, {0, 1}, anchors.to(torch::kInt64), proposals, {}, {}, prefixes, 3, stream_)
                     .packed.tokens.defined());
}

TEST_F(MtpRoundHeadPrepareTest, BudgetedVerifyCaptureOwnsOrderAndReplayRefreshesEveryDeviceInput) {
    const auto           i32   = cudaInt32();
    constexpr int64_t    batch = 48, gamma = 5, budget = batch * 3;
    auto                 rates     = torch::rand({batch, gamma}, i32.dtype(torch::kFloat32));
    auto                 anchors   = torch::arange(batch, i32);
    auto                 proposals = torch::arange(batch * gamma, i32).reshape({batch, gamma});
    auto                 history   = torch::arange(batch * 4, i32).reshape({batch, 4});
    auto                 bases     = torch::arange(batch * 3, i32).reshape({batch, 3});
    auto                 prefixes  = torch::arange(batch, i32) + 100;
    std::vector<int64_t> expected_order(batch);
    std::iota(expected_order.rbegin(), expected_order.rend(), 0);
    auto warm = tryMtpDSparkBudgetedVerifyPrepare(
        rates, expected_order, anchors, proposals, history, bases, prefixes, budget, stream_, true);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    at::cuda::CUDAGraph           graph(true);
    MtpDSparkBudgetedVerifyInputs captured;
    {
        auto temporary_order = expected_order;
        graph.capture_begin();
        captured = tryMtpDSparkBudgetedVerifyPrepare(
            rates, temporary_order, anchors, proposals, history, bases, prefixes, budget, stream_, true);
        graph.capture_end();
    }
    size_t nodes = 0;
    ASSERT_EQ(cudaGraphGetNodes(graph.raw_cuda_graph(), nullptr, &nodes), cudaSuccess);
    EXPECT_EQ(nodes, 1);
    for (int replay = 0; replay < 3; ++replay) {
        rates.fill_(replay == 0 ? 1.0f : 0.5f);
        rates.select(0, replay).fill_(std::numeric_limits<float>::quiet_NaN());
        anchors.add_(37);
        proposals.add_(71);
        history.add_(101);
        bases.add_(13);
        prefixes.add_(17);
        graph.replay();
        auto expected =
            oldBudgetedVerify(rates, expected_order, anchors, proposals, history, bases, prefixes, budget, stream_);
        expectBudgetedVerifyEqual(captured, expected);
    }
    auto saved_tokens    = captured.dense_tokens.clone();
    auto saved_proposals = captured.packed.proposals.clone();
    proposals.fill_(-999);
    EXPECT_TRUE(torch::equal(captured.dense_tokens, saved_tokens));
    EXPECT_TRUE(torch::equal(captured.packed.proposals, saved_proposals));
}

TEST_F(MtpRoundHeadPrepareTest, BudgetedVerifyCleanMicrobenchmarkAndOptionalTrace) {
    const auto           i32   = cudaInt32();
    constexpr int64_t    batch = 48, gamma = 5, budget = batch * 3;
    std::vector<int64_t> order(batch);
    std::iota(order.rbegin(), order.rend(), 0);
    auto          rates     = torch::rand({batch, gamma}, i32.dtype(torch::kFloat32));
    auto          anchors   = torch::arange(batch, i32);
    auto          proposals = torch::arange(batch * gamma, i32).reshape({batch, gamma});
    auto          history   = torch::arange(batch * 4, i32).reshape({batch, 4});
    torch::Tensor bases;  // Original B48 trace has Engram history and no positions.
    auto          prefixes = torch::arange(batch, i32) + 100;
    auto          run      = [&](bool fused) {
        return fused ? tryMtpDSparkBudgetedVerifyPrepare(
                           rates, order, anchors, proposals, history, bases, prefixes, budget, stream_, true) :
                                     oldBudgetedVerify(rates, order, anchors, proposals, history, bases, prefixes, budget, stream_);
    };
    expectBudgetedVerifyEqual(run(true), run(false));
    for (int warmup = 0; warmup < 10; ++warmup) {
        run(false);
        run(true);
    }
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    for (bool fused : {false, true}) {
        std::vector<double> cpu_us, event_us;
        for (int sample = 0; sample < 7; ++sample) {
            cudaEvent_t begin, end;
            ASSERT_EQ(cudaEventCreate(&begin), cudaSuccess);
            ASSERT_EQ(cudaEventCreate(&end), cudaSuccess);
            ASSERT_EQ(cudaEventRecord(begin, stream_), cudaSuccess);
            const auto start = std::chrono::steady_clock::now();
            for (int repeat = 0; repeat < 32; ++repeat) {
                run(fused);
            }
            cpu_us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count()
                             / 32);
            ASSERT_EQ(cudaEventRecord(end, stream_), cudaSuccess);
            ASSERT_EQ(cudaEventSynchronize(end), cudaSuccess);
            float milliseconds = 0;
            ASSERT_EQ(cudaEventElapsedTime(&milliseconds, begin, end), cudaSuccess);
            event_us.push_back(milliseconds * 1000.0 / 32);
            ASSERT_EQ(cudaEventDestroy(begin), cudaSuccess);
            ASSERT_EQ(cudaEventDestroy(end), cudaSuccess);
        }
        std::sort(cpu_us.begin(), cpu_us.end());
        std::sort(event_us.begin(), event_us.end());
        std::cout << "dspark_postsample_bench B=48 gamma=5 budget=144 queries=192 mode=" << (fused ? "fused" : "old")
                  << " cpu_enqueue_median_us=" << cpu_us[3] << " cuda_event_interval_median_us=" << event_us[3]
                  << " warmup=10 samples=7 repeats=32 profiler=off" << std::endl;
    }
    const char* trace_path = std::getenv("RTP_DSPARK_POSTSAMPLE_TRACE");
    if (trace_path && trace_path[0]) {
        using namespace torch::autograd::profiler;
        const ProfilerConfig         config(ProfilerState::KINETO);
        const std::set<ActivityType> activities{ActivityType::CPU, ActivityType::CUDA};
        prepareProfiler(config, activities);
        enableProfiler(config, activities);
        struct ProfilerCleanup {
            bool active = true;
            ~ProfilerCleanup() {
                if (active) {
                    try {
                        torch::autograd::profiler::disableProfiler();
                    } catch (...) {}
                }
            }
        } cleanup;
        for (int repeat = 0; repeat < 3; ++repeat) {
            {
                RECORD_FUNCTION("dspark_postsample.old_chain", std::vector<c10::IValue>());
                run(false);
            }
            {
                RECORD_FUNCTION("dspark_postsample.fused", std::vector<c10::IValue>());
                run(true);
            }
        }
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        auto result    = disableProfiler();
        cleanup.active = false;
        result->save(trace_path);
        std::cout << "dspark_postsample_trace=" << trace_path << std::endl;
    }
}

TEST_F(MtpRoundHeadPrepareTest, BudgetedVerifyGraphGpuBenchmark) {
    const auto           i32   = cudaInt32();
    constexpr int64_t    batch = 48, gamma = 5, budget = batch * 3;
    constexpr int        replays_per_batch = 128, warmup_batches = 10, samples = 7;
    std::vector<int64_t> order(batch);
    std::iota(order.rbegin(), order.rend(), 0);
    auto rates     = torch::rand({batch, gamma}, i32.dtype(torch::kFloat32));
    auto anchors   = torch::arange(batch, i32);
    auto proposals = torch::arange(batch * gamma, i32).reshape({batch, gamma});
    auto history   = torch::arange(batch * 4, i32).reshape({batch, 4});
    auto prefixes  = torch::arange(batch, i32) + 100;
    // Both order tensors outlive every captured graph. The baseline captures
    // the production H2D upload explicitly, while its helper consumes the
    // already allocated device tensor and creates no temporary host source.
    auto order_host   = torch::tensor(order, torch::kInt64).pin_memory();
    auto order_device = torch::empty({batch}, i32.dtype(torch::kInt64));
    order_device.copy_(order_host, true);
    auto warm_old =
        oldBudgetedVerify(rates, order, anchors, proposals, history, {}, prefixes, budget, stream_, true, order_device);
    auto warm_new = tryMtpDSparkBudgetedVerifyPrepare(
        rates, order, anchors, proposals, history, {}, prefixes, budget, stream_, true);
    expectBudgetedVerifyEqual(warm_new, warm_old);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    at::cuda::CUDAGraph           old_step(true), fused_step(true);
    MtpDSparkBudgetedVerifyInputs old_outputs, fused_outputs;
    old_step.capture_begin();
    order_device.copy_(order_host, true);
    old_outputs =
        oldBudgetedVerify(rates, order, anchors, proposals, history, {}, prefixes, budget, stream_, true, order_device);
    old_step.capture_end();
    old_step.instantiate();
    fused_step.capture_begin();
    fused_outputs = tryMtpDSparkBudgetedVerifyPrepare(
        rates, order, anchors, proposals, history, {}, prefixes, budget, stream_, true);
    fused_step.capture_end();
    fused_step.instantiate();
    ASSERT_EQ(cudaGraphLaunch(old_step.raw_cuda_graph_exec(), stream_), cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(fused_step.raw_cuda_graph_exec(), stream_), cudaSuccess);
    expectBudgetedVerifyEqual(fused_outputs, old_outputs);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    struct StepNodeCounts {
        size_t total = 0, kernels = 0, h2d = 0, d2d = 0, d2h = 0, other_copies = 0, other = 0;
    };
    auto count_nodes = [](cudaGraph_t graph) {
        StepNodeCounts counts;
        TORCH_CHECK(cudaGraphGetNodes(graph, nullptr, &counts.total) == cudaSuccess, "cannot count step graph nodes");
        std::vector<cudaGraphNode_t> nodes(counts.total);
        TORCH_CHECK(cudaGraphGetNodes(graph, nodes.data(), &counts.total) == cudaSuccess,
                    "cannot read step graph nodes");
        for (auto node : nodes) {
            cudaGraphNodeType type;
            TORCH_CHECK(cudaGraphNodeGetType(node, &type) == cudaSuccess, "cannot read step node type");
            if (type == cudaGraphNodeTypeKernel) {
                ++counts.kernels;
            } else if (type == cudaGraphNodeTypeMemcpy) {
                cudaMemcpy3DParms copy{};
                TORCH_CHECK(cudaGraphMemcpyNodeGetParams(node, &copy) == cudaSuccess, "cannot read graph copy kind");
                switch (copy.kind) {
                    case cudaMemcpyHostToDevice:
                        ++counts.h2d;
                        break;
                    case cudaMemcpyDeviceToDevice:
                        ++counts.d2d;
                        break;
                    case cudaMemcpyDeviceToHost:
                        ++counts.d2h;
                        break;
                    default:
                        ++counts.other_copies;
                        break;
                }
            } else {
                ++counts.other;
            }
        }
        return counts;
    };
    const StepNodeCounts counts[2] = {count_nodes(old_step.raw_cuda_graph()), count_nodes(fused_step.raw_cuda_graph())};
    EXPECT_EQ(counts[1].total, 1);
    EXPECT_EQ(counts[1].kernels, 1);
    EXPECT_EQ(counts[0].h2d, 1);

    // Measurement only: a single native launch runs 128 dependent copies of
    // the step graph. This keeps even the one-kernel candidate supplied with
    // GPU work without CPU launch gaps between individual step replays.
    // CapturedGraph releases exec before graph on every return/exception path.
    CapturedGraph old_batch, fused_batch;
    auto          make_batch = [](cudaGraph_t step, CapturedGraph& output) {
        TORCH_CHECK(cudaGraphCreate(&output.graph, 0) == cudaSuccess, "cannot create measurement graph");
        cudaGraphNode_t previous = nullptr;
        for (int replay = 0; replay < replays_per_batch; ++replay) {
            cudaGraphNode_t child;
            TORCH_CHECK(
                cudaGraphAddChildGraphNode(&child, output.graph, previous ? &previous : nullptr, previous ? 1 : 0, step)
                    == cudaSuccess,
                "cannot append measurement replay");
            previous = child;
        }
        TORCH_CHECK(cudaGraphInstantiate(&output.executable, output.graph, nullptr, nullptr, 0) == cudaSuccess,
                    "cannot instantiate measurement graph");
    };
    make_batch(old_step.raw_cuda_graph(), old_batch);
    make_batch(fused_step.raw_cuda_graph(), fused_batch);
    for (int warmup = 0; warmup < warmup_batches; ++warmup) {
        ASSERT_EQ(cudaGraphLaunch(old_batch.executable, stream_), cudaSuccess);
        ASSERT_EQ(cudaGraphLaunch(fused_batch.executable, stream_), cudaSuccess);
    }
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    expectBudgetedVerifyEqual(fused_outputs, old_outputs);

    struct TimingEvents {
        cudaEvent_t begin = nullptr, end = nullptr;
        ~TimingEvents() {
            if (end) {
                cudaEventDestroy(end);
            }
            if (begin) {
                cudaEventDestroy(begin);
            }
        }
    } events;
    ASSERT_EQ(cudaEventCreate(&events.begin), cudaSuccess);
    ASSERT_EQ(cudaEventCreate(&events.end), cudaSuccess);
    std::vector<double>   gpu_us[2];
    const cudaGraphExec_t executables[2] = {old_batch.executable, fused_batch.executable};
    for (int sample = 0; sample < samples; ++sample) {
        for (int mode = 0; mode < 2; ++mode) {
            ASSERT_EQ(cudaEventRecord(events.begin, stream_), cudaSuccess);
            ASSERT_EQ(cudaGraphLaunch(executables[mode], stream_), cudaSuccess);
            ASSERT_EQ(cudaEventRecord(events.end, stream_), cudaSuccess);
            ASSERT_EQ(cudaEventSynchronize(events.end), cudaSuccess);
            float milliseconds = 0;
            ASSERT_EQ(cudaEventElapsedTime(&milliseconds, events.begin, events.end), cudaSuccess);
            gpu_us[mode].push_back(milliseconds * 1000.0 / replays_per_batch);
        }
    }
    expectBudgetedVerifyEqual(fused_outputs, old_outputs);
    for (int mode = 0; mode < 2; ++mode) {
        std::sort(gpu_us[mode].begin(), gpu_us[mode].end());
        std::cout << "dspark_postsample_graph_gpu_bench B=48 gamma=5 budget=144 queries=192 mode="
                  << (mode ? "fused" : "old") << " pipeline_median_us=" << gpu_us[mode][samples / 2]
                  << " pipeline_min_us=" << gpu_us[mode].front() << " pipeline_max_us=" << gpu_us[mode].back()
                  << " step_nodes=" << counts[mode].total << " kernel_nodes=" << counts[mode].kernels
                  << " h2d_nodes=" << counts[mode].h2d << " d2d_nodes=" << counts[mode].d2d
                  << " d2h_nodes=" << counts[mode].d2h << " other_copy_nodes=" << counts[mode].other_copies
                  << " other_nodes=" << counts[mode].other << " child_replays=" << replays_per_batch
                  << " warmup_batches=" << warmup_batches << " samples=" << samples
                  << " copies=included gpu_graph_scheduling=included host_per_op_dispatch=excluded profiler=off"
                  << std::endl;
    }
}

}  // namespace
}  // namespace rtp_llm
