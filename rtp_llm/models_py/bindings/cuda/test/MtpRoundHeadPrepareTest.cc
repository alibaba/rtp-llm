#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <memory>
#include <vector>

#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_target_verify_prepare.h"

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
