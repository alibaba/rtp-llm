#include <algorithm>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

#include "rtp_llm/models_py/bindings/cuda/kernels/cuda_graph_prepare.h"

namespace rtp_llm {
namespace {

constexpr int32_t kSentinel = -1234567;

class FlashInferDecodeParamsTest: public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream_));
    }

    void TearDown() override {
        if (graph_exec_ != nullptr) {
            EXPECT_EQ(cudaSuccess, cudaGraphExecDestroy(graph_exec_));
        }
        if (graph_ != nullptr) {
            EXPECT_EQ(cudaSuccess, cudaGraphDestroy(graph_));
        }
        for (int32_t* allocation : allocations_) {
            EXPECT_EQ(cudaSuccess, cudaFree(allocation));
        }
        EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream_));
    }

    int32_t* allocate(size_t count) {
        int32_t* result = nullptr;
        EXPECT_EQ(cudaSuccess, cudaMalloc(&result, count * sizeof(int32_t)));
        allocations_.push_back(result);
        return result;
    }

    void initialize(int32_t batch_size, int32_t max_blocks_per_batch, int32_t page_size) {
        batch_size_           = batch_size;
        max_blocks_per_batch_ = max_blocks_per_batch;
        page_size_            = page_size;
        lengths_              = allocate(batch_size);
        block_ids_            = allocate(batch_size * max_blocks_per_batch);
        for (int32_t size : {batch_size,
                             batch_size * max_blocks_per_batch,
                             batch_size + 1,
                             batch_size,
                             batch_size + 1,
                             batch_size,
                             batch_size}) {
            output_sizes_.push_back(size);
            outputs_.push_back(allocate(size + 2));
        }
    }

    void invoke() {
        invokePrepareFlashInferDecodeParams(lengths_,
                                            block_ids_,
                                            outputs_[0] + 1,
                                            outputs_[1] + 1,
                                            outputs_[2] + 1,
                                            outputs_[3] + 1,
                                            outputs_[4] + 1,
                                            outputs_[5] + 1,
                                            outputs_[6] + 1,
                                            batch_size_,
                                            max_blocks_per_batch_,
                                            page_size_,
                                            stream_);
    }

    void capture() {
        ASSERT_EQ(cudaSuccess, cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
        invoke();
        ASSERT_EQ(cudaSuccess, cudaStreamEndCapture(stream_, &graph_));
        ASSERT_EQ(cudaSuccess, cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0));
    }

    void check(const std::vector<int32_t>& lengths, int32_t block_id_base, bool replay = false) {
        ASSERT_EQ(batch_size_, lengths.size());
        std::vector<int32_t> blocks(batch_size_ * max_blocks_per_batch_);
        std::iota(blocks.begin(), blocks.end(), block_id_base);
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpyAsync(
                      lengths_, lengths.data(), lengths.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpyAsync(
                      block_ids_, blocks.data(), blocks.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream_));

        // Guard both ends and the unused page-capacity tail, including after replay.
        std::vector<std::vector<int32_t>> initial;
        for (int32_t size : output_sizes_) {
            initial.emplace_back(size + 2, kSentinel);
        }
        for (size_t i = 0; i < outputs_.size(); ++i) {
            ASSERT_EQ(cudaSuccess,
                      cudaMemcpyAsync(outputs_[i],
                                      initial[i].data(),
                                      initial[i].size() * sizeof(int32_t),
                                      cudaMemcpyHostToDevice,
                                      stream_));
        }
        if (replay) {
            ASSERT_NE(nullptr, graph_exec_);
            ASSERT_EQ(cudaSuccess, cudaGraphLaunch(graph_exec_, stream_));
        } else {
            invoke();
        }
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));

        auto          expected  = initial;
        const int32_t page_size = std::max(page_size_, 1);
        int32_t       offset    = 0;
        expected[2][1] = expected[4][1] = 0;
        for (int32_t row = 0; row < batch_size_; ++row) {
            const int32_t length = std::max(lengths[row], 1);
            const int32_t pages  = std::min((length - 1) / page_size + 1, max_blocks_per_batch_);
            expected[0][row + 1] = row;
            expected[3][row + 1] = (length - 1) % page_size + 1;
            expected[4][row + 2] = row + 1;
            expected[5][row + 1] = length;
            expected[6][row + 1] = length - 1;
            std::copy_n(blocks.begin() + row * max_blocks_per_batch_, pages, expected[1].begin() + offset + 1);
            offset += pages;
            expected[2][row + 2] = offset;
        }
        for (size_t i = 0; i < outputs_.size(); ++i) {
            SCOPED_TRACE(i);
            auto actual = initial[i];
            ASSERT_EQ(cudaSuccess,
                      cudaMemcpy(actual.data(), outputs_[i], actual.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
            EXPECT_EQ(expected[i], actual);
        }
    }

    cudaStream_t          stream_     = nullptr;
    cudaGraph_t           graph_      = nullptr;
    cudaGraphExec_t       graph_exec_ = nullptr;
    std::vector<int32_t*> allocations_;
    std::vector<int32_t*> outputs_;
    std::vector<int32_t>  output_sizes_;
    int32_t*              lengths_              = nullptr;
    int32_t*              block_ids_            = nullptr;
    int32_t               batch_size_           = 0;
    int32_t               max_blocks_per_batch_ = 0;
    int32_t               page_size_            = 0;
};

TEST_F(FlashInferDecodeParamsTest, MixedLengthsClipPageCopiesAndPreserveMetadata) {
    initialize(7, 4, 64);
    check({0, -7, 1, 64, 65, 256, 81921}, 100);
}

TEST_F(FlashInferDecodeParamsTest, NonpositivePageSizeUsesOne) {
    initialize(4, 8, 0);
    check({0, 1, 3, 10}, 200);
}

TEST_F(FlashInferDecodeParamsTest, LongContextGraphReplayReadsNewLengthsAndBlockIds) {
    initialize(16, 1296, 64);
    std::vector<int32_t> lengths(16, 81921);
    check(lengths, 1000);
    capture();
    check(lengths, 2000, true);
    lengths = {0, 1, 64, 65, 81920, 81921, 82944, 90000, -1, 0, 0, 0, 0, 0, 0, 0};
    check(lengths, 3000, true);
    check(std::vector<int32_t>(16, 81921), 4000, true);
}

}  // namespace
}  // namespace rtp_llm
