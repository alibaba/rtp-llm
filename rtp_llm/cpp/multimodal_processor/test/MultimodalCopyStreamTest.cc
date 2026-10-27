#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>

#include "gtest/gtest.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"
#include "rtp_llm/cpp/multimodal_processor/transport/rdma/MMRdmaReader.h"
#include "rtp_llm/cpp/testing/TestBase.h"

#if USING_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>

namespace rtp_llm {
namespace {

class CudaOutputProcessor: public MultimodalProcessor {
public:
    CudaOutputProcessor(): MultimodalProcessor(py::none(), MMModelConfig{true, {{1}}, false}, 100) {
        output_.mm_features       = {torch::arange(8, torch::kFloat32).reshape({2, 4}).cuda()};
        output_.mm_extra_input    = std::vector<torch::Tensor>{torch::arange(6, torch::kInt64).cuda()};
        output_.mm_feature_hashes = std::vector<torch::Tensor>{torch::tensor({-10, 11}, torch::kInt32)};
    }

    bool fail_next = false;

private:
    ErrorResult<MultimodalOutput>
    MultimodalEmbedding(const std::vector<MultimodalInput>, std::string, int64_t, grpc::ServerContext*) override {
        if (fail_next) {
            fail_next = false;
            return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, "injected multimodal transport failure");
        }
        return MultimodalOutput(output_);
    }

    MultimodalOutput output_;
};

std::shared_ptr<GenerateInput> multimodalInput() {
    auto input               = std::make_shared<GenerateInput>();
    input->input_ids         = torch::tensor({0, 1, 2}, torch::kInt32);
    input->multimodal_inputs = std::vector<MultimodalInput>{MultimodalInput("image")};
    return input;
}

class PendingInference {
public:
    ~PendingInference() {
        release();
        if (stream_ != nullptr) {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
        }
    }

    cudaError_t start() {
        auto status = cudaStreamCreateWithFlags(&stream_, cudaStreamDefault);
        if (status != cudaSuccess) {
            return status;
        }
        return cudaLaunchHostFunc(stream_, [](void* state) { static_cast<PendingInference*>(state)->wait(); }, this);
    }

    bool waitUntilEntered() {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, std::chrono::seconds(2), [this]() { return entered_; });
    }

    void release() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        cv_.notify_all();
    }

private:
    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        entered_ = true;
        cv_.notify_all();
        // Even the unfixed implementation must finish instead of hanging the test process.
        cv_.wait_for(lock, std::chrono::seconds(5), [this]() { return released_; });
    }

    cudaStream_t            stream_ = nullptr;
    std::mutex              mutex_;
    std::condition_variable cv_;
    bool                    entered_  = false;
    bool                    released_ = false;
};

class MultimodalCopyStreamTest: public DeviceTestBase {};

TEST_F(MultimodalCopyStreamTest, PinnedHostCopyDoesNotWaitForUnrelatedInference) {
    CudaOutputProcessor processor;
    auto                warmup = multimodalInput();
    ASSERT_TRUE(processor.updateMultimodalFeatures(warmup).ok());
    warmup.reset();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int device_id = -1;
    ASSERT_EQ(cudaGetDevice(&device_id), cudaSuccess);
    auto             input = multimodalInput();
    PendingInference inference;
    ASSERT_EQ(inference.start(), cudaSuccess);
    ASSERT_TRUE(inference.waitUntilEntered());

    auto       result = std::async(std::launch::async, [&]() {
        c10::cuda::CUDAGuard device_guard(device_id);
        return processor.updateMultimodalFeatures(input);
    });
    const auto ready  = result.wait_for(std::chrono::seconds(2));
    inference.release();
    const auto status = result.get();
    EXPECT_EQ(ready, std::future_status::ready);
    ASSERT_TRUE(status.ok()) << status.ToString();
    ASSERT_TRUE(input->multimodal_features.has_value());
    ASSERT_TRUE(input->mm_extra_input.has_value());
    const auto& embedding = input->multimodal_features->at(0);
    const auto& extra     = input->mm_extra_input->at(0);
    EXPECT_TRUE(embedding.is_cpu());
    EXPECT_TRUE(embedding.is_pinned());
    EXPECT_TRUE(extra.is_cpu());
    EXPECT_TRUE(extra.is_pinned());
    EXPECT_TRUE(torch::equal(embedding, torch::arange(8, torch::kFloat32).reshape({2, 4})));
    EXPECT_TRUE(torch::equal(extra, torch::arange(6, torch::kInt64)));
}

TEST_F(MultimodalCopyStreamTest, RequestErrorAllowsNextCudaOutputRequest) {
    CudaOutputProcessor processor;
    processor.fail_next = true;
    auto       failed   = multimodalInput();
    const auto error    = processor.updateMultimodalFeatures(failed);
    EXPECT_EQ(error.code(), ErrorCode::MM_PROCESS_ERROR);

    auto       valid  = multimodalInput();
    const auto status = processor.updateMultimodalFeatures(valid);
    ASSERT_TRUE(status.ok()) << status.ToString();
    ASSERT_TRUE(valid->multimodal_features.has_value());
    EXPECT_TRUE(valid->multimodal_features->at(0).is_pinned());
    EXPECT_TRUE(torch::equal(valid->multimodal_features->at(0), torch::arange(8, torch::kFloat32).reshape({2, 4})));
}

TEST_F(MultimodalCopyStreamTest, RdmaAssemblyWithPositionIdsDoesNotWaitForUnrelatedInference) {
    const auto                       embedding = torch::arange(12, torch::kFloat32).reshape({3, 4});
    const auto                       positions = torch::arange(6, torch::kInt64).reshape({3, 2});
    const std::vector<torch::Tensor> tensors   = {
        embedding.narrow(0, 0, 2).cuda(), embedding.narrow(0, 2, 1).cuda(), positions.cuda()};
    const std::vector<MMRdmaSlotPB::Role> roles = {
        MMRdmaSlotPB::EMBEDDING, MMRdmaSlotPB::EMBEDDING, MMRdmaSlotPB::POS_ID};
    MultimodalOutputPB receipt;
    receipt.add_split_size(2);
    receipt.add_split_size(1);
    {
        MultimodalOutput warmup;
        ASSERT_TRUE(assembleMMRdmaOutput(tensors, roles, &receipt, &warmup));
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    int device_id = -1;
    ASSERT_EQ(cudaGetDevice(&device_id), cudaSuccess);

    MultimodalOutput output;
    PendingInference inference;
    ASSERT_EQ(inference.start(), cudaSuccess);
    ASSERT_TRUE(inference.waitUntilEntered());
    auto       result = std::async(std::launch::async, [&]() {
        c10::cuda::CUDAGuard device_guard(device_id);
        return assembleMMRdmaOutput(tensors, roles, &receipt, &output);
    });
    const auto ready  = result.wait_for(std::chrono::seconds(2));
    inference.release();
    const auto success = result.get();
    EXPECT_EQ(ready, std::future_status::ready);
    ASSERT_TRUE(success);
    ASSERT_EQ(output.mm_features.size(), 2u);
    ASSERT_TRUE(output.mm_position_ids.has_value());
    ASSERT_EQ(output.mm_position_ids->size(), 2u);
    EXPECT_TRUE(torch::equal(output.mm_features[0].cpu(), embedding.narrow(0, 0, 2)));
    EXPECT_TRUE(torch::equal(output.mm_features[1].cpu(), embedding.narrow(0, 2, 1)));
    for (const auto& position : *output.mm_position_ids) {
        EXPECT_TRUE(position.is_cpu());
        EXPECT_TRUE(position.is_pinned());
    }
    EXPECT_TRUE(torch::equal(output.mm_position_ids->at(0), positions.narrow(0, 0, 2)));
    EXPECT_TRUE(torch::equal(output.mm_position_ids->at(1), positions.narrow(0, 2, 1)));
}

TEST_F(MultimodalCopyStreamTest, InvalidRdmaManifestAllowsNextAssembly) {
    const std::vector<torch::Tensor>      tensors = {torch::ones({2, 4}).cuda()};
    const std::vector<MMRdmaSlotPB::Role> roles   = {MMRdmaSlotPB::EMBEDDING};
    MultimodalOutputPB                    receipt;
    receipt.add_split_size(3);
    MultimodalOutput output;
    EXPECT_FALSE(assembleMMRdmaOutput(tensors, roles, &receipt, &output));

    receipt.set_split_size(0, 2);
    ASSERT_TRUE(assembleMMRdmaOutput(tensors, roles, &receipt, &output));
    ASSERT_EQ(output.mm_features.size(), 1u);
    EXPECT_TRUE(torch::equal(output.mm_features[0].cpu(), torch::ones({2, 4})));
}

}  // namespace
}  // namespace rtp_llm
#endif
