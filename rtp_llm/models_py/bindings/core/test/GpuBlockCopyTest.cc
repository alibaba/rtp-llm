#include "rtp_llm/models_py/bindings/core/GpuBlockCopy.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <gtest/gtest.h>
#include <cstring>

namespace rtp_llm {
namespace {

torch::Tensor mapping(const std::vector<int32_t>& ids) {
    return torch::tensor(ids, torch::kInt32).reshape({-1, 3});
}

torch::Tensor pattern(int rows, int stride, int seed = 0) {
    return (torch::arange(rows * stride, torch::kInt64).reshape({rows, stride}) * 13
            + torch::arange(rows, torch::kInt64).reshape({rows, 1}) * 17 + seed).to(torch::kUInt8);
}

void applyReference(torch::Tensor& expected, const torch::Tensor& ids, size_t copy_bytes) {
    auto original = expected.clone();
    const auto* entries = ids.data_ptr<int32_t>();
    for (int64_t i = 0; i < ids.size(0); ++i) {
        std::memcpy(expected.data_ptr<uint8_t>() + entries[3*i+2] * expected.stride(0),
                    original.data_ptr<uint8_t>() + entries[3*i+1] * original.stride(0), copy_bytes);
    }
}

TEST(GpuBlockCopyTest, BeamFanoutAllLayersAndScale) {
    c10::InferenceMode inference;
    std::vector<GpuBlockCopyPlane> planes;
    std::vector<torch::Tensor> expected;
    std::vector<int32_t> ids;
    for (int beam = 0; beam < 1024; ++beam) {
        ids.insert(ids.end(), {0, beam % 160, 160 + beam});
    }
    auto mappings = mapping(ids);
    for (int layer = 0; layer < 24; ++layer) {
        for (int bytes : {8192, 32}) {
            auto host = pattern(1186, bytes, layer);
            planes.push_back({host.cuda(), static_cast<size_t>(bytes)});
            applyReference(host, mappings, bytes);
            expected.push_back(host);
        }
    }
    auto op = GpuBlockCopy::create(planes);
    op->copy(mappings);
    for (size_t i = 0; i < planes.size(); ++i) {
        EXPECT_TRUE(torch::equal(planes[i].blocks.cpu(), expected[i])) << "plane " << i;
    }
}

TEST(GpuBlockCopyTest, PaddingUnalignedPointersAndByteTails) {
    for (auto sizes : {std::pair<int,int>{64, 37}, {47, 35}, {16, 3}}) {
        auto host = pattern(9, sizes.first);
        auto device = host.cuda();
        auto op = GpuBlockCopy::create({{device, static_cast<size_t>(sizes.second)}});
        auto ids = mapping({0, 0, 5, 0, 1, 6, 0, 1, 7, 0, 8, 8});
        applyReference(host, ids, sizes.second);
        op->copy(ids);
        EXPECT_TRUE(torch::equal(device.cpu(), host));
    }
}

TEST(GpuBlockCopyTest, VariableCountAndImmediateHostMutation) {
    auto host = pattern(1200, 64);
    auto device = host.cuda();
    auto op = GpuBlockCopy::create({{device, 64}});
    for (int count : {1, 160, 1024, 0, 12, 1024, 1}) {
        std::vector<int32_t> entries;
        for (int i = 0; i < count; ++i) {
            entries.insert(entries.end(), {0, (i + count) % 160, i + 160});
        }
        auto ids = mapping(entries);
        applyReference(host, ids, 64);
        op->copy(ids);
        // The op must own H2D staging instead of borrowing the caller's buffer.
        ids.fill_(-1);
    }
    EXPECT_TRUE(torch::equal(device.cpu(), host));
}

TEST(GpuBlockCopyTest, SlotReuseAndSwitchingCudaStreams) {
    auto host = pattern(8, 8192);
    auto device = host.cuda();
    auto op = GpuBlockCopy::create({{device, 8192}});
    const auto first = at::cuda::getStreamFromPool();
    const auto second = at::cuda::getStreamFromPool();
    for (int i = 0; i < 100; ++i) {
        c10::cuda::CUDAStreamGuard guard(i % 2 ? first : second);
        const int src = i % 8, dst = (i + 1) % 8;
        auto ids = mapping({0, src, dst});
        applyReference(host, ids, 8192);
        op->copy(ids);
    }
    // Destructor also has to drain non-default-stream copies.
    op.reset();
    EXPECT_TRUE(torch::equal(device.cpu(), host));
}

TEST(GpuBlockCopyTest, CopyPrecedesCudaGraphReplay) {
    auto host = pattern(6, 256);
    auto device = host.cuda();
    auto observed = torch::empty({256}, device.options());
    auto op = GpuBlockCopy::create({{device, 256}});
    const auto stream = at::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard guard(stream);
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    C10_CUDA_CHECK(cudaStreamBeginCapture(stream.stream(), cudaStreamCaptureModeThreadLocal));
    C10_CUDA_CHECK(cudaMemcpyAsync(observed.data_ptr(), device.data_ptr<uint8_t>() + 5 * 256,
                                   256, cudaMemcpyDeviceToDevice, stream.stream()));
    C10_CUDA_CHECK(cudaStreamEndCapture(stream.stream(), &graph));
    C10_CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int source : {0, 3, 1, 4, 2}) {
        op->copy(mapping({0, source, 5}));
        C10_CUDA_CHECK(cudaGraphLaunch(exec, stream.stream()));
        EXPECT_TRUE(torch::equal(observed.cpu(), host[source]));
    }
    C10_CUDA_CHECK(cudaGraphExecDestroy(exec));
    C10_CUDA_CHECK(cudaGraphDestroy(graph));
}

TEST(GpuBlockCopyTest, RejectInvalidMappingsBeforeEnqueue) {
    auto host = pattern(8, 16);
    auto device = host.cuda();
    auto op = GpuBlockCopy::create({{device, 16}});
    EXPECT_THROW(op->copy(mapping({1, 0, 1})), c10::Error);
    EXPECT_THROW(op->copy(mapping({0, -1, 1})), c10::Error);
    EXPECT_THROW(op->copy(mapping({0, 0, 8})), c10::Error);
    EXPECT_THROW(op->copy(torch::zeros({2, 3}, torch::kInt64)), c10::Error);
    EXPECT_THROW(op->copy(torch::zeros({2, 2}, torch::kInt32)), c10::Error);
    EXPECT_THROW(op->copy(torch::zeros({3, 2}, torch::kInt32).t()), c10::Error);
    EXPECT_THROW(op->copy(mapping({0, 0, 1}).cuda()), c10::Error);
    EXPECT_TRUE(torch::equal(device.cpu(), host));
}

}  // namespace
}  // namespace rtp_llm
