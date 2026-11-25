#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <array>
#include <cstring>
#include <exception>
#include <thread>

namespace rtp_llm {
namespace {

torch::TensorOptions bytesOn(torch::Device device) {
    return torch::TensorOptions().dtype(torch::kUInt8).device(device);
}

TEST(PpuNoBlockCopyTest, CopiesAllDirectionsAndInteriorViews) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto host        = torch::arange(256, torch::kInt32).to(torch::kUInt8).pin_memory();
    auto device      = torch::zeros({512}, bytesOn(torch::Device(torch::kCUDA, 0)));
    auto device_copy = torch::zeros_like(device);
    auto result      = torch::zeros_like(host);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    MultiCopyParams h2d;
    h2d.multi_src = {host.slice(0, 17, 217)};
    h2d.multi_dst = {device.slice(0, 129, 329)};
    auto owner    = acquireDeviceHostCopyStreams(0);
    execNoBlockCopy(h2d, {owner, DeviceHostCopyDirection::H2D});
    execNoBlockCopy({{device_copy.slice(0, 31, 231)}, h2d.multi_dst});
    execNoBlockCopy({{result.slice(0, 17, 217)}, {device_copy.slice(0, 31, 231)}},
                    {owner, DeviceHostCopyDirection::D2H});
    EXPECT_TRUE(torch::equal(result.slice(0, 17, 217), host.slice(0, 17, 217)));
    EXPECT_EQ(result.slice(0, 0, 17).sum().item<int64_t>(), 0);
    EXPECT_EQ(result.slice(0, 217).sum().item<int64_t>(), 0);
    auto host_copy = torch::empty_like(result);
    execNoBlockCopy({{host_copy}, {result}});
    EXPECT_TRUE(torch::equal(host_copy, result));
}

TEST(PpuNoBlockCopyTest, ValidatesBeforeSubmittingAndHandlesEmptyCopies) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto src = torch::full({16}, 7, bytesOn(torch::Device(torch::kCUDA, 0)));
    auto dst = torch::zeros({16}, torch::kUInt8).pin_memory();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    // The first item must not be copied when a later item is invalid.
    EXPECT_THROW(execNoBlockCopy({{dst, dst.slice(0, 0, 15)}, {src, src}}), c10::Error);
    EXPECT_EQ(dst.sum().item<int64_t>(), 0);
    EXPECT_THROW(execNoBlockCopy({{dst}, {src.slice(0, 0, 16, 2)}}), c10::Error);
    EXPECT_THROW(execNoBlockCopy({{dst}, {src}}, {acquireDeviceHostCopyStreams(0), DeviceHostCopyDirection::H2D}),
                 c10::Error);
    EXPECT_NO_THROW(execNoBlockCopy({}));
    EXPECT_NO_THROW(execNoBlockCopy({{dst.slice(0, 0, 0)}, {src.slice(0, 0, 0)}}));
}

TEST(PpuNoBlockCopyTest, HostKvRoundTripDoesNotUsePooledDmaOrConvertBytes) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto owner = acquireDeviceHostCopyStreams(0);
    auto gpu = torch::zeros({15}, bytesOn(torch::Device(torch::kCUDA, 0)));
    auto backing = torch::tensor({0, 513, 1027, 0}, torch::kInt32).pin_memory();
    auto words = backing.narrow(0, 1, 2);
    auto payload = words.view(torch::kUInt8);
    auto output = torch::full({11}, 93, torch::kUInt8);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Reject the known-bad submission path. Both actual transfers must still
    // succeed, and a larger destination must retain its untouched suffix.
    struct RestoreHook {
        decltype(no_block_copy_detail::cuda_result_for_test) previous;
        ~RestoreHook() { no_block_copy_detail::cuda_result_for_test = previous; }
    } restore{no_block_copy_detail::cuda_result_for_test};
    no_block_copy_detail::cuda_result_for_test = [](const char* phase, int result) {
        return std::strcmp(phase, "generic_submit") == 0 ? static_cast<int>(cudaErrorInvalidValue) : result;
    };
    execNoBlockCopy({{gpu.narrow(0, 3, 10)}, {words}}, {owner, DeviceHostCopyDirection::H2D});
    execNoBlockCopy({{output.narrow(0, 1, 9)}, {gpu.narrow(0, 3, 8)}}, {owner, DeviceHostCopyDirection::D2H});
    EXPECT_TRUE(torch::equal(output.narrow(0, 1, 8), payload));
    EXPECT_EQ(output[0].item<int>(), 93);
    EXPECT_EQ(output[9].item<int>(), 93);
    EXPECT_EQ(gpu[11].item<int>(), 0);
}

TEST(PpuNoBlockCopyTest, ConcurrentLayerStagingKeepsBytesAndOwnerLifetimes) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    constexpr int layers = 24;
    constexpr int bytes  = 8192;
    auto          pool   = torch::empty({layers, bytes + 256}, bytesOn(torch::Device(torch::kCUDA, 0)));
    for (int layer = 0; layer < layers; ++layer) {
        pool[layer].fill_(layer + 1);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::array<std::exception_ptr, layers> errors{};
    std::array<torch::Tensor, layers>      results;
    std::vector<std::thread>               workers;
    for (int layer = 0; layer < layers; ++layer) {
        workers.emplace_back([&, layer]() {
            try {
                auto            pinned = torch::empty({bytes}, torch::kUInt8).pin_memory();
                MultiCopyParams params;
                for (int block = 0; block < 4; ++block) {
                    // Match makeValidBlocks: non-owning byte tensors into one
                    // pinned allocation and an interior layer of a GPU pool.
                    params.multi_src.push_back(torch::from_blob(pool[layer].data_ptr<uint8_t>() + 128 + block * 2048,
                                                                {2048},
                                                                bytesOn(torch::Device(torch::kCUDA, 0))));
                    params.multi_dst.push_back(
                        torch::from_blob(pinned.data_ptr<uint8_t>() + block * 2048, {2048}, bytesOn(torch::kCPU)));
                }
                execNoBlockCopy(params);
                results[layer] = pinned;
            } catch (...) { errors[layer] = std::current_exception(); }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    pool = torch::Tensor();
    for (int layer = 0; layer < layers; ++layer) {
        ASSERT_EQ(errors[layer], nullptr) << "layer=" << layer;
        EXPECT_TRUE(torch::equal(results[layer], torch::full({bytes}, layer + 1, torch::kUInt8)));
    }
}

TEST(PpuNoBlockCopyTest, PrefillReloadsInteriorHostSlotIntoDeviceBlockOnWorker) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    // Match pd_reuse_prefill_memory: host slot 1 is 24 x 4 KiB into the
    // host backing; layer 0, device block 3 is 12 KiB into the KV pool.
    constexpr int64_t bytes = 4096, host_slot = 24 * bytes, device_block = 3 * bytes;
    auto host_opts = bytesOn(torch::kCPU);
    auto gpu_opts  = bytesOn(torch::Device(torch::kCUDA, 0));
    auto host      = torch::empty({host_slot * 2}, host_opts.pinned_memory(true)).fill_(19);
    auto gpu       = torch::full({device_block + 2 * bytes}, 37, gpu_opts);
    auto owner     = acquireDeviceHostCopyStreams(0);
    auto host_view = torch::from_blob(host.data_ptr<uint8_t>() + host_slot, {bytes}, host_opts);
    auto gpu_view  = torch::from_blob(gpu.data_ptr<uint8_t>() + device_block, {bytes}, gpu_opts);
    auto expected  = torch::arange(bytes, torch::kInt32).to(torch::kUInt8);
    host_view.copy_(expected);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    // Negative case: a short second destination must reject the entire H2D
    // batch rather than partially restoring the first (valid) KV block.
    EXPECT_THROW(execNoBlockCopy({{gpu_view, gpu_view.narrow(0, 0, bytes - 1)}, {host_view, host_view}},
                                 {owner, DeviceHostCopyDirection::H2D}),
                 c10::Error);
    ASSERT_TRUE(torch::equal(gpu.cpu(), torch::full({gpu.numel()}, 37, host_opts)));
    std::exception_ptr error;
    std::thread worker([&] {
        try {
            execNoBlockCopy({{gpu_view}, {host_view}}, {owner, DeviceHostCopyDirection::H2D});
        } catch (...) { error = std::current_exception(); }
    });
    worker.join();
    if (error) {
        std::rethrow_exception(error);
    }
    auto expected_gpu = torch::full({gpu.numel()}, 37, host_opts);
    expected_gpu.narrow(0, device_block, bytes).copy_(expected);
    EXPECT_TRUE(torch::equal(gpu.cpu(), expected_gpu));
}

TEST(PpuNoBlockCopyTest, CopiesBeyondFourGiBAllocationOffset) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    // The smoke pool is 91 GiB; layer bases pass 32-bit offsets. Reserve only
    // 4 GiB here and touch a small tail, without filling the entire allocation.
    constexpr int64_t offset = (int64_t{1} << 32) + 256;
    auto              pool   = torch::empty({offset + 4096}, bytesOn(torch::Device(torch::kCUDA, 0)));
    auto              tail   = pool.slice(0, offset, offset + 4096);
    tail.fill_(173);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto pinned = torch::zeros({4096}, torch::kUInt8).pin_memory();
    auto source = torch::from_blob(tail.data_ptr(), {4096}, tail.options());
    execNoBlockCopy({{pinned}, {source}});
    EXPECT_TRUE(torch::equal(pinned, torch::full({4096}, 173, torch::kUInt8)));
}

}  // namespace
}  // namespace rtp_llm
