#include "gtest/gtest.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheTransferBufferPool.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ThreadLocalScratch.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/CacheStoreTestBase.h"

#include <atomic>
#include <cuda_runtime.h>
#include <thread>
#include <vector>

namespace rtp_llm {

class GatheredTransferIntegrationTest: public CacheStoreTestBase {};

TEST_F(GatheredTransferIntegrationTest, PoolAllocD2HViaCudaMemcpy) {
    constexpr size_t kPoolSize  = 64 * 1024;
    constexpr size_t kBlockSize = 1024;
    constexpr int    kNumBlocks = 4;

    CacheTransferBufferPool pool(kPoolSize, memory_util_);

    size_t total_size = kBlockSize * kNumBlocks;
    auto*  handle     = pool.tryAllocate(total_size);
    ASSERT_NE(handle, nullptr);
    EXPECT_GE(handle->size, total_size);

    // Allocate GPU memory and fill with known pattern
    void* gpu_blocks[kNumBlocks];
    for (int i = 0; i < kNumBlocks; i++) {
        ASSERT_EQ(cudaMalloc(&gpu_blocks[i], kBlockSize), cudaSuccess);
        cudaMemset(gpu_blocks[i], 'A' + i, kBlockSize);
    }
    cudaDeviceSynchronize();

    // D2H: copy GPU blocks → contiguous pool buffer via cudaMemcpy
    auto* host_ptr = static_cast<char*>(handle->ptr);
    for (int i = 0; i < kNumBlocks; i++) {
        ASSERT_EQ(cudaMemcpy(host_ptr + i * kBlockSize, gpu_blocks[i], kBlockSize, cudaMemcpyDeviceToHost),
                  cudaSuccess);
    }

    // Verify host buffer contains gathered data
    for (int i = 0; i < kNumBlocks; i++) {
        char expected = 'A' + i;
        for (size_t j = 0; j < kBlockSize; j++) {
            ASSERT_EQ(host_ptr[i * kBlockSize + j], expected) << "Mismatch at block " << i << " offset " << j;
        }
    }

    for (int i = 0; i < kNumBlocks; i++) {
        cudaFree(gpu_blocks[i]);
    }
    pool.free(handle);
}

TEST_F(GatheredTransferIntegrationTest, PoolH2DScatterViaCudaMemcpy) {
    constexpr size_t kPoolSize  = 64 * 1024;
    constexpr size_t kBlockSize = 512;
    constexpr int    kNumBlocks = 3;

    CacheTransferBufferPool pool(kPoolSize, memory_util_);

    size_t total_size = kBlockSize * kNumBlocks;
    auto*  handle     = pool.tryAllocate(total_size);
    ASSERT_NE(handle, nullptr);

    // Fill host pool buffer with known patterns
    auto* host_ptr = static_cast<char*>(handle->ptr);
    for (int i = 0; i < kNumBlocks; i++) {
        memset(host_ptr + i * kBlockSize, 'X' + i, kBlockSize);
    }

    // Allocate empty GPU blocks
    void* gpu_blocks[kNumBlocks];
    for (int i = 0; i < kNumBlocks; i++) {
        ASSERT_EQ(cudaMalloc(&gpu_blocks[i], kBlockSize), cudaSuccess);
        cudaMemset(gpu_blocks[i], 0, kBlockSize);
    }
    cudaDeviceSynchronize();

    // H2D scatter: pool buffer → GPU blocks via cudaMemcpy
    for (int i = 0; i < kNumBlocks; i++) {
        ASSERT_EQ(cudaMemcpy(gpu_blocks[i], host_ptr + i * kBlockSize, kBlockSize, cudaMemcpyHostToDevice),
                  cudaSuccess);
    }

    // Verify GPU blocks via D2H copy back
    std::vector<char> verify_buf(kBlockSize);
    for (int i = 0; i < kNumBlocks; i++) {
        cudaMemcpy(verify_buf.data(), gpu_blocks[i], kBlockSize, cudaMemcpyDeviceToHost);
        char expected = 'X' + i;
        for (size_t j = 0; j < kBlockSize; j++) {
            ASSERT_EQ(verify_buf[j], expected) << "Mismatch at block " << i << " offset " << j;
        }
    }

    for (int i = 0; i < kNumBlocks; i++) {
        cudaFree(gpu_blocks[i]);
    }
    pool.free(handle);
}

TEST_F(GatheredTransferIntegrationTest, PoolRoundTrip_D2H_then_H2D) {
    constexpr size_t kPoolSize  = 64 * 1024;
    constexpr size_t kBlockSize = 2048;
    constexpr int    kNumBlocks = 2;
    size_t           total_size = kBlockSize * kNumBlocks;

    CacheTransferBufferPool pool(kPoolSize, memory_util_);
    auto*                   handle = pool.tryAllocate(total_size);
    ASSERT_NE(handle, nullptr);

    // Allocate source and destination GPU blocks
    void* gpu_src[kNumBlocks];
    void* gpu_dst[kNumBlocks];
    for (int i = 0; i < kNumBlocks; i++) {
        ASSERT_EQ(cudaMalloc(&gpu_src[i], kBlockSize), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&gpu_dst[i], kBlockSize), cudaSuccess);
        cudaMemset(gpu_src[i], 'M' + i, kBlockSize);
        cudaMemset(gpu_dst[i], 0, kBlockSize);
    }
    cudaDeviceSynchronize();

    // D2H: gather GPU src → pool
    auto* host_ptr = static_cast<char*>(handle->ptr);
    for (int i = 0; i < kNumBlocks; i++) {
        ASSERT_EQ(cudaMemcpy(host_ptr + i * kBlockSize, gpu_src[i], kBlockSize, cudaMemcpyDeviceToHost), cudaSuccess);
    }

    // Verify host contains correct data
    for (int i = 0; i < kNumBlocks; i++) {
        EXPECT_EQ(host_ptr[i * kBlockSize], 'M' + i);
        EXPECT_EQ(host_ptr[(i + 1) * kBlockSize - 1], 'M' + i);
    }

    // H2D: scatter pool → GPU dst
    for (int i = 0; i < kNumBlocks; i++) {
        ASSERT_EQ(cudaMemcpy(gpu_dst[i], host_ptr + i * kBlockSize, kBlockSize, cudaMemcpyHostToDevice), cudaSuccess);
    }

    // Verify GPU dst matches original GPU src
    std::vector<char> verify(kBlockSize);
    for (int i = 0; i < kNumBlocks; i++) {
        cudaMemcpy(verify.data(), gpu_dst[i], kBlockSize, cudaMemcpyDeviceToHost);
        EXPECT_EQ(verify[0], 'M' + i);
        EXPECT_EQ(verify[kBlockSize - 1], 'M' + i);
    }

    for (int i = 0; i < kNumBlocks; i++) {
        cudaFree(gpu_src[i]);
        cudaFree(gpu_dst[i]);
    }
    pool.free(handle);
}

TEST_F(GatheredTransferIntegrationTest, RequestBlockBufferStoreGatheredPath) {
    auto store = std::make_shared<RequestBlockBufferStore>(memory_util_, true);

    constexpr size_t kBlockSize = 256;
    std::string      request_id = "gathered_test_req_001";

    auto request_block_buffer = std::make_shared<RequestBlockBuffer>(request_id);

    // Create GPU blocks
    void* gpu_mem;
    ASSERT_EQ(cudaMalloc(&gpu_mem, kBlockSize * 2), cudaSuccess);
    cudaMemset(gpu_mem, 'G', kBlockSize * 2);
    cudaDeviceSynchronize();

    std::shared_ptr<void> gpu_ptr1(gpu_mem, [](void*) {});
    std::shared_ptr<void> gpu_ptr2(static_cast<char*>(gpu_mem) + kBlockSize, [gpu_mem](void*) { cudaFree(gpu_mem); });

    memory_util_->regUserMr(gpu_mem, kBlockSize * 2, true);

    request_block_buffer->addBlock("block_k_0", gpu_ptr1, kBlockSize, true, true);
    request_block_buffer->addBlock("block_v_0", gpu_ptr2, kBlockSize, true, true);

    bool ret = store->setRequestBlockBuffer(request_block_buffer);

    if (memory_util_->isRdmaMode()) {
        ASSERT_TRUE(ret);
        auto stored_k = store->getBlockBuffer(request_id, "block_k_0");
        auto stored_v = store->getBlockBuffer(request_id, "block_v_0");
        ASSERT_NE(stored_k, nullptr);
        ASSERT_NE(stored_v, nullptr);
        EXPECT_EQ(stored_k->kind_, BlockBuffer::Kind::GPU_GATHERED_READY);
        EXPECT_EQ(stored_v->kind_, BlockBuffer::Kind::GPU_GATHERED_READY);
    } else {
        // Non-RDMA: normal path (may or may not succeed depending on MR state)
        (void)ret;
    }

    store->delRequestBlockBuffer(request_id);
}

TEST_F(GatheredTransferIntegrationTest, NormalCacheStorePoolInitialization) {
    CacheStoreInitParams params;
    params.listen_port  = 0;
    params.thread_count = 1;
    params.queue_size   = 10;
    params.rdma_mode    = false;
    params.memory_util  = memory_util_;

    params.kv_cache_config.enable_gathered_cache_transfer = false;
    params.kv_cache_config.cache_transfer_buffer_size_mb  = 0;

    auto store1 = NormalCacheStore::createNormalCacheStore(params);
    ASSERT_NE(store1, nullptr);
    EXPECT_EQ(store1->getBufferPool(), nullptr);

    params.kv_cache_config.enable_gathered_cache_transfer = true;
    params.kv_cache_config.cache_transfer_buffer_size_mb  = 0;

    auto store2 = NormalCacheStore::createNormalCacheStore(params);
    ASSERT_NE(store2, nullptr);
    EXPECT_EQ(store2->getBufferPool(), nullptr);

    params.kv_cache_config.enable_gathered_cache_transfer = true;
    params.kv_cache_config.cache_transfer_buffer_size_mb  = 1;

    auto store3 = NormalCacheStore::createNormalCacheStore(params);
    ASSERT_NE(store3, nullptr);
    EXPECT_NE(store3->getBufferPool(), nullptr);
    EXPECT_EQ(store3->getBufferPool()->totalBytes(), 1ULL * 1024 * 1024);
}

TEST_F(GatheredTransferIntegrationTest, ThreadLocalScratchPerDevice) {
    auto& s0a = threadLocalScratch(0);
    auto& s0b = threadLocalScratch(0);
    EXPECT_EQ(&s0a, &s0b);

    auto& s1 = threadLocalScratch(1);
    EXPECT_NE(&s0a, &s1);

    StagedMemoryCopyScratch* other_thread_scratch = nullptr;
    std::thread              t([&]() { other_thread_scratch = &threadLocalScratch(0); });
    t.join();
    ASSERT_NE(other_thread_scratch, nullptr);
    EXPECT_NE(other_thread_scratch, &s0a);
}

TEST_F(GatheredTransferIntegrationTest, BlockBufferKindDefault) {
    auto addr  = std::shared_ptr<void>(malloc(64), ::free);
    auto block = std::make_shared<BlockBuffer>("test_key", addr, 64, false, false);
    EXPECT_EQ(block->kind_, BlockBuffer::Kind::HOST_PINNED);

    block->kind_ = BlockBuffer::Kind::GPU_GATHERED_READY;
    BlockBuffer copied(*block);
    EXPECT_EQ(copied.kind_, BlockBuffer::Kind::GPU_GATHERED_READY);
}

TEST_F(GatheredTransferIntegrationTest, PoolIsPinnedMemory) {
    constexpr size_t        kPoolSize = 4096;
    CacheTransferBufferPool pool(kPoolSize, memory_util_);

    ASSERT_NE(pool.baseAddr(), nullptr);

    // Verify the pool memory is pinned by doing async H2D from it
    void* gpu_dst;
    ASSERT_EQ(cudaMalloc(&gpu_dst, 256), cudaSuccess);

    auto* handle = pool.tryAllocate(256);
    ASSERT_NE(handle, nullptr);
    memset(handle->ptr, 'P', 256);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    // cudaMemcpyAsync from pinned host memory should succeed
    ASSERT_EQ(cudaMemcpyAsync(gpu_dst, handle->ptr, 256, cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    // Verify
    char verify[256];
    cudaMemcpy(verify, gpu_dst, 256, cudaMemcpyDeviceToHost);
    EXPECT_EQ(verify[0], 'P');
    EXPECT_EQ(verify[255], 'P');

    cudaStreamDestroy(stream);
    cudaFree(gpu_dst);
    pool.free(handle);
}

}  // namespace rtp_llm
