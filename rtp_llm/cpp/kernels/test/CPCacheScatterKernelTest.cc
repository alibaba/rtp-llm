#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include "rtp_llm/cpp/kernels/cp_cache_scatter_kernel.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {
namespace test {

class CPCacheScatterKernelTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        rtp_llm::ParallelismConfig           parallelism_config;
        rtp_llm::ModelConfig                 model_config;
        rtp_llm::EPLBConfig                  eplb_config;
        rtp_llm::FMHAConfig                  fmha_config;
        rtp_llm::DeviceResourceConfig        device_resource_config;
        rtp_llm::MoeConfig                   moe_config;
        rtp_llm::SpeculativeExecutionConfig  sp_config;
        rtp_llm::MiscellaneousConfig         misc_config;
        rtp_llm::ProfilingDebugLoggingConfig profiling_debug_logging_config;
        rtp_llm::HWKernelConfig              hw_kernel_config;
        rtp_llm::ConcurrencyConfig           concurrency_config;
        rtp_llm::FfnDisAggregateConfig       ffn_disaggregate_config;
        rtp_llm::RuntimeConfig               runtime_config;
        rtp_llm::ModelSpecificConfig         model_specific_config;

        device_resource_config.device_reserve_memory_bytes = 2048000000;
        device_resource_config.host_reserve_memory_bytes   = 0;

        rtp_llm::DeviceFactory::initDevices(parallelism_config,
                                            model_config,
                                            eplb_config,
                                            fmha_config,
                                            device_resource_config,
                                            moe_config,
                                            sp_config,
                                            misc_config,
                                            profiling_debug_logging_config,
                                            hw_kernel_config,
                                            concurrency_config,
                                            ffn_disaggregate_config,
                                            runtime_config,
                                            model_specific_config,
                                            rtp_llm::NcclCommConfig{});
        device_ = rtp_llm::DeviceFactory::getDefaultDevice();
        ASSERT_NE(device_, nullptr);
    }

    /// Build a temp buffer simulating RDMA-received data from cp_size prefill peers,
    /// run the scatter kernel, and verify decode blocks contain contiguous tokens.
    ///
    /// @param total_tokens  Actual token count (may be < virtual_block_count * cp_size * block_size)
    void
    runScatterTest(int virtual_block_count, int cp_size, int block_size, int elem_stride_bytes, int total_tokens = -1) {
        ASSERT_EQ(elem_stride_bytes % 16, 0);

        const int tokens_per_vb = block_size * cp_size;
        if (total_tokens < 0) {
            total_tokens = virtual_block_count * tokens_per_vb;  // full
        }
        const int    decode_blocks = (total_tokens + block_size - 1) / block_size;
        const int    temp_slots    = virtual_block_count * cp_size;
        const size_t block_bytes   = static_cast<size_t>(block_size) * elem_stride_bytes;

        // --- Build temp buffer (simulates RDMA receive) ---
        // Layout: [virtual_block_count * cp_size] consecutive block-sized regions.
        // For vblock v, peer p: temp_slot = v * cp_size + p.
        // Peer p's slot s holds the token at global offset (s * cp_size + p) within the vblock.
        const size_t         temp_size = static_cast<size_t>(temp_slots) * block_bytes;
        std::vector<uint8_t> temp_host(temp_size, 0);

        for (int v = 0; v < virtual_block_count; ++v) {
            for (int p = 0; p < cp_size; ++p) {
                int      slot_idx = v * cp_size + p;
                uint8_t* slot_ptr = temp_host.data() + slot_idx * block_bytes;
                for (int s = 0; s < block_size; ++s) {
                    int global_token = v * tokens_per_vb + s * cp_size + p;
                    if (global_token >= total_tokens)
                        continue;
                    uint8_t tag = static_cast<uint8_t>(global_token & 0xFF);
                    std::memset(slot_ptr + s * elem_stride_bytes, tag, elem_stride_bytes);
                }
            }
        }

        auto temp_gpu = device_->allocateBuffer({DataType::TYPE_BYTES, {temp_size}, AllocationType::DEVICE}, {});
        device_->copy({*temp_gpu, Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_BYTES, {temp_size}, temp_host.data())});

        // --- Allocate decode blocks (identity block IDs for simplicity) ---
        const size_t dst_total = static_cast<size_t>(decode_blocks) * block_bytes;
        auto         dst_gpu = device_->allocateBuffer({DataType::TYPE_BYTES, {dst_total}, AllocationType::DEVICE}, {});

        std::vector<void*> dst_addrs(decode_blocks);
        std::vector<int>   dst_ids(decode_blocks);
        for (int i = 0; i < decode_blocks; ++i) {
            dst_addrs[i] = static_cast<char*>(dst_gpu->data()) + i * block_bytes;
            dst_ids[i]   = i;
        }

        auto dst_addrs_gpu =
            device_->allocateBuffer({DataType::TYPE_UINT64, {(size_t)decode_blocks}, AllocationType::DEVICE}, {});
        auto dst_ids_gpu =
            device_->allocateBuffer({DataType::TYPE_INT32, {(size_t)decode_blocks}, AllocationType::DEVICE}, {});
        device_->copy(
            {*dst_addrs_gpu,
             Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_UINT64, {(size_t)decode_blocks}, dst_addrs.data())});
        device_->copy({*dst_ids_gpu,
                       Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_INT32, {(size_t)decode_blocks}, dst_ids.data())});

        // --- Run kernel ---
        invokeCPCacheScatter(reinterpret_cast<void**>(dst_addrs_gpu->data()),
                             dst_ids_gpu->data<int>(),
                             temp_gpu->data(),
                             virtual_block_count,
                             cp_size,
                             block_size,
                             total_tokens,
                             elem_stride_bytes,
                             nullptr);
        device_->syncAndCheck();

        // --- Verify ---
        std::vector<uint8_t> result(dst_total);
        device_->copy({Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_BYTES, {dst_total}, result.data()), *dst_gpu});
        device_->syncAndCheck();

        for (int t = 0; t < total_tokens; ++t) {
            int            blk      = t / block_size;
            int            slot     = t % block_size;
            uint8_t        expected = static_cast<uint8_t>(t & 0xFF);
            const uint8_t* ptr      = result.data() + blk * block_bytes + slot * elem_stride_bytes;
            for (int b = 0; b < elem_stride_bytes; ++b) {
                ASSERT_EQ(ptr[b], expected) << "token=" << t << " blk=" << blk << " slot=" << slot << " byte=" << b;
            }
        }
    }

    DeviceBase* device_ = nullptr;
};

// Full virtual blocks
TEST_F(CPCacheScatterKernelTest, Basic_1VB_CP2_BS4) {
    runScatterTest(1, 2, 4, 32);
}

TEST_F(CPCacheScatterKernelTest, MultiVB_CP2_BS4) {
    runScatterTest(3, 2, 4, 32);
}

TEST_F(CPCacheScatterKernelTest, CP4_BS4) {
    runScatterTest(2, 4, 4, 64);
}

TEST_F(CPCacheScatterKernelTest, CP2_BS8) {
    runScatterTest(2, 2, 8, 32);
}

// Partial last virtual block (total_tokens < vblock_count * tokens_per_vb)
TEST_F(CPCacheScatterKernelTest, Partial_75tokens_CP4_BS64) {
    // cp_size=4, block_size=64, virtual_block_size=256, 1 vblock, only 75 tokens
    runScatterTest(1, 4, 64, 576, /*total_tokens=*/75);
}

TEST_F(CPCacheScatterKernelTest, Partial_130tokens_CP2_BS64) {
    // 2 vblocks (128 per vb), but only 130 tokens — second vblock has 2 tokens
    runScatterTest(2, 2, 64, 32, /*total_tokens=*/130);
}

TEST_F(CPCacheScatterKernelTest, Partial_1token_CP4_BS4) {
    runScatterTest(1, 4, 4, 32, /*total_tokens=*/1);
}

// Realistic MLA dimensions
TEST_F(CPCacheScatterKernelTest, RealisticMLA_BS64_CP2_Stride576) {
    runScatterTest(4, 2, 64, 576);
}

TEST_F(CPCacheScatterKernelTest, RealisticMLA_BS64_CP4_Stride576) {
    runScatterTest(8, 4, 64, 576);
}

// Edge cases
TEST_F(CPCacheScatterKernelTest, CP1_NoOp) {
    invokeCPCacheScatter(nullptr, nullptr, nullptr, 5, 1, 4, 20, 32, nullptr);
}

TEST_F(CPCacheScatterKernelTest, ZeroVB_NoOp) {
    invokeCPCacheScatter(nullptr, nullptr, nullptr, 0, 2, 4, 0, 32, nullptr);
}

TEST_F(CPCacheScatterKernelTest, ZeroTokens_NoOp) {
    invokeCPCacheScatter(nullptr, nullptr, nullptr, 1, 2, 4, 0, 32, nullptr);
}

TEST_F(CPCacheScatterKernelTest, MinAlignment_Stride16) {
    runScatterTest(2, 2, 4, 16);
}

TEST_F(CPCacheScatterKernelTest, LargeScale_32VB_CP4_BS64) {
    runScatterTest(32, 4, 64, 576);
}

// Non-contiguous decode block IDs
TEST_F(CPCacheScatterKernelTest, NonContiguousBlockIds) {
    const int    vblock_count      = 1;
    const int    cp_size           = 4;
    const int    block_size        = 4;
    const int    elem_stride_bytes = 32;
    const int    total_tokens      = 13;                                            // partial
    const int    tokens_per_vb     = block_size * cp_size;                          // 16
    const int    decode_blocks     = (total_tokens + block_size - 1) / block_size;  // 4
    const size_t block_bytes       = static_cast<size_t>(block_size) * elem_stride_bytes;

    // Build temp buffer
    const int            temp_slots = vblock_count * cp_size;
    const size_t         temp_size  = static_cast<size_t>(temp_slots) * block_bytes;
    std::vector<uint8_t> temp_host(temp_size, 0);
    for (int p = 0; p < cp_size; ++p) {
        uint8_t* slot_ptr = temp_host.data() + p * block_bytes;
        for (int s = 0; s < block_size; ++s) {
            int global_token = s * cp_size + p;
            if (global_token >= total_tokens)
                continue;
            uint8_t tag = static_cast<uint8_t>(global_token & 0xFF);
            std::memset(slot_ptr + s * elem_stride_bytes, tag, elem_stride_bytes);
        }
    }
    auto temp_gpu = device_->allocateBuffer({DataType::TYPE_BYTES, {temp_size}, AllocationType::DEVICE}, {});
    device_->copy({*temp_gpu, Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_BYTES, {temp_size}, temp_host.data())});

    // Shuffled decode block IDs: [7, 3, 11, 1]
    std::vector<int> dst_ids   = {7, 3, 11, 1};
    int              max_bid   = 12;
    const size_t     dst_total = static_cast<size_t>(max_bid) * block_bytes;
    auto             dst_gpu = device_->allocateBuffer({DataType::TYPE_BYTES, {dst_total}, AllocationType::DEVICE}, {});

    std::vector<void*> dst_addrs(max_bid);
    for (int i = 0; i < max_bid; ++i) {
        dst_addrs[i] = static_cast<char*>(dst_gpu->data()) + i * block_bytes;
    }

    auto dst_addrs_gpu =
        device_->allocateBuffer({DataType::TYPE_UINT64, {(size_t)max_bid}, AllocationType::DEVICE}, {});
    auto dst_ids_gpu =
        device_->allocateBuffer({DataType::TYPE_INT32, {(size_t)decode_blocks}, AllocationType::DEVICE}, {});
    device_->copy(
        {*dst_addrs_gpu, Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_UINT64, {(size_t)max_bid}, dst_addrs.data())});
    device_->copy(
        {*dst_ids_gpu, Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_INT32, {(size_t)decode_blocks}, dst_ids.data())});

    invokeCPCacheScatter(reinterpret_cast<void**>(dst_addrs_gpu->data()),
                         dst_ids_gpu->data<int>(),
                         temp_gpu->data(),
                         vblock_count,
                         cp_size,
                         block_size,
                         total_tokens,
                         elem_stride_bytes,
                         nullptr);
    device_->syncAndCheck();

    std::vector<uint8_t> result(dst_total);
    device_->copy({Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_BYTES, {dst_total}, result.data()), *dst_gpu});
    device_->syncAndCheck();

    for (int t = 0; t < total_tokens; ++t) {
        int            blk_idx  = t / block_size;
        int            slot     = t % block_size;
        int            phys_id  = dst_ids[blk_idx];
        uint8_t        expected = static_cast<uint8_t>(t & 0xFF);
        const uint8_t* ptr      = result.data() + phys_id * block_bytes + slot * elem_stride_bytes;
        for (int b = 0; b < elem_stride_bytes; ++b) {
            ASSERT_EQ(ptr[b], expected) << "token=" << t << " phys_id=" << phys_id << " slot=" << slot << " byte=" << b;
        }
    }
}

}  // namespace test
}  // namespace rtp_llm
