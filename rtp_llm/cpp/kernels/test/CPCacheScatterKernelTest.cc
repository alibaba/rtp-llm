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

class CPCacheScatterKernelTest : public ::testing::Test {
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

        rtp_llm::DeviceFactory::initDevices(parallelism_config, model_config, eplb_config, fmha_config,
                                            device_resource_config, moe_config, sp_config, misc_config,
                                            profiling_debug_logging_config, hw_kernel_config, concurrency_config,
                                            ffn_disaggregate_config, runtime_config, model_specific_config,
                                            rtp_llm::NcclCommConfig{});
        device_ = rtp_llm::DeviceFactory::getDefaultDevice();
        ASSERT_NE(device_, nullptr);
    }

    /// Run the scatter kernel and verify correctness.
    ///
    /// @param virtual_block_count  Number of virtual blocks
    /// @param cp_size              Number of CP peers
    /// @param block_size           Tokens per physical block
    /// @param elem_stride_bytes    Bytes per token (must be multiple of 16)
    void runScatterTest(int virtual_block_count, int cp_size, int block_size, int elem_stride_bytes) {
        ASSERT_EQ(elem_stride_bytes % 16, 0);

        const int total_blocks   = virtual_block_count * cp_size;
        const int tokens_per_vb  = block_size * cp_size;

        // Allocate one contiguous GPU buffer for all blocks.
        // block_data[block_id] starts at offset block_id * block_size * elem_stride_bytes.
        const size_t block_bytes = static_cast<size_t>(block_size) * elem_stride_bytes;
        const size_t total_bytes = static_cast<size_t>(total_blocks) * block_bytes;

        // Host-side data: fill each physical block with a known pattern.
        // For peer p, virtual block v, physical slot s:
        //   The byte value at position (byte_offset) within that slot is:
        //     (v * cp_size + p) * block_size + s  (i.e., a unique token-level tag)
        // We fill all elem_stride_bytes of each token with the same tag byte for simplicity.
        std::vector<uint8_t> host_data(total_bytes, 0);

        for (int v = 0; v < virtual_block_count; ++v) {
            for (int p = 0; p < cp_size; ++p) {
                int block_id = v * cp_size + p;
                uint8_t* block_ptr = host_data.data() + block_id * block_bytes;
                for (int s = 0; s < block_size; ++s) {
                    // Tag = global token index within the virtual block
                    // global_token = s * cp_size + p (interleaved order from prefill)
                    uint8_t tag = static_cast<uint8_t>((v * tokens_per_vb + s * cp_size + p) & 0xFF);
                    std::memset(block_ptr + s * elem_stride_bytes, tag, elem_stride_bytes);
                }
            }
        }

        // Upload block data to GPU
        auto gpu_data = device_->allocateBuffer(
            {DataType::TYPE_BYTES, {total_bytes}, AllocationType::DEVICE}, {});
        device_->copy({*gpu_data, Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_BYTES, {total_bytes}, host_data.data())});

        // Build block_addrs array: block_addrs[i] = gpu_data + i * block_bytes
        std::vector<void*> block_addrs_host(total_blocks);
        for (int i = 0; i < total_blocks; ++i) {
            block_addrs_host[i] = static_cast<char*>(gpu_data->data()) + i * block_bytes;
        }

        // Block IDs: identity mapping [0, 1, 2, ..., total_blocks-1]
        std::vector<int> block_ids_host(total_blocks);
        std::iota(block_ids_host.begin(), block_ids_host.end(), 0);

        // Upload to GPU
        auto block_addrs_gpu = device_->allocateBuffer(
            {DataType::TYPE_UINT64, {(size_t)total_blocks}, AllocationType::DEVICE}, {});
        auto block_ids_gpu = device_->allocateBuffer(
            {DataType::TYPE_INT32, {(size_t)total_blocks}, AllocationType::DEVICE}, {});

        device_->copy({*block_addrs_gpu,
                       Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_UINT64, {(size_t)total_blocks}, block_addrs_host.data())});
        device_->copy({*block_ids_gpu,
                       Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_INT32, {(size_t)total_blocks}, block_ids_host.data())});

        // Allocate temp buffer
        size_t temp_size = (size_t)virtual_block_count * block_size * cp_size * elem_stride_bytes;
        auto temp_buffer = device_->allocateBuffer(
            {DataType::TYPE_BYTES, {temp_size}, AllocationType::DEVICE}, {});

        // Run scatter kernel
        invokeCPCacheScatter(
            reinterpret_cast<void**>(block_addrs_gpu->data()),
            block_ids_gpu->data<int>(),
            temp_buffer->data(),
            virtual_block_count,
            cp_size,
            block_size,
            elem_stride_bytes,
            nullptr);  // default stream

        device_->syncAndCheck();

        // Read back result
        std::vector<uint8_t> result(total_bytes);
        device_->copy({Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_BYTES, {total_bytes}, result.data()), *gpu_data});
        device_->syncAndCheck();

        // Verify: After scatter, decode block d (relative to VB v) should hold
        // tokens [d*block_size .. (d+1)*block_size - 1] in contiguous order.
        // Token t within VB v has tag = (v * tokens_per_vb + t) & 0xFF.
        for (int v = 0; v < virtual_block_count; ++v) {
            for (int d = 0; d < cp_size; ++d) {
                int block_id = v * cp_size + d;
                const uint8_t* block_ptr = result.data() + block_id * block_bytes;
                for (int slot = 0; slot < block_size; ++slot) {
                    int global_token = d * block_size + slot;
                    uint8_t expected_tag = static_cast<uint8_t>((v * tokens_per_vb + global_token) & 0xFF);
                    for (int b = 0; b < elem_stride_bytes; ++b) {
                        ASSERT_EQ(block_ptr[slot * elem_stride_bytes + b], expected_tag)
                            << "Mismatch at vb=" << v << " decode_block=" << d
                            << " slot=" << slot << " byte=" << b
                            << " expected=" << (int)expected_tag
                            << " got=" << (int)block_ptr[slot * elem_stride_bytes + b];
                    }
                }
            }
        }
    }

    DeviceBase* device_ = nullptr;
};

/// Test with non-contiguous (shuffled) block IDs to verify indirect addressing.
/// In real decode, block IDs come from KV cache allocator and are not sequential.
TEST_F(CPCacheScatterKernelTest, NonContiguousBlockIds) {
    const int virtual_block_count = 3;
    const int cp_size             = 2;
    const int block_size          = 4;
    const int elem_stride_bytes   = 32;
    const int total_blocks        = virtual_block_count * cp_size;  // 6
    const int tokens_per_vb       = block_size * cp_size;           // 8
    const size_t block_bytes      = static_cast<size_t>(block_size) * elem_stride_bytes;

    // Shuffled block IDs: not identity, with gaps
    // Logical blocks 0..5 map to physical IDs: [5, 2, 9, 0, 7, 3]
    std::vector<int> block_ids_host = {5, 2, 9, 0, 7, 3};
    int max_block_id = 10;  // need at least max+1 entries in block_addrs

    // Allocate GPU memory for max_block_id+1 blocks (some unused)
    const size_t total_bytes = static_cast<size_t>(max_block_id + 1) * block_bytes;
    auto gpu_data = device_->allocateBuffer(
        {DataType::TYPE_BYTES, {total_bytes}, AllocationType::DEVICE}, {});

    // Fill only the blocks referenced by block_ids_host with interleaved pattern
    std::vector<uint8_t> host_data(total_bytes, 0xDD);  // 0xDD = unused sentinel
    for (int v = 0; v < virtual_block_count; ++v) {
        for (int p = 0; p < cp_size; ++p) {
            int logical_idx = v * cp_size + p;
            int phys_id     = block_ids_host[logical_idx];
            uint8_t* block_ptr = host_data.data() + phys_id * block_bytes;
            for (int s = 0; s < block_size; ++s) {
                uint8_t tag = static_cast<uint8_t>((v * tokens_per_vb + s * cp_size + p) & 0xFF);
                std::memset(block_ptr + s * elem_stride_bytes, tag, elem_stride_bytes);
            }
        }
    }

    device_->copy({*gpu_data, Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_BYTES, {total_bytes}, host_data.data())});

    // Build block_addrs
    std::vector<void*> block_addrs_host(max_block_id + 1);
    for (int i = 0; i <= max_block_id; ++i) {
        block_addrs_host[i] = static_cast<char*>(gpu_data->data()) + i * block_bytes;
    }

    auto block_addrs_gpu = device_->allocateBuffer(
        {DataType::TYPE_UINT64, {(size_t)(max_block_id + 1)}, AllocationType::DEVICE}, {});
    auto block_ids_gpu = device_->allocateBuffer(
        {DataType::TYPE_INT32, {(size_t)total_blocks}, AllocationType::DEVICE}, {});

    device_->copy({*block_addrs_gpu,
                   Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_UINT64, {(size_t)(max_block_id + 1)}, block_addrs_host.data())});
    device_->copy({*block_ids_gpu,
                   Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_INT32, {(size_t)total_blocks}, block_ids_host.data())});

    size_t temp_size = (size_t)virtual_block_count * block_size * cp_size * elem_stride_bytes;
    auto temp_buffer = device_->allocateBuffer(
        {DataType::TYPE_BYTES, {temp_size}, AllocationType::DEVICE}, {});

    invokeCPCacheScatter(
        reinterpret_cast<void**>(block_addrs_gpu->data()),
        block_ids_gpu->data<int>(),
        temp_buffer->data(),
        virtual_block_count, cp_size, block_size, elem_stride_bytes, nullptr);

    device_->syncAndCheck();

    // Read back and verify
    std::vector<uint8_t> result(total_bytes);
    device_->copy({Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_BYTES, {total_bytes}, result.data()), *gpu_data});
    device_->syncAndCheck();

    for (int v = 0; v < virtual_block_count; ++v) {
        for (int d = 0; d < cp_size; ++d) {
            int logical_idx = v * cp_size + d;
            int phys_id     = block_ids_host[logical_idx];
            const uint8_t* block_ptr = result.data() + phys_id * block_bytes;
            for (int slot = 0; slot < block_size; ++slot) {
                int global_token = d * block_size + slot;
                uint8_t expected_tag = static_cast<uint8_t>((v * tokens_per_vb + global_token) & 0xFF);
                for (int b = 0; b < elem_stride_bytes; ++b) {
                    ASSERT_EQ(block_ptr[slot * elem_stride_bytes + b], expected_tag)
                        << "Mismatch at vb=" << v << " decode_block=" << d
                        << " phys_id=" << phys_id << " slot=" << slot << " byte=" << b;
                }
            }
        }
    }
}

// Basic test: 1 virtual block, cp_size=2, block_size=4, 32 bytes per token
TEST_F(CPCacheScatterKernelTest, Basic_1VB_CP2_BS4) {
    runScatterTest(/*virtual_block_count=*/1, /*cp_size=*/2, /*block_size=*/4, /*elem_stride_bytes=*/32);
}

// Multiple virtual blocks
TEST_F(CPCacheScatterKernelTest, MultiVB_CP2_BS4) {
    runScatterTest(/*virtual_block_count=*/3, /*cp_size=*/2, /*block_size=*/4, /*elem_stride_bytes=*/32);
}

// Larger cp_size
TEST_F(CPCacheScatterKernelTest, CP4_BS4) {
    runScatterTest(/*virtual_block_count=*/2, /*cp_size=*/4, /*block_size=*/4, /*elem_stride_bytes=*/64);
}

// Larger block_size
TEST_F(CPCacheScatterKernelTest, CP2_BS8) {
    runScatterTest(/*virtual_block_count=*/2, /*cp_size=*/2, /*block_size=*/8, /*elem_stride_bytes=*/32);
}

// Realistic MLA dimensions: block_size=1, elem_stride=512 bytes (e.g., compressed_kv_dim=256, fp16)
TEST_F(CPCacheScatterKernelTest, MLA_Realistic_BS1_CP2) {
    runScatterTest(/*virtual_block_count=*/4, /*cp_size=*/2, /*block_size=*/1, /*elem_stride_bytes=*/512);
}

// Larger realistic: block_size=16, cp_size=2, 1024 bytes per token
TEST_F(CPCacheScatterKernelTest, Large_BS16_CP2) {
    runScatterTest(/*virtual_block_count=*/2, /*cp_size=*/2, /*block_size=*/16, /*elem_stride_bytes=*/1024);
}

// Edge case: cp_size=1 should be a no-op (invokeCPCacheScatter returns early)
TEST_F(CPCacheScatterKernelTest, CP1_NoOp) {
    // cp_size=1 means no scatter needed; the function should return immediately.
    // Just verify it doesn't crash.
    invokeCPCacheScatter(nullptr, nullptr, nullptr, 5, 1, 4, 32, nullptr);
}

// Edge case: virtual_block_count=0 should be a no-op
TEST_F(CPCacheScatterKernelTest, ZeroVB_NoOp) {
    invokeCPCacheScatter(nullptr, nullptr, nullptr, 0, 2, 4, 32, nullptr);
}

// Realistic MLA: block_size=64, cp_size=2, elem_stride=576 (DeepSeek compressed_kv_dim=576, FP8)
TEST_F(CPCacheScatterKernelTest, RealisticMLA_BS64_CP2_Stride576) {
    runScatterTest(/*virtual_block_count=*/4, /*cp_size=*/2, /*block_size=*/64, /*elem_stride_bytes=*/576);
}

// Realistic MLA: block_size=64, cp_size=4, elem_stride=576
TEST_F(CPCacheScatterKernelTest, RealisticMLA_BS64_CP4_Stride576) {
    runScatterTest(/*virtual_block_count=*/8, /*cp_size=*/4, /*block_size=*/64, /*elem_stride_bytes=*/576);
}

// Minimum alignment: elem_stride_bytes=16 (single int4 per token)
TEST_F(CPCacheScatterKernelTest, MinAlignment_Stride16) {
    runScatterTest(/*virtual_block_count=*/2, /*cp_size=*/2, /*block_size=*/4, /*elem_stride_bytes=*/16);
}

// Larger scale: 32 virtual blocks (simulates 8K seq, block_size=64, cp_size=4)
TEST_F(CPCacheScatterKernelTest, LargeScale_32VB_CP4_BS64) {
    runScatterTest(/*virtual_block_count=*/32, /*cp_size=*/4, /*block_size=*/64, /*elem_stride_bytes=*/576);
}

}  // namespace test
}  // namespace rtp_llm
