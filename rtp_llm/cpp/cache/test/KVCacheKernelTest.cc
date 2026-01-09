#include "rtp_llm/cpp/kernels/kv_cache_kernels.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include <c10/cuda/CUDAStream.h>

#include <memory>
#include <vector>
#include <cstring>
#include <random>

using namespace std;

namespace rtp_llm {

class KVCacheKernelTest: public DeviceTestBase {
protected:
    CacheConfig initConfig(rtp_llm::DataType dtype = rtp_llm::TYPE_FP16) {
        // layer_num, block_nums, local_head_num_kv, size_per_head, seq_size_per_block, dtype
        CacheConfig config(KVCacheParam({4, 10, 4, 64, 128, dtype}));
        return config;
    }

    CacheConfig initSmallConfig() {
        // 较小的配置用于测试边界情况
        CacheConfig config(KVCacheParam({2, 5, 2, 32, 64, rtp_llm::TYPE_FP16}));
        return config;
    }

    template<typename T>
    void fillRandomData(T* data, size_t size) {
        std::random_device                    rd;
        std::mt19937                          gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        for (size_t i = 0; i < size; i++) {
            data[i] = static_cast<T>(dis(gen));
        }
    }

    template<typename T>
    bool verifyAllZero(T* data, size_t size) {
        for (size_t i = 0; i < size; i++) {
            if (data[i] != T(0)) {
                return false;
            }
        }
        return true;
    }
};

#if (defined(USING_CUDA) && USING_CUDA) || (defined(USING_ROCM) && USING_ROCM)

// 测试 ClearIncompleteBlock kernel - FP16
TEST_F(KVCacheKernelTest, testClearIncompleteBlockFP16) {
    auto config = initConfig(rtp_llm::TYPE_FP16);

    // 分配 k 和 v blocks 的内存
    size_t k_total_size_bytes = config.getKBlockSize();
    size_t v_total_size_bytes = config.getVBlockSize();
    size_t k_total_elements   = k_total_size_bytes / rtp_llm::getTypeSize(rtp_llm::TYPE_FP16);
    size_t v_total_elements   = v_total_size_bytes / rtp_llm::getTypeSize(rtp_llm::TYPE_FP16);

    auto k_blocks_buffer =
        device_->allocateBuffer({rtp_llm::TYPE_FP16, {k_total_elements}, rtp_llm::AllocationType::DEVICE});
    auto v_blocks_buffer =
        device_->allocateBuffer({rtp_llm::TYPE_FP16, {v_total_elements}, rtp_llm::AllocationType::DEVICE});

    // 在 host 上创建随机数据
    std::vector<__half> k_host_data(k_total_elements);
    std::vector<__half> v_host_data(v_total_elements);
    fillRandomData(k_host_data.data(), k_host_data.size());
    fillRandomData(v_host_data.data(), v_host_data.size());

    // 复制到 device
    device_->copy(
        {*k_blocks_buffer, Buffer(MemoryType::MEMORY_CPU, TYPE_FP16, {k_host_data.size()}, k_host_data.data())});
    device_->copy(
        {*v_blocks_buffer, Buffer(MemoryType::MEMORY_CPU, TYPE_FP16, {v_host_data.size()}, v_host_data.data())});

    // 选择要清理的 block (最后一个 block)
    int  block_index = config.block_nums - 1;
    auto stream      = c10::cuda::getCurrentCUDAStream().stream();

    // 调用 kernel 清理最后一个 block
    invokeclearIncompleteBlocks<__half>(reinterpret_cast<__half*>(k_blocks_buffer->data()),
                                        reinterpret_cast<__half*>(v_blocks_buffer->data()),
                                        block_index,
                                        config.layer_num,
                                        config.getKeyBlockStride(),
                                        config.getValueBlockStride(),
                                        config.getKeyLayerStride(),
                                        config.getValueLayerStride(),
                                        config.getKeyShape(),
                                        config.getValueShape(),
                                        stream);

    // 同步等待 kernel 完成
    cudaStreamSynchronize(stream);

    // 复制回 host 验证
    std::vector<__half> k_result(k_host_data.size());
    std::vector<__half> v_result(v_host_data.size());
    device_->copy({Buffer(MemoryType::MEMORY_CPU, TYPE_FP16, {k_result.size()}, k_result.data()), *k_blocks_buffer});
    device_->copy({Buffer(MemoryType::MEMORY_CPU, TYPE_FP16, {v_result.size()}, v_result.data()), *v_blocks_buffer});

    // 验证：除了最后一个 block，其他 block 的数据应该保持不变
    // 验证最后一个 block 的所有 layer 都被清零
    size_t element_size = rtp_llm::getTypeSize(config.dtype);
    for (uint32_t layer_id = 0; layer_id < config.layer_num; layer_id++) {
        size_t k_offset_bytes    = config.getKeyOffset(block_index, layer_id);
        size_t v_offset_bytes    = config.getValueOffset(block_index, layer_id);
        size_t k_offset_elements = k_offset_bytes / element_size;
        size_t v_offset_elements = v_offset_bytes / element_size;
        size_t k_size            = config.getKeyShape();
        size_t v_size            = config.getValueShape();

        // 检查 k block 是否被清零
        bool k_cleared = verifyAllZero(&k_result[k_offset_elements], k_size);
        EXPECT_TRUE(k_cleared) << "K block at layer " << layer_id << " should be cleared";

        // 检查 v block 是否被清零
        bool v_cleared = verifyAllZero(&v_result[v_offset_elements], v_size);
        EXPECT_TRUE(v_cleared) << "V block at layer " << layer_id << " should be cleared";
    }

    // 验证其他 block 的数据没有被改变（除了最后一个 block）
    for (int other_block = 1; other_block < block_index; other_block++) {
        for (uint32_t layer_id = 0; layer_id < config.layer_num; layer_id++) {
            size_t k_offset_bytes    = config.getKeyOffset(other_block, layer_id);
            size_t v_offset_bytes    = config.getValueOffset(other_block, layer_id);
            size_t k_offset_elements = k_offset_bytes / element_size;
            size_t v_offset_elements = v_offset_bytes / element_size;
            size_t k_size            = config.getKeyShape();
            size_t v_size            = config.getValueShape();

            // 检查数据是否保持不变
            bool k_unchanged = std::equal(&k_host_data[k_offset_elements],
                                          &k_host_data[k_offset_elements + k_size],
                                          &k_result[k_offset_elements]);
            bool v_unchanged = std::equal(&v_host_data[v_offset_elements],
                                          &v_host_data[v_offset_elements + v_size],
                                          &v_result[v_offset_elements]);

            EXPECT_TRUE(k_unchanged) << "K block at block " << other_block << " layer " << layer_id
                                     << " should be unchanged";
            EXPECT_TRUE(v_unchanged) << "V block at block " << other_block << " layer " << layer_id
                                     << " should be unchanged";
        }
    }
}

// 测试 ClearIncompleteBlock kernel - BF16
TEST_F(KVCacheKernelTest, testClearIncompleteBlockBF16) {
    auto config = initConfig(rtp_llm::TYPE_BF16);

    size_t k_total_size_bytes = config.getKBlockSize();
    size_t v_total_size_bytes = config.getVBlockSize();
    size_t k_total_elements   = k_total_size_bytes / rtp_llm::getTypeSize(rtp_llm::TYPE_BF16);
    size_t v_total_elements   = v_total_size_bytes / rtp_llm::getTypeSize(rtp_llm::TYPE_BF16);

    auto k_blocks_buffer =
        device_->allocateBuffer({rtp_llm::TYPE_BF16, {k_total_elements}, rtp_llm::AllocationType::DEVICE});
    auto v_blocks_buffer =
        device_->allocateBuffer({rtp_llm::TYPE_BF16, {v_total_elements}, rtp_llm::AllocationType::DEVICE});

    std::vector<__nv_bfloat16> k_host_data(k_total_elements);
    std::vector<__nv_bfloat16> v_host_data(v_total_elements);
    fillRandomData(k_host_data.data(), k_host_data.size());
    fillRandomData(v_host_data.data(), v_host_data.size());

    device_->copy(
        {*k_blocks_buffer, Buffer(MemoryType::MEMORY_CPU, TYPE_BF16, {k_host_data.size()}, k_host_data.data())});
    device_->copy(
        {*v_blocks_buffer, Buffer(MemoryType::MEMORY_CPU, TYPE_BF16, {v_host_data.size()}, v_host_data.data())});

    int  block_index = config.block_nums - 1;
    auto stream      = c10::cuda::getCurrentCUDAStream().stream();

    invokeclearIncompleteBlocks<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16*>(k_blocks_buffer->data()),
                                               reinterpret_cast<__nv_bfloat16*>(v_blocks_buffer->data()),
                                               block_index,
                                               config.layer_num,
                                               config.getKeyBlockStride(),
                                               config.getValueBlockStride(),
                                               config.getKeyLayerStride(),
                                               config.getValueLayerStride(),
                                               config.getKeyShape(),
                                               config.getValueShape(),
                                               stream);

    cudaStreamSynchronize(stream);

    std::vector<__nv_bfloat16> k_result(k_host_data.size());
    std::vector<__nv_bfloat16> v_result(v_host_data.size());
    device_->copy({Buffer(MemoryType::MEMORY_CPU, TYPE_BF16, {k_result.size()}, k_result.data()), *k_blocks_buffer});
    device_->copy({Buffer(MemoryType::MEMORY_CPU, TYPE_BF16, {v_result.size()}, v_result.data()), *v_blocks_buffer});

    // 验证最后一个 block 的所有 layer 都被清零
    size_t element_size = rtp_llm::getTypeSize(config.dtype);
    for (uint32_t layer_id = 0; layer_id < config.layer_num; layer_id++) {
        size_t k_offset_bytes    = config.getKeyOffset(block_index, layer_id);
        size_t v_offset_bytes    = config.getValueOffset(block_index, layer_id);
        size_t k_offset_elements = k_offset_bytes / element_size;
        size_t v_offset_elements = v_offset_bytes / element_size;
        size_t k_size            = config.getKeyShape();
        size_t v_size            = config.getValueShape();

        EXPECT_TRUE(verifyAllZero(&k_result[k_offset_elements], k_size))
            << "K block at layer " << layer_id << " should be cleared";
        EXPECT_TRUE(verifyAllZero(&v_result[v_offset_elements], v_size))
            << "V block at layer " << layer_id << " should be cleared";
    }
}

// 测试 ClearIncompleteBlock kernel - 小配置
TEST_F(KVCacheKernelTest, testClearIncompleteBlockSmallConfig) {
    auto config = initSmallConfig();

    size_t k_total_size_bytes = config.getKBlockSize();
    size_t v_total_size_bytes = config.getVBlockSize();
    size_t k_total_elements   = k_total_size_bytes / rtp_llm::getTypeSize(rtp_llm::TYPE_FP16);
    size_t v_total_elements   = v_total_size_bytes / rtp_llm::getTypeSize(rtp_llm::TYPE_FP16);

    auto k_blocks_buffer =
        device_->allocateBuffer({rtp_llm::TYPE_FP16, {k_total_elements}, rtp_llm::AllocationType::DEVICE});
    auto v_blocks_buffer =
        device_->allocateBuffer({rtp_llm::TYPE_FP16, {v_total_elements}, rtp_llm::AllocationType::DEVICE});

    std::vector<__half> k_host_data(k_total_elements);
    std::vector<__half> v_host_data(v_total_elements);
    fillRandomData(k_host_data.data(), k_host_data.size());
    fillRandomData(v_host_data.data(), v_host_data.size());

    device_->copy(
        {*k_blocks_buffer, Buffer(MemoryType::MEMORY_CPU, TYPE_FP16, {k_host_data.size()}, k_host_data.data())});
    device_->copy(
        {*v_blocks_buffer, Buffer(MemoryType::MEMORY_CPU, TYPE_FP16, {v_host_data.size()}, v_host_data.data())});

    int  block_index = config.block_nums - 1;
    auto stream      = c10::cuda::getCurrentCUDAStream().stream();

    invokeclearIncompleteBlocks<__half>(reinterpret_cast<__half*>(k_blocks_buffer->data()),
                                        reinterpret_cast<__half*>(v_blocks_buffer->data()),
                                        block_index,
                                        config.layer_num,
                                        config.getKeyBlockStride(),
                                        config.getValueBlockStride(),
                                        config.getKeyLayerStride(),
                                        config.getValueLayerStride(),
                                        config.getKeyShape(),
                                        config.getValueShape(),
                                        stream);

    cudaStreamSynchronize(stream);

    std::vector<__half> k_result(k_host_data.size());
    std::vector<__half> v_result(v_host_data.size());
    device_->copy({Buffer(MemoryType::MEMORY_CPU, TYPE_FP16, {k_result.size()}, k_result.data()), *k_blocks_buffer});
    device_->copy({Buffer(MemoryType::MEMORY_CPU, TYPE_FP16, {v_result.size()}, v_result.data()), *v_blocks_buffer});

    // 验证最后一个 block 的所有 layer 都被清零
    size_t element_size = rtp_llm::getTypeSize(config.dtype);
    for (uint32_t layer_id = 0; layer_id < config.layer_num; layer_id++) {
        size_t k_offset_bytes    = config.getKeyOffset(block_index, layer_id);
        size_t v_offset_bytes    = config.getValueOffset(block_index, layer_id);
        size_t k_offset_elements = k_offset_bytes / element_size;
        size_t v_offset_elements = v_offset_bytes / element_size;
        size_t k_size            = config.getKeyShape();
        size_t v_size            = config.getValueShape();

        EXPECT_TRUE(verifyAllZero(&k_result[k_offset_elements], k_size))
            << "K block at layer " << layer_id << " should be cleared";
        EXPECT_TRUE(verifyAllZero(&v_result[v_offset_elements], v_size))
            << "V block at layer " << layer_id << " should be cleared";
    }
}

// 测试 ClearIncompleteBlock kernel - 边界情况：第一个 block
TEST_F(KVCacheKernelTest, testClearIncompleteBlockFirstBlock) {
    auto config = initConfig();

    size_t k_total_size_bytes = config.getKBlockSize();
    size_t v_total_size_bytes = config.getVBlockSize();
    size_t k_total_elements   = k_total_size_bytes / rtp_llm::getTypeSize(rtp_llm::TYPE_FP16);
    size_t v_total_elements   = v_total_size_bytes / rtp_llm::getTypeSize(rtp_llm::TYPE_FP16);

    auto k_blocks_buffer =
        device_->allocateBuffer({rtp_llm::TYPE_FP16, {k_total_elements}, rtp_llm::AllocationType::DEVICE});
    auto v_blocks_buffer =
        device_->allocateBuffer({rtp_llm::TYPE_FP16, {v_total_elements}, rtp_llm::AllocationType::DEVICE});

    std::vector<__half> k_host_data(k_total_elements);
    std::vector<__half> v_host_data(v_total_elements);
    fillRandomData(k_host_data.data(), k_host_data.size());
    fillRandomData(v_host_data.data(), v_host_data.size());

    device_->copy(
        {*k_blocks_buffer, Buffer(MemoryType::MEMORY_CPU, TYPE_FP16, {k_host_data.size()}, k_host_data.data())});
    device_->copy(
        {*v_blocks_buffer, Buffer(MemoryType::MEMORY_CPU, TYPE_FP16, {v_host_data.size()}, v_host_data.data())});

    // 测试清理第一个 block (block 1, 因为 block 0 是保留的)
    int  block_index = 1;
    auto stream      = c10::cuda::getCurrentCUDAStream().stream();

    invokeclearIncompleteBlocks<__half>(reinterpret_cast<__half*>(k_blocks_buffer->data()),
                                        reinterpret_cast<__half*>(v_blocks_buffer->data()),
                                        block_index,
                                        config.layer_num,
                                        config.getKeyBlockStride(),
                                        config.getValueBlockStride(),
                                        config.getKeyLayerStride(),
                                        config.getValueLayerStride(),
                                        config.getKeyShape(),
                                        config.getValueShape(),
                                        stream);

    cudaStreamSynchronize(stream);

    std::vector<__half> k_result(k_host_data.size());
    std::vector<__half> v_result(v_host_data.size());
    device_->copy({Buffer(MemoryType::MEMORY_CPU, TYPE_FP16, {k_result.size()}, k_result.data()), *k_blocks_buffer});
    device_->copy({Buffer(MemoryType::MEMORY_CPU, TYPE_FP16, {v_result.size()}, v_result.data()), *v_blocks_buffer});

    // 验证第一个 block 的所有 layer 都被清零
    for (uint32_t layer_id = 0; layer_id < config.layer_num; layer_id++) {
        size_t k_offset = config.getKeyOffset(block_index, layer_id);
        size_t v_offset = config.getValueOffset(block_index, layer_id);
        size_t k_size   = config.getKeyShape();
        size_t v_size   = config.getValueShape();

        EXPECT_TRUE(verifyAllZero(&k_result[k_offset], k_size))
            << "K block at layer " << layer_id << " should be cleared";
        EXPECT_TRUE(verifyAllZero(&v_result[v_offset], v_size))
            << "V block at layer " << layer_id << " should be cleared";
    }
}

#endif  // USING_CUDA || USING_ROCM

}  // namespace rtp_llm
