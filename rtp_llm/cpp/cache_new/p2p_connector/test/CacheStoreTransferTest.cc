#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>
#include <atomic>
#include "rtp_llm/cpp/cache_new/p2p_connector/CacheStoreClient.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/CacheStoreServer.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/TcpClient.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/TcpServer.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/CommonDefs.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/LayerCacheBuffer.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/test/MockKVCacheAllocator.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/test/DeviceUtil.h"
#include "rtp_llm/cpp/core/Event.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
namespace cache_store {

class CacheStoreTransferTest: public ::testing::Test {
public:
    CacheStoreTransferTest() = default;
    ~CacheStoreTransferTest();

protected:
    void SetUp() override;
    void TearDown() override {};

protected:
    std::shared_ptr<MockKVCacheAllocator> kv_cache_allocator_;
    std::shared_ptr<TcpClient>            server_tcp_client_;
    std::shared_ptr<TcpServer>            server_tcp_server_;
    std::shared_ptr<TcpClient>            client_tcp_client_;
    std::shared_ptr<TcpServer>            client_tcp_server_;
    std::shared_ptr<CacheStoreClient>     cache_store_client_;
    std::shared_ptr<CacheStoreServer>     cache_store_server_;
    std::shared_ptr<DeviceUtil>           device_util_;
    std::string                           server_ip_   = "127.0.0.1";
    uint32_t                              server_port_ = 12345;
    uint32_t                              client_port_ = 12346;
};

void CacheStoreTransferTest::SetUp() {
    server_port_ = autil::NetUtil::randomPort();
    client_port_ = autil::NetUtil::randomPort();

    device_util_        = std::make_shared<DeviceUtil>();
    kv_cache_allocator_ = std::make_shared<MockKVCacheAllocator>();

    // Server 端的 TCP client 和 server
    server_tcp_client_ = std::make_shared<TcpClient>();
    server_tcp_server_ = std::make_shared<TcpServer>();
    ASSERT_TRUE(server_tcp_client_->init(1));
    ASSERT_TRUE(server_tcp_server_->init(1, 1, server_port_, false));
    ASSERT_TRUE(server_tcp_server_->start());

    // Client 端的 TCP client 和 server
    client_tcp_client_ = std::make_shared<TcpClient>();
    client_tcp_server_ = std::make_shared<TcpServer>();
    ASSERT_TRUE(client_tcp_client_->init(1));
    ASSERT_TRUE(client_tcp_server_->init(1, 1, client_port_, false));
    ASSERT_TRUE(client_tcp_server_->start());

    cache_store_client_ =
        std::make_shared<CacheStoreClient>(client_tcp_client_, client_tcp_server_, device_util_->device_);
    ASSERT_TRUE(cache_store_client_->init());

    std::vector<CacheStoreServerWorker> worker_addrs = {CacheStoreServerWorker(server_ip_, server_port_, 0)};
    cache_store_server_                              = std::make_shared<CacheStoreServer>(
        server_tcp_client_, server_tcp_server_, 1, kv_cache_allocator_, worker_addrs, device_util_->device_);
    ASSERT_TRUE(cache_store_server_->init());
}

CacheStoreTransferTest::~CacheStoreTransferTest() {
    cache_store_server_.reset();
    cache_store_client_.reset();
    server_tcp_server_.reset();
    server_tcp_client_.reset();
    client_tcp_server_.reset();
    client_tcp_client_.reset();
    device_util_.reset();
}

// 创建测试用的 LayerCacheBuffer，包含指定数量的 block
std::shared_ptr<LayerCacheBuffer> createTestLayerCacheBuffer(int layer_id, int num_blocks, int block_size) {
    auto layer_cache_buffer = std::make_shared<LayerCacheBuffer>(layer_id);
    for (int i = 0; i < num_blocks; i++) {
        int64_t key      = 1000 + i;
        int     block_id = i;
        layer_cache_buffer->addBlockCacheBuffer(key, block_id);
    }
    return layer_cache_buffer;
}

// 为 LayerCacheBuffer 的 block 分配 buffer
void allocateBlockBuffers(std::shared_ptr<LayerCacheBuffer> layer_cache_buffer,
                          DeviceUtil*                       device_util,
                          int                               block_size) {
    for (auto& [key, block] : layer_cache_buffer->blockCacheBuffers()) {
        auto buffer1 =
            device_util->device_->allocateBuffer({DataType::TYPE_UINT8, {block_size}, AllocationType::DEVICE});
        auto buffer2 =
            device_util->device_->allocateBuffer({DataType::TYPE_UINT8, {block_size}, AllocationType::DEVICE});

        // 初始化 buffer 数据用于测试
        std::vector<uint8_t> test_data1(block_size, 0);
        std::vector<uint8_t> test_data2(block_size, 0);
        for (size_t i = 0; i < block_size; i++) {
            test_data1[i] = static_cast<uint8_t>((key * 100 + i) % 256);
            test_data2[i] = static_cast<uint8_t>((key * 200 + i) % 256);
        }

        auto cpu_buffer1 =
            device_util->device_->allocateBuffer({DataType::TYPE_UINT8, {block_size}, AllocationType::HOST});
        auto cpu_buffer2 =
            device_util->device_->allocateBuffer({DataType::TYPE_UINT8, {block_size}, AllocationType::HOST});
        memcpy(cpu_buffer1->data(), test_data1.data(), block_size);
        memcpy(cpu_buffer2->data(), test_data2.data(), block_size);

        device_util->device_->copy({*buffer1, *cpu_buffer1});
        device_util->device_->copy({*buffer2, *cpu_buffer2});
        device_util->device_->syncAndCheck();

        block->buffer1 = buffer1;
        block->buffer2 = buffer2;
    }
}

// 验证 client 端 buffer 的数据是否正确
bool verifyBlockBuffer(BufferPtr buffer, int64_t key, int block_size, int multiplier, DeviceUtil* device_util) {
    auto cpu_buffer = device_util->device_->allocateBuffer({DataType::TYPE_UINT8, {block_size}, AllocationType::HOST});
    device_util->device_->copy({*cpu_buffer, *buffer});
    device_util->device_->syncAndCheck();

    for (size_t i = 0; i < block_size; i++) {
        uint8_t expected = static_cast<uint8_t>((key * multiplier + i) % 256);
        uint8_t actual   = static_cast<uint8_t*>(cpu_buffer->data())[i];
        if (actual != expected) {
            return false;
        }
    }
    return true;
}

TEST_F(CacheStoreTransferTest, TransferMultipleBlocks) {
    const int layer_id        = 0;
    const int num_blocks      = 3;
    const int block_size      = 2048;
    const int partition_count = 1;
    const int partition_id    = 0;

    // 1. 在 server 端创建 LayerCacheBuffer 并分配 buffer
    auto server_layer_buffer = createTestLayerCacheBuffer(layer_id, num_blocks, block_size);
    allocateBlockBuffers(server_layer_buffer, device_util_.get(), block_size);

    // 2. 设置 MockKVCacheAllocator 的 convertIndexToBuffer 返回值
    for (auto& [key, block] : server_layer_buffer->blockCacheBuffers()) {
        BlockBufferInfo buffer_info;
        buffer_info.k_addr = block->buffer1;
        buffer_info.v_addr = block->buffer2;
        EXPECT_CALL(*kv_cache_allocator_,
                    convertIndexToBuffer(layer_id, block->block_id, partition_count, partition_id))
            .WillRepeatedly(::testing::Return(buffer_info));
    }

    // 3. 在 client 端创建 LayerCacheBuffer 并分配空的 buffer
    auto client_layer_buffer = createTestLayerCacheBuffer(layer_id, num_blocks, block_size);
    for (auto& [key, block] : client_layer_buffer->blockCacheBuffers()) {
        auto buffer1 =
            device_util_->device_->allocateBuffer({DataType::TYPE_UINT8, {block_size}, AllocationType::DEVICE});
        auto buffer2 =
            device_util_->device_->allocateBuffer({DataType::TYPE_UINT8, {block_size}, AllocationType::DEVICE});
        block->buffer1 = buffer1;
        block->buffer2 = buffer2;
    }

    // 4. 在 client 端调用 asyncLoad 创建 LoadContext
    std::vector<std::shared_ptr<LayerCacheBuffer>> client_buffers = {client_layer_buffer};
    auto                                           load_context =
        cache_store_client_->asyncLoad(client_buffers, 5000, server_ip_, server_port_, partition_count, partition_id);
    ASSERT_NE(load_context, nullptr);

    // 5. 在 server 端触发 notify（通过 asyncStore）
    auto null_event = DeviceEventPtr(nullptr);
    cache_store_server_->asyncStore(server_layer_buffer, null_event, 1000);

    // 6. 等待传输完成
    // std::this_thread::sleep_for(std::chrono::milliseconds(500));
    load_context->waitDone();

    // 7. 验证 client 端所有 block 的数据是否正确
    for (auto& [key, block] : client_layer_buffer->blockCacheBuffers()) {
        EXPECT_TRUE(verifyBlockBuffer(block->buffer1, key, block_size, 100, device_util_.get()));
        EXPECT_TRUE(verifyBlockBuffer(block->buffer2, key, block_size, 200, device_util_.get()));
    }
}

TEST_F(CacheStoreTransferTest, TransferSingleBufferOnly) {
    const int layer_id        = 0;
    const int num_blocks      = 1;
    const int block_size      = 512;
    const int partition_count = 1;
    const int partition_id    = 0;

    // 1. 在 server 端创建 LayerCacheBuffer，只分配 buffer1
    auto server_layer_buffer = createTestLayerCacheBuffer(layer_id, num_blocks, block_size);
    for (auto& [key, block] : server_layer_buffer->blockCacheBuffers()) {
        auto buffer1 =
            device_util_->device_->allocateBuffer({DataType::TYPE_UINT8, {block_size}, AllocationType::DEVICE});
        std::vector<uint8_t> test_data(block_size, 0);
        for (size_t i = 0; i < block_size; i++) {
            test_data[i] = static_cast<uint8_t>((key * 100 + i) % 256);
        }
        auto cpu_buffer =
            device_util_->device_->allocateBuffer({DataType::TYPE_UINT8, {block_size}, AllocationType::HOST});
        memcpy(cpu_buffer->data(), test_data.data(), block_size);
        device_util_->device_->copy({*buffer1, *cpu_buffer});
        device_util_->device_->syncAndCheck();
        block->buffer1 = buffer1;
        block->buffer2 = nullptr;  // 只有 buffer1
    }

    // 2. 设置 MockKVCacheAllocator 的 convertIndexToBuffer 返回值（只有 k_addr）
    for (auto& [key, block] : server_layer_buffer->blockCacheBuffers()) {
        BlockBufferInfo buffer_info;
        buffer_info.k_addr = block->buffer1;
        buffer_info.v_addr = nullptr;  // 没有 v_addr
        EXPECT_CALL(*kv_cache_allocator_,
                    convertIndexToBuffer(layer_id, block->block_id, partition_count, partition_id))
            .WillRepeatedly(::testing::Return(buffer_info));
    }

    // 3. 在 client 端创建 LayerCacheBuffer，只分配 buffer1
    auto client_layer_buffer = createTestLayerCacheBuffer(layer_id, num_blocks, block_size);
    for (auto& [key, block] : client_layer_buffer->blockCacheBuffers()) {
        auto buffer1 =
            device_util_->device_->allocateBuffer({DataType::TYPE_UINT8, {block_size}, AllocationType::DEVICE});
        block->buffer1 = buffer1;
        block->buffer2 = nullptr;  // 只有 buffer1
    }

    // 4. 在 client 端调用 asyncLoad 创建 LoadContext
    std::vector<std::shared_ptr<LayerCacheBuffer>> client_buffers = {client_layer_buffer};
    auto                                           load_context =
        cache_store_client_->asyncLoad(client_buffers, 5000, server_ip_, server_port_, partition_count, partition_id);
    ASSERT_NE(load_context, nullptr);

    // 5. 在 server 端触发 notify（通过 asyncStore）
    auto null_event = DeviceEventPtr(nullptr);
    cache_store_server_->asyncStore(server_layer_buffer, null_event, 1000);

    // 6. 等待传输完成
    load_context->waitDone();

    // 7. 验证 client 端 buffer1 的数据是否正确
    for (auto& [key, block] : client_layer_buffer->blockCacheBuffers()) {
        EXPECT_TRUE(verifyBlockBuffer(block->buffer1, key, block_size, 100, device_util_.get()));
    }
}

TEST_F(CacheStoreTransferTest, TransferMultipleBlocksWithTimeout) {
    const int layer_id        = 0;
    const int num_blocks      = 3;
    const int block_size      = 2048;
    const int partition_count = 1;
    const int partition_id    = 0;

    // 1. 在 server 端创建 LayerCacheBuffer 并分配 buffer
    auto server_layer_buffer = createTestLayerCacheBuffer(layer_id, num_blocks, block_size);
    allocateBlockBuffers(server_layer_buffer, device_util_.get(), block_size);

    // 2. 在 client 端创建 LayerCacheBuffer 并分配空的 buffer
    auto client_layer_buffer = createTestLayerCacheBuffer(layer_id, num_blocks, block_size);
    for (auto& [key, block] : client_layer_buffer->blockCacheBuffers()) {
        auto buffer1 =
            device_util_->device_->allocateBuffer({DataType::TYPE_UINT8, {block_size}, AllocationType::DEVICE});
        auto buffer2 =
            device_util_->device_->allocateBuffer({DataType::TYPE_UINT8, {block_size}, AllocationType::DEVICE});
        block->buffer1 = buffer1;
        block->buffer2 = buffer2;
    }

    // 4. 在 client 端调用 asyncLoad 创建 LoadContext
    std::vector<std::shared_ptr<LayerCacheBuffer>> client_buffers = {client_layer_buffer};
    auto                                           load_context =
        cache_store_client_->asyncLoad(client_buffers, 5000, server_ip_, server_port_, partition_count, partition_id);
    ASSERT_NE(load_context, nullptr);

    // 5. 等待传输完成
    load_context->waitDone();

    // 6. 验证 load context 是否失败
    EXPECT_FALSE(load_context->success());
}

TEST_F(CacheStoreTransferTest, TransferMultipleBlocksWithFailed) {
    const int layer_id        = 0;
    const int num_blocks      = 3;
    const int block_size      = 2048;
    const int partition_count = 1;
    const int partition_id    = 0;

    // 1. 在 server 端创建 LayerCacheBuffer 并分配 buffer
    auto server_layer_buffer = createTestLayerCacheBuffer(layer_id, num_blocks, block_size);
    allocateBlockBuffers(server_layer_buffer, device_util_.get(), block_size);

    // 2. 在 client 端创建 LayerCacheBuffer 并分配空的 buffer
    auto client_layer_buffer = createTestLayerCacheBuffer(layer_id, num_blocks, block_size);
    for (auto& [key, block] : client_layer_buffer->blockCacheBuffers()) {
        auto buffer1 =
            device_util_->device_->allocateBuffer({DataType::TYPE_UINT8, {block_size}, AllocationType::DEVICE});
        auto buffer2 =
            device_util_->device_->allocateBuffer({DataType::TYPE_UINT8, {block_size}, AllocationType::DEVICE});
        block->buffer1 = buffer1;
        block->buffer2 = buffer2;
    }

    // 3. stop server
    server_tcp_server_->stop();

    // 4. 在 client 端调用 asyncLoad 创建 LoadContext
    std::vector<std::shared_ptr<LayerCacheBuffer>> client_buffers = {client_layer_buffer};
    auto                                           load_context =
        cache_store_client_->asyncLoad(client_buffers, 5000, server_ip_, server_port_, partition_count, partition_id);
    ASSERT_NE(load_context, nullptr);

    // 5. 等待传输完成
    load_context->waitDone();

    // 6. 验证 load context 是否失败
    EXPECT_FALSE(load_context->success());
}

}  // namespace cache_store
}  // namespace rtp_llm
