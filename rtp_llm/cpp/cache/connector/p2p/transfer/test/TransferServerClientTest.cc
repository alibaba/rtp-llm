#include <gtest/gtest.h>
#include "rtp_llm/cpp/cache/connector/p2p/transfer/test/TransferServerClientTest.h"

namespace rtp_llm {

void MockLayerBlockConvertor::addBuffer(int layer_id, int block_id, BufferPtr buffer) {
    buffer_map_[layer_id][block_id].push_back(buffer);
}

void MockLayerBlockConvertor::removeBuffer(int layer_id, int block_id) {
    auto iter = buffer_map_.find(layer_id);
    if (iter == buffer_map_.end()) {
        return;
    }
    iter->second.erase(block_id);
}

std::vector<BufferPtr>
MockLayerBlockConvertor::convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const {
    auto iter = buffer_map_.find(layer_id);
    if (iter == buffer_map_.end()) {
        return {};
    }
    auto block_iter = iter->second.find(block_id);
    if (block_iter == iter->second.end()) {
        return {};
    }
    return block_iter->second;
}

void TransferServerClientTest::SetUp() {
    DeviceTestBase::SetUp();
    mock_client_layer_block_convertor_ = std::make_shared<MockLayerBlockConvertor>(device_);
    mock_server_layer_block_convertor_ = std::make_shared<MockLayerBlockConvertor>(device_);

    layer_cache_buffer0_ = createLayerCacheBuffer(0, 2);
    layer_cache_buffer1_ = createLayerCacheBuffer(1, 2);

    addBufferToConvertor(layer_cache_buffer0_, mock_client_layer_block_convertor_, 'A');
    addBufferToConvertor(layer_cache_buffer0_, mock_server_layer_block_convertor_, 'B');
    addBufferToConvertor(layer_cache_buffer1_, mock_client_layer_block_convertor_, 'C');
    addBufferToConvertor(layer_cache_buffer1_, mock_server_layer_block_convertor_, 'D');
}

void TransferServerClientTest::TearDown() {
    if (transfer_server_) {
        transfer_server_.reset();
    }
    if (transfer_client_) {
        transfer_client_.reset();
    }
    DeviceTestBase::TearDown();
}

// 创建测试用的 LayerCacheBuffer
std::shared_ptr<LayerCacheBuffer> TransferServerClientTest::createLayerCacheBuffer(int layer_id, int num_blocks) {
    auto buffer = std::make_shared<LayerCacheBuffer>(layer_id);
    for (int i = 0; i < num_blocks; ++i) {
        int64_t cache_key = layer_id * 1000 + i;
        int     block_id  = i;
        buffer->addBlockId(cache_key, block_id);
    }
    return buffer;
}

// 创建测试用的 GPU Buffer
BufferPtr TransferServerClientTest::createTestBuffer(size_t size, char fill_value) {
    auto buffer = device_->allocateBuffer({DataType::TYPE_UINT8, {size}, AllocationType::DEVICE}, {});
    if (buffer == nullptr) {
        return nullptr;
    }
    device_->bufMemset(*buffer, static_cast<int>(fill_value));
    device_->syncAndCheck();
    return buffer;
}

bool TransferServerClientTest::addBufferToConvertor(
    const std::shared_ptr<LayerCacheBuffer>&        layer_cache_buffer,
    const std::shared_ptr<MockLayerBlockConvertor>& layer_block_convertor,
    char                                            fill_value) {
    for (const auto& [cache_key, block_id] : layer_cache_buffer->blockIdMap()) {
        auto buffer0 = createTestBuffer(1024, fill_value);
        auto buffer1 = createTestBuffer(1024, fill_value + 1);
        layer_block_convertor->addBuffer(layer_cache_buffer->getLayerId(), block_id, buffer0);
        layer_block_convertor->addBuffer(layer_cache_buffer->getLayerId(), block_id, buffer1);
    }
    return true;
}

void TransferServerClientTest::verifyBufferContent(
    const std::shared_ptr<LayerCacheBuffer>&        layer_cache_buffer,
    char                                            fill_value,
    const std::shared_ptr<MockLayerBlockConvertor>& layer_block_convertor) {
    for (const auto& [cache_key, block_id] : layer_cache_buffer->blockIdMap()) {
        auto buffers = layer_block_convertor->convertIndexToBuffer(layer_cache_buffer->getLayerId(), block_id, 1, 0);
        ASSERT_EQ(buffers.size(), 2);
        ASSERT_EQ(buffers[0]->size(), 1024);
        ASSERT_EQ(buffers[1]->size(), 1024);
        verifyBufferContent(buffers[0], fill_value);
        verifyBufferContent(buffers[1], fill_value + 1);
    }
}

void TransferServerClientTest::verifyBufferContent(const BufferPtr& buffer, char fill_value) {
    BufferPtr cpu_buffer = device_->allocateBuffer({DataType::TYPE_UINT8, {buffer->size()}, AllocationType::HOST});
    device_->noBlockCopy({*cpu_buffer, *buffer});
    device_->syncAndCheck();
    for (int i = 0; i < cpu_buffer->size(); ++i) {
        ASSERT_EQ(*(cpu_buffer->dataWithOffset<uint8_t>(i)), static_cast<uint8_t>(fill_value));
    }
}

}  // namespace rtp_llm
