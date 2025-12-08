#pragma once

#include "gtest/gtest.h"

#include "rtp_llm/cpp/disaggregate/transfer/TransferServer.h"
#include "rtp_llm/cpp/disaggregate/transfer/TransferClient.h"
#include "rtp_llm/cpp/disaggregate/transfer/LayerBlockConvertor.h"
#include "rtp_llm/cpp/disaggregate/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "autil/NetUtil.h"

namespace rtp_llm {

// Mock LayerBlockConvertor for testing
class MockLayerBlockConvertor: public LayerBlockConvertor {
public:
    MockLayerBlockConvertor(DeviceBase* device): device_(device) {}
    ~MockLayerBlockConvertor() = default;

    void addBuffer(int layer_id, int block_id, BufferPtr buffer);
    void removeBuffer(int layer_id, int block_id);
    std::vector<BufferPtr>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count = 1, int partition_id = 0) const override;

private:
    std::unordered_map<int, std::unordered_map<int, std::vector<BufferPtr>>> buffer_map_;
    DeviceBase*                                                              device_ = nullptr;
};

class TransferServerClientTest: public DeviceTestBase {
protected:
    void SetUp() override;
    void TearDown() override;

    // 创建测试用的 LayerCacheBuffer
    std::shared_ptr<LayerCacheBuffer> createLayerCacheBuffer(int layer_id, int num_blocks = 3);

    // 创建测试用的 GPU Buffer
    BufferPtr createTestBuffer(size_t size, char fill_value = 'A');

    bool addBufferToConvertor(const std::shared_ptr<LayerCacheBuffer>&        layer_cache_buffer,
                              const std::shared_ptr<MockLayerBlockConvertor>& layer_block_convertor,
                              char                                            fill_value = 'A');

    void verifyBufferContent(const std::shared_ptr<LayerCacheBuffer>&        layer_cache_buffer,
                             char                                            fill_value,
                             const std::shared_ptr<MockLayerBlockConvertor>& layer_block_convertor);

    void verifyBufferContent(const BufferPtr& buffer, char fill_value);

protected:
    std::shared_ptr<TransferClient>          transfer_client_;
    std::shared_ptr<MockLayerBlockConvertor> mock_client_layer_block_convertor_;

    std::shared_ptr<TransferServer>          transfer_server_;
    std::shared_ptr<MockLayerBlockConvertor> mock_server_layer_block_convertor_;
    uint32_t                                 listen_port_;

    std::shared_ptr<LayerCacheBuffer> layer_cache_buffer0_;
    std::shared_ptr<LayerCacheBuffer> layer_cache_buffer1_;
};
}  // namespace rtp_llm