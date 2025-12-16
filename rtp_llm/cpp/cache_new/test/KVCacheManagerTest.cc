#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "rtp_llm/cpp/cache_new/KVCacheManager.h"
#include "rtp_llm/cpp/cache_new/KVCacheMemoryConnector.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"

namespace rtp_llm {
namespace test {

class MockKVCacheMemoryConnector: public KVCacheMemoryConnector {
public:
    MockKVCacheMemoryConnector(const CacheConfig&                       cache_config,
                               const std::shared_ptr<KVCacheAllocator>& allocator,
                               rtp_llm::DeviceBase*                     device,
                               const std::vector<std::string>&          tp_addrs):
        KVCacheMemoryConnector(cache_config, allocator, device, tp_addrs) {}

    MOCK_METHOD(bool, copyCache, (const MemoryCopyCacheRequestPB&, MemoryCopyCacheResponsePB&), (override));
};

class KVCacheManagerTest: public DeviceTestBase {
protected:
    void SetUp() override {
        // Use default cache config
        CacheConfig config;
        kv_cache_manager_ = std::make_shared<KVCacheManager>(config, device_);
    }

    std::shared_ptr<KVCacheManager> kv_cache_manager_;
};

TEST_F(KVCacheManagerTest, CopyCache_ReturnFalse_WhenNoMemRequest) {
    CopyCacheRequestPB  request;
    CopyCacheResponsePB response;

    // Request has no mem_request
    // Should return false and log warning
    EXPECT_FALSE(kv_cache_manager_->copyCache(request, response));
}

TEST_F(KVCacheManagerTest, CopyCache_ReturnFalse_WhenMemoryConnectorIsNull) {
    CopyCacheRequestPB request;
    request.mutable_mem_request();  // Add mem_request
    CopyCacheResponsePB response;

    // memory_connector_ is null by default
    kv_cache_manager_->memory_connector_ = nullptr;

    EXPECT_FALSE(kv_cache_manager_->copyCache(request, response));
    EXPECT_FALSE(response.mem_response().success());
}

TEST_F(KVCacheManagerTest, CopyCache_DelegatesToMemoryConnector_AndReturnsTrue) {
    CopyCacheRequestPB request;
    request.mutable_mem_request();
    CopyCacheResponsePB response;

    // Create mock connector
    CacheConfig              config;
    std::vector<std::string> tp_addrs;
    auto mock_connector = std::make_shared<MockKVCacheMemoryConnector>(config, nullptr, device_, tp_addrs);

    // Inject mock
    kv_cache_manager_->memory_connector_ = mock_connector;

    // Expect call
    EXPECT_CALL(*mock_connector, copyCache(testing::_, testing::_))
        .WillOnce(testing::Invoke([](const MemoryCopyCacheRequestPB& req, MemoryCopyCacheResponsePB& resp) {
            resp.set_success(true);
            return true;
        }));

    EXPECT_TRUE(kv_cache_manager_->copyCache(request, response));
    EXPECT_TRUE(response.mem_response().success());
}

TEST_F(KVCacheManagerTest, CopyCache_DelegatesToMemoryConnector_AndReturnsFalse) {
    CopyCacheRequestPB request;
    request.mutable_mem_request();
    CopyCacheResponsePB response;

    // Create mock connector
    CacheConfig              config;
    std::vector<std::string> tp_addrs;
    auto mock_connector = std::make_shared<MockKVCacheMemoryConnector>(config, nullptr, device_, tp_addrs);

    // Inject mock
    kv_cache_manager_->memory_connector_ = mock_connector;

    // Expect call
    EXPECT_CALL(*mock_connector, copyCache(testing::_, testing::_))
        .WillOnce(testing::Invoke([](const MemoryCopyCacheRequestPB& req, MemoryCopyCacheResponsePB& resp) {
            resp.set_success(false);
            return false;
        }));

    EXPECT_FALSE(kv_cache_manager_->copyCache(request, response));
    EXPECT_FALSE(response.mem_response().success());
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    rtp_llm::initLogger();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
