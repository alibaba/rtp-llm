#pragma once

#include <gtest/gtest.h>
#include "autil/Log.h"
#include "rtp_llm/cpp/disaggregate/cache_store/Interface.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/test_util/MockMemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/test_util/BlockBufferUtil.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {
class CacheStoreTestBase: public ::testing::Test {
public:
    void SetUp() override {
        CacheStoreConfig cache_store_config;
        memory_util_       = createMemoryUtilImpl(cache_store_config.cache_store_rdma_mode);
        device_util_       = std::make_shared<DeviceUtil>();
        block_buffer_util_ = std::make_shared<BlockBufferUtil>(memory_util_, device_util_);
    }
    void TearDown() override {}

protected:
    bool initMockMemoryUtil() {
        CacheStoreConfig cache_store_config;
        mock_memory_util_ = new MockMemoryUtil(createMemoryUtilImpl(cache_store_config.cache_store_rdma_mode));
        memory_util_.reset(mock_memory_util_);
        block_buffer_util_ = std::make_shared<BlockBufferUtil>(memory_util_, device_util_);
        return true;
    }

    void initTestDataDir() {
        const auto test_src_dir    = getenv("TEST_SRCDIR");
        const auto test_work_space = getenv("TEST_WORKSPACE");
        const auto test_binary     = getenv("TEST_BINARY");
        if (!(test_src_dir && test_work_space && test_binary)) {
            std::cerr << "Unable to retrieve TEST_SRCDIR / TEST_WORKSPACE / TEST_BINARY env!" << std::endl;
            return;
        }

        std::string test_binary_str = std::string(test_binary);
        RTP_LLM_CHECK(*test_binary_str.rbegin() != '/');
        size_t filePos         = test_binary_str.rfind('/');
        auto   test_data_path_ = std::string(test_src_dir) + "/" + std::string(test_work_space) + "/"
                               + test_binary_str.substr(0, filePos) + "/";

        std::cout << "test_src_dir [" << test_src_dir << "]" << std::endl;
        std::cout << "test_work_space [" << test_work_space << "]" << std::endl;
        std::cout << "test_binary [" << test_binary << "]" << std::endl;
        std::cout << "test using data path [" << test_data_path_ << "]" << std::endl;
    }

protected:
    std::shared_ptr<MemoryUtil>      memory_util_;
    MockMemoryUtil*                  mock_memory_util_{nullptr};
    std::shared_ptr<DeviceUtil>      device_util_;
    std::shared_ptr<BlockBufferUtil> block_buffer_util_;
};
}  // namespace rtp_llm