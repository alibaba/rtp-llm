
#include "gtest/gtest.h"

#define private public
#include "maga_transformer/cpp/cache/CacheManager.h"
#include "maga_transformer/cpp/dataclass/GenerateStream.h"
#include "maga_transformer/cpp/dataclass/Query.h"
#include "maga_transformer/cpp/dataclass/StreamCacheResource.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/devices/testing/TestBase.h"

#include <chrono>
#include <memory>
#include <thread>

using namespace std;
using namespace fastertransformer;

namespace rtp_llm {

class StreamCacheResourceTest: public DeviceTestBase {
protected:
    CacheConfig init_config() {
        CacheConfig config(3, 5, 1, 1, 2, TYPE_INT8);
        return config;
    }

protected:
};

TEST_F(StreamCacheResourceTest, testSimple) {
    auto            cache_config = init_config();
    ft::DeviceBase* device;
    CacheManagerPtr cache_manager = std::make_shared<CacheManager>(cache_config, nullptr, device_);
    ASSERT_EQ(cache_manager->freeBlockNums(), 4);

    std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
    std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
    auto                            vec   = vector<int>{1, 2, 3, 4, 5, 6};
    std::vector<size_t>             shape = {6};
    generate_input->input_ids =
        std::make_unique<ft::Buffer>(ft::MEMORY_CPU, ft::TYPE_INT32, shape, (void*)(vec.data()));
    generate_input->generate_config = generate_config;
    int max_seq_len                 = 2048;

    GenerateStream stream(generate_input, max_seq_len);
    stream.setCacheManager(cache_manager);

    auto& resource = stream.streamCacheResource();
    ASSERT_EQ(resource.nextNeedBlockNums(), 4);

    ASSERT_EQ(resource.initalKVCacheCount(), 3);

    ASSERT_TRUE(resource.initKVBlock());
    ASSERT_EQ(cache_manager->freeBlockNums(), 1);
    ASSERT_EQ(resource.maxBlockSize(), 3);
    const BatchKVCacheBlockAddr& blocks     = resource.kvCache();
    auto                         CHECK_FUNC = [](const auto& block_vec, auto outter_size, auto inner_size) {
        ASSERT_EQ(block_vec.size(), outter_size);
        ASSERT_EQ(block_vec[0][0].size(), inner_size);
    };
    CHECK_FUNC(blocks.k_ptr, 1, 3);
    CHECK_FUNC(blocks.v_ptr, 1, 3);
    CHECK_FUNC(blocks.k_scale_ptr, 1, 3);
    CHECK_FUNC(blocks.v_scale_ptr, 1, 3);

    ASSERT_EQ(resource.nextNeedBlockNums(), 1);
    ASSERT_TRUE(resource.incrKVBlock());
    ASSERT_EQ(cache_manager->freeBlockNums(), 0);

    ASSERT_EQ(blocks.k_ptr[0][0].size(), 4);
    ASSERT_EQ(blocks.v_ptr[0][0].size(), 4);
    ASSERT_EQ(blocks.k_scale_ptr[0][0].size(), 4);
    ASSERT_EQ(blocks.v_scale_ptr[0][0].size(), 4);

    stream.seq_length_ = 7;
    ASSERT_EQ(resource.nextNeedBlockNums(), 0);
    ASSERT_TRUE(resource.incrKVBlock());
    ASSERT_EQ(cache_manager->freeBlockNums(), 0);

    CHECK_FUNC(blocks.k_ptr, 1, 4);
    CHECK_FUNC(blocks.v_ptr, 1, 4);
    CHECK_FUNC(blocks.k_scale_ptr, 1, 4);
    CHECK_FUNC(blocks.v_scale_ptr, 1, 4);

    stream.releaseResource();
    // ASSERT_EQ(cache_manager->freeBlockNums(), 4);

    // ASSERT_EQ(blocks.k_ptr.size(), 0);
    // ASSERT_EQ(blocks.v_ptr.size(), 0);
    // ASSERT_EQ(blocks.k_scale_ptr.size(), 0);
    // ASSERT_EQ(blocks.v_scale_ptr.size(), 0);
}

TEST_F(StreamCacheResourceTest, testError) {}

}  // namespace rtp_llm
