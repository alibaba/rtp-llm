#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "autil/NetUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverterImpl.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorker.h"
#include "rtp_llm/cpp/cache/smoke/CacheSmokeTestUtils.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm::test {

namespace {

struct PdEndpoint {
    PdEndpoint(const CacheConfig& config, uint32_t listen_port):
        allocator(makeCacheSmokeAllocatorForConfig(config, /*enable_prefix_cache=*/false)), port(listen_port) {}

    bool init() {
        if (!allocator->init()) {
            return false;
        }
        converter = std::make_shared<LayerBlockConverterImpl>(allocator);

        P2PConnectorWorkerConfig worker_config;
        worker_config.transfer_backend_config.cache_store_rdma_mode           = false;
        worker_config.transfer_backend_config.cache_store_listen_port         = port;
        worker_config.transfer_backend_config.messager_io_thread_count        = 1;
        worker_config.transfer_backend_config.messager_worker_thread_count    = 1;
        worker_config.transfer_backend_config.rdma_transfer_wait_timeout_ms   = 5000;
        worker_config.transfer_backend_config.transfer_wait_check_interval_us = 1000;
        worker_config.tp_size                                                 = 1;
        worker_config.tp_rank                                                 = 0;
        worker_config.layer_all_num                                           = config_layer_num;
        worker_config.p2p_read_steal_before_deadline_ms                       = 250;
        worker_config.p2p_read_return_before_deadline_ms                      = 100;

        worker = std::make_unique<P2PConnectorWorker>(worker_config, converter, nullptr);
        return worker->init(/*store_wait_timeout_ms=*/5000);
    }

    void setLayerNum(uint32_t layer_num) {
        config_layer_num = layer_num;
    }

    KVCacheAllocatorPtr                  allocator;
    std::shared_ptr<LayerBlockConverter> converter;
    std::unique_ptr<P2PConnectorWorker>  worker;
    uint32_t                             port;
    uint32_t                             config_layer_num{0};
};

}  // namespace

class CachePDSmokeTest: public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        initCacheSmokeRuntime();
    }
};

TEST_F(CachePDSmokeTest, WorkerTcpRoundTripPreservesPayloadAndConnectorLifetime) {
    const auto config = makeCacheSmokeConfig(/*block_num=*/8, DataType::TYPE_INT8);

    uint32_t prefill_port = autil::NetUtil::randomPort();
    uint32_t decode_port  = autil::NetUtil::randomPort();
    while (decode_port == prefill_port) {
        decode_port = autil::NetUtil::randomPort();
    }

    PdEndpoint prefill(config, prefill_port);
    PdEndpoint decode(config, decode_port);
    prefill.setLayerNum(config.layer_all_num);
    decode.setLayerNum(config.layer_all_num);
    ASSERT_TRUE(prefill.init());
    ASSERT_TRUE(decode.init());

    const size_t prefill_baseline_free = prefill.allocator->freeBlocksNum();
    const size_t decode_baseline_free  = decode.allocator->freeBlocksNum();
    const auto   cache_keys            = makeCacheKeys(/*begin=*/700, /*count=*/3);
    auto         tokens                = makeCacheSmokeTokenIds(makeTokenRange(/*begin=*/1, /*count=*/12));
    auto         prefill_batch         = makeCacheSmokeResource(config, cache_keys);
    auto         decode_batch          = makeCacheSmokeResource(config, cache_keys);

    ASSERT_TRUE(allocateCacheSmokeResource(prefill.allocator, prefill_batch, tokens).success);
    ASSERT_TRUE(allocateCacheSmokeResource(decode.allocator, decode_batch, tokens).success);
    fillAllocatorResource(*prefill.allocator, prefill_batch->cacheResource(0), /*seed=*/53);
    fillAllocatorResource(*decode.allocator, decode_batch->cacheResource(0), /*seed=*/0);

    auto prefill_hold =
        prefill.allocator->incrKVCacheRef(prefill_batch->cacheResource(0), cache_keys, /*is_connector=*/true);
    auto decode_hold =
        decode.allocator->incrKVCacheRef(decode_batch->cacheResource(0), cache_keys, /*is_connector=*/true);
    ASSERT_NE(prefill_hold, nullptr);
    ASSERT_NE(decode_hold, nullptr);

    prefill.allocator->free(FreeInfo{prefill_batch, tokens});
    decode.allocator->free(FreeInfo{decode_batch, tokens});
    EXPECT_EQ(prefill.allocator->requestRefBlocksNum(), 0u);
    EXPECT_EQ(decode.allocator->requestRefBlocksNum(), 0u);
    EXPECT_EQ(prefill.allocator->connectorRefBlocksNum(), cache_keys.size());
    EXPECT_EQ(decode.allocator->connectorRefBlocksNum(), cache_keys.size());
    EXPECT_EQ(prefill.allocator->freeBlocksNum(), prefill_baseline_free - cache_keys.size());
    EXPECT_EQ(decode.allocator->freeBlocksNum(), decode_baseline_free - cache_keys.size());

    const int64_t     request_id     = 9001;
    const std::string unique_key     = "cache-smoke-pd-9001";
    const int64_t     deadline_ms    = currentTimeMs() + 10000;
    auto              decode_buffers = LayerCacheBufferUtil::convert(*decode_hold,
                                                        /*batch_id=*/0,
                                                        /*start_block_idx=*/0,
                                                        /*block_count=*/-1,
                                                        /*cp_rank=*/0,
                                                        /*cp_size=*/1);
    ASSERT_EQ(decode_buffers.size(), config.layer_num);

    auto decode_result_future = std::async(std::launch::async, [&]() {
        return decode.worker->read(request_id, unique_key, deadline_ms, decode_buffers, /*remote_tp_size=*/1);
    });

    for (int layer_id = 0; layer_id < static_cast<int>(config.layer_num); ++layer_id) {
        ASSERT_TRUE(prefill.worker->writeByLayer(layer_id, prefill_hold, request_id, std::nullopt));
    }

    const auto send_result =
        prefill.worker->sendKVCache(request_id, unique_key, deadline_ms, {{"127.0.0.1", decode_port}});
    ASSERT_TRUE(send_result.ok()) << send_result.ToString();
    ASSERT_EQ(decode_result_future.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    const auto decode_result = decode_result_future.get();
    ASSERT_TRUE(decode_result.ok()) << decode_result.ToString();

    expectAllocatorResourcesEqual(*prefill.allocator, *prefill_hold, *decode.allocator, *decode_hold);

    prefill_hold.reset();
    decode_hold.reset();
    EXPECT_EQ(prefill.allocator->connectorRefBlocksNum(), 0u);
    EXPECT_EQ(decode.allocator->connectorRefBlocksNum(), 0u);
    EXPECT_EQ(prefill.allocator->freeBlocksNum(), prefill_baseline_free);
    EXPECT_EQ(decode.allocator->freeBlocksNum(), decode_baseline_free);
    EXPECT_EQ(prefill.allocator->availableBlocksNum(), prefill_baseline_free);
    EXPECT_EQ(decode.allocator->availableBlocksNum(), decode_baseline_free);
}

TEST_F(CachePDSmokeTest, MultiTypeGroupsTcpRoundTripPreservesLayerTags) {
    const auto config       = makeMultiGroupCacheSmokeConfig();
    uint32_t   prefill_port = autil::NetUtil::randomPort();
    uint32_t   decode_port  = autil::NetUtil::randomPort();
    while (decode_port == prefill_port)
        decode_port = autil::NetUtil::randomPort();
    PdEndpoint prefill(config, prefill_port), decode(config, decode_port);
    prefill.setLayerNum(config.layer_all_num);
    decode.setLayerNum(config.layer_all_num);
    ASSERT_TRUE(prefill.init());
    ASSERT_TRUE(decode.init());
    auto prefill_baseline =
        snapshotCacheSmokePools(*dynamic_cast<HybridPoolKVCacheAllocator*>(prefill.allocator.get()));
    auto decode_baseline = snapshotCacheSmokePools(*dynamic_cast<HybridPoolKVCacheAllocator*>(decode.allocator.get()));
    auto keys            = makeCacheKeys(900, 5);
    auto tokens          = makeCacheSmokeTokenIds(makeTokenRange(1, 20));
    auto prefill_batch   = makeCacheSmokeResource(config, keys);
    auto decode_batch    = makeCacheSmokeResource(config, keys);
    ASSERT_TRUE(allocateCacheSmokeResource(prefill.allocator, prefill_batch, tokens, true).success);
    ASSERT_TRUE(allocateCacheSmokeResource(decode.allocator, decode_batch, tokens, true).success);
    fillAllocatorResource(*prefill.allocator, prefill_batch->cacheResource(0), 71);
    fillAllocatorResource(*decode.allocator, decode_batch->cacheResource(0), 0);
    auto prefill_hold = prefill.allocator->incrKVCacheRef(prefill_batch->cacheResource(0), keys, true);
    auto decode_hold  = decode.allocator->incrKVCacheRef(decode_batch->cacheResource(0), keys, true);
    ASSERT_NE(prefill_hold, nullptr);
    ASSERT_NE(decode_hold, nullptr);
    prefill.allocator->free(FreeInfo{prefill_batch, tokens});
    decode.allocator->free(FreeInfo{decode_batch, tokens});
    auto buffers = LayerCacheBufferUtil::convert(*decode_hold, 0);
    ASSERT_EQ(buffers.size(), 7u);
    auto future = std::async(std::launch::async, [&] {
        return decode.worker->read(9101, "cache-smoke-multi", currentTimeMs() + 10000, buffers, 1);
    });
    for (int layer = 0; layer < 3; ++layer) {
        ASSERT_TRUE(prefill.worker->writeByLayer(layer, prefill_hold, 9101, std::nullopt));
    }
    ASSERT_TRUE(
        prefill.worker->sendKVCache(9101, "cache-smoke-multi", currentTimeMs() + 10000, {{"127.0.0.1", decode_port}})
            .ok());
    ASSERT_EQ(future.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    ASSERT_TRUE(future.get().ok());
    expectAllocatorResourcesEqual(*prefill.allocator, *prefill_hold, *decode.allocator, *decode_hold);
    prefill_hold.reset();
    decode_hold.reset();
    expectCacheSmokePoolsEqual(*dynamic_cast<HybridPoolKVCacheAllocator*>(prefill.allocator.get()), prefill_baseline);
    expectCacheSmokePoolsEqual(*dynamic_cast<HybridPoolKVCacheAllocator*>(decode.allocator.get()), decode_baseline);
}

}  // namespace rtp_llm::test
