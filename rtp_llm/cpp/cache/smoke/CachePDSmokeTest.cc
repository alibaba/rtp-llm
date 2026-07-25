#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <initializer_list>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "autil/NetUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverterImpl.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorker.h"
#include "rtp_llm/cpp/cache/smoke/CacheSmokeTestUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm::test {

namespace {

constexpr int kEndpointInitAttempts = 10;

struct PdEndpoint {
    explicit PdEndpoint(const CacheConfig& config, int64_t tp_size = 1, int64_t tp_rank = 0):
        allocator(makeCacheSmokeAllocatorForConfig(config, /*enable_prefix_cache=*/false)),
        layer_all_num(config.layer_all_num),
        tp_size(tp_size),
        tp_rank(tp_rank) {}

    bool init(std::initializer_list<uint32_t> excluded_ports = {}) {
        if (!allocator->init()) {
            return false;
        }
        converter = std::make_shared<LayerBlockConverterImpl>(allocator);

        std::set<uint32_t> attempted_ports(excluded_ports);
        for (int attempt = 0; attempt < kEndpointInitAttempts; ++attempt) {
            uint32_t candidate_port = 0;
            do {
                candidate_port = autil::NetUtil::randomPort();
            } while (!attempted_ports.insert(candidate_port).second);

            P2PConnectorWorkerConfig worker_config;
            // Hermetic smoke uses TCP. Production RDMA requires separate validation on an RDMA-enabled host.
            worker_config.transfer_backend_config.cache_store_rdma_mode           = false;
            worker_config.transfer_backend_config.cache_store_listen_port         = candidate_port;
            worker_config.transfer_backend_config.messager_io_thread_count        = 1;
            worker_config.transfer_backend_config.messager_worker_thread_count    = 1;
            worker_config.transfer_backend_config.rdma_transfer_wait_timeout_ms   = 5000;
            worker_config.transfer_backend_config.transfer_wait_check_interval_us = 1000;
            worker_config.tp_size                                                 = tp_size;
            worker_config.tp_rank                                                 = tp_rank;
            worker_config.layer_all_num                                           = layer_all_num;
            worker_config.p2p_read_steal_before_deadline_ms                       = 250;
            worker_config.p2p_read_return_before_deadline_ms                      = 100;

            auto candidate_worker = std::make_unique<P2PConnectorWorker>(worker_config, converter, nullptr);
            if (candidate_worker->init(/*store_wait_timeout_ms=*/5000)) {
                port   = candidate_port;
                worker = std::move(candidate_worker);
                return true;
            }
            RTP_LLM_LOG_WARNING("cache PD smoke endpoint init failed, port=%u, attempt=%d/%d",
                                candidate_port,
                                attempt + 1,
                                kEndpointInitAttempts);
        }
        return false;
    }

    KVCacheAllocatorPtr                  allocator;
    std::shared_ptr<LayerBlockConverter> converter;
    std::unique_ptr<P2PConnectorWorker>  worker;
    uint32_t                             port{0};
    uint32_t                             layer_all_num;
    int64_t                              tp_size;
    int64_t                              tp_rank;
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

    PdEndpoint prefill(config);
    PdEndpoint decode(config);
    ASSERT_TRUE(prefill.init());
    ASSERT_TRUE(decode.init({prefill.port}));

    const size_t prefill_baseline_free = prefill.allocator->freeBlocksNum();
    const size_t decode_baseline_free  = decode.allocator->freeBlocksNum();
    const auto   cache_keys            = makeCacheKeys(/*begin=*/700, /*count=*/3);
    auto         tokens                = makeCacheSmokeTokenIds(config, makeTokenRange(/*begin=*/1, /*count=*/12));
    auto         prefill_batch         = makeCacheSmokeResource(config, cache_keys);
    auto         decode_batch          = makeCacheSmokeResource(config, cache_keys);

    ASSERT_TRUE(allocateCacheSmokeResource(prefill.allocator, prefill_batch, tokens).success);
    ASSERT_TRUE(allocateCacheSmokeResource(decode.allocator, decode_batch, tokens).success);
    ASSERT_NO_FATAL_FAILURE(fillAllocatorResource(*prefill.allocator, prefill_batch->cacheResource(0), /*seed=*/53));
    ASSERT_NO_FATAL_FAILURE(fillAllocatorResource(*decode.allocator, decode_batch->cacheResource(0), /*seed=*/0));

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
        prefill.worker->sendKVCache(request_id, unique_key, deadline_ms, {{"127.0.0.1", decode.port}});
    ASSERT_TRUE(send_result.ok()) << send_result.ToString();
    ASSERT_EQ(decode_result_future.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    const auto decode_result = decode_result_future.get();
    ASSERT_TRUE(decode_result.ok()) << decode_result.ToString();

    ASSERT_NO_FATAL_FAILURE(
        expectAllocatorResourcesEqual(*prefill.allocator, *prefill_hold, *decode.allocator, *decode_hold));

    prefill_hold.reset();
    decode_hold.reset();
    EXPECT_EQ(prefill.allocator->connectorRefBlocksNum(), 0u);
    EXPECT_EQ(decode.allocator->connectorRefBlocksNum(), 0u);
    EXPECT_EQ(prefill.allocator->freeBlocksNum(), prefill_baseline_free);
    EXPECT_EQ(decode.allocator->freeBlocksNum(), decode_baseline_free);
    EXPECT_EQ(prefill.allocator->availableBlocksNum(), prefill_baseline_free);
    EXPECT_EQ(decode.allocator->availableBlocksNum(), decode_baseline_free);
}

TEST_F(CachePDSmokeTest, WorkerTcpReadTimeoutReleasesConnectorHoldAndRestoresPool) {
    const auto config = makeCacheSmokeConfig(/*block_num=*/8, DataType::TYPE_INT8);
    PdEndpoint decode(config);
    ASSERT_TRUE(decode.init());

    const size_t baseline_free      = decode.allocator->freeBlocksNum();
    const size_t baseline_available = decode.allocator->availableBlocksNum();
    const auto   cache_keys         = makeCacheKeys(/*begin=*/750, /*count=*/2);
    auto         tokens             = makeCacheSmokeTokenIds(config, makeTokenRange(/*begin=*/1, /*count=*/8));
    auto         decode_batch       = makeCacheSmokeResource(config, cache_keys);
    ASSERT_TRUE(allocateCacheSmokeResource(decode.allocator, decode_batch, tokens).success);

    auto decode_hold =
        decode.allocator->incrKVCacheRef(decode_batch->cacheResource(0), cache_keys, /*is_connector=*/true);
    ASSERT_NE(decode_hold, nullptr);
    decode.allocator->free(FreeInfo{decode_batch, tokens});
    EXPECT_EQ(decode.allocator->requestRefBlocksNum(), 0u);
    EXPECT_EQ(decode.allocator->connectorRefBlocksNum(), cache_keys.size());
    EXPECT_EQ(decode.allocator->freeBlocksNum(), baseline_free - cache_keys.size());

    auto decode_buffers = LayerCacheBufferUtil::convert(*decode_hold, /*batch_id=*/0);
    ASSERT_EQ(decode_buffers.size(), config.layer_num);
    const auto read_result = decode.worker->read(/*request_id=*/9051,
                                                 /*unique_key=*/"cache-smoke-pd-timeout-9051",
                                                 /*deadline_ms=*/currentTimeMs() + 500,
                                                 decode_buffers,
                                                 /*remote_tp_size=*/1);
    EXPECT_TRUE(read_result.hasError());
    EXPECT_EQ(decode.allocator->requestRefBlocksNum(), 0u);
    EXPECT_EQ(decode.allocator->connectorRefBlocksNum(), cache_keys.size());

    decode_hold.reset();
    EXPECT_EQ(decode.allocator->connectorRefBlocksNum(), 0u);
    EXPECT_EQ(decode.allocator->freeBlocksNum(), baseline_free);
    EXPECT_EQ(decode.allocator->availableBlocksNum(), baseline_available);
}

TEST_F(CachePDSmokeTest, WorkerTcp2P1DRoundTripAssemblesDecodePartitions) {
    constexpr int64_t kPrefillTpSize = 2;
    const auto        prefill_config = makeSimpleMhaCacheConfig(/*layer_num=*/2,
                                                         /*block_num=*/8,
                                                         /*tokens_per_block=*/4,
                                                         DataType::TYPE_FP16,
                                                         /*local_head_num_kv=*/1,
                                                         /*size_per_head=*/8);
    const auto        decode_config  = makeSimpleMhaCacheConfig(/*layer_num=*/2,
                                                        /*block_num=*/8,
                                                        /*tokens_per_block=*/4,
                                                        DataType::TYPE_FP16,
                                                        /*local_head_num_kv=*/2,
                                                        /*size_per_head=*/8);

    PdEndpoint prefill_rank0(prefill_config, kPrefillTpSize, /*tp_rank=*/0);
    PdEndpoint prefill_rank1(prefill_config, kPrefillTpSize, /*tp_rank=*/1);
    PdEndpoint decode(decode_config);
    ASSERT_TRUE(prefill_rank0.init());
    ASSERT_TRUE(prefill_rank1.init({prefill_rank0.port}));
    ASSERT_TRUE(decode.init({prefill_rank0.port, prefill_rank1.port}));

    const size_t prefill_rank0_baseline_free = prefill_rank0.allocator->freeBlocksNum();
    const size_t prefill_rank1_baseline_free = prefill_rank1.allocator->freeBlocksNum();
    const size_t decode_baseline_free        = decode.allocator->freeBlocksNum();
    const auto   cache_keys                  = makeCacheKeys(/*begin=*/800, /*count=*/3);
    auto         tokens = makeCacheSmokeTokenIds(prefill_config, makeTokenRange(/*begin=*/1, /*count=*/12));
    auto         prefill_rank0_batch = makeCacheSmokeResource(prefill_config, cache_keys);
    auto         prefill_rank1_batch = makeCacheSmokeResource(prefill_config, cache_keys);
    auto         decode_batch        = makeCacheSmokeResource(decode_config, cache_keys);

    ASSERT_TRUE(allocateCacheSmokeResource(prefill_rank0.allocator, prefill_rank0_batch, tokens).success);
    ASSERT_TRUE(allocateCacheSmokeResource(prefill_rank1.allocator, prefill_rank1_batch, tokens).success);
    ASSERT_TRUE(allocateCacheSmokeResource(decode.allocator, decode_batch, tokens).success);
    ASSERT_NO_FATAL_FAILURE(
        fillAllocatorResource(*prefill_rank0.allocator, prefill_rank0_batch->cacheResource(0), /*seed=*/29));
    ASSERT_NO_FATAL_FAILURE(
        fillAllocatorResource(*prefill_rank1.allocator, prefill_rank1_batch->cacheResource(0), /*seed=*/137));
    ASSERT_NO_FATAL_FAILURE(fillAllocatorResource(*decode.allocator, decode_batch->cacheResource(0), /*seed=*/0));

    auto prefill_rank0_hold = prefill_rank0.allocator->incrKVCacheRef(prefill_rank0_batch->cacheResource(0),
                                                                      cache_keys,
                                                                      /*is_connector=*/true);
    auto prefill_rank1_hold = prefill_rank1.allocator->incrKVCacheRef(prefill_rank1_batch->cacheResource(0),
                                                                      cache_keys,
                                                                      /*is_connector=*/true);
    auto decode_hold =
        decode.allocator->incrKVCacheRef(decode_batch->cacheResource(0), cache_keys, /*is_connector=*/true);
    ASSERT_NE(prefill_rank0_hold, nullptr);
    ASSERT_NE(prefill_rank1_hold, nullptr);
    ASSERT_NE(decode_hold, nullptr);

    prefill_rank0.allocator->free(FreeInfo{prefill_rank0_batch, tokens});
    prefill_rank1.allocator->free(FreeInfo{prefill_rank1_batch, tokens});
    decode.allocator->free(FreeInfo{decode_batch, tokens});

    const int64_t     request_id     = 9201;
    const std::string unique_key     = "cache-smoke-pd-2p1d-9201";
    const int64_t     deadline_ms    = currentTimeMs() + 10000;
    auto              decode_buffers = LayerCacheBufferUtil::convert(*decode_hold, /*batch_id=*/0);
    ASSERT_EQ(decode_buffers.size(), decode_config.layer_num);

    auto decode_result_future = std::async(std::launch::async, [&]() {
        return decode.worker->read(
            request_id, unique_key, deadline_ms, decode_buffers, /*remote_tp_size=*/kPrefillTpSize);
    });

    for (int layer_id = 0; layer_id < static_cast<int>(prefill_config.layer_num); ++layer_id) {
        ASSERT_TRUE(prefill_rank0.worker->writeByLayer(layer_id, prefill_rank0_hold, request_id, std::nullopt));
        ASSERT_TRUE(prefill_rank1.worker->writeByLayer(layer_id, prefill_rank1_hold, request_id, std::nullopt));
    }

    const std::vector<std::pair<std::string, uint32_t>> decode_servers = {{"127.0.0.1", decode.port}};
    auto prefill_rank0_send_future                                     = std::async(std::launch::async, [&]() {
        return prefill_rank0.worker->sendKVCache(request_id, unique_key, deadline_ms, decode_servers);
    });
    auto prefill_rank1_send_future                                     = std::async(std::launch::async, [&]() {
        return prefill_rank1.worker->sendKVCache(request_id, unique_key, deadline_ms, decode_servers);
    });

    ASSERT_EQ(prefill_rank0_send_future.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    ASSERT_EQ(prefill_rank1_send_future.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    const auto prefill_rank0_send_result = prefill_rank0_send_future.get();
    const auto prefill_rank1_send_result = prefill_rank1_send_future.get();
    ASSERT_TRUE(prefill_rank0_send_result.ok()) << prefill_rank0_send_result.ToString();
    ASSERT_TRUE(prefill_rank1_send_result.ok()) << prefill_rank1_send_result.ToString();
    ASSERT_EQ(decode_result_future.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    const auto decode_result = decode_result_future.get();
    ASSERT_TRUE(decode_result.ok()) << decode_result.ToString();

    ASSERT_NO_FATAL_FAILURE(expectAllocatorResourcePartitionsEqual(*prefill_rank0.allocator,
                                                                   *prefill_rank0_hold,
                                                                   /*src_partition_count=*/1,
                                                                   /*src_partition_id=*/0,
                                                                   *decode.allocator,
                                                                   *decode_hold,
                                                                   /*dst_partition_count=*/2,
                                                                   /*dst_partition_id=*/0));
    ASSERT_NO_FATAL_FAILURE(expectAllocatorResourcePartitionsEqual(*prefill_rank1.allocator,
                                                                   *prefill_rank1_hold,
                                                                   /*src_partition_count=*/1,
                                                                   /*src_partition_id=*/0,
                                                                   *decode.allocator,
                                                                   *decode_hold,
                                                                   /*dst_partition_count=*/2,
                                                                   /*dst_partition_id=*/1));

    prefill_rank0_hold.reset();
    prefill_rank1_hold.reset();
    decode_hold.reset();
    EXPECT_EQ(prefill_rank0.allocator->connectorRefBlocksNum(), 0u);
    EXPECT_EQ(prefill_rank1.allocator->connectorRefBlocksNum(), 0u);
    EXPECT_EQ(decode.allocator->connectorRefBlocksNum(), 0u);
    EXPECT_EQ(prefill_rank0.allocator->freeBlocksNum(), prefill_rank0_baseline_free);
    EXPECT_EQ(prefill_rank1.allocator->freeBlocksNum(), prefill_rank1_baseline_free);
    EXPECT_EQ(decode.allocator->freeBlocksNum(), decode_baseline_free);
    EXPECT_EQ(prefill_rank0.allocator->availableBlocksNum(), prefill_rank0_baseline_free);
    EXPECT_EQ(prefill_rank1.allocator->availableBlocksNum(), prefill_rank1_baseline_free);
    EXPECT_EQ(decode.allocator->availableBlocksNum(), decode_baseline_free);
}

TEST_F(CachePDSmokeTest, MultiTypeGroupsTcpRoundTripPreservesLayerTags) {
    const auto config = makeMultiGroupCacheSmokeConfig();
    PdEndpoint prefill(config);
    PdEndpoint decode(config);
    ASSERT_TRUE(prefill.init());
    ASSERT_TRUE(decode.init({prefill.port}));
    auto* prefill_hybrid = dynamic_cast<HybridPoolKVCacheAllocator*>(prefill.allocator.get());
    auto* decode_hybrid  = dynamic_cast<HybridPoolKVCacheAllocator*>(decode.allocator.get());
    ASSERT_NE(prefill_hybrid, nullptr);
    ASSERT_NE(decode_hybrid, nullptr);
    const auto prefill_baseline = snapshotCacheSmokePools(*prefill_hybrid);
    const auto decode_baseline  = snapshotCacheSmokePools(*decode_hybrid);
    const auto keys             = makeCacheKeys(/*begin=*/900, /*count=*/5);
    auto       tokens           = makeCacheSmokeTokenIds(config, makeTokenRange(/*begin=*/1, /*count=*/20));
    auto       prefill_batch    = makeCacheSmokeResource(config, keys);
    auto       decode_batch     = makeCacheSmokeResource(config, keys);
    ASSERT_TRUE(
        allocateCacheSmokeResource(prefill.allocator, prefill_batch, tokens, /*enable_device_cache=*/true).success);
    ASSERT_TRUE(
        allocateCacheSmokeResource(decode.allocator, decode_batch, tokens, /*enable_device_cache=*/true).success);
    ASSERT_NO_FATAL_FAILURE(fillAllocatorResource(*prefill.allocator, prefill_batch->cacheResource(0), /*seed=*/71));
    ASSERT_NO_FATAL_FAILURE(fillAllocatorResource(*decode.allocator, decode_batch->cacheResource(0), /*seed=*/0));
    auto prefill_hold = prefill.allocator->incrKVCacheRef(prefill_batch->cacheResource(0), keys, /*is_connector=*/true);
    auto decode_hold  = decode.allocator->incrKVCacheRef(decode_batch->cacheResource(0), keys, /*is_connector=*/true);
    ASSERT_NE(prefill_hold, nullptr);
    ASSERT_NE(decode_hold, nullptr);
    prefill.allocator->free(FreeInfo{prefill_batch, tokens});
    decode.allocator->free(FreeInfo{decode_batch, tokens});
    auto buffers = LayerCacheBufferUtil::convert(*decode_hold, /*batch_id=*/0);
    // Layer-tag buffers: layer 0 has 2 groups, layer 1 has 3, and layer 2 has 2.
    ASSERT_EQ(buffers.size(), 7u);
    const int64_t     request_id           = 9101;
    const std::string unique_key           = "cache-smoke-multi";
    const int64_t     deadline_ms          = currentTimeMs() + 10000;
    auto              decode_result_future = std::async(std::launch::async, [&]() {
        return decode.worker->read(request_id, unique_key, deadline_ms, buffers, /*remote_tp_size=*/1);
    });
    for (int layer_id = 0; layer_id < static_cast<int>(config.layer_num); ++layer_id) {
        ASSERT_TRUE(prefill.worker->writeByLayer(layer_id, prefill_hold, request_id, std::nullopt));
    }
    const auto send_result =
        prefill.worker->sendKVCache(request_id, unique_key, deadline_ms, {{"127.0.0.1", decode.port}});
    ASSERT_TRUE(send_result.ok()) << send_result.ToString();
    ASSERT_EQ(decode_result_future.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    const auto decode_result = decode_result_future.get();
    ASSERT_TRUE(decode_result.ok()) << decode_result.ToString();
    ASSERT_NO_FATAL_FAILURE(
        expectAllocatorResourcesEqual(*prefill.allocator, *prefill_hold, *decode.allocator, *decode_hold));
    prefill_hold.reset();
    decode_hold.reset();
    ASSERT_NO_FATAL_FAILURE(expectCacheSmokePoolsEqual(*prefill_hybrid, prefill_baseline));
    ASSERT_NO_FATAL_FAILURE(expectCacheSmokePoolsEqual(*decode_hybrid, decode_baseline));
}

}  // namespace rtp_llm::test
