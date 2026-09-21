#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/SingleTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorReadWriteContext.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnector.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorDecode.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PSchedulerDecodeRead.h"
#include "rtp_llm/cpp/cache/connector/p2p/test/MockGenerateStream.h"
#include "rtp_llm/cpp/cache/connector/p2p/test/TestRpcServer.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
namespace {

// Supply the allocator's logical match separately from device-ready reuse.
// The existing StreamCacheResource tests cover how this value is produced;
// these tests cover its propagation through the manager into the wire routes.
class PrefixReadContext: public KVCacheConnectorReadWriteContext {
public:
    PrefixReadContext(std::shared_ptr<Meta> meta, KVCacheResourcePtr resource, size_t covered):
        meta_(std::move(meta)), resource_(std::move(resource)), covered_(covered) {}

    const std::shared_ptr<Meta>& meta() const override {
        return meta_;
    }
    const KVCacheResource& kvCacheResource() const override {
        return *resource_;
    }
    size_t treeCoveredBlockNum() const override {
        return covered_;
    }

private:
    std::shared_ptr<Meta> meta_;
    KVCacheResourcePtr resource_;
    size_t covered_;
};

struct PrefixReadCase {
    const char* name;
    size_t covered;
    size_t expected_start;
    size_t expected_count;
};

class KVCacheManagerP2PReadTest: public ::testing::TestWithParam<PrefixReadCase> {
protected:
    void SetUp() override {
        worker_ = std::make_unique<TestRpcServer>(std::make_unique<TestRpcService>());
        prefill_ = std::make_unique<TestRpcServer>(std::make_unique<TestRpcService>());
        ASSERT_TRUE(worker_->start());
        ASSERT_TRUE(prefill_->start());
        worker_->service()->setLeaseStatus(true, 0, 0, true);
        prefill_->service()->setFirstGenerateTokenId(42);

        const auto config = test::makeSimpleMhaCacheConfig(
            /*layer_num=*/2, /*block_num=*/8, /*tokens_per_block=*/1, DataType::TYPE_FP16);
        allocator_ = std::make_shared<SingleTypeKVCacheAllocator>(config, AllocationType::HOST);
        ASSERT_TRUE(allocator_->init());
        const auto pool = allocator_->getDeviceBlockPool();
        const auto allocated = pool->malloc(5);
        ASSERT_TRUE(allocated.has_value());
        pool->incRef(*allocated);
        KVCacheResource source;
        source.initGroups(config.topologyPtr());
        // Neither key values nor physical IDs encode logical prefix positions.
        source.cacheKeys() = {901, 103, 705, 207, 509};
        source.mutableBlockIds(0).assign(
            {(*allocated)[4], (*allocated)[2], (*allocated)[0], (*allocated)[3], (*allocated)[1]});
        resource_ = allocator_->incrKVCacheRef(source, source.cacheKeys(), /*is_connector=*/false);
        pool->decRef(*allocated);
        ASSERT_NE(resource_, nullptr);
        resource_->setDeviceReuseBlockNum(GetParam().covered == 0 ? 0 : 1);

        P2PConnectorConfig connector_config;
        connector_config.role_type = RoleType::DECODE;
        auto& scheduler_config = connector_config.scheduler_config;
        scheduler_config.role_type = RoleType::DECODE;
        scheduler_config.parallelism_config.role_type = RoleType::DECODE;
        scheduler_config.topology = config.topologyPtr();
        scheduler_config.load_cache_timeout_ms = 10000;
        scheduler_config.worker_grpc_addrs = {"127.0.0.1:" + std::to_string(worker_->listenPort())};
        scheduler_config.worker_addrs = {"127.0.0.1:12345:" + std::to_string(worker_->listenPort())};
        auto broadcast = std::make_shared<P2PBroadcastClient>(scheduler_config.worker_grpc_addrs);
        ASSERT_TRUE(broadcast->init());
        auto connector = std::make_shared<P2PConnector>(connector_config, nullptr, nullptr);
        connector->decode_ = std::make_unique<P2PConnectorDecode>(connector_config, nullptr, nullptr);
        connector->decode_->scheduler_ =
            std::make_unique<P2PSchedulerDecodeRead>(scheduler_config, nullptr, broadcast);
        ASSERT_TRUE(connector->decode_->scheduler_->init("manager_prefix_read_test"));

        // Keep the manager, allocator reference handling, connector, planner and
        // scheduler real. Only remote worker/Prefill responses are test services;
        // no transfer backend is needed to inspect the emitted RPC payloads.
        manager_ = std::make_unique<KVCacheManager>(config);
        manager_->allocator_ = allocator_;
        manager_->pd_sep_config_.role_type = RoleType::DECODE;
        manager_->p2p_connector_ = std::move(connector);

        auto input = std::make_shared<GenerateInput>();
        input->request_id = 701;
        input->begin_time_us = currentTimeUs();
        input->request_deadline_ms = currentTimeMs() + 10000;
        input->generate_config = std::make_shared<GenerateConfig>();
        input->generate_config->unique_key = GetParam().name;
        input->generate_config->timeout_ms = 10000;
        input->input_ids = torch::zeros({5}, torch::kInt32);
        stream_ = std::make_shared<MockGenerateStream>(input);
        meta_ = std::make_shared<MockMeta>();
        meta_->setUniqueKey(GetParam().name);
        meta_->setRequestId(input->request_id);
        meta_->setDeadlineMs(input->request_deadline_ms);
        meta_->setPrefillAddr("127.0.0.1", prefill_->listenPort());
        meta_->setPrefillTpSize(1);
        meta_->setPrefillCpSize(1);
        meta_->setGenerateStream(stream_.get());
    }

    void TearDown() override {
        if (context_ && !context_->done()) {
            manager_->p2p_connector_->cancelRead(context_);
        }
        // Stop the scheduler while its RPC endpoints and input resources live.
        manager_.reset();
        context_.reset();
        stream_.reset();
        resource_.reset();
        allocator_.reset();
        worker_.reset();
        prefill_.reset();
    }

    std::unique_ptr<TestRpcServer> worker_, prefill_;
    std::shared_ptr<SingleTypeKVCacheAllocator> allocator_;
    KVCacheResourcePtr resource_;
    std::unique_ptr<KVCacheManager> manager_;
    std::shared_ptr<MockGenerateStream> stream_;
    std::shared_ptr<MockMeta> meta_;
    std::shared_ptr<P2PConnectorAsyncReadContext> context_;
};

TEST_P(KVCacheManagerP2PReadTest, TreePrefixDeterminesWireRoutes) {
    const auto& scenario = GetParam();
    auto read_context = std::make_shared<PrefixReadContext>(meta_, resource_, scenario.covered);
    // Do not construct a block_range here: it must be selected by the manager.
    context_ = std::dynamic_pointer_cast<P2PConnectorAsyncReadContext>(manager_->asyncLoadCache(read_context));
    ASSERT_NE(context_, nullptr);
    const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!context_->done() && std::chrono::steady_clock::now() < limit) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    ASSERT_TRUE(context_->done());
    ASSERT_TRUE(context_->success()) << context_->errorInfo().ToString();
    ASSERT_NE(context_->sideChannelPayload(), nullptr);
    EXPECT_EQ(context_->sideChannelPayload()->first_token_id, 42);
    ASSERT_EQ(prefill_->service()->getStartLoadCallCount(), 1);
    const auto start_load = prefill_->service()->getLastStartLoadRequest();
    EXPECT_EQ(start_load.unique_key(), scenario.name);
    EXPECT_EQ(start_load.no_transfer(), scenario.expected_count == 0);

    if (scenario.expected_count == 0) {
        // This asserts manager selection of no-transfer, beyond the scheduler's
        // existing test that passes no_transfer=true directly.
        EXPECT_EQ(worker_->service()->getBroadcastTpCallCount(), 0);
        EXPECT_EQ(start_load.active_route_ids_size(), 0);
        return;
    }

    ASSERT_EQ(worker_->service()->getBroadcastTpCallCount(), 1);
    const auto request = worker_->service()->getLastBroadcastTpRequest();
    EXPECT_EQ(request.type(), P2PConnectorBroadcastType::READ);
    ASSERT_EQ(request.routes_size(), 1);
    const auto& route = request.routes(0);
    ASSERT_EQ(route.layer_blocks_size(), 2);
    ASSERT_EQ(start_load.active_route_ids_size(), 1);
    EXPECT_EQ(start_load.active_route_ids(0), route.route_id());

    std::map<int64_t, int> expected;
    for (size_t i = scenario.expected_start; i < scenario.expected_start + scenario.expected_count; ++i) {
        expected.emplace(resource_->cacheKeys()[i], resource_->blocks(0)[i]);
    }
    std::set<int> layers;
    for (const auto& layer : route.layer_blocks()) {
        layers.insert(layer.layer_id());
        EXPECT_EQ(layer.cache_tag(), manager_->cacheConfig().tagForGroup(0));
        ASSERT_EQ(layer.cache_keys_size(), static_cast<int>(scenario.expected_count));
        ASSERT_EQ(layer.block_ids_size(), static_cast<int>(scenario.expected_count));
        std::map<int64_t, int> actual;
        for (int i = 0; i < layer.cache_keys_size(); ++i) {
            EXPECT_TRUE(actual.emplace(layer.cache_keys(i), layer.block_ids(i)).second);
        }
        EXPECT_EQ(actual, expected);  // Exact keys and IDs; includes prefix exclusion.
    }
    EXPECT_EQ(layers, (std::set<int>{0, 1}));
}

INSTANTIATE_TEST_SUITE_P(
    PrefixCoverage,
    KVCacheManagerP2PReadTest,
    ::testing::Values(PrefixReadCase{"NoHit", 0, 0, 5},
                      PrefixReadCase{"PartialHitIncludesPendingTreeLoad", 2, 2, 3},
                      PrefixReadCase{"OnlyFinalBlockMissing", 4, 4, 1},
                      PrefixReadCase{"AllHitStillHandshakes", 5, 5, 0},
                      PrefixReadCase{"CoverageBeyondKeysClampsToAllHit", 7, 5, 0}),
    [](const ::testing::TestParamInfo<PrefixReadCase>& info) { return std::string(info.param.name); });

}  // namespace
}  // namespace rtp_llm
