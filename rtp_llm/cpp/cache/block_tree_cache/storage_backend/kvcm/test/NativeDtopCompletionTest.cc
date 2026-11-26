#include "rtp_llm/cpp/utils/DecodePublicationTest.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BoundedThreadTestUtils.h"
#include <cuda_runtime_api.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include "rtp_llm/cpp/utils/RemoteCacheConfig.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/PayloadDispatchProof.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/PublicationSettlement.h"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <future>
#include <functional>
#include <map>
#include <tuple>
#include <typeinfo>
#include <stdexcept>
#include <vector>
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/ClientFactory.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/ClientWrapper.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"

namespace rtp_llm {
namespace {
TEST(DecodePublicationProofTest, AbortRequiresNoStartOrTerminalProofAndNoFinishAttempt) {
    using Proof = kvcm::PayloadDispatchProof;
    using Action = kvcm::FailedPublicationAction;
    Proof proof;
    EXPECT_EQ(proof.state(), Proof::State::NOT_HANDED_OFF);
    EXPECT_EQ(kvcm::failedPublicationAction(!proof.neverStarted(), false), Action::ABORT_UNCOMMITTED);
    proof.pending();
    EXPECT_EQ(proof.state(), Proof::State::HANDOFF_PENDING);
    EXPECT_EQ(kvcm::failedPublicationAction(!proof.neverStarted(), false), Action::RETAIN_SESSION);
    proof.notStarted();
    EXPECT_EQ(proof.state(), Proof::State::PROVEN_NOT_STARTED);
    EXPECT_EQ(kvcm::failedPublicationAction(!proof.neverStarted(), false), Action::ABORT_UNCOMMITTED);
    // A FinishWrite attempt alone still forbids abort or replay, even without payload.
    EXPECT_EQ(kvcm::failedPublicationAction(!proof.neverStarted(), true), Action::RETAIN_SESSION);
    Proof possible;
    possible.pending();
    possible.started();
    possible.notStarted();
    possible.pending();
    EXPECT_EQ(possible.state(), Proof::State::MAY_HAVE_STARTED);
    EXPECT_FALSE(possible.neverStarted());
    EXPECT_EQ(kvcm::failedPublicationAction(!possible.neverStarted(), false), Action::RETAIN_SESSION);
    // The sticky handoff marker remains true, but a later all-peer terminal
    // proof can permit cleanup. Missing proof and attempted Finish still win.
    EXPECT_EQ(kvcm::failedPublicationAction(!possible.neverStarted(), false, true), Action::ABORT_UNCOMMITTED);
    EXPECT_EQ(kvcm::failedPublicationAction(!possible.neverStarted(), true, true), Action::RETAIN_SESSION);
    EXPECT_EQ(kvcm::failedPublicationAction(!possible.neverStarted(), false, false), Action::RETAIN_SESSION);
}

TEST(ClientFactoryFeatureBoundaryTest, DisabledAndUnsetDelegateConfigurationErrorsToTheSdk) {
    using namespace kv_cache_manager;
    // Invalid JSON cannot initiate network I/O. Compare against the SDK itself,
    // whose legacy error/exception contract the disabled feature must preserve.
    // Our SSD parser used to throw before the SDK was called, even when off.
    const std::string         config = "{";
    InitParams                params{};
    ClientMemoryRegistrations registrations;
    kvcm::ClientFactory       factory;
    const auto                observe = [](auto call) -> std::pair<bool, std::string> {
        try {
            return {static_cast<bool>(call()), {}};
        } catch (const std::exception& e) {
            return {false, std::string(typeid(e).name()) + ":" + e.what()};
        } catch (...) {
            return {false, "non-standard exception"};
        }
    };
    for (bool unset : {false, true}) {
        autil::EnvGuard off("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
        ::rtp_llm::test::withDecodePublicationStartup([&] {
        if (unset) {
            ASSERT_EQ(unsetenv("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH"), 0);
        }
        EXPECT_EQ(observe([&] { return factory.createTransferClient(config, params); }),
                  observe([&] { return TransferClient::Create(config, params); }))
            << unset;
        EXPECT_EQ(observe([&] { return factory.createTransferClientWithMemory(config, params, registrations); }),
                  observe([&] { return TransferClient::Create(config, params, registrations); }))
            << unset;
        });
    }
}
TEST(FeatureOffNativeTest, PrefillRemoteRoundTripPreservesLocalPrefixPayloadAndReferences) {
    for (const bool unset : {false, true}) {
        SCOPED_TRACE(unset);
        autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
        ::rtp_llm::test::withDecodePublicationStartup([&] {
        if (unset) {
            ASSERT_EQ(unsetenv("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH"), 0);
        }
        auto env = makeBackendEnvironment("feature_off_prefill_payload");
        ScopedReferencedBlocks pages(env.device_pool, 3);
        const auto ids = pages.get();
        const auto pool = env.device_pool;
        std::vector<size_t> refs;
        for (size_t i = 0; i < ids.size(); ++i) {
            refs.push_back(pool->refCount(ids[i]));
            for (const auto& part : environmentBuffers(env, 0, "default", ids[i]))
                ASSERT_EQ(cudaMemset(part.addr, 0x41 + i, part.size_bytes), cudaSuccess);
        }
        auto client = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*client, shutdown()).Times(1);
        auto parallel = singleRankConfig();
        parallel.role_type = RoleType::PREFILL;
        auto backend = makeBackend(env, parallel, client);
        ASSERT_TRUE(initSingleRank(*backend.backend, env));
        kv_cache_manager::Locations locations{{{"tp0_Fdefault", "off://102"}},
                                             {{"tp0_Fdefault", "off://103"}}};
        EXPECT_CALL(*client, match(_, _, _, std::vector<int64_t>({101, 102, 103}), _, _))
            .WillOnce(Return(std::make_pair(true, locations)));
        auto request = makeStorageRequest(env, {101, 102, 103}, 1, ids);
        auto hit = match(*backend.backend, request);
        ASSERT_TRUE(hit.success);
        ASSERT_EQ(hit.matched_blocks_num, 3u);
        ASSERT_NE(hit.match_meta, nullptr);
        EXPECT_CALL(*client, loadKvCachesForTag("default", kv_cache_manager::UriStrVec({"off://102", "off://103"}), _, _))
            .WillOnce(Invoke([](const auto&, const auto&, auto& buffers, const auto&) {
                if (buffers.size() != 2u) return false;
                for (size_t i = 0; i < buffers.size(); ++i)
                    for (const auto& iov : buffers[i].iovs)
                        if (cudaMemset(iov.base, 0x52 + i, iov.size) != cudaSuccess) return false;
                return true;
            }));
        ASSERT_TRUE(read(*backend.backend, request, std::move(hit.match_meta)));
        ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
        for (size_t i = 0; i < ids.size(); ++i) {
            EXPECT_EQ(pool->refCount(ids[i]), refs[i]);
            EXPECT_FALSE(pool->isFrozen(ids[i]));
            for (const auto& part : environmentBuffers(env, 0, "default", ids[i])) {
                std::vector<unsigned char> bytes(part.size_bytes);
                ASSERT_EQ(cudaMemcpy(bytes.data(), part.addr, bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
                const auto expected = i == 0 ? 0x41 : 0x51 + i;
                EXPECT_TRUE(std::all_of(bytes.begin(), bytes.end(), [&](auto value) { return value == expected; }));
            }
        }
        kv_cache_manager::WriteLocation allocation;
        allocation.write_session_id = "feature-off-acknowledged";
        allocation.block_mask = kv_cache_manager::BlockMaskOffset{0};
        allocation.locations = locations;
        EXPECT_CALL(*client, getWriteLocation(_, _, std::vector<int64_t>({102, 103}), _, _, _, 0))
            .WillOnce(Return(std::make_pair(true, allocation)));
        EXPECT_CALL(*client, saveKvCachesForTag("default", kv_cache_manager::UriStrVec({"off://102", "off://103"}), _, _))
            .WillOnce(Invoke([](const auto&, const auto& uris, const auto& buffers, const auto&) {
                if (buffers.size() != 2u) return std::make_pair(false, uris);
                for (size_t i = 0; i < buffers.size(); ++i)
                    for (const auto& iov : buffers[i].iovs) {
                        std::vector<unsigned char> bytes(iov.size);
                        if (cudaMemcpy(bytes.data(), iov.base, bytes.size(), cudaMemcpyDeviceToHost) != cudaSuccess)
                            return std::make_pair(false, uris);
                        EXPECT_TRUE(std::all_of(bytes.begin(), bytes.end(), [&](auto value) { return value == 0x52 + i; }));
                    }
                return std::make_pair(true, uris);
            }));
        EXPECT_CALL(*client, finishWrite(_, _, "feature-off-acknowledged",
                    ::testing::Eq(kv_cache_manager::BlockMask{kv_cache_manager::BlockMaskOffset{2}}), ::testing::IsEmpty()))
            .WillOnce(Return(true));
        auto task = backend.backend->prepareWrite(makeStorageRequest(env, {102, 103}, 0, {ids[1], ids[2]}));
        ASSERT_TRUE(static_cast<bool>(task));
        ASSERT_TRUE(backend.backend->write(std::move(task)));
        ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
        backend.backend->shutdown();
        for (size_t i = 0; i < ids.size(); ++i) {
            EXPECT_EQ(pool->refCount(ids[i]), refs[i]);
            EXPECT_FALSE(pool->isFrozen(ids[i]));
        }
        });
    }
}

class EosGpuCompletionTest: public ::testing::TestWithParam<std::tuple<int, bool, int>> {
protected:
    autil::EnvGuard enable{"RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1"},
        pending{"RTP_LLM_REMOTE_CACHE_DECODE_PENDING_JOBS", "1"},
        // This tiny fixture intentionally holds most of its synthetic pool.
        // Production's 1/8 default is covered separately; allow these positive
        // lifetime/projection cases to reach transfer rather than admission.
        pending_bytes{"RTP_LLM_REMOTE_CACHE_DECODE_PENDING_MIB", "1"};
    // Failure cases deliberately freeze a large fraction of these tiny pools.
    autil::EnvGuard quarantine_abort{"RTP_LLM_REMOTE_CACHE_QUARANTINE_ABORT_RATIO", "0"};
    struct Scenario {
        static constexpr int kCheckpointCount = 3;
        int                                 cp, shards, count = kCheckpointCount;
        BackendEnvironment                  env;
        BlockIndicesType                    full, state;
        BatchKVCacheResource                resource;
        std::shared_ptr<MockClientWrapper>  client = std::make_shared<MockClientWrapper>();
        std::unique_ptr<MockQuiescentKVCMBackend> backend;
        std::map<std::string, std::string>  specs;
        std::atomic<int>                    starts{0}, saves{0}, finishes{0}, aborts{0};
        std::atomic<size_t>                 requested_keys{0};
        bool                                block_start = false, fail_save = false, lose_commit = false;
        std::promise<void>                  entered, released;
        std::shared_future<void>            release = released.get_future().share();
        std::future<void>                   started = entered.get_future();
        explicit Scenario(
            int target_cp, bool sharded, bool bounded = true, bool native_ssd_gdr = false, int decode_dp = 1,
            int minimum_pool_blocks = 32):
            cp(target_cp), shards(sharded ? cp : 1) {
            // The target CP comes only from ParallelismConfig, not a CP-specific environment flag.
            auto f                     = defaultCacheGroupPolicy(CacheGroupType::FULL);
            auto s                     = defaultCacheGroupPolicy(CacheGroupType::SWA);
            s.enable_prefix_reuse      = true;
            s.cp_mapping               = sharded ? CpBlockMappingMode::COMPACT_LAST_RANK : CpBlockMappingMode::NONE;
            s.cp_slice                 = sharded ? CpBlockSliceMode::EQUAL_BYTES : CpBlockSliceMode::NONE;
            s.sliding_window_size      = bounded ? 256 * shards : 0;
            env.cache_config.dtype     = DataType::TYPE_FP16;
            env.cache_config.layer_num = 2;
            env.cache_config.seq_size_per_block = 128;
            env.cache_config.fromGroupedSpecs({test::makeMhaSpec("full", 128, DataType::TYPE_FP16, 1, 2),
                                               test::makeMhaSpec("state", 128 * shards, DataType::TYPE_FP16, 1, 2)},
                                              {{0}, {1}},
                                              {CacheGroupType::FULL, CacheGroupType::SWA},
                                              {"full", "state"},
                                              {f, s});
            const int pool_blocks = std::max(minimum_pool_blocks, (count + 1) * shards + 8);
            env.cache_config.finalizeBlockNums(pool_blocks, RuntimeConfig{});
            initializeEnvironmentPools(env, "eos_native_gpu", pool_blocks);
            full  = *env.pools_by_tag.at("full")->malloc((count + 1) * shards);
            state = *env.pools_by_tag.at("state")->malloc(count + 1);
            env.pools_by_tag.at("full")->incRef(full);
            env.pools_by_tag.at("state")->incRef(state);
            for (size_t i = 0; i < full.size(); ++i)
                fill(0, "full", full[i], i + 1);
            for (size_t i = 0; i < state.size(); ++i)
                fill(1, "state", state[i], 100 + i);
            resource.resetBatchSize(1);
            resource.initGroups(env.cache_config.topologyPtr());
            resource.setBatchBlocks(0, "full", full);
            resource.setBatchBlocks(0, "state", state);
            CacheKeysType keys;
            for (size_t i = 0; i < full.size(); ++i)
                keys.push_back(101 + i);
            resource.setBatchCacheKeys(0, keys);
            EXPECT_CALL(*client, initForPools(_, _, _, _))
                .WillOnce(Invoke([&](const auto&, auto, const auto& registrations, const auto& tags) {
                    EXPECT_EQ(registrations.size(), tags.size());
                    for (size_t i = 0; i < tags.size(); ++i)
                        specs[tags[i]] = registrations[i].location_spec_name;
                    return true;
                }));
            EXPECT_CALL(*client, shutdown()).Times(1);
            EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, _))
                .WillRepeatedly(
                    Invoke([&](const auto&, const auto&, const auto& keys, const auto&, const auto&, auto, auto) {
                        ++starts;
                        requested_keys = keys.size();
                        if (block_start) {
                            entered.set_value();
                            EXPECT_EQ(release.wait_for(std::chrono::seconds(5)), std::future_status::ready);
                        }
                        kv_cache_manager::WriteLocation location;
                        location.write_session_id = "EOS";
                        location.block_mask       = kv_cache_manager::BlockMaskOffset{0};
                        for (size_t key = 0; key < keys.size(); ++key) {
                            EXPECT_EQ(keys[key], 100 + (key + 1) * shards);
                            kv_cache_manager::Location one;
                            for (int peer = 0; peer < cp; ++peer) {
                                const auto suffix = std::to_string(key) + ":" + std::to_string(peer);
                                one.push_back({specs.at("full#cp" + std::to_string(peer)), "pace://F:" + suffix});
                                if (keys.size() - key
                                    <= env.cache_config.topology().group("state").reuseBlockCount(keys.size()))
                                    one.push_back({specs.at("state#cp" + std::to_string(peer)), "pace://S:" + suffix});
                            }
                            location.locations.push_back(one);
                        }
                        return std::make_pair(true, location);
                    }));
            EXPECT_CALL(*client, saveKvCachesForTag(_, _, _, _))
                .WillRepeatedly(Invoke([&](const auto& tag, const auto& uris, const auto& buffers, const auto&) {
                    ++saves;
                    EXPECT_EQ(uris.size(), buffers.size());
                    kv_cache_manager::UriStrVec actual;
                    for (size_t i = 0; i < uris.size(); ++i) {
                        const auto payload = uris[i].substr(7);
                        const int key = std::stoi(payload.substr(2)), peer = std::stoi(payload.substr(4));
                        // The first registered PACE client owns every GPU span.
                        EXPECT_EQ(tag, "full#cp0");
                        const int expected = payload[0] == 'F' ? key * shards + (shards > 1 ? peer : 0) + 1 : 100 + key;
                        for (const auto& iov : buffers[i].iovs) {
                            EXPECT_EQ(iov.type, kv_cache_manager::MemoryType::GPU);
                            std::vector<unsigned char> bytes(iov.size);
                            EXPECT_EQ(cudaMemcpy(bytes.data(), iov.base, iov.size, cudaMemcpyDeviceToHost),
                                      cudaSuccess);
                            EXPECT_TRUE(std::all_of(bytes.begin(), bytes.end(), [&](auto b) { return b == expected; }));
                        }
                        actual.push_back("actual_" + uris[i]);
                    }
                    return std::make_pair(!fail_save, actual);
                }));
            EXPECT_CALL(*client, finishWrite(_, _, _, _, _))
                .WillRepeatedly(
                    Invoke([&](const auto&, const auto&, const auto&, const auto& mask, const auto& locations) {
                        if (locations.empty()) {
                            ++aborts;
                            return true;
                        }
                        ++finishes;
                        EXPECT_EQ(std::get<kv_cache_manager::BlockMaskOffset>(mask), locations.size());
                        for (const auto& row : locations)
                            for (const auto& item : row)
                                EXPECT_EQ(item.uri.rfind("actual_", 0), 0u);
                        return !lose_commit;
                    }));
            KVCacheConfig k;
            k.kvcm_server_address   = "unused";
            k.kvcm_instance_group   = "decode_remote_eos_gpu_test";
            k.kvcm_instance_id_salt = "eos_native_gpu";
            k.kvcm_model_sdk_config = "[{\"type\":\"pace\"}]";
            if (native_ssd_gdr) {
                k.kvcm_model_sdk_config  = "[{\"type\":\"pace_ssd\"}]";
                k.kvcm_read_backend_type = 9;
            }
            ParallelismConfig parallel;
            parallel.tp_size                            = 1;
            parallel.dp_size                            = decode_dp;
            parallel.ep_size                            = decode_dp;
            parallel.role_type                          = RoleType::DECODE;
            parallel.prefill_cp_config.prefill_cp_size  = cp;
            parallel.prefill_cp_config.kv_cache_sharded = sharded;
            RuntimeConfig rt;
            rt.model_name = "eos_gpu";
            backend       = std::make_unique<MockQuiescentKVCMBackend>(
                env.cache_config, k, rt, parallel, SpeculativeExecutionConfig{}, nullptr, true, nullptr, client);
            EXPECT_TRUE(initSingleRank(*backend, env));
        }
        ~Scenario() {
            unblock();
            if (backend) {
                backend->proveMockQuiescence();
                backend->shutdown();
            }
            dropRequestRefs();
        }
        void unblock() {
            try {
                released.set_value();
            } catch (...) {}
        }
        void fill(int layer, const std::string& tag, int block, int value) {
            for (const auto& part : environmentBuffers(env, layer, tag, block))
                EXPECT_EQ(cudaMemset(part.addr, value, part.size_bytes), cudaSuccess);
        }
        size_t end() const {
            return count * shards * 128;
        }
        bool refs = true;
        void dropRequestRefs() {
            if (refs) {
                env.pools_by_tag.at("full")->decRef(full);
                env.pools_by_tag.at("state")->decRef(state);
                refs = false;
            }
        }
    };
};

static void verifyReleasedDsparkStatePublication(bool all_resident_state) {
    autil::EnvGuard enabled("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    autil::EnvGuard budget("RTP_LLM_REMOTE_CACHE_DECODE_PENDING_MIB", "1");
    test::withDecodePublicationStartup([&] {
        createDevice();
        CacheConfig cache;
        cache.dtype = DataType::TYPE_FP16;
        cache.layer_num = 2;
        cache.seq_size_per_block = 8;
        const auto full = test::makeMhaSpec("csa_kv", 8, cache.dtype, 1, 2);
        const auto state = test::makeResolvedOpaqueSpec(true, "csa_state", DataType::TYPE_FP32, 32, 8);
        const auto swa = test::makeResolvedOpaqueSpec(true, "swa_kv", cache.dtype, 32, 8);
        const auto decoder = test::makeResolvedOpaqueSpec(true, "decoder_swa_kv", cache.dtype, 32, 8);
        const auto full_policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
        auto tail_policy = defaultCacheGroupPolicy(CacheGroupType::SWA);
        tail_policy.enable_prefix_reuse = true;
        tail_policy.sliding_window_size = 8; // Latest H alone cannot recover the earlier H's state.
        auto decoder_policy = tail_policy;
        decoder_policy.enable_prefix_reuse = false;
        cache.fromGroupedSpecs({full, state, swa, decoder}, {{0}, {0}, {0}, {1}},
            {CacheGroupType::FULL, CacheGroupType::SWA, CacheGroupType::SWA, CacheGroupType::SWA},
            {"csa_kv", "csa_state", "swa_kv", "decoder_swa_kv"},
            {full_policy, tail_policy, tail_policy, decoder_policy});
        CacheConfig draft;
        draft.dtype = cache.dtype;
        draft.layer_num = 3;
        draft.seq_size_per_block = 8;
        draft.fromGroupedSpecs({decoder}, {{0, 1, 2}}, {CacheGroupType::SWA},
                               {"decoder_swa_kv"}, {decoder_policy});
        cache.mtp_sub_configs.push_back(cache.mergeMTPModule(draft, 0, cache.layer_num));
        cache.finalizeBlockNums(16, RuntimeConfig{});
        auto parallel = singleRankConfig();
        parallel.role_type = RoleType::DECODE;
        parallel.prefill_cp_config.prefill_cp_size = 1;
        auto manager = std::make_shared<KVCacheManager>(cache, false, nullptr, KVCacheConfig{}, parallel,
            RuntimeConfig{}, SpeculativeExecutionConfig{}, PDSepConfig{}, CacheStoreConfig{},
            /*use_device_malloc_block_pool=*/true);
        ASSERT_TRUE(manager->init());
        BackendEnvironment env;
        env.cache_config = manager->cacheConfig();
        const auto& tags = env.cache_config.groupTags();
        const auto& pools = manager->coordinator_manager_->groupBlockPools();
        ASSERT_EQ(tags.size(), pools.size());
        for (size_t group = 0; group < tags.size(); ++group)
            env.pools_by_tag.emplace(tags[group], pools[group]);
        auto pages = std::make_unique<ScopedReferencedBlocks>(env, 3);
        const auto blocks = pages->get();
        BatchKVCacheResource resource;
        resource.resetBatchSize(1);
        resource.initGroups(env.cache_config.topologyPtr());
        resource.setBatchCacheKeys(0, {101, 102, 103});
        auto ids = torch::full({static_cast<int64_t>(tags.size()), 1, 3}, NULL_BLOCK_IDX, torch::kInt32);
        std::map<std::string, unsigned char> expected_bytes;
        for (size_t group = 0; group < tags.size(); ++group) {
            resource.setBatchBlocks(0, tags[group], blocks);
            for (size_t slot = 0; slot < blocks.size(); ++slot) {
                ids.data_ptr<int32_t>()[group * 3 + slot] = blocks[slot];
                const unsigned char value = 0x40 + 4 * group + slot;
                expected_bytes.emplace("pace://endpoint/" + tags[group] + "/" + std::to_string(slot), value);
                for (int layer : env.cache_config.layerIdsForGroup(tags[group]))
                    for (const auto& part : environmentBuffers(env, layer, tags[group], blocks[slot]))
                        ASSERT_EQ(cudaMemsetAsync(part.addr, value, part.size_bytes,
                            at::cuda::getCurrentCUDAStream().stream()), cudaSuccess);
            }
        }
        // Real capture owns/fences these synthetic writers. This is not a
        // model-kernel, CUDA-call-count, probe-error or performance test.
        ASSERT_EQ(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream().stream()), cudaSuccess);
        resource.dspark_decode_publication_unit = 8;
        resource.dspark_release_computed_end = 19;
        resource.decode_endpoint = manager->retainReleasedDSparkPages(resource, ids, ids, all_resident_state);
        ASSERT_TRUE(resource.decode_endpoint);
        std::weak_ptr<const DecodeEndpoint> endpoint = resource.decode_endpoint;
        block_tree_cache_test::CallbackBarrier barrier;
        std::map<std::string, std::string> specs;
        auto client = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Invoke(
            [&](const auto&, auto, const auto& registrations, const auto& registered_tags) {
                for (size_t i = 0; i < registered_tags.size(); ++i)
                    specs.emplace(registered_tags[i], registrations[i].location_spec_name);
                return true;
            }));
        EXPECT_CALL(*client, shutdown()).Times(1);
        EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, _)).WillOnce(Invoke(
            [&](const auto&, const auto&, const auto& keys, const auto&, const auto&, auto, auto) {
                EXPECT_EQ(keys, (CacheKeysType{101, 102}));
                barrier.enterAndWait();
                kv_cache_manager::WriteLocation location;
                location.write_session_id = "captured-endpoint";
                location.block_mask = kv_cache_manager::BlockMaskOffset{0};
                for (size_t key = 0; key < keys.size(); ++key) {
                    kv_cache_manager::Location row;
                    for (const auto& group : env.cache_config.topology().groups()) {
                        if (!group.policy.enable_prefix_reuse
                            || (key == 0 && !all_resident_state && group.policy.group_type != CacheGroupType::FULL)) continue;
                        row.push_back({specs.at(group.tag + "#cp0"),
                            "pace://endpoint/" + group.tag + "/" + std::to_string(key)});
                    }
                    location.locations.push_back(std::move(row));
                }
                return std::make_pair(true, location);
            }));
        std::atomic<size_t> saved{0}, finished{0};
        EXPECT_CALL(*client, saveKvCachesForTag(_, _, _, _)).WillRepeatedly(Invoke(
            [&](const auto&, const auto& uris, const auto& buffers, const auto&) {
                EXPECT_FALSE(endpoint.expired());
                EXPECT_EQ(uris.size(), buffers.size());
                kv_cache_manager::UriStrVec actual;
                for (size_t i = 0; i < uris.size(); ++i) {
                    const auto uri = uris[i].substr(0, uris[i].find('?'));
                    for (const auto& iov : buffers[i].iovs) {
                        EXPECT_EQ(iov.type, kv_cache_manager::MemoryType::GPU);
                        std::vector<unsigned char> bytes(iov.size);
                        EXPECT_EQ(cudaMemcpy(bytes.data(), iov.base, iov.size, cudaMemcpyDeviceToHost), cudaSuccess);
                        EXPECT_TRUE(std::all_of(bytes.begin(), bytes.end(),
                            [&](auto byte) { return byte == expected_bytes.at(uri); }));
                    }
                    actual.push_back("actual_" + uris[i]);
                }
                saved += uris.size();
                return std::make_pair(true, actual);
            }));
        EXPECT_CALL(*client, finishWrite(_, _, _, _, _)).WillOnce(Invoke(
            [&](const auto&, const auto&, const auto&, const auto& mask, const auto& locations) {
                EXPECT_EQ(std::get<kv_cache_manager::BlockMaskOffset>(mask), 2u);
                EXPECT_EQ(locations.size(), 2u);
                if (locations.size() == 2) {
                    EXPECT_EQ(locations[0].size(), all_resident_state ? 3u : 1u);
                    EXPECT_EQ(locations[1].size(), 3u);
                }
                for (const auto& row : locations)
                    for (const auto& item : row) EXPECT_EQ(item.uri.rfind("actual_", 0), 0u);
                ++finished;
                return true;
            }));
        KVCacheConfig settings;
        settings.kvcm_server_address = "unused-test-address";
        settings.kvcm_model_sdk_config = R"([{"type":"pace"}])";
        RuntimeConfig runtime;
        runtime.model_name = "DeepSeekV41";
        SpeculativeExecutionConfig speculation;
        speculation.type = SP_TYPE_DSPARK;
        BackendHandle backend(std::make_unique<MockQuiescentKVCMBackend>(
            env.cache_config, settings, runtime, parallel, speculation, nullptr, true, nullptr, client));
        block_tree_cache_detail::ScopeRollback unblock([&] { barrier.release(); });
        ASSERT_TRUE(initSingleRank(*backend.backend, env));
        BatchKVCacheResource expired;
        expired.recordDecodePublicationEos(BatchKVCacheResource::DecodePublicationClock::now() - std::chrono::hours(24));
        EXPECT_FALSE(backend->canStartDecodePublication(expired));
        ASSERT_TRUE(backend->captureCompletedDecode(resource, 8, false));
        ASSERT_TRUE(backend->captureCompletedDecode(resource, 16, false));
        EXPECT_EQ(saved.load(), 0u);
        ASSERT_TRUE(backend->captureCompletedDecode(resource, 19, true));
        ASSERT_TRUE(barrier.waitUntilEnteredFor(1, std::chrono::seconds(5)));
        ASSERT_TRUE(backend->captureCompletedDecode(resource, 19, true)); // Once guard covers all endpoints.
        resource.clearBlocks();
        pages.reset();
        EXPECT_FALSE(endpoint.expired()); // Only the actual publication task owns the endpoint now.
        barrier.release();
        ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
        backend->shutdown(); // Joins callbacks before checking final owner/ref release.
        EXPECT_EQ(saved.load(), all_resident_state ? 6u : 4u);
        EXPECT_EQ(finished.load(), 1u);
        EXPECT_TRUE(endpoint.expired());
        for (const auto& [tag, pool] : env.pools_by_tag) {
            EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::REMOTE_CONNECTOR), 0u) << tag;
            EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 0u) << tag;
        }
    });
}

TEST(NativeEndpointPublicationTest, OnlyTerminalDsparkStateIsPublishedAndOutlivesTheStream) {
    verifyReleasedDsparkStatePublication(false);
}

TEST(NativeEndpointPublicationTest, AllResidentDsparkStatesUseTheirOwnBytesInOnePublication) {
    verifyReleasedDsparkStatePublication(true);
}

TEST_P(EosGpuCompletionTest, DisabledAtInitializationDoesNotPublishPinOrChangeNativePages) {
    for (const bool unset : {false, true}) {
        SCOPED_TRACE(unset);
        autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
        ::rtp_llm::test::withDecodePublicationStartup([&] {
        if (unset) {
            ASSERT_EQ(unsetenv("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH"), 0);
        }
        Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()));
        std::map<std::string, std::vector<size_t>> refs;
        for (const auto& tag : {std::string("full"), std::string("state")}) {
            const auto& blocks = tag == "full" ? s.full : s.state;
            for (const auto block : blocks) refs[tag].push_back(s.env.pools_by_tag.at(tag)->refCount(block));
        }
        ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), false));
        ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
        ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
        EXPECT_EQ(s.starts, 0);
        EXPECT_EQ(s.saves, 0);
        EXPECT_EQ(s.finishes, 0);
        EXPECT_EQ(s.aborts, 0);
        EXPECT_FALSE(s.resource.decodeCompletionSubmitted());
        EXPECT_FALSE(s.resource.decode_endpoint);
        EXPECT_TRUE(s.resource.decode_provisional_endpoints.empty());
        for (const auto& tag : {std::string("full"), std::string("state")}) {
            const auto& blocks = tag == "full" ? s.full : s.state;
            const auto pool = s.env.pools_by_tag.at(tag);
            for (size_t i = 0; i < blocks.size(); ++i) {
                EXPECT_EQ(pool->refCount(blocks[i]), refs[tag][i]);
                EXPECT_FALSE(pool->isFrozen(blocks[i]));
                for (const auto& part : environmentBuffers(s.env, tag == "full" ? 0 : 1, tag, blocks[i])) {
                    std::vector<unsigned char> bytes(part.size_bytes);
                    ASSERT_EQ(cudaMemcpy(bytes.data(), part.addr, bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
                    const auto expected = tag == "full" ? i + 1 : 100 + i;
                    EXPECT_TRUE(std::all_of(bytes.begin(), bytes.end(), [&](auto value) { return value == expected; }));
                }
            }
        }
        s.dropRequestRefs();
        for (const auto block : s.full) EXPECT_FALSE(s.env.pools_by_tag.at("full")->isAllocated(block));
        for (const auto block : s.state) EXPECT_FALSE(s.env.pools_by_tag.at("state")->isAllocated(block));
        });
    }
}

TEST_P(EosGpuCompletionTest, RuntimeDisableSkipsEosBeforePinsAndAllowsLaterPublication) {
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()));
    const auto pool = s.env.pools_by_tag.at("full");
    const auto refs = pool->refCount(s.full.front());
    s.backend->setRemoteCacheEnabled(false);
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    EXPECT_EQ(s.starts, 0);
    EXPECT_EQ(s.saves, 0);
    EXPECT_EQ(s.finishes, 0);
    EXPECT_FALSE(s.resource.decodeCompletionSubmitted());
    EXPECT_EQ(pool->refCount(s.full.front()), refs);

    s.backend->setRemoteCacheEnabled(true);
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    EXPECT_EQ(s.starts, 1);
    EXPECT_GT(s.saves, 0);
    EXPECT_EQ(s.finishes, 1);
    EXPECT_EQ(s.aborts, 0);
    EXPECT_EQ(pool->refCount(s.full.front()), refs);
    });
}

TEST_P(EosGpuCompletionTest, NoProgressSnapshotAndNativePagesSurviveRequestRelease) {
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()));
    s.block_start = true;
    for (size_t i = 0; i < s.state.size(); ++i)
        s.fill(1, "state", s.state[i], 0xee);
    for (size_t n = 128; n <= s.end(); n += 128)
        ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, n, false));
    EXPECT_EQ(s.starts, 0);
    EXPECT_EQ(s.saves, 0);
    for (size_t i = 0; i < s.state.size(); ++i)
        s.fill(1, "state", s.state[i], 100 + i);
    // Change the final state after all progress callbacks. EOS must read this
    // GPU state, not any early host snapshot. Values remain same per-key here.
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    ASSERT_EQ(s.started.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    ASSERT_TRUE(s.resource.decodeCompletionSubmitted());
    s.dropRequestRefs();
    s.resource.clearBlocks();
    for (int i = 0; i < s.count * s.shards; ++i)
        EXPECT_TRUE(s.env.pools_by_tag.at("full")->isAllocated(s.full[i]));
    EXPECT_TRUE(s.env.pools_by_tag.at("state")->isAllocated(s.state[1]));
    EXPECT_TRUE(s.env.pools_by_tag.at("state")->isAllocated(s.state[2]));
    EXPECT_FALSE(s.env.pools_by_tag.at("state")->isAllocated(s.state[0]));
    s.unblock();
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    EXPECT_EQ(s.finishes, 1);
    EXPECT_EQ(s.aborts, 0);
    for (auto block : s.full)
        EXPECT_FALSE(s.env.pools_by_tag.at("full")->isAllocated(block));
    for (auto block : s.state)
        EXPECT_FALSE(s.env.pools_by_tag.at("state")->isAllocated(block));
    });
}
TEST_P(EosGpuCompletionTest, PartialTailUsesEarlierImmutableCheckpointNotFinalState) {
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()));
    // Active, incomplete final slot intentionally differs from complete state.
    s.fill(1, "state", s.state.back(), 0xee);
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end() + 128 * s.shards - 1, true));
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    EXPECT_EQ(s.finishes, 1);
    EXPECT_EQ(s.aborts, 0);
    });
}
TEST_P(EosGpuCompletionTest, NativeSsdGdrKeepsGpuPagesThroughEosTransfer) {
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, true, std::get<2>(GetParam()));
    s.block_start = true;
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    ASSERT_EQ(s.started.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    s.dropRequestRefs();
    s.resource.clearBlocks();
    for (int i = 0; i < s.count * s.shards; ++i)
        EXPECT_TRUE(s.env.pools_by_tag.at("full")->isAllocated(s.full[i]));
    s.unblock();
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    EXPECT_EQ(s.finishes, 1);
    EXPECT_EQ(s.aborts, 0);
    for (auto block : s.full)
        EXPECT_FALSE(s.env.pools_by_tag.at("full")->isAllocated(block));
    });
}
TEST_P(EosGpuCompletionTest, MissingStateBacksOffWithoutRelabeling) {
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()));
    auto     state = s.state;
    state[2]       = NULL_BLOCK_IDX;
    s.resource.setBatchBlocks(0, "state", state);
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    EXPECT_EQ(s.finishes, 1);
    EXPECT_EQ(s.aborts, 0);
    });
}
TEST_P(EosGpuCompletionTest, FullPrefixHolesAndTruncationKeepLargestCompleteCheckpoint) {
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    for (int variant = 0; variant < 8; ++variant) {
        SCOPED_TRACE(variant);
        Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()));
        auto full = s.full;
        size_t expected_keys = 0;
        switch (variant) {
        case 0: full[0] = NULL_BLOCK_IDX; break;
        case 1: full[s.shards] = NULL_BLOCK_IDX; expected_keys = 1; break;
        case 2: full[3 * s.shards - 1] = NULL_BLOCK_IDX; expected_keys = 2; break;
        case 3: full.resize(2 * s.shards); expected_keys = 2; break;
        case 4:
            // An odd/native base column inside a CP canonical block, not a
            // canonical boundary. CP1 is only the degenerate control.
            full[s.shards > 1 ? s.shards + 1 : 1] = NULL_BLOCK_IDX;
            expected_keys = 1;
            break;
        case 5: {
            full[3 * s.shards - 1] = NULL_BLOCK_IDX; // FULL permits two canonical keys.
            auto state = s.state;
            state[1] = NULL_BLOCK_IDX; // The unchanged terminal check must back off to one.
            s.resource.setBatchBlocks(0, "state", state);
            expected_keys = 1;
            break;
        }
        case 6: full[3 * s.shards] = NULL_BLOCK_IDX; expected_keys = 3; break; // Uncomputed tail ignored.
        case 7: full.clear(); break;
        }
        s.resource.setBatchBlocks(0, "full", full);
        ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
        ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
        EXPECT_EQ(s.requested_keys.load(), expected_keys);
        EXPECT_EQ(s.starts.load(), expected_keys ? 1 : 0);
        EXPECT_EQ(s.finishes.load(), expected_keys ? 1 : 0);
        EXPECT_EQ(s.aborts.load(), 0);
        EXPECT_EQ(s.resource.decodeCompletionSubmitted(), expected_keys != 0);
        if (expected_keys) {
            EXPECT_GT(s.saves.load(), 0);
        } else {
            EXPECT_EQ(s.saves.load(), 0);
        }
        for (const auto& [tag, pool] : s.env.pools_by_tag) {
            EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::REMOTE_CONNECTOR), 0u) << tag;
            EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 0u) << tag;
        }
    }
    });
}
TEST_P(EosGpuCompletionTest, KnownPrefillBoundarySkipsCheckpointWithoutNewDecodeKv) {
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    for (const bool checkpoint_backoff : {false, true}) {
        SCOPED_TRACE(checkpoint_backoff);
        Scenario     s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()));
        const size_t unit    = 128 * s.shards;
        const size_t handoff = checkpoint_backoff ? s.end() - unit : s.end();
        ASSERT_TRUE(s.resource.recordPrefillComputedEnd(handoff));
        if (checkpoint_backoff) {
            auto state = s.state;
            state[2]   = NULL_BLOCK_IDX;
            s.resource.setBatchBlocks(0, "state", state);
        }
        // D has computed a tail beyond P. In the second case the newest full
        // boundary lacks its state page, so only P's older checkpoint is valid.
        ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end() + 17, true));
        ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
        EXPECT_EQ(s.starts, 0);
        EXPECT_EQ(s.saves, 0);
        EXPECT_EQ(s.finishes, 0);
        EXPECT_EQ(s.aborts, 0);
        EXPECT_FALSE(s.resource.decodeCompletionSubmitted());
        for (const auto& [tag, pool] : s.env.pools_by_tag) {
            EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 0u) << tag;
        }
        s.dropRequestRefs();
        for (const auto block : s.full) {
            EXPECT_FALSE(s.env.pools_by_tag.at("full")->isAllocated(block));
        }
        for (const auto block : s.state) {
            EXPECT_FALSE(s.env.pools_by_tag.at("state")->isAllocated(block));
        }
    }
    });
}

TEST_P(EosGpuCompletionTest, UnknownSdkCompletionUsesNativeFrozenCustodyAcrossShutdown) {
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    // Leave the native guard enabled. Even the maximum FULL prefix
    // occupies at most 25% of this fixture's pool, below its 30% lower bound.
    const int physical_shards = std::get<1>(GetParam()) ? std::get<0>(GetParam()) : 1;
    const int pool_blocks = std::max(64, 4 * Scenario::kCheckpointCount * physical_shards);
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()), pool_blocks);
    ASSERT_GE(s.env.pools_by_tag.at("full")->totalBlocksNum(), static_cast<size_t>(4 * s.count * s.shards));
    // initializeEnvironmentPools owns a separate REQUEST sentinel per pool.
    // Account for it independently from this operation's full/state sources.
    std::map<std::string, size_t> fixture_requests;
    for (const auto& [tag, pool] : s.env.pools_by_tag) {
        const auto& sources = tag == "full" ? s.full : s.state;
        const auto count = pool->referencedBlocksNum(DeviceBlockRefType::REQUEST);
        ASSERT_GE(count, sources.size());
        fixture_requests[tag] = count - sources.size();
    }
    auto endpoint = std::make_shared<DecodeEndpoint>();
    endpoint->end = s.end();
    endpoint->blocks["full"] = s.full;
    endpoint->blocks["state"] = s.state;
    auto pin = [&](const std::string& tag, int first, int end, const BlockIndicesType& blocks) {
        for (int i = first; i < end; ++i) {
            auto page = std::make_shared<DecodeEndpoint::PageHold>();
            page->pool = s.env.pools_by_tag.at(tag);
            page->block = blocks[i];
            page->pin = page->pool->pinForRemotePublication(page->block);
            endpoint->pages[tag].push_back(std::move(page));
        }
    };
    pin("full", 0, s.count * s.shards, s.full);
    const auto state_count = s.env.cache_config.topology().group("state").reuseBlockCount(s.count);
    const int first_state = s.count - static_cast<int>(std::min<size_t>(s.count, state_count));
    ASSERT_GT(first_state, 0);
    pin("state", first_state, s.count, s.state);
    s.resource.decode_endpoint = std::move(endpoint);
    s.fail_save = true;
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    s.resource.decode_endpoint.reset(); // Only the in-flight write owns the terminal endpoint now.
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    s.dropRequestRefs();
    EXPECT_EQ(s.finishes, 0);
    // Unknown DMA completion must retain the remote WRITING session as well as
    // local GPU pages. An immediate abort could recycle its destination.
    EXPECT_EQ(s.aborts, 0);
    const auto& state_pool = s.env.pools_by_tag.at("state");
    EXPECT_FALSE(state_pool->isAllocated(s.state.front()));
    for (int i = first_state; i < s.count; ++i) {
        EXPECT_TRUE(state_pool->isFrozen(s.state[i]));
    }
    for (int i = 0; i < s.count * s.shards; ++i) {
        EXPECT_TRUE(s.env.pools_by_tag.at("full")->isAllocated(s.full[i]));
        EXPECT_TRUE(s.env.pools_by_tag.at("full")->isFrozen(s.full[i]));
        EXPECT_EQ(s.env.pools_by_tag.at("full")->treeRefCount(s.full[i]), 1u);
    }
    EXPECT_FALSE(s.backend->protected_quarantine_);
    EXPECT_EQ(s.backend->quarantined_task_count_, 1u);
    EXPECT_FALSE(s.backend->protectUnknownCompletion());
    std::vector<std::function<void()>> release_frozen;
    for (const auto& [tag, pool] : s.env.pools_by_tag) {
        EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::REQUEST), fixture_requests.at(tag)) << tag;
        EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::REMOTE_CONNECTOR), 0u) << tag;
        BlockIdList frozen;
        for (auto block : tag == "full" ? s.full : s.state) {
            if (!pool->isAllocated(block)) continue;
            EXPECT_FALSE(pool->hasRequestRef(block));
            EXPECT_TRUE(pool->isFrozen(block));
            EXPECT_EQ(pool->treeRefCount(block), 1u);
            if (pool->isFrozen(block)) frozen.push_back(block);
        }
        if (!frozen.empty()) release_frozen.push_back(pool->prepareFrozenReferenceRelease(std::move(frozen)));
    }
    EXPECT_EQ(s.starts, 1);
    auto stopped = std::async(std::launch::async, [&] { s.backend->shutdown(); });
    EXPECT_EQ(stopped.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    stopped.get();
    EXPECT_FALSE(s.backend->protected_quarantine_);
    for (int i = 0; i < s.count * s.shards; ++i)
        EXPECT_TRUE(s.env.pools_by_tag.at("full")->isFrozen(s.full[i]));
    // This fixture's synchronous CUDA/SDK mocks have no detached DMA. Only
    // that test-owned proof permits explicitly releasing our native holds.
    for (auto& release : release_frozen) release();
    for (int i = 0; i < s.count * s.shards; ++i)
        EXPECT_FALSE(s.env.pools_by_tag.at("full")->isAllocated(s.full[i]));
    });
}
TEST_P(EosGpuCompletionTest, DuplicateEosAndFullAdmissionDoNotDoublePinOrPublish) {
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()));
    s.block_start = true;
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    ASSERT_EQ(s.started.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    const auto held = s.env.pools_by_tag.at("full")->refCount(s.full[0]);
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    BatchKVCacheResource other        = s.resource;
    other.decodeCompletionSubmitted() = false;
    ASSERT_TRUE(s.backend->captureCompletedDecode(other, s.end(), true));
    EXPECT_FALSE(other.decodeCompletionSubmitted());
    EXPECT_EQ(s.env.pools_by_tag.at("full")->refCount(s.full[0]), held);
    s.unblock();
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    EXPECT_EQ(s.starts, 1);
    EXPECT_EQ(s.finishes, 1);
    });
}
TEST_P(EosGpuCompletionTest, LostCommitAckDoesNotAbortOrQuarantineCompletedGpuTransfer) {
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()));
    s.lose_commit = true;
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    s.dropRequestRefs();
    EXPECT_EQ(s.finishes, 1);
    EXPECT_EQ(s.aborts, 0);
    for (auto block : s.full)
        EXPECT_FALSE(s.env.pools_by_tag.at("full")->isAllocated(block));
    });
}
TEST_P(EosGpuCompletionTest, UnboundedSwaNeverPretendsTwoTailPagesAreEnough) {
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), false, false, std::get<2>(GetParam()));
    auto     state = s.state;
    state[0]       = NULL_BLOCK_IDX;
    s.resource.setBatchBlocks(0, "state", state);
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    EXPECT_EQ(s.starts, 0);
    EXPECT_FALSE(s.resource.decodeCompletionSubmitted());
    });
}
INSTANTIATE_TEST_SUITE_P(GenericCPAndIndependentDP,
                         EosGpuCompletionTest,
                         // Keep every CP/layout policy, including both CP1
                         // policies despite equal geometry, and add CP4/DP4.
                         // DP is independently varied instead of a Cartesian product.
                         ::testing::Values(std::make_tuple(1, false, 1),
                                           std::make_tuple(1, true, 1),
                                           std::make_tuple(2, false, 3),
                                           std::make_tuple(2, true, 1),
                                           std::make_tuple(4, false, 4),
                                           std::make_tuple(4, true, 4),
                                           std::make_tuple(8, false, 8),
                                           std::make_tuple(8, true, 3),
                                           std::make_tuple(16, false, 1),
                                           std::make_tuple(16, true, 8)));
}  // namespace
}  // namespace rtp_llm
