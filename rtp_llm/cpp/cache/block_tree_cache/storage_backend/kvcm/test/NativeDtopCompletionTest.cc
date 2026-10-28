// Native CUDA pages plus a recording SDK. Host copies are assertion oracles,
// never an application snapshot or a real-model/GDR acceptance claim.
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/WorkerPayloadProof.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include <cuda_runtime.h>
#include <map>
#include <sstream>
#include <tuple>

namespace rtp_llm {
namespace {
class NativeDtopCompletionTest: public ::testing::TestWithParam<std::tuple<int, bool, int>> {
protected:
    autil::EnvGuard feature{"RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1"};
    autil::EnvGuard jobs{"RTP_LLM_REMOTE_CACHE_DECODE_PENDING_JOBS", "1"};
    autil::EnvGuard bytes{"RTP_LLM_REMOTE_CACHE_DECODE_PENDING_MIB", "1"};
    struct Scenario {
        int cp, shards;
        static constexpr size_t count = 3;
        BackendEnvironment env;
        BatchKVCacheResource resource;
        std::map<std::string, BlockIndicesType> blocks;
        std::map<std::string, std::string> specs;
        std::shared_ptr<MockClientWrapper> client = std::make_shared<MockClientWrapper>();
        std::unique_ptr<KVCMStorageBackend> backend;
        std::atomic<size_t> starts{0}, saves{0}, finish_attempts{0};
        bool refs{true}, pause{false}, fail_commit{false};
        std::promise<void> entered, released;
        std::future<void> started = entered.get_future();
        std::shared_future<void> release = released.get_future().share();
        std::map<std::string, std::vector<std::vector<std::vector<uint8_t>>>> expected;
        Scenario(int target_cp, bool sharded, int dp, bool enabled = true, bool bounded = true,
                 bool record_handoff = true):
            cp(target_cp), shards(sharded ? cp : 1) {
            auto full = defaultCacheGroupPolicy(CacheGroupType::FULL);
            auto state = defaultCacheGroupPolicy(CacheGroupType::LINEAR);
            auto swa = defaultCacheGroupPolicy(CacheGroupType::SWA);
            for (auto* policy : {&state, &swa}) {
                policy->enable_prefix_reuse = true;
                policy->cp_mapping = sharded ? CpBlockMappingMode::COMPACT_LAST_RANK : CpBlockMappingMode::NONE;
                policy->cp_slice = sharded ? CpBlockSliceMode::EQUAL_BYTES : CpBlockSliceMode::NONE;
            }
            swa.cp_slice = sharded ? CpBlockSliceMode::PAYLOAD_BYTES : CpBlockSliceMode::NONE;
            swa.sliding_window_size = bounded ? 256 * shards : 0;
            auto full_spec = test::makeMhaSpec("full", 128, DataType::TYPE_FP8_E4M3, 1, 2);
            auto state_spec = test::makeMhaSpec("state", 128 * shards, DataType::TYPE_FP16, 1, 2);
            auto swa_base = test::makeMhaSpec("swa", 128 * shards, DataType::TYPE_FP16, 1, 2);
            auto swa_spec = std::make_shared<test::TestLayoutSpec>(*swa_base, swa_base->block_size_bytes() + 64, 0);
            // Packed payload followed by a canary trailer. PAYLOAD_BYTES must
            // not include that trailer in sharded transfer slices.
            swa_spec->k_payload = swa_base->block_size_bytes();
            env.cache_config.dtype = DataType::TYPE_FP16;
            env.cache_config.layer_num = 4;
            env.cache_config.seq_size_per_block = 128;
            env.cache_config.fromGroupedSpecs({full_spec, state_spec, swa_spec}, {{0, 1}, {2}, {3}},
                {CacheGroupType::FULL, CacheGroupType::LINEAR, CacheGroupType::SWA},
                {"full", "state", "swa"}, {full, state, swa});
            // More than 1 MiB of actual native capacity, including CP1. Do not
            // bypass production's explicit-budget <= physical-pool protection.
            env.cache_config.finalizeBlockNums(512, RuntimeConfig{});
            initializeEnvironmentPools(env, "eos_dram_native", 512);
            resource.resetBatchSize(1);
            resource.initGroups(env.cache_config.topologyPtr());
            for (const std::string tag : {"full", "state", "swa"}) {
                const size_t n = (count + 1) * (tag == "full" ? shards : 1);
                blocks[tag] = *env.pools_by_tag.at(tag)->malloc(n);
                env.pools_by_tag.at(tag)->incRef(blocks[tag]);
                resource.setBatchBlocks(0, tag, blocks[tag]);
                for (size_t i = 0; i < n; ++i) {
                    auto& oracle = expected[tag].emplace_back();
                    size_t part_number = 0;
                    for (int layer : env.cache_config.layerIdsForGroup(tag)) {
                        for (const auto& part : environmentBuffers(env, layer, tag, blocks[tag][i])) {
                            auto& data = oracle.emplace_back(part.size_bytes);
                            for (size_t offset = 0; offset < data.size(); ++offset) {
                                data[offset] = static_cast<uint8_t>((layer * 43 + i * 17 + part_number * 7
                                                                   + offset * 13 + offset / 16) % 251);
                            }
                            if (tag == "swa") {
                                std::fill(data.end() - 64, data.end(), 0xdd);
                            }
                            EXPECT_EQ(cudaMemcpy(part.addr, data.data(), data.size(), cudaMemcpyHostToDevice), cudaSuccess);
                            ++part_number;
                        }
                    }
                }
            }
            CacheKeysType keys;
            for (size_t i = 0; i < blocks["full"].size(); ++i) {
                keys.push_back(101 + i);
            }
            resource.setBatchCacheKeys(0, keys);
            if (record_handoff) {
                EXPECT_TRUE(resource.decodeCompletionLifecycle().recordPrefillComputedEnd(unit()));
            }
            EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Invoke(
                [&](const auto&, auto, const auto& registrations, const auto& tags) {
                    EXPECT_EQ(registrations.size(), tags.size());
                    EXPECT_EQ(tags.size(), static_cast<size_t>(3 * (enabled ? cp : 1)));
                    for (size_t i = 0; i < tags.size(); ++i) {
                        specs[tags[i]] = registrations[i].location_spec_name;
                        if (enabled) {
                            EXPECT_TRUE(registrations[i].memory_registrations.has_value());
                            if (registrations[i].memory_registrations) {
                                EXPECT_FALSE(registrations[i].memory_registrations->gpu.empty());
                            }
                        }
                    }
                    return true;
                }));
            EXPECT_CALL(*client, shutdown()).Times(1);
            EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, _)).WillRepeatedly(Invoke(
                [&](const auto&, const auto&, const auto& keys, const auto&, const auto&, auto, auto) {
                    ++starts;
                    if (pause) {
                        entered.set_value();
                        EXPECT_EQ(release.wait_for(std::chrono::seconds(5)), std::future_status::ready);
                    }
                    kv_cache_manager::WriteLocation result;
                    result.write_session_id = "EOS";
                    result.block_mask = kv_cache_manager::BlockMaskOffset{0};
                    for (size_t k = 0; k < keys.size(); ++k) {
                        EXPECT_EQ(keys[k], 100 + (k + 1) * shards);
                        kv_cache_manager::Location row;
                        for (int peer = 0; peer < cp; ++peer) {
                            for (const std::string tag : {"full", "state", "swa"}) {
                                const size_t tail = env.cache_config.topology().group(tag).reuseBlockCount(keys.size());
                                if (keys.size() - k <= tail) {
                                    row.push_back({specs.at(tag + "#cp" + std::to_string(peer)),
                                                   "pace://" + tag + "/" + std::to_string(k) + "/" + std::to_string(peer)});
                                }
                            }
                        }
                        result.locations.push_back(std::move(row));
                    }
                    return std::make_pair(true, result);
                }));
            EXPECT_CALL(*client, saveKvCachesForTag(_, _, _, _)).WillRepeatedly(Invoke(
                [&](const auto& tag, const auto& uris, const auto& buffers, const auto&) {
                    ++saves;
                    // Preserve upstream7cf: ONE PACE call for the whole peer,
                    // across FULL/state/SWA, through the first registered client.
                    EXPECT_EQ(tag, "full#cp0");
                    EXPECT_EQ(uris.size(), buffers.size());
                    kv_cache_manager::UriStrVec actual;
                    for (size_t i = 0; i < uris.size(); ++i) {
                        std::istringstream route(uris[i].substr(7));
                        std::string group, key_text, peer_text;
                        std::getline(route, group, '/'); std::getline(route, key_text, '/'); std::getline(route, peer_text);
                        const size_t key = std::stoul(key_text), peer = std::stoul(peer_text);
                        const size_t slot = group == "full" ? key * shards + (shards > 1 ? peer : 0) : key;
                        const auto& oracle = expected.at(group).at(slot);
                        EXPECT_EQ(buffers[i].iovs.size(), oracle.size());
                        for (size_t p = 0; p < buffers[i].iovs.size(); ++p) {
                            const auto& iov = buffers[i].iovs[p];
                            EXPECT_EQ(iov.type, kv_cache_manager::MemoryType::GPU);
                            size_t bytes = oracle[p].size(), offset = 0;
                            if (group != "full" && shards > 1) {
                                bytes = (group == "swa" ? bytes - 64 : bytes) / shards;
                                offset = bytes * peer;
                            }
                            EXPECT_EQ(iov.size, bytes);
                            std::vector<uint8_t> observed(iov.size);
                            EXPECT_EQ(cudaMemcpy(observed.data(), iov.base, iov.size, cudaMemcpyDeviceToHost), cudaSuccess);
                            EXPECT_EQ(observed, std::vector<uint8_t>(oracle[p].begin() + offset,
                                                                    oracle[p].begin() + offset + bytes));
                        }
                        actual.push_back(uris[i] + "_actual");
                    }
                    return std::make_pair(true, actual);
                }));
            EXPECT_CALL(*client, finishWrite(_, _, _, _, _)).WillRepeatedly(Invoke(
                [&](const auto&, const auto&, const auto&, const auto& mask, const auto& locations) {
                    if (!locations.empty()) {
                        ++finish_attempts;
                        EXPECT_EQ(std::get<kv_cache_manager::BlockMaskOffset>(mask), locations.size());
                        for (const auto& row : locations) {
                            for (const auto& spec : row) {
                                EXPECT_NE(spec.uri.find("_actual"), std::string::npos);
                            }
                        }
                    }
                    return !fail_commit;
                }));
            KVCacheConfig kv;
            kv.kvcm_server_address = "unused";
            kv.kvcm_instance_group = "decode_eos_dram_test";
            kv.kvcm_instance_id_salt = "eos_dram";
            kv.kvcm_model_sdk_config = "[{\"type\":\"pace\",\"sdk_log_level\":\"INFO\"}]";
            ParallelismConfig parallel;
            parallel.tp_size = 1; parallel.dp_size = dp; parallel.ep_size = dp;
            parallel.role_type = RoleType::DECODE;
            parallel.prefill_cp_config.prefill_cp_size = cp;
            parallel.prefill_cp_config.kv_cache_sharded = sharded;
            RuntimeConfig rt; rt.model_name = "eos_dram";
            // GDR intentionally OFF: application IOVs remain native GPU pages;
            // the unchanged original SDK may internally stage its transport.
            backend = std::make_unique<KVCMStorageBackend>(env.cache_config, kv, rt, parallel,
                SpeculativeExecutionConfig{}, nullptr, false, nullptr, client);
            EXPECT_TRUE(initSingleRank(*backend, env));
        }
        ~Scenario() {
            unblock();
            if (backend) backend->shutdown();
            dropRefs();
        }
        size_t unit() const { return 128 * shards; }
        size_t end() const { return count * unit(); }
        void unblock() { try { released.set_value(); } catch (...) {} }
        void dropRefs() {
            if (refs) {
                for (const auto& entry : blocks) env.pools_by_tag.at(entry.first)->decRef(entry.second);
                refs = false;
            }
        }
    };
};
TEST_P(NativeDtopCompletionTest, GpuSourcesAndNonuniformSlicesReuseOnePaceCallPerPeer) {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()));
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end()));
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    EXPECT_EQ(s.finish_attempts, 1u); EXPECT_EQ(s.saves, static_cast<size_t>(s.cp));
}
TEST_P(NativeDtopCompletionTest, RequestReleaseCannotRecycleQueuedSourcePages) {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()));
    s.pause = true;
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end()));
    ASSERT_EQ(s.started.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    s.dropRefs(); s.resource.clearBlocks();
    for (size_t i = 0; i < Scenario::count * s.shards; ++i)
        EXPECT_TRUE(s.env.pools_by_tag.at("full")->isAllocated(s.blocks.at("full")[i]));
    s.unblock(); ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    EXPECT_EQ(s.finish_attempts, 1u);
}
TEST_P(NativeDtopCompletionTest, PartialTailPublishesPriorCheckpointAndState) {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()));
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end() + s.unit() - 1));
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend)); EXPECT_EQ(s.finish_attempts, 1u);
}
TEST_P(NativeDtopCompletionTest, MissingTerminalStateBacksOffWithoutRelabeling) {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()));
    auto slots = s.blocks.at("state"); slots[2] = NULL_BLOCK_IDX;
    s.resource.setBatchBlocks(0, "state", slots);
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end()));
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend)); EXPECT_EQ(s.finish_attempts, 1u);
}
TEST_P(NativeDtopCompletionTest, NoProgressOrNoNewCheckpointStartsNoIo) {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()));
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), false));
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.unit() + 1));
    EXPECT_EQ(s.starts, 0u); EXPECT_FALSE(s.resource.decodeCompletionSubmitted());
}
TEST_P(NativeDtopCompletionTest, DuplicateEosAndJobBudgetCannotDoublePublish) {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()));
    s.pause = true; ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end()));
    ASSERT_EQ(s.started.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end()));
    auto another = s.resource; another.decodeCompletionSubmitted() = false;
    ASSERT_TRUE(s.backend->captureCompletedDecode(another, s.end()));
    EXPECT_FALSE(another.decodeCompletionSubmitted());
    s.unblock(); ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend)); EXPECT_EQ(s.starts, 1u);
}
TEST_P(NativeDtopCompletionTest, LostCommitAckIsNotUnknownSourceDma) {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()));
    s.fail_commit = true; ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end()));
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    s.dropRefs();
    for (const auto& entry : s.blocks)
        for (auto block : entry.second) EXPECT_FALSE(s.env.pools_by_tag.at(entry.first)->isFrozen(block));
}
TEST_P(NativeDtopCompletionTest, UnboundedSwaRequiresEveryHistoricalState) {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()), true, false);
    auto slots = s.blocks.at("swa"); slots[0] = NULL_BLOCK_IDX; s.resource.setBatchBlocks(0, "swa", slots);
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end()));
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend)); EXPECT_EQ(s.starts, 0u);
}
TEST_P(NativeDtopCompletionTest, MissingHandoffAndMultiBatchDoNotGuessSourceBoundary) {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()), true, true, false);
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end()));
    EXPECT_EQ(s.starts, 0u);
    s.resource.resetBatchSize(2);
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end()));
    EXPECT_EQ(s.starts, 0u);
}
TEST_P(NativeDtopCompletionTest, DisabledFeatureIgnoresUnusedBudgetAndRetainsNoResources) {
    autil::EnvGuard off("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    autil::EnvGuard invalid("RTP_LLM_REMOTE_CACHE_DECODE_PENDING_MIB", "invalid");
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()), false);
    EXPECT_FALSE(s.backend->protectUnknownCompletion());
    EXPECT_EQ(s.backend->completionResourceOwner(), nullptr);
    const auto before = s.env.pools_by_tag.at("full")->refCount(s.blocks.at("full").front());
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end()));
    EXPECT_EQ(s.starts, 0u);
    EXPECT_EQ(s.env.pools_by_tag.at("full")->refCount(s.blocks.at("full").front()), before);
}
INSTANTIATE_TEST_SUITE_P(GenericCPAndIndependentDP, NativeDtopCompletionTest,
    ::testing::Combine(::testing::Values(1, 2, 4, 8), ::testing::Bool(), ::testing::Values(1, 3)));

TEST(WorkerPayloadProofTest, AllWorkersMustExplicitlyProveSafety) {
    kvcm::WorkerPayloadProof proof;
    proof.observe(true, true, false);
    proof.observe(false, true, false);  // old worker's default field is false
    EXPECT_EQ(proof.result(), kvcm::WorkerPayloadProof::Result::UNKNOWN);
}
TEST(WorkerPayloadProofTest, SafeFailureDoesNotMaskAnotherWorkersUnknown) {
    kvcm::WorkerPayloadProof proof;
    proof.observe(true, false, true);  // NOT_SUBMITTED or completed payload
    proof.observe(false, false, true); // genuinely unknown payload completion
    EXPECT_EQ(proof.result(), kvcm::WorkerPayloadProof::Result::UNKNOWN);
}
TEST(WorkerPayloadProofTest, AllSafeFailedWorkersAllowSourceRelease) {
    kvcm::WorkerPayloadProof proof;
    proof.observe(true, true, false);
    proof.observe(true, false, true);
    EXPECT_EQ(proof.result(), kvcm::WorkerPayloadProof::Result::BUFFER_SAFE_FAILURE);
}
TEST(WorkerPayloadProofTest, AllSafeSuccessfulWorkersComplete) {
    kvcm::WorkerPayloadProof proof;
    proof.observe(true, true, false);
    proof.observe(true, true, false);
    EXPECT_EQ(proof.result(), kvcm::WorkerPayloadProof::Result::SUCCESS);
}
TEST(WorkerPayloadProofTest, LegacyUnspecifiedStatusIsNotProofWhenEnabled) {
    kvcm::WorkerPayloadProof proof;
    proof.observe(true, false, false);
    EXPECT_EQ(proof.result(), kvcm::WorkerPayloadProof::Result::UNKNOWN);
}

// Exercise the production proof-to-exception selection through StorageBackend's
// real callback error mapping. No SDK is submitted; this is not a DMA fence.
class ProofErrorBackend final: public StorageBackend {
public:
    explicit ProofErrorBackend(bool timeout): timeout_(timeout) {}
    ~ProofErrorBackend() override { shutdown(); }
protected:
    bool initImpl() override { return true; }
    StorageMatchResult matchImpl(const StorageRequest&) override { return {}; }
    void writeImpl(const StorageRequest&) override {}
    void readImpl(const StorageRequest&, const std::shared_ptr<StorageBackendMatchMeta>&) override {
        kvcm::WorkerPayloadProof proof;
        proof.observe(true, true, false);
        proof.observe(false, false, !timeout_);
        std::rethrow_exception(proof.failure(timeout_, safe_, unknown_, timed_out_));
    }
private:
    bool timeout_;
    const std::exception_ptr safe_ = std::make_exception_ptr(StorageOperationBufferSafeFailure("safe"));
    const std::exception_ptr unknown_ = std::make_exception_ptr(
        StorageOperationCompletionUnknown("unknown", StorageQuarantineReason::INVALID_RESPONSE));
    const std::exception_ptr timed_out_ = std::make_exception_ptr(
        StorageOperationTimeout("SDK timeout", StorageQuarantineReason::SDK_TIMEOUT));
};
TEST(WorkerPayloadProofTest, UnknownSdkTimeoutKeepsDeadlineExceededCallback) {
    for (bool timeout : {false, true}) {
        auto env = makeBackendEnvironment(timeout ? "worker_proof_timeout" : "worker_proof_unknown");
        ProofErrorBackend backend(timeout);
        ASSERT_TRUE(backend.init(env.cache_config.topologyPtr(), env.pools_by_tag,
            [&](int layer, const std::string& tag, int block) { return environmentBuffers(env, layer, tag, block); }));
        std::promise<ErrorCode> completed;
        auto result = completed.get_future();
        backend.read(makeGroupedStorageRequest(env, {101}, 0, {env.block_id}), nullptr,
                     [&](ErrorInfo error) { completed.set_value(error.code()); });
        ASSERT_EQ(result.wait_for(std::chrono::seconds(5)), std::future_status::ready);
        EXPECT_EQ(result.get(), timeout ? ErrorCode::DEADLINE_EXCEEDED : ErrorCode::EXECUTION_EXCEPTION);
        backend.shutdown();
    }
}
TEST_P(NativeDtopCompletionTest, WorkerPreflightFailureProvesNoPayloadSubmission) {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()));
    RemoteOperationRequestPB request;
    request.set_op(REMOTE_OPERATION_WRITE);
    request.add_group_tags("unregistered");
    request.add_block_ids(0);
    request.add_uris("pace://unused");
    RemoteOperationResponsePB response;
    EXPECT_FALSE(s.backend->execute(request, response));
    EXPECT_EQ(response.transfer_status(), REMOTE_TRANSFER_STATUS_FAILED);
    EXPECT_TRUE(response.payload_buffer_safe());
    EXPECT_EQ(s.saves, 0u);
    EXPECT_FALSE(s.backend->hasUnresolvedExternalIo());
}
TEST_P(NativeDtopCompletionTest, CompletedWorkerPayloadHasExplicitSafeProof) {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()));
    RemoteOperationRequestPB request;
    request.set_op(REMOTE_OPERATION_WRITE);
    request.add_group_tags("full");
    request.add_block_ids(s.blocks.at("full").front());
    request.add_uris("pace://full/0/0");
    RemoteOperationResponsePB response;
    ASSERT_TRUE(s.backend->execute(request, response));
    EXPECT_EQ(response.transfer_status(), REMOTE_TRANSFER_STATUS_SUCCESS);
    EXPECT_TRUE(response.payload_buffer_safe());
    EXPECT_EQ(s.saves, 1u);
    EXPECT_FALSE(s.backend->hasUnresolvedExternalIo());
}
TEST_P(NativeDtopCompletionTest, DisabledWorkerDoesNotRequireNewResponseProof) {
    autil::EnvGuard off("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()), false);
    RemoteOperationRequestPB request;
    request.set_op(REMOTE_OPERATION_WRITE);
    RemoteOperationResponsePB response;
    EXPECT_TRUE(s.backend->execute(request, response));
    EXPECT_EQ(response.transfer_status(), REMOTE_TRANSFER_STATUS_SUCCESS);
    EXPECT_FALSE(response.payload_buffer_safe()); // default proto value, accepted on legacy route
    EXPECT_EQ(s.saves, 0u);
}
} // namespace
} // namespace rtp_llm
