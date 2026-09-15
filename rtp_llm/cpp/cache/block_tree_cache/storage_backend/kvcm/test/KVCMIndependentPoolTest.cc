#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"
#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheFactory.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/DirectSubscriber.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include <cuda_runtime.h>

namespace rtp_llm {
namespace {

struct PoolTransferState {
    std::vector<kv_cache_manager::RegistSpan>        spans;
    std::vector<const kv_cache_manager::RegistSpan*> retained_spans;
    std::vector<std::string>                         spec_names;
    std::map<std::string, std::vector<uint8_t>>      stored;
    std::vector<size_t>                              writes;
    std::vector<size_t>                              reads;
    size_t                                           destroyed{0};
    int                                              fail_pool{-1};
    kv_cache_manager::Locations                      locations;
};

class RecordingPoolTransfer final: public kv_cache_manager::TransferClient {
public:
    RecordingPoolTransfer(std::shared_ptr<PoolTransferState> state, size_t pool):
        state_(std::move(state)), pool_(pool) {}
    ~RecordingPoolTransfer() override {
        ++state_->destroyed;
    }

    kv_cache_manager::ClientErrorCode LoadKvCaches(const kv_cache_manager::UriStrVec&    uris,
                                                   const kv_cache_manager::BlockBuffers& buffers,
                                                   std::shared_ptr<kv_cache_manager::TransferTraceInfo>) override {
        if (static_cast<int>(pool_) == state_->fail_pool) {
            return kv_cache_manager::ClientErrorCode::ER_INVALID_PARAMS;
        }
        for (size_t index = 0; index < buffers.size(); ++index) {
            const auto& bytes  = state_->stored.at(uris.at(index));
            size_t      offset = 0;
            for (const auto& iov : buffers[index].iovs) {
                if (!checkSpan(iov) || offset > bytes.size() || iov.size > bytes.size() - offset) {
                    ADD_FAILURE() << "read buffer exceeds its registration or stored payload";
                    return kv_cache_manager::ClientErrorCode::ER_INVALID_PARAMS;
                }
                EXPECT_EQ(cudaMemcpy(iov.base, bytes.data() + offset, iov.size, cudaMemcpyHostToDevice), cudaSuccess);
                offset += iov.size;
            }
            EXPECT_EQ(offset, bytes.size());
            ++state_->reads.at(pool_);
        }
        return kv_cache_manager::ClientErrorCode::ER_OK;
    }

    std::pair<kv_cache_manager::ClientErrorCode, kv_cache_manager::UriStrVec>
    SaveKvCaches(const kv_cache_manager::UriStrVec&    uris,
                 const kv_cache_manager::BlockBuffers& buffers,
                 std::shared_ptr<kv_cache_manager::TransferTraceInfo>) override {
        if (static_cast<int>(pool_) == state_->fail_pool) {
            return {kv_cache_manager::ClientErrorCode::ER_INVALID_PARAMS, {}};
        }
        auto actual = uris;
        for (size_t index = 0; index < buffers.size(); ++index) {
            if (pool_ == 0) {
                actual[index] += "_actual";
            }
            auto& bytes = state_->stored[actual[index]];
            bytes.clear();
            for (const auto& iov : buffers[index].iovs) {
                if (!checkSpan(iov)) {
                    return {kv_cache_manager::ClientErrorCode::ER_INVALID_PARAMS, {}};
                }
                const auto offset = bytes.size();
                bytes.resize(offset + iov.size);
                EXPECT_EQ(cudaMemcpy(bytes.data() + offset, iov.base, iov.size, cudaMemcpyDeviceToHost), cudaSuccess);
            }
            ++state_->writes.at(pool_);
        }
        // Both SDK conventions for unchanged URIs must survive batch merging.
        if (pool_ == 1) {
            actual.clear();
        }
        return {kv_cache_manager::ClientErrorCode::ER_OK, std::move(actual)};
    }

protected:
    kv_cache_manager::ClientErrorCode Init(const std::string&, const kv_cache_manager::InitParams&) override {
        return kv_cache_manager::ClientErrorCode::ER_OK;
    }

private:
    bool checkSpan(const kv_cache_manager::Iov& iov) const {
        const auto& span    = state_->spans.at(pool_);
        const auto  begin   = reinterpret_cast<uintptr_t>(span.base);
        const auto  address = reinterpret_cast<uintptr_t>(iov.base);
        const bool  valid =
            address >= begin && address - begin <= span.size && iov.size <= span.size - (address - begin);
        EXPECT_TRUE(valid) << "payload was routed to a client registered for another pool";
        EXPECT_EQ(iov.type, kv_cache_manager::MemoryType::GPU);
        return valid;
    }
    std::shared_ptr<PoolTransferState> state_;
    size_t                             pool_;
};

class KVCMIndependentPoolTest: public DeviceTestBase {
protected:
    void SetUp() override {
        DeviceTestBase::SetUp();
        auto environment                    = makeMultiGroupBackendEnvironment("independent_config", 2, 1, 1);
        config_                             = environment.cache_config;
        config_.use_independent_block_pools = true;
        allocator_                          = std::make_shared<HybridPoolKVCacheAllocator>(config_);
        ASSERT_TRUE(allocator_->init());
        state_ = std::make_shared<PoolTransferState>();
        pools_ = allocator_->groupBlockPools();
        ASSERT_EQ(pools_.size(), 3u);
        for (size_t group = 0; group < pools_.size(); ++group) {
            refs_.push_back(std::make_unique<ScopedReferencedBlocks>(pools_[group], group + 2));
            blocks_.push_back(refs_.back()->get().back());
        }
    }

    void TearDown() override {
        if (backend_) {
            backend_->shutdown();
        }
        cache_.reset();
        backend_.reset();
        wrapper_.reset();
        refs_.clear();
        allocator_.reset();
        DeviceTestBase::TearDown();
    }

    bool initialize(bool fail_second = false, int rank = 0) {
        auto factory = std::make_unique<kvcm::MockClientFactory>();
        auto meta    = std::make_unique<kv_cache_manager::MockMetaClient>();
        meta_        = meta.get();
        EXPECT_CALL(*factory, createSubscriber(false)).WillOnce(Invoke([](bool) {
            return std::make_unique<kvcm::DirectSubscriber>();
        }));
        EXPECT_CALL(*factory, createMetaClient(_, _))
            .WillOnce(Invoke([&, rank](const std::string& json, const auto& params) {
                EXPECT_EQ(params.role_type,
                          rank == 0 ? kv_cache_manager::RoleType::HYBRID : kv_cache_manager::RoleType::SCHEDULER);
                autil::legacy::json::JsonMap object;
                autil::legacy::FromJsonString(object, json);
                std::map<std::string, size_t> sizes;
                autil::legacy::FromJson(sizes, object.at("location_spec_infos"));
                for (size_t group = 0; group < pools_.size(); ++group) {
                    const auto& topology_group = config_.topology().groupById(group);
                    const auto  prefix         = group < 2 ? "F" : "L";
                    for (int tp = 0; tp < (rank == 0 ? 1 : 2); ++tp) {
                        EXPECT_EQ(sizes.at("tp" + std::to_string(tp) + "_" + prefix + topology_group.tag),
                                  config_.blockSizeBytesForGroup(group));
                    }
                }
                EXPECT_NE(config_.blockSizeBytesForGroup(0), config_.blockSizeBytesForGroup(2));
                return std::move(meta);
            }));
        static const std::string storage_config = R"({"sdk_backend_configs":[]})";
        EXPECT_CALL(*meta_, GetStorageConfig()).WillOnce(::testing::ReturnRef(storage_config));
        EXPECT_CALL(*factory, createTransferClient(_, _))
            .Times(fail_second ? 2 : 3)
            .WillRepeatedly(
                Invoke([&, fail_second, rank](const std::string&, const kv_cache_manager::InitParams& params)
                           -> std::unique_ptr<kv_cache_manager::TransferClient> {
                    const auto index = state_->spans.size();
                    if (fail_second && index == 1) {
                        return nullptr;
                    }
                    EXPECT_EQ(params.role_type,
                              index == 0 && rank == 0 ? kv_cache_manager::RoleType::HYBRID :
                                                        kv_cache_manager::RoleType::WORKER);
                    EXPECT_NE(params.regist_span, nullptr);
                    if (!params.regist_span) {
                        return nullptr;
                    }
                    EXPECT_EQ(params.regist_span->base, pools_[index]->getBaseAddress());
                    EXPECT_EQ(params.regist_span->size, pools_[index]->getTotalSizeBytes());
                    state_->spans.push_back(*params.regist_span);
                    state_->retained_spans.push_back(params.regist_span);
                    state_->spec_names.push_back(params.self_location_spec_name);
                    state_->writes.push_back(0);
                    state_->reads.push_back(0);
                    return std::make_unique<RecordingPoolTransfer>(state_, index);
                }));
        wrapper_ = std::make_shared<kvcm::ClientWrapper>(std::move(factory));
        KVCacheConfig options;
        options.enable_remote_cache = true;
        options.kvcm_server_address = "direct";
        auto parallel               = singleRankConfig();
        if (rank != 0) {
            parallel.tp_size = 2;
            parallel.tp_rank = rank;
        }
        backend_ = std::make_shared<KVCMStorageBackend>(
            config_, options, RuntimeConfig{}, parallel, SpeculativeExecutionConfig{}, nullptr, wrapper_);
        if (fail_second || rank != 0) {
            return backend_->init(config_.topologyPtr(), pools_, [&](int layer, int group, int block) {
                return allocator_->convertIndexToBuffer(layer, group, block);
            });
        }
        cache_ = createBlockTreeCache(config_, options, allocator_, parallel, backend_);
        return cache_ != nullptr;
    }

    StorageRequest request() const {
        StorageRequest result;
        result.keys    = std::make_shared<CacheKeysType>(CacheKeysType{101});
        result.handles = {{{2, blocks_[2]}, {1, blocks_[1]}, {0, blocks_[0]}}};
        return result;
    }

    void fill(uint8_t value) {
        for (size_t group = 0; group < pools_.size(); ++group) {
            for (int layer : config_.topology().groupById(group).layer_ids) {
                for (const auto& buffer :
                     allocator_->convertIndexToBuffer(layer, static_cast<int>(group), blocks_[group])) {
                    ASSERT_EQ(cudaMemset(buffer.addr, value == 0 ? 0 : value + group, buffer.size_bytes), cudaSuccess);
                }
            }
        }
    }

    CacheConfig                                          config_;
    std::shared_ptr<HybridPoolKVCacheAllocator>          allocator_;
    std::vector<DeviceBlockPoolPtr>                      pools_;
    std::vector<std::unique_ptr<ScopedReferencedBlocks>> refs_;
    std::vector<BlockIdxType>                            blocks_;
    std::shared_ptr<PoolTransferState>                   state_;
    std::shared_ptr<kvcm::ClientWrapper>                 wrapper_;
    kv_cache_manager::MockMetaClient*                    meta_{nullptr};
    std::shared_ptr<KVCMStorageBackend>                  backend_;
    BlockTreeCachePtr                                    cache_;
};

TEST_F(KVCMIndependentPoolTest, FactoryPublishesHeterogeneousSpecsAndRoundTripsEachPool) {
    ASSERT_TRUE(initialize());
    ASSERT_TRUE(cache_->isRemoteCacheEnabled());
    ASSERT_EQ(state_->spans.size(), 3u);
    for (size_t pool = 0; pool < pools_.size(); ++pool) {
        EXPECT_EQ(state_->retained_spans[pool]->base, pools_[pool]->getBaseAddress());
        EXPECT_EQ(state_->retained_spans[pool]->size, pools_[pool]->getTotalSizeBytes());
    }
    kv_cache_manager::WriteLocation proposal;
    proposal.write_session_id = "session";
    proposal.block_mask       = kv_cache_manager::BlockMaskOffset{0};
    proposal.locations        = {{{"tp0_Llinear0", "linear"}, {"tp0_Ffull1", "full1"}, {"tp0_Ffull0", "full0"}}};
    EXPECT_CALL(*meta_, StartWrite(_, _, _, _, _))
        .WillOnce(Return(std::make_pair(kv_cache_manager::ClientErrorCode::ER_OK, proposal)));
    EXPECT_CALL(*meta_, FinishWrite(_, "session", _, _))
        .WillOnce(Invoke([&](const auto&, const auto&, const auto&, const auto& locations) {
            state_->locations = locations;
            return kv_cache_manager::ClientErrorCode::ER_OK;
        }));
    fill(17);
    backend_->write(backend_->prepareWrite(request()));
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend_));
    ASSERT_EQ(state_->locations.size(), 1u);
    ASSERT_EQ(state_->locations[0].size(), 3u);
    EXPECT_EQ(state_->locations[0][0].uri, "linear");
    EXPECT_EQ(state_->locations[0][1].uri, "full1");
    EXPECT_EQ(state_->locations[0][2].uri, "full0_actual");
    EXPECT_EQ(state_->writes, (std::vector<size_t>{1, 1, 1}));
    EXPECT_CALL(*meta_, MatchLocation(_, _, _, _, _, _, _))
        .WillOnce(Return(std::make_pair(kv_cache_manager::ClientErrorCode::ER_OK, state_->locations)));
    fill(0);
    auto matched = match(*backend_, request());
    ASSERT_EQ(matched.matched_blocks_num, 1u);
    ASSERT_TRUE(read(*backend_, request(), matched.match_meta));
    EXPECT_EQ(state_->reads, (std::vector<size_t>{1, 1, 1}));
    for (size_t group = 0; group < pools_.size(); ++group) {
        for (int layer : config_.topology().groupById(group).layer_ids) {
            for (const auto& buffer :
                 allocator_->convertIndexToBuffer(layer, static_cast<int>(group), blocks_[group])) {
                std::vector<uint8_t> bytes(buffer.size_bytes);
                ASSERT_EQ(cudaMemcpy(bytes.data(), buffer.addr, bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
                EXPECT_EQ(bytes, std::vector<uint8_t>(bytes.size(), 17 + group));
            }
        }
        EXPECT_EQ(pools_[group]->refCount(blocks_[group]), 1u);
    }
    backend_->shutdown();
    EXPECT_EQ(state_->destroyed, 3u);
}

TEST_F(KVCMIndependentPoolTest, PartialWriteFailureAbortsSessionAndReleasesPins) {
    ASSERT_TRUE(initialize());
    kv_cache_manager::WriteLocation proposal;
    proposal.write_session_id = "failed_session";
    proposal.block_mask       = kv_cache_manager::BlockMaskOffset{0};
    proposal.locations        = {{{"tp0_Llinear0", "linear"}, {"tp0_Ffull1", "full1"}, {"tp0_Ffull0", "full0"}}};
    EXPECT_CALL(*meta_, StartWrite(_, _, _, _, _))
        .WillOnce(Return(std::make_pair(kv_cache_manager::ClientErrorCode::ER_OK, proposal)));
    EXPECT_CALL(*meta_,
                FinishWrite(_,
                            "failed_session",
                            ::testing::VariantWith<kv_cache_manager::BlockMaskOffset>(::testing::Eq(0u)),
                            ::testing::IsEmpty()))
        .WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_OK));
    state_->fail_pool = 1;
    fill(29);
    backend_->write(backend_->prepareWrite(request()));
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend_));
    EXPECT_EQ(state_->writes, (std::vector<size_t>{1, 0, 0}));
    for (size_t group = 0; group < pools_.size(); ++group) {
        EXPECT_EQ(pools_[group]->refCount(blocks_[group]), 1u);
    }
    backend_->shutdown();
    EXPECT_EQ(state_->destroyed, 3u);
}

TEST_F(KVCMIndependentPoolTest, FailedSecondRegistrationReleasesAcceptedClients) {
    EXPECT_FALSE(initialize(true));
    EXPECT_EQ(state_->destroyed, 1u);
    wrapper_->shutdown();
    EXPECT_EQ(state_->destroyed, 1u);
}

TEST_F(KVCMIndependentPoolTest, WorkerRegistersAllPoolsAndPropagatesTransferFailure) {
    ASSERT_TRUE(initialize(false, 1));
    EXPECT_EQ(state_->spec_names, (std::vector<std::string>{"tp1_Ffull0", "tp1_Ffull1", "tp1_Llinear0"}));
    fill(31);
    RemoteOperationRequestPB operation;
    operation.set_op(REMOTE_OPERATION_WRITE);
    for (size_t group : {2u, 0u, 1u, 0u}) {
        operation.add_group_tags(config_.topology().groupById(group).tag);
        operation.add_block_ids(blocks_[group]);
        operation.add_uris("group_" + std::to_string(group) + "_" + std::to_string(operation.uris_size()));
    }
    RemoteOperationResponsePB response;
    ASSERT_TRUE(backend_->execute(operation, response));
    ASSERT_EQ(response.actual_uris_size(), 4);
    EXPECT_EQ(response.actual_uris(0), operation.uris(0));
    EXPECT_EQ(response.actual_uris(1), operation.uris(1) + "_actual");
    EXPECT_EQ(response.actual_uris(2), operation.uris(2));
    EXPECT_EQ(response.actual_uris(3), operation.uris(3) + "_actual");
    state_->fail_pool = 1;
    RemoteOperationResponsePB failed;
    EXPECT_FALSE(backend_->execute(operation, failed));
    EXPECT_EQ(failed.actual_uris_size(), 0);
    backend_->shutdown();
    EXPECT_EQ(state_->destroyed, 3u);
}

}  // namespace
}  // namespace rtp_llm
