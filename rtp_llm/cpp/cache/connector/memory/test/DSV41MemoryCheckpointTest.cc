#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <grpcpp/grpcpp.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <map>
#include <numeric>
#include <set>
#include <tuple>

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/DSV41CacheState.h"
#include "rtp_llm/cpp/cache/DSV41KVCacheSpec.h"
#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.h"
#include "rtp_llm/cpp/cache/connector/memory/MemoryAsyncContext.h"
#include "rtp_llm/cpp/utils/HashUtil.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm::test {
namespace {

DSV41CacheIdentity identity() {
    return {"2bc89ac599031fa673cab993f1df02fc4a98c673", "test-layout", DSV41ReplayMode::FULL, 1, 128, 136};
}

DSV41CheckpointMetadata checkpoint(const DSV41CacheIdentity& id, int64_t end) {
    DSV41CheckpointMetadata value;
    value.identity         = id;
    value.materialized_end = value.encoder_materialized_end = value.decoder_checkpoint_end = end;
    value.aux_valid_start                                                                  = end - 128;
    value.aux_valid_end                                                                    = end;
    value.global_entries = value.index_entries = {end / 2, end / 2, end / 2, end};
    for (size_t layer = 0; layer < 43; ++layer) {
        const int64_t floor = id.replay_mode == DSV41ReplayMode::BOUNDED_CHECKPOINT_V1 && layer > 20 ? end - 128 : 0;
        value.swa[layer]    = {end - 128, end, floor};
    }
    value.history_token_ids  = {1042, 2043, 3044};
    value.history_image_mask = {0, 1, 0};
    value.history_ready = value.draft_committed = value.pair_empty = true;
    return value;
}

ModelConfig model(bool draft) {
    ModelConfig value;
    value.num_layers                = draft ? 3 : 40;
    value.hidden_size               = 5120;
    value.max_seq_len               = 1048576;
    value.dsv41_model_revision      = identity().model_revision;
    auto& attn                      = value.attn_config;
    attn.dsv41_cache_layout_version = 1;
    attn.head_num                   = 64;
    attn.kv_head_num                = 1;
    attn.size_per_head              = 512;
    attn.sliding_window             = 128;
    attn.indexer_head_dim           = 128;
    attn.indexer_head_num           = 32;
    attn.indexer_topk               = 512;
    for (int layer = 0; layer < value.num_layers; ++layer)
        attn.layer_compress_ratios.push_back(draft || layer < 2 ? 0 : (layer < 20 ? 2 : 1));
    return value;
}

class MemoryMeta final: public Meta {
public:
    bool enableMemoryCache() const override {
        return true;
    }
    bool enableRemoteCache() const override {
        return false;
    }
    const std::string& trace_id() const override {
        return id_;
    }
    const std::string& unique_id() const override {
        return id_;
    }
    const std::vector<int64_t>& tokens() const override {
        return tokens_;
    }

private:
    std::string          id_{"dsv41-memory-gpu-test"};
    std::vector<int64_t> tokens_;
};

// The production RPC performs a real GPU copy before optional failure injection.
class CopyService final: public RpcService::Service {
public:
    KVCacheMemoryConnector* connector{nullptr};
    std::atomic<bool>       fail_next{false};
    std::atomic<size_t>     copied_requests{0};
    grpc::Status
    ExecuteFunction(grpc::ServerContext*, const FunctionRequestPB* request, FunctionResponsePB* response) override {
        try {
            if (!connector || !request->has_mem_request())
                return {grpc::StatusCode::INVALID_ARGUMENT, "missing copy"};
            if (connector->copyCache(request->mem_request(), *response->mutable_mem_response()))
                ++copied_requests;
            if (fail_next.exchange(false))
                response->mutable_mem_response()->set_success(false);
            return grpc::Status::OK;
        } catch (const std::exception& error) {
            return {grpc::StatusCode::INTERNAL, error.what()};
        }
    }
};

void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

}  // namespace

TEST(DSV41CacheStateTest, MissingOwnerAuxHistoryPairOrOneDraftSwaRejectsCheckpoint) {
    const auto valid = checkpoint(identity(), 256);
    auto       value = valid;
    value.index_entries[3]--;
    EXPECT_THROW(value.validate(128), std::invalid_argument);
    value = valid;
    value.global_entries[0]--;
    EXPECT_THROW(value.validate(128), std::invalid_argument);
    value = valid;
    value.swa[42].valid_start++;
    EXPECT_THROW(value.validate(128), std::invalid_argument);
    value = valid;
    value.aux_valid_end--;
    EXPECT_THROW(value.validate(128), std::invalid_argument);
    value            = valid;
    value.pair_empty = false;
    EXPECT_THROW(value.validate(128), std::invalid_argument);
    value               = valid;
    value.history_ready = false;
    EXPECT_THROW(value.validate(128), std::invalid_argument);
    EXPECT_NO_THROW(valid.validate(128));
}

TEST(DSV41CacheStateTest, ModeTailPolicyAndActualRangesFollowRestoredState) {
    auto full           = identity();
    auto bounded        = full;
    bounded.replay_mode = DSV41ReplayMode::BOUNDED_CHECKPOINT_V1;
    DSV41CacheState state(bounded);
    EXPECT_THROW(state.restore(checkpoint(full, 256), 128), std::invalid_argument);
    const auto metadata = checkpoint(bounded, 256);
    state.restore(metadata, 128);
    EXPECT_TRUE(*state.view().completed == metadata);
    EXPECT_EQ(state.view().completed->swa[21].replay_floor, 128);
    EXPECT_EQ(state.view().target_ready_end, 0);
    bounded.tail_policy_version = 2;
    EXPECT_THROW(DSV41CacheState invalid(bounded), std::invalid_argument);
}

TEST(DSV41CacheStateTest, SwaRangeUsesThePhysicalCapacityInItsLayoutIdentity) {
    auto id                   = identity();
    id.physical_swa_entries   = 134;
    auto value                = checkpoint(id, 256);
    value.swa[42].valid_start = 256 - 134;
    EXPECT_NO_THROW(value.validate(128));
    value.swa[42].valid_start--;
    EXPECT_THROW(value.validate(128), std::invalid_argument);
    id.physical_swa_entries = 0;
    EXPECT_THROW(DSV41CacheState invalid(id), std::invalid_argument);
}

class DSV41MemoryCheckpointGpuTest: public ::testing::Test {
protected:
    using ByteKey = std::tuple<size_t, int, size_t>;
    using ByteMap = std::map<ByteKey, std::vector<uint8_t>>;

    virtual uint32_t blockSize() const {
        return 128;
    }
    virtual uint32_t poolBlocks() const {
        return 16;
    }
    virtual bool withDraft() const {
        return true;
    }
    virtual ParallelismConfig parallelism() const {
        return {};
    }
    void SetUp() override {
        int count = 0;
        ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
        ASSERT_GT(count, 0) << "required real GPU memory cache test";
        cudaDeviceProp properties{};
        ASSERT_EQ(cudaGetDeviceProperties(&properties, 0), cudaSuccess);
        ASSERT_EQ(properties.major, 10) << "required CUDA13 Blackwell memory cache test";
        initRuntime(0, false, false, MlaOpsType::AUTO);
        kv_.seq_size_per_block                      = blockSize();
        kv_.test_block_num                          = poolBlocks();
        kv_.dsv4_fixed_pool_blocks                  = 16;
        kv_.enable_memory_cache                     = true;
        kv_.enable_memory_cache_disk                = false;
        kv_.memory_cache_size_mb                    = 64;
        kv_.memory_cache_sync_timeout_ms            = 30000;
        kv_.prefix_tree_memory_state_swa_pool_ratio = 50;
        SpeculativeExecutionConfig sp;
        sp.type              = SP_TYPE_DSPARK;
        sp.gen_num_per_cycle = 5;
        if (withDraft())
            config_ = CacheConfigCreator::createSpConfig(
                model(false), model(true), parallelism(), RuntimeConfig(), kv_, sp, std::nullopt, true, false);
        else
            config_ = CacheConfigCreator::createConfig(model(false), parallelism(), RuntimeConfig(), kv_);
        allocator_ = std::make_shared<HybridPoolKVCacheAllocator>(config_, AllocationType::DEVICE);
        ASSERT_TRUE(allocator_->init());
        grpc::ServerBuilder builder;
        int                 port = 0;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(&service_);
        server_ = builder.BuildAndStart();
        ASSERT_TRUE(server_);
        connector_ = std::make_shared<KVCacheMemoryConnector>(
            config_, kv_, allocator_, std::vector<std::string>{"127.0.0.1:" + std::to_string(port)});
        ASSERT_TRUE(connector_->init());
        service_.connector = connector_.get();
        meta_              = std::make_shared<MemoryMeta>();
    }
    void TearDown() override {
        if (server_) {
            server_->Shutdown();
            server_->Wait();
        }
        service_.connector = nullptr;
        connector_.reset();
        allocator_.reset();
    }
    DSV41CacheIdentity cacheIdentity(DSV41ReplayMode mode = DSV41ReplayMode::FULL) const {
        return connector_->dsv41CacheIdentity(identity().model_revision, mode);
    }
    CacheKeysType keys(size_t blocks, int32_t first, const DSV41CacheIdentity& id) const {
        CacheKeysType result;
        const size_t  cp   = connector_->dsv41DataUnit() / blockSize();
        int64_t       hash = id.cacheKeySeed();
        for (size_t i = 0; i < (blocks + 1) * cp; ++i) {
            int32_t token = first + i;
            hash          = hashInt64Array(hash, &token, &token + 1);
            if ((i + 1) % cp == 0)
                result.push_back(hash);
        }
        return result;
    }
    std::shared_ptr<KVCacheResource>
    resource(size_t blocks, int32_t first, DSV41ReplayMode mode = DSV41ReplayMode::FULL) {
        const auto pools = allocator_->groupBlockPools();
        auto       owned = std::make_shared<std::array<BlockIndicesType, 6>>();
        auto value = std::shared_ptr<KVCacheResource>(new KVCacheResource(), [pools, owned](KVCacheResource* resource) {
            delete resource;
            for (size_t group = 0; group < owned->size(); ++group)
                if (!(*owned)[group].empty())
                    pools[group]->requestFree((*owned)[group]);
        });
        value->initGroups(6,
                          config_.layer_all_num,
                          config_.layer_to_group_id,
                          1,
                          config_.group_types,
                          config_.layer_region_to_group_id);
        value->setCacheKeys(keys(blocks, first, cacheIdentity(mode)));
        value->setCacheKeysAreCpCanonical(connector_->dsv41DataUnit() != blockSize());
        value->setLastBlockAligned(false);
        value->setBlockIdsKeyAligned(true);
        for (size_t group = 0; group < 6; ++group) {
            const bool full = group < 4;
            auto&      ids  = (*owned)[group];
            ids             = pools[group]->malloc(full ? blocks : 1);
            if (ids.size() != (full ? blocks : 1))
                throw std::runtime_error("test GPU pool exhausted");
            BlockIndicesType mapping(blocks + 1, NULL_BLOCK_IDX);
            if (full)
                std::copy(ids.begin(), ids.end(), mapping.begin());
            else
                mapping[blocks - 1] = ids.front();
            value->mutableBlockIds(group).assign(mapping);
        }
        value->setDsv41CacheState(std::make_shared<DSV41CacheState>(cacheIdentity(mode)));
        return value;
    }
    void fill(const std::shared_ptr<KVCacheResource>& value, uint32_t salt, bool fixed_only = false) {
        for (size_t group = fixed_only ? 4 : 0; group < 6; ++group) {
            for (int owner : config_.global_layer_ids[group]) {
                for (size_t index = 0; index < value->blocks(group).size(); ++index) {
                    const auto block = value->blocks(group)[index];
                    if (block <= 0)
                        continue;
                    for (const auto& segment :
                         allocator_->convertIndexToBuffer(owner, config_.group_region_names[group], block)) {
                        std::vector<uint8_t> data(segment.size_bytes);
                        const auto           key = static_cast<uint64_t>(value->cacheKeys()[index]);
                        for (size_t i = 0; i < data.size(); ++i)
                            data[i] = (i * 37 + owner * 11 + group * 17 + key % 251 + salt) % 251;
                        cudaCheck(cudaMemcpy(segment.addr, data.data(), data.size(), cudaMemcpyHostToDevice));
                    }
                }
            }
        }
    }
    ByteMap bytes(const std::shared_ptr<KVCacheResource>& value, size_t count, bool reset_pair) {
        ByteMap result;
        for (size_t group = 0; group < 6; ++group) {
            for (int owner : config_.global_layer_ids[group]) {
                for (size_t index = group < 4 ? 0 : count - 1; index < count; ++index) {
                    auto&        data = result[{group, owner, index}];
                    const size_t slot_index =
                        group < 4 || value->blockIdsAreKeyAligned() ?
                            index :
                            count * connector_->dsv41DataUnit() / connector_->dsv41ReuseUnit() - 1;
                    const auto block = value->blocks(group).at(slot_index);
                    for (const auto& segment :
                         allocator_->convertIndexToBuffer(owner, config_.group_region_names[group], block)) {
                        const auto offset = data.size();
                        data.resize(offset + segment.size_bytes, 0);
                        if (!(group == 4 && reset_pair))
                            cudaCheck(cudaMemcpy(
                                data.data() + offset, segment.addr, segment.size_bytes, cudaMemcpyDeviceToHost));
                    }
                }
            }
        }
        return result;
    }
    void tail(const std::shared_ptr<KVCacheResource>& value, size_t index) {
        const auto end = (uint64_t{value->blockDependencies()[index].ordinal} + 1) * connector_->dsv41DataUnit();
        value->setDsv41RecoveryMetadata(
            index,
            std::make_shared<DSV41CheckpointMetadata>(checkpoint(value->dsv41CacheState()->view().identity, end)));
    }
    bool done(const std::shared_ptr<AsyncContext>& context) {
        if (!context)
            return false;
        context->waitDone();
        return context->success();
    }
    bool write(const std::shared_ptr<KVCacheResource>& value) {
        return done(connector_->asyncWrite(value, meta_));
    }
    bool restore(const std::shared_ptr<KVCacheResource>& value, size_t end) {
        auto match = connector_->asyncMatch(value, meta_);
        if (!match || match->matchedBlockCount() != end)
            return false;
        const int start = value->reuseBlockNum();
        return done(connector_->asyncRead(value, meta_, match, start, end - start));
    }
    std::shared_ptr<KVCacheResource> suffix(const std::shared_ptr<KVCacheResource>& source, size_t begin) {
        auto value = std::make_shared<KVCacheResource>();
        value->initGroups(6,
                          config_.layer_all_num,
                          config_.layer_to_group_id,
                          1,
                          config_.group_types,
                          config_.layer_region_to_group_id);
        value->setCacheKeys(CacheKeysType(source->cacheKeys().begin() + begin, source->cacheKeys().end()));
        value->setBlockDependencies(
            BlockDependenciesType(source->blockDependencies().begin() + begin, source->blockDependencies().end()));
        value->setCacheKeysAreCpCanonical(source->cacheKeysAreCpCanonical());
        value->setLastBlockAligned(source->lastBlockAligned());
        value->setBlockIdsKeyAligned(true);
        value->setDsv41CacheState(source->dsv41CacheState());
        for (int group = 0; group < 6; ++group)
            value->mutableBlockIds(group).assign(
                BlockIndicesType(source->blocks(group).begin() + begin, source->blocks(group).end()));
        for (size_t i = begin; i < source->cacheKeys().size(); ++i)
            value->setDsv41RecoveryMetadata(i - begin, source->dsv41RecoveryMetadata(i));
        return value;
    }
    void copyGpuPrefix(const std::shared_ptr<KVCacheResource>& source,
                       const std::shared_ptr<KVCacheResource>& destination,
                       size_t                                  count) {
        for (size_t group = 0; group < 4; ++group)
            for (int owner : config_.global_layer_ids[group])
                for (size_t index = 0; index < count; ++index) {
                    const auto src = allocator_->convertIndexToBuffer(
                        owner, config_.group_region_names[group], source->blocks(group)[index]);
                    const auto dst = allocator_->convertIndexToBuffer(
                        owner, config_.group_region_names[group], destination->blocks(group)[index]);
                    for (size_t s = 0; s < src.size(); ++s)
                        cudaCheck(cudaMemcpy(dst[s].addr, src[s].addr, src[s].size_bytes, cudaMemcpyDeviceToDevice));
                }
        destination->setDeviceReuseBlockNum(count);
    }

    KVCacheConfig                               kv_;
    CacheConfig                                 config_;
    std::shared_ptr<HybridPoolKVCacheAllocator> allocator_;
    std::shared_ptr<KVCacheMemoryConnector>     connector_;
    std::shared_ptr<MemoryMeta>                 meta_;
    CopyService                                 service_;
    std::unique_ptr<grpc::Server>               server_;
};

TEST_F(DSV41MemoryCheckpointGpuTest, PhysicalSlotsCountOwnersOnceAndKeepAllTargetDraftSwa) {
    const auto slots = connector_->layerRegionSlots();
    ASSERT_EQ(slots.size(), 54);
    size_t global = 0, state = 0;
    for (const auto& slot : slots) {
        if (connector_->kindForSlot(slot) == CacheBlockKind::COMPRESSED_KV)
            global += slot.stride_bytes;
        else
            state += slot.stride_bytes;
    }
    EXPECT_EQ(global, config_.block_size_bytes);
    EXPECT_EQ(state, config_.group_block_size_bytes[4] + config_.group_block_size_bytes[5]);
    EXPECT_TRUE(connector_->usePrefixTreeMemoryCache());
    EXPECT_EQ(connector_->block_pool_, nullptr);
}

TEST_F(DSV41MemoryCheckpointGpuTest, KvOnlyHitPreservesWritableStateAndBothExecutionAxes) {
    auto source = resource(3, 10000);
    fill(source, 7);
    ASSERT_TRUE(write(source));
    auto destination = resource(3, 10000);
    fill(destination, 29);
    auto expected = bytes(destination, 3, false);
    for (const auto& [key, data] : bytes(source, 3, false))
        if (std::get<0>(key) < 4)
            expected[key] = data;
    ASSERT_TRUE(restore(destination, 3));
    EXPECT_EQ(bytes(destination, 3, false), expected);
    EXPECT_EQ(destination->memoryReuseBlockNum(), 3);
    const auto view = destination->dsv41CacheState()->view();
    EXPECT_EQ(view.encoder_materialized_end, 0);
    EXPECT_EQ(view.decoder_checkpoint_end, 0);
    EXPECT_EQ(view.target_ready_end, 0);
    EXPECT_FALSE(view.completed.has_value());
    EXPECT_EQ(destination->dsv41RecoveryMetadata(2), nullptr);
}

TEST_F(DSV41MemoryCheckpointGpuTest, GpuPrefixAndCpuSuffixUseOriginalGlobalOrdinals) {
    auto source = resource(3, 20000);
    fill(source, 11);
    tail(source, 2);
    const auto expected      = bytes(source, 3, true);
    auto       stored_suffix = suffix(source, 1);
    stored_suffix->setDsv41CacheState(nullptr);
    ASSERT_TRUE(write(stored_suffix));
    EXPECT_FALSE(connector_->prefix_block_cache_->contains(source->cacheKeys()[0], CacheBlockKind::COMPRESSED_KV));
    auto destination = resource(3, 20000);
    fill(destination, 89);
    copyGpuPrefix(source, destination, 1);
    auto match = std::dynamic_pointer_cast<MemoryAsyncMatchContext>(connector_->asyncMatch(destination, meta_));
    ASSERT_NE(match, nullptr);
    EXPECT_EQ(match->startReadBlockIndex(), 1);
    EXPECT_EQ(match->readBlockNum(), 2);
    ASSERT_TRUE(done(connector_->asyncRead(destination, meta_, match, 1, 2)));
    EXPECT_EQ(bytes(destination, 3, false), expected);
    EXPECT_EQ(destination->reuseBlockNum(), 3);
    EXPECT_EQ(destination->memoryReuseBlockNum(), 2);
    ASSERT_NE(destination->dsv41RecoveryMetadata(2), nullptr);
    EXPECT_EQ(destination->dsv41RecoveryMetadata(2)->materialized_end, 384);
    EXPECT_EQ(destination->dsv41CacheState()->view().decoder_checkpoint_end, 384);
    EXPECT_EQ(destination->dsv41CacheState()->view().target_ready_end, 384);
}

TEST_F(DSV41MemoryCheckpointGpuTest, SharedPrefixAndDivergingSuffixRemainReadableAcrossConcurrentPlans) {
    auto    first          = resource(2, 30000);
    auto    second         = resource(2, 30000);
    int32_t branch         = 333;
    second->cacheKeys()[1] = hashInt64Array(second->cacheKeys()[0], &branch, &branch + 1);
    branch++;
    second->cacheKeys()[2] = hashInt64Array(second->cacheKeys()[1], &branch, &branch + 1);
    second->rebuildLinearBlockDependencies();
    fill(first, 13);
    fill(second, 13);
    tail(first, 1);
    tail(second, 1);
    ASSERT_TRUE(write(first));
    ASSERT_TRUE(write(second));
    auto a = resource(2, 30000);
    auto b = resource(2, 30000);
    b->setCacheKeys(second->cacheKeys());
    auto a_match = connector_->asyncMatch(a, meta_);
    auto b_match = connector_->asyncMatch(b, meta_);
    ASSERT_NE(a_match, nullptr);
    ASSERT_NE(b_match, nullptr);
    EXPECT_TRUE(
        connector_->prefix_block_cache_->popOldestJointEvictable(CacheBlockKind::STATE_SWA_KV, CacheBackingType::MEMORY)
            .empty());
    ASSERT_TRUE(done(connector_->asyncRead(a, meta_, a_match, 0, 2)));
    ASSERT_TRUE(done(connector_->asyncRead(b, meta_, b_match, 0, 2)));
    EXPECT_EQ(bytes(a, 2, false), bytes(first, 2, true));
    EXPECT_EQ(bytes(b, 2, false), bytes(second, 2, true));
}

TEST_F(DSV41MemoryCheckpointGpuTest, InvalidTailMetadataDoesNotHideKvHit) {
    for (int bad = 0; bad < 3; ++bad) {
        auto source = resource(1, 40000 + bad * 100);
        fill(source, 17);
        auto metadata = std::make_shared<DSV41CheckpointMetadata>(checkpoint(cacheIdentity(), 128));
        if (bad == 0)
            metadata->aux_valid_end--;
        else if (bad == 1)
            metadata->identity.model_revision += "-other";
        else
            *metadata = checkpoint(cacheIdentity(), 256);
        source->setDsv41RecoveryMetadata(0, metadata);
        ASSERT_TRUE(write(source));
        auto destination = resource(1, 40000 + bad * 100);
        ASSERT_TRUE(restore(destination, 1));
        EXPECT_EQ(destination->dsv41CacheState()->view().encoder_materialized_end, 0);
        EXPECT_EQ(destination->dsv41CacheState()->view().decoder_checkpoint_end, 0);
        EXPECT_EQ(destination->dsv41RecoveryMetadata(0), nullptr);
    }
}

TEST_F(DSV41MemoryCheckpointGpuTest, MissingTailPreservesAnEarlierRecoveredBoundary) {
    auto source = resource(2, 50000);
    fill(source, 19);
    ASSERT_TRUE(write(source));
    auto       destination = resource(2, 50000);
    const auto old         = checkpoint(cacheIdentity(), 128);
    destination->dsv41CacheState()->restore(old, 128);
    destination->setDsv41RecoveryMetadata(0, std::make_shared<DSV41CheckpointMetadata>(old));
    ASSERT_TRUE(restore(destination, 2));
    EXPECT_EQ(destination->dsv41CacheState()->view().encoder_materialized_end, 128);
    EXPECT_EQ(destination->dsv41CacheState()->view().decoder_checkpoint_end, 128);
    EXPECT_EQ(destination->dsv41RecoveryMetadata(1), nullptr);
    EXPECT_EQ(destination->dsv41RecoveryMetadata(0)->materialized_end, 128);
}

TEST_F(DSV41MemoryCheckpointGpuTest, SelectedEarlierReadKeepsKvHitWithoutInventingIntermediateTail) {
    auto source = resource(3, 50500);
    fill(source, 19);
    tail(source, 2);
    ASSERT_TRUE(write(source));
    auto destination = resource(3, 50500);
    auto match       = connector_->asyncMatch(destination, meta_);
    ASSERT_NE(match, nullptr);
    EXPECT_EQ(match->matchedBlockCount(), 3);
    ASSERT_TRUE(done(connector_->asyncRead(destination, meta_, match, 0, 2)));
    EXPECT_EQ(destination->reuseBlockNum(), 2);
    EXPECT_EQ(destination->dsv41CacheState()->view().decoder_checkpoint_end, 0);
    EXPECT_EQ(destination->dsv41RecoveryMetadata(1), nullptr);
    ASSERT_TRUE(done(connector_->asyncRead(destination, meta_, match, 2, 1)));
    EXPECT_EQ(bytes(destination, 3, false), bytes(source, 3, true));
    EXPECT_EQ(destination->dsv41CacheState()->view().decoder_checkpoint_end, 384);
    EXPECT_EQ(destination->reuseBlockNum(), 3);
}

TEST_F(DSV41MemoryCheckpointGpuTest, UnassignedDestinationSwaDoesNotBlockKvCopy) {
    auto source = resource(1, 51000);
    fill(source, 19);
    tail(source, 0);
    ASSERT_TRUE(write(source));
    auto destination = resource(1, 51000);
    destination->mutableBlockIds(5).setAt(0, NULL_BLOCK_IDX);
    ASSERT_TRUE(restore(destination, 1));
    EXPECT_EQ(destination->reuseBlockNum(), 1);
    EXPECT_EQ(destination->dsv41CacheState()->view().encoder_materialized_end, 0);
    EXPECT_EQ(destination->dsv41CacheState()->view().decoder_checkpoint_end, 0);
    EXPECT_EQ(destination->dsv41RecoveryMetadata(0), nullptr);
}

TEST_F(DSV41MemoryCheckpointGpuTest, FailedWritePublishesNothingAndPreservesSourceBytes) {
    auto source = resource(3, 60000);
    fill(source, 23);
    tail(source, 2);
    const auto expected    = bytes(source, 3, false);
    const auto global_free = connector_->compressed_pool_->freeBlocksNum();
    const auto state_free  = connector_->state_swa_pool_->freeBlocksNum();
    service_.fail_next     = true;
    EXPECT_FALSE(write(source));
    EXPECT_TRUE(connector_->cacheKeys().empty());
    EXPECT_EQ(connector_->compressed_pool_->freeBlocksNum(), global_free);
    EXPECT_EQ(connector_->state_swa_pool_->freeBlocksNum(), state_free);
    EXPECT_EQ(bytes(source, 3, false), expected);
    ASSERT_TRUE(write(source));
}

TEST_F(DSV41MemoryCheckpointGpuTest, InvalidWriteMappingReportsFailureInsteadOfNoOp) {
    auto source = resource(1, 61000);
    fill(source, 23);
    const auto original = source->blocks(0)[0];
    source->mutableBlockIds(0).setAt(0, NULL_BLOCK_IDX);
    auto result = connector_->asyncWrite(source, meta_);
    ASSERT_NE(result, nullptr);
    EXPECT_FALSE(done(result));
    EXPECT_TRUE(connector_->cacheKeys().empty());
    source->mutableBlockIds(0).setAt(0, original);
    ASSERT_TRUE(write(source));
}

TEST_F(DSV41MemoryCheckpointGpuTest, AllocationFailureRollsBackEarlierKindsAndReportsFailure) {
    auto source = resource(3, 62000);
    fill(source, 23);
    tail(source, 2);
    const auto global_free = connector_->compressed_pool_->freeBlocksNum();
    const auto held        = connector_->state_swa_pool_->malloc(connector_->state_swa_pool_->freeBlocksNum());
    auto       result      = connector_->asyncWrite(source, meta_);
    ASSERT_NE(result, nullptr);
    EXPECT_FALSE(done(result));
    EXPECT_EQ(connector_->compressed_pool_->freeBlocksNum(), global_free);
    EXPECT_TRUE(connector_->cacheKeys().empty());
    connector_->state_swa_pool_->requestFree(held);
    ASSERT_TRUE(write(source));
}

TEST_F(DSV41MemoryCheckpointGpuTest, FailedReadRetainsCpuSourceAndDoesNotPublishProgress) {
    auto source = resource(1, 70000);
    fill(source, 29);
    tail(source, 0);
    ASSERT_TRUE(write(source));
    auto destination = resource(1, 70000);
    auto match       = connector_->asyncMatch(destination, meta_);
    ASSERT_NE(match, nullptr);
    service_.fail_next = true;
    EXPECT_FALSE(done(connector_->asyncRead(destination, meta_, match, 0, 1)));
    EXPECT_EQ(destination->reuseBlockNum(), 0);
    EXPECT_EQ(destination->dsv41RecoveryMetadata(0), nullptr);
    EXPECT_EQ(destination->dsv41CacheState()->view().decoder_checkpoint_end, 0);
    EXPECT_TRUE(connector_->prefix_block_cache_->contains(source->cacheKeys()[0], CacheBlockKind::COMPRESSED_KV));
    EXPECT_TRUE(connector_->prefix_block_cache_->contains(source->cacheKeys()[0], CacheBlockKind::STATE_SWA_KV));
    ASSERT_TRUE(done(connector_->asyncRead(destination, meta_, match, 0, 1)));
    EXPECT_EQ(bytes(destination, 1, false), bytes(source, 1, true));
}

TEST_F(DSV41MemoryCheckpointGpuTest, PinnedMatchProtectsBothKindsAndAbandonAllowsJointEviction) {
    auto source = resource(1, 80000);
    fill(source, 31);
    tail(source, 0);
    ASSERT_TRUE(write(source));
    auto destination = resource(1, 80000);
    auto match       = connector_->asyncMatch(destination, meta_);
    ASSERT_NE(match, nullptr);
    for (auto kind : {CacheBlockKind::COMPRESSED_KV, CacheBlockKind::STATE_SWA_KV})
        EXPECT_TRUE(connector_->prefix_block_cache_->popOldestJointEvictable(kind, CacheBackingType::MEMORY).empty());
    match.reset();
    const auto evicted = connector_->prefix_block_cache_->popOldestJointEvictable(CacheBlockKind::STATE_SWA_KV,
                                                                                  CacheBackingType::MEMORY);
    ASSERT_EQ(evicted.size(), 2);
    for (const auto& item : evicted)
        connector_->releasePrefixCacheBacking(item);
    EXPECT_TRUE(connector_->cacheKeys().empty());
}

TEST_F(DSV41MemoryCheckpointGpuTest, DuplicateDifferentBytesCannotRelabelExistingRecovery) {
    auto source = resource(1, 90000);
    fill(source, 37);
    tail(source, 0);
    ASSERT_TRUE(write(source));
    auto conflict = resource(1, 90000);
    fill(conflict, 67);
    auto changed = std::make_shared<DSV41CheckpointMetadata>(checkpoint(cacheIdentity(), 128));
    changed->history_token_ids[0]++;
    conflict->setDsv41RecoveryMetadata(0, changed);
    auto rejected = connector_->asyncWrite(conflict, meta_);
    ASSERT_NE(rejected, nullptr);
    EXPECT_FALSE(done(rejected));
    auto destination = resource(1, 90000);
    ASSERT_TRUE(restore(destination, 1));
    EXPECT_EQ(bytes(destination, 1, false), bytes(source, 1, true));
    EXPECT_EQ(destination->dsv41RecoveryMetadata(0)->history_token_ids,
              source->dsv41RecoveryMetadata(0)->history_token_ids);
}

TEST_F(DSV41MemoryCheckpointGpuTest, ReadRefreshesDestinationMappingsAndKeepsSwaPrivate) {
    auto source = resource(1, 100000);
    fill(source, 41);
    tail(source, 0);
    ASSERT_TRUE(write(source));
    auto destination = resource(1, 100000);
    fill(destination, 43);
    const auto stale       = bytes(destination, 1, false);
    auto       old_mapping = std::make_shared<KVCacheResource>(*destination);
    old_mapping->initGroups(
        6, config_.layer_all_num, config_.layer_to_group_id, 1, config_.group_types, config_.layer_region_to_group_id);
    for (int group = 0; group < 6; ++group)
        old_mapping->mutableBlockIds(group).assign(destination->blocks(group));
    auto match       = connector_->asyncMatch(destination, meta_);
    auto replacement = resource(1, 100000);
    fill(replacement, 47);
    for (int group = 0; group < 6; ++group)
        destination->mutableBlockIds(group).assign(replacement->blocks(group));
    ASSERT_TRUE(done(connector_->asyncRead(destination, meta_, match, 0, 1)));
    EXPECT_EQ(bytes(destination, 1, false), bytes(source, 1, true));
    EXPECT_EQ(bytes(old_mapping, 1, false), stale);
    const auto source_before_write = bytes(source, 1, false);
    fill(destination, 53, true);
    EXPECT_EQ(bytes(source, 1, false), source_before_write);
}

TEST_F(DSV41MemoryCheckpointGpuTest, ProtectedNUsesOriginalAsyncCopyAndSurvivesLiveRingAdvance) {
    auto source = resource(1, 110000);
    fill(source, 59);
    const auto expected = bytes(source, 1, true);
    auto       state    = source->dsv41CacheState();
    state->requireProtectedPrefix(128, 300);
    state->advanceEncoder(128);
    state->completeDecoder(checkpoint(cacheIdentity(), 128), 128);
    ASSERT_TRUE(connector_->stageDsv41Checkpoint(source, [] { cudaCheck(cudaDeviceSynchronize()); }, meta_));
    EXPECT_FALSE(connector_->cacheKeys().empty());
    state->advanceEncoder(300);
    fill(source, 61, true);
    state->completeHandoff(300, true, true, true);
    state->finish(300);
    auto destination = resource(1, 110000);
    ASSERT_TRUE(restore(destination, 1));
    EXPECT_EQ(bytes(destination, 1, false), expected);
    EXPECT_EQ(destination->dsv41CacheState()->view().decoder_checkpoint_end, 128);
}

TEST_F(DSV41MemoryCheckpointGpuTest, ProtectedSnapshotKeepsCopyOwnerUntilItsPinsAreReleased) {
    auto source = resource(1, 110500);
    fill(source, 59);
    auto state = source->dsv41CacheState();
    state->advanceEncoder(128);
    state->completeDecoder(checkpoint(cacheIdentity(), 128), 128);
    ASSERT_TRUE(connector_->stageDsv41Checkpoint(source, [] { cudaCheck(cudaDeviceSynchronize()); }, meta_));
    auto                                  snapshot = state->view().snapshots.at(0);
    std::weak_ptr<KVCacheMemoryConnector> owner    = connector_;
    state->cancel();
    state.reset();
    source.reset();
    service_.connector = nullptr;
    connector_.reset();
    EXPECT_FALSE(owner.expired());
    EXPECT_EQ(snapshot->metadata().materialized_end, 128);
    snapshot.reset();
    EXPECT_TRUE(owner.expired());
}

TEST_F(DSV41MemoryCheckpointGpuTest, StaleNIsRejectedBeforeCopyAndCancellationKeepsValidatedBlocks) {
    auto stale = resource(1, 111000);
    fill(stale, 59);
    auto stale_state = stale->dsv41CacheState();
    stale_state->advanceEncoder(128);
    stale_state->completeDecoder(checkpoint(cacheIdentity(), 128), 128);
    EXPECT_FALSE(connector_->stageDsv41Checkpoint(stale, [&] { stale_state->advanceEncoder(300); }, meta_));
    EXPECT_TRUE(connector_->cacheKeys().empty());
    EXPECT_EQ(service_.copied_requests.load(), 0);

    auto source = resource(1, 112000);
    fill(source, 61);
    auto state = source->dsv41CacheState();
    state->advanceEncoder(128);
    state->completeDecoder(checkpoint(cacheIdentity(), 128), 128);
    ASSERT_TRUE(connector_->stageDsv41Checkpoint(source, [] { cudaCheck(cudaDeviceSynchronize()); }, meta_));
    state->cancel();
    EXPECT_TRUE(state->view().snapshots.empty());
    EXPECT_FALSE(connector_->cacheKeys().empty());
    auto destination = resource(1, 112000);
    ASSERT_TRUE(restore(destination, 1));
    EXPECT_EQ(bytes(destination, 1, false), bytes(source, 1, true));
}

TEST_F(DSV41MemoryCheckpointGpuTest, PrefixMatchDoesNotRequireTailOrDestinationPages) {
    auto source = resource(2, 120000);
    fill(source, 71);
    ASSERT_TRUE(write(source));
    auto destination = std::make_shared<KVCacheResource>();
    destination->initGroups(
        6, config_.layer_all_num, config_.layer_to_group_id, 1, config_.group_types, config_.layer_region_to_group_id);
    destination->setCacheKeys(source->cacheKeys());
    destination->setDsv41CacheState(std::make_shared<DSV41CacheState>(cacheIdentity()));
    auto match = connector_->asyncMatch(destination, meta_);
    ASSERT_NE(match, nullptr);
    EXPECT_EQ(match->matchedBlockCount(), 2);
    EXPECT_FALSE(done(connector_->asyncRead(destination, meta_, match, 0, 2)));
    auto allocated = resource(2, 120000);
    for (int group = 0; group < 6; ++group)
        destination->mutableBlockIds(group).assign(allocated->blocks(group));
    ASSERT_TRUE(done(connector_->asyncRead(destination, meta_, match, 0, 2)));
    EXPECT_EQ(destination->dsv41CacheState()->view().decoder_checkpoint_end, 0);
}

TEST_F(DSV41MemoryCheckpointGpuTest, NamespacedBlockKeysSeparateReplayModeAndModelIdentity) {
    auto source = resource(1, 130000);
    fill(source, 73);
    tail(source, 0);
    ASSERT_TRUE(write(source));
    auto bounded = resource(1, 130000, DSV41ReplayMode::BOUNDED_CHECKPOINT_V1);
    EXPECT_NE(bounded->cacheKeys(), source->cacheKeys());
    EXPECT_EQ(connector_->asyncMatch(bounded, meta_), nullptr);
    auto other = resource(1, 130000);
    auto id    = cacheIdentity();
    id.model_revision += "-other";
    other->setDsv41CacheState(std::make_shared<DSV41CacheState>(id));
    other->setCacheKeys(keys(1, 130000, id));
    EXPECT_EQ(connector_->asyncMatch(other, meta_), nullptr);
}

class DSV41MemoryTargetOnlyGpuTest: public DSV41MemoryCheckpointGpuTest {
protected:
    bool withDraft() const override {
        return false;
    }
};

TEST_F(DSV41MemoryTargetOnlyGpuTest, FortyLayerLayoutStoresKvWithoutInventingDraftState) {
    ASSERT_EQ(config_.layer_all_num, 40);
    EXPECT_EQ(connector_->layerRegionSlots().size(), 51);
    EXPECT_EQ(connector_->dsv41LayoutFingerprint(), config_.dsv41LayoutFingerprint());
    auto source = resource(1, 135000);
    fill(source, 73);
    ASSERT_TRUE(write(source));
    auto destination = resource(1, 135000);
    ASSERT_TRUE(restore(destination, 1));
    const auto copied = bytes(destination, 1, false);
    for (const auto& [key, data] : bytes(source, 1, false)) {
        if (std::get<0>(key) < 4) {
            EXPECT_EQ(copied.at(key), data);
        }
    }
    EXPECT_EQ(destination->dsv41CacheState()->view().target_ready_end, 0);
}

class DSV41MemoryFullPageGpuTest: public DSV41MemoryCheckpointGpuTest, public ::testing::WithParamInterface<uint32_t> {
protected:
    uint32_t blockSize() const override {
        return GetParam();
    }
    uint32_t poolBlocks() const override {
        return 32;
    }
    ParallelismConfig parallelism() const override {
        ParallelismConfig value;
        value.role_type                          = RoleType::DECODE;
        value.tp_size                            = 1;
        value.prefill_cp_config.method           = CPRotateMethod::PREFILL_CP;
        value.prefill_cp_config.kv_cache_sharded = true;
        value.prefill_cp_config.prefill_cp_size  = 8;
        return value;
    }
};

TEST_P(DSV41MemoryFullPageGpuTest, DataPagesUseBAndRawFixedSlotsUseEightB) {
    EXPECT_EQ(connector_->dsv41DataUnit(), blockSize());
    EXPECT_EQ(connector_->dsv41ReuseUnit(), blockSize() * 8);
    EXPECT_EQ(connector_->cacheKeyTokensPerBlockForMetrics(), blockSize());
    auto source = resource(8, 136000);
    fill(source, 79);
    tail(source, 7);
    const auto expected = bytes(source, 8, true);
    for (int group : {4, 5})
        source->mutableBlockIds(group).assign(BlockIndicesType{source->blocks(group)[7]});
    source->setBlockIdsKeyAligned(false);
    EXPECT_FALSE(source->cacheKeysAreCpCanonical());
    ASSERT_TRUE(write(source));
    auto destination = resource(8, 136000);
    fill(destination, 83);
    copyGpuPrefix(source, destination, 4);
    for (int group : {4, 5})
        destination->mutableBlockIds(group).assign(BlockIndicesType{destination->blocks(group)[7]});
    destination->setBlockIdsKeyAligned(false);
    ASSERT_TRUE(restore(destination, 8));
    EXPECT_EQ(bytes(destination, 8, false), expected);
    EXPECT_EQ(destination->memoryReuseBlockNum(), 4);
    ASSERT_NE(destination->dsv41RecoveryMetadata(7), nullptr);
    EXPECT_EQ(destination->dsv41RecoveryMetadata(7)->materialized_end, blockSize() * 8);
    EXPECT_NE(destination->blocks(5)[0], source->blocks(5)[0]);
    const auto unchanged = bytes(source, 8, false);
    fill(destination, 89, true);
    EXPECT_EQ(bytes(source, 8, false), unchanged);
}

INSTANTIATE_TEST_SUITE_P(PhysicalBlocks, DSV41MemoryFullPageGpuTest, ::testing::Values(128u, 256u));

class DSV41MemoryCheckpointCp8GpuTest:
    public DSV41MemoryCheckpointGpuTest,
    public ::testing::WithParamInterface<uint32_t> {
protected:
    uint32_t blockSize() const override {
        return GetParam();
    }
    ParallelismConfig parallelism() const override {
        ParallelismConfig value;
        value.role_type                          = RoleType::PREFILL;
        value.tp_size                            = 8;
        value.prefill_cp_config.method           = CPRotateMethod::PREFILL_CP;
        value.prefill_cp_config.kv_cache_sharded = true;
        value.prefill_cp_config.prefill_cp_size  = 8;
        return value;
    }
};

// Local shard transport qualification; distributed CP recovery remains an engine gate.
TEST_P(DSV41MemoryCheckpointCp8GpuTest, CanonicalSuffixCopiesExactOwnerBytesAtOriginalOrdinal) {
    auto source = resource(2, 140000, DSV41ReplayMode::BOUNDED_CHECKPOINT_V1);
    fill(source, 79);
    tail(source, 1);
    auto stored_suffix = suffix(source, 1);
    ASSERT_TRUE(write(stored_suffix));
    auto destination = resource(2, 140000, DSV41ReplayMode::BOUNDED_CHECKPOINT_V1);
    fill(destination, 83);
    copyGpuPrefix(source, destination, 1);
    destination->setCacheKeysAreCpCanonical(false);
    EXPECT_EQ(connector_->asyncMatch(destination, meta_), nullptr);
    destination->setCacheKeysAreCpCanonical(true);
    ASSERT_TRUE(restore(destination, 2));
    EXPECT_EQ(bytes(destination, 2, false), bytes(source, 2, true));
    ASSERT_NE(destination->dsv41RecoveryMetadata(1), nullptr);
    EXPECT_EQ(destination->dsv41RecoveryMetadata(1)->materialized_end, blockSize() * 8 * 2);
    EXPECT_EQ(destination->dsv41CacheState()->view().decoder_checkpoint_end, blockSize() * 8 * 2);
    // A copied aux range declaration does not restore the L20 replay source tensors.
    EXPECT_GT(destination->dsv41RecoveryMetadata(1)->aux_valid_end, 0);
    EXPECT_EQ(destination->dsv41CacheState()->view().target_ready_end, 0);
    EXPECT_EQ(destination->memoryReuseBlockNum(), 1);
}

INSTANTIATE_TEST_SUITE_P(PhysicalBlocks, DSV41MemoryCheckpointCp8GpuTest, ::testing::Values(128u, 256u));

}  // namespace rtp_llm::test
