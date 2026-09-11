#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <grpcpp/grpcpp.h>

#include <algorithm>
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

std::vector<PrefixTreeMemoryBlockCache::DSV41Entry> treeEntries(size_t count, const std::set<size_t>& boundaries) {
    std::vector<PrefixTreeMemoryBlockCache::DSV41Entry> entries;
    for (size_t index = 0; index < count; ++index) {
        PrefixTreeMemoryBlockCache::DSV41Entry entry;
        entry.global.cache_key   = 500 + index;
        entry.global.kind        = CacheBlockKind::COMPRESSED_KV;
        entry.global.block_index = index * 2 + 1;
        entry.global.block_size  = 64;
        entry.global.slot_valid_mask.resize(54, 0);
        std::fill_n(entry.global.slot_valid_mask.begin(), 8, 1);
        entry.dependency = {index > 0, static_cast<int64_t>(499 + index), static_cast<uint32_t>(index)};
        if (boundaries.count(index + 1)) {
            auto swa = entry.global;
            swa.kind = CacheBlockKind::STATE_SWA_KV;
            swa.block_index++;
            for (auto& bit : swa.slot_valid_mask)
                bit = 1 - bit;
            entry.swa        = swa;
            entry.checkpoint = checkpoint(identity(), (index + 1) * 128);
        }
        entries.push_back(std::move(entry));
    }
    return entries;
}

ModelConfig model(bool draft) {
    ModelConfig value;
    value.num_layers                = draft ? 3 : 40;
    value.hidden_size               = 5120;
    value.max_seq_len               = 1048576;
    auto& attn                      = value.attn_config;
    attn.dsv41_cache_layout_version = 1;
    attn.head_num                   = 64;
    attn.kv_head_num                = 1;
    attn.size_per_head              = 512;
    attn.sliding_window             = 128;
    attn.indexer_head_dim           = 128;
    attn.indexer_head_num           = 32;
    attn.indexer_topk               = 512;
    for (int layer = 0; layer < value.num_layers; ++layer) {
        attn.layer_compress_ratios.push_back(draft || layer < 2 ? 0 : (layer < 20 ? 2 : 1));
    }
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

// The RPC performs the production connector's real GPU copy. The failure flag
// is only fault injection after that copy, to check transaction rollback.
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
            const bool copied = connector->copyCache(request->mem_request(), *response->mutable_mem_response());
            if (copied)
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

TEST(DSV41CacheStateTest, TwoAxesAndPrivateNMustCompleteBeforeT) {
    DSV41CacheState state(identity());
    state.requireProtectedPrefix(128, 300);
    state.advanceEncoder(128);
    EXPECT_EQ(state.view().decoder_checkpoint_end, 0);
    EXPECT_THROW(state.advanceEncoder(300), std::logic_error);
    state.completeDecoder(checkpoint(identity(), 128), 128);
    EXPECT_THROW(state.advanceEncoder(300), std::logic_error);
    state.protect(std::make_shared<DSV41CheckpointSnapshot>(checkpoint(identity(), 128)));
    state.advanceEncoder(300);
    EXPECT_THROW(state.finish(300), std::logic_error);
    EXPECT_THROW(state.completeHandoff(300, true, false, true), std::invalid_argument);
    state.completeHandoff(300, true, true, true);
    state.finish(300);
    EXPECT_EQ(state.view().decoder_checkpoint_end, 300);
    EXPECT_TRUE(state.view().finished);
}

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
    bounded.tail_policy_version = 2;
    EXPECT_THROW(DSV41CacheState invalid(bounded), std::invalid_argument);
}

TEST(DSV41CacheStateTest, CancellationCannotPublishPrivateSnapshots) {
    DSV41CacheState state(identity());
    state.advanceEncoder(128);
    state.completeDecoder(checkpoint(identity(), 128), 128);
    state.protect(std::make_shared<DSV41CheckpointSnapshot>(checkpoint(identity(), 128)));
    state.cancel();
    bool published = false;
    EXPECT_FALSE(state.publishSnapshots([&](const auto&) {
        published = true;
        return true;
    }));
    EXPECT_FALSE(published);
    EXPECT_TRUE(state.view().snapshots.empty());
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

TEST(DSV41CacheStateTest, SelectedNAndTCannotBeReplacedAfterSelection) {
    DSV41CacheState state(identity());
    state.requireProtectedPrefix(128, 300);
    EXPECT_NO_THROW(state.requireProtectedPrefix(128, 300));
    EXPECT_THROW(state.requireProtectedPrefix(0, 300), std::invalid_argument);
    EXPECT_THROW(state.requireProtectedPrefix(128, 400), std::invalid_argument);
}

TEST(DSV41CacheStateTest, FinishedPublicationIsIdempotentAndFailedPublicationCanRetry) {
    DSV41CacheState state(identity());
    state.advanceEncoder(128);
    state.completeDecoder(checkpoint(identity(), 128), 128);
    state.protect(std::make_shared<DSV41CheckpointSnapshot>(checkpoint(identity(), 128)));
    state.finish(128);
    EXPECT_FALSE(state.publishSnapshots([](const auto&) { return false; }));
    EXPECT_FALSE(state.view().snapshots.empty());
    size_t calls = 0;
    for (int i = 0; i < 2; ++i) {
        EXPECT_TRUE(state.publishSnapshots([&](const auto&) {
            ++calls;
            return true;
        }));
    }
    EXPECT_EQ(calls, 1);
    EXPECT_TRUE(state.view().published);
    EXPECT_TRUE(state.view().snapshots.empty());
    EXPECT_THROW(state.cancel(), std::logic_error);
}

TEST(DSV41PrefixTreeMemoryTest, EveryPublishedCheckpointUsesOneLogicalReuseUnit) {
    PrefixTreeMemoryBlockCache tree;
    auto                       entries = treeEntries(2, {1, 2});
    entries[0].checkpoint              = checkpoint(identity(), 256);
    EXPECT_FALSE(tree.putDsv41Committed(identity(), entries, {}).success);
    EXPECT_TRUE(tree.dsv41CacheKeys().empty());
    entries[0].checkpoint = checkpoint(identity(), 128);
    EXPECT_TRUE(tree.putDsv41Committed(identity(), entries, {}).success);
}

TEST(DSV41PrefixTreeMemoryTest, MissingOrOverlappingRegionValidityRejectsWholePublication) {
    PrefixTreeMemoryBlockCache tree;
    auto                       entries  = treeEntries(2, {2});
    entries[1].swa->slot_valid_mask[53] = 0;
    EXPECT_FALSE(tree.putDsv41Committed(identity(), entries, {}).success);
    entries[1].swa->slot_valid_mask[0] = 1;
    EXPECT_FALSE(tree.putDsv41Committed(identity(), entries, {}).success);
    EXPECT_TRUE(tree.dsv41CacheKeys().empty());
}

TEST(DSV41PrefixTreeMemoryTest, ResidentDescendantProtectsItsGlobalAncestorsAndOtherCheckpoints) {
    PrefixTreeMemoryBlockCache tree;
    auto                       entries = treeEntries(3, {1, 3});
    entries.back().swa->is_resident    = true;
    ASSERT_TRUE(tree.putDsv41Committed(identity(), entries, {}).success);
    EXPECT_TRUE(tree.popOldestDsv41JointEvictable().empty());
    CacheKeysType         keys{500, 501, 502};
    BlockDependenciesType dependencies;
    for (const auto& entry : entries)
        dependencies.push_back(entry.dependency);
    auto match = tree.matchDsv41AndMarkInFlight(identity(), keys, dependencies, keys.size());
    EXPECT_EQ(match.matched_blocks, 3);
    tree.releaseDsv41InFlight(identity(), match);
    EXPECT_TRUE(tree.popOldestDsv41JointEvictable().empty());
}

class DSV41MemoryCheckpointGpuTest: public ::testing::Test {
protected:
    using ByteKey = std::tuple<size_t, int, size_t>;
    using ByteMap = std::map<ByteKey, std::vector<uint8_t>>;

    virtual uint32_t blockSize() const {
        return 128;
    }
    virtual ParallelismConfig parallelism() const {
        return {};
    }
    void SetUp() override {
        int count = 0;
        ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
        ASSERT_GT(count, 0) << "required real GPU checkpoint test";
        cudaDeviceProp properties{};
        ASSERT_EQ(cudaGetDeviceProperties(&properties, 0), cudaSuccess);
        ASSERT_EQ(properties.major, 10) << "required CUDA13 Blackwell checkpoint test";
        initRuntime(0, false, false, MlaOpsType::AUTO);
        kv_.seq_size_per_block                      = blockSize();
        kv_.test_block_num                          = 16;
        kv_.dsv4_fixed_pool_blocks                  = 16;
        kv_.enable_memory_cache                     = true;
        kv_.enable_memory_cache_disk                = false;
        kv_.memory_cache_size_mb                    = 64;
        kv_.memory_cache_sync_timeout_ms            = 30000;
        kv_.prefix_tree_memory_state_swa_pool_ratio = 50;
        SpeculativeExecutionConfig sp;
        sp.type              = SP_TYPE_DSPARK;
        sp.gen_num_per_cycle = 5;
        config_              = CacheConfigCreator::createSpConfig(
            model(false), model(true), parallelism(), RuntimeConfig(), kv_, sp, std::nullopt, true, false);
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
    std::shared_ptr<KVCacheResource>
    resource(size_t blocks, int64_t first_key, DSV41ReplayMode mode = DSV41ReplayMode::FULL) {
        auto value = std::make_shared<KVCacheResource>();
        value->initGroups(6, 43, config_.layer_to_group_id, 1, config_.group_types, config_.layer_region_to_group_id);
        CacheKeysType keys(blocks + 1);
        std::iota(keys.begin(), keys.end(), first_key);
        value->setCacheKeys(keys);
        value->setCacheKeysAreCpCanonical(connector_->dsv41ReuseUnit() != kv_.seq_size_per_block);
        value->setLastBlockAligned(false);
        for (size_t group = 0; group < 6; ++group) {
            const bool full = config_.group_types[group] == CacheGroupType::FULL;
            const auto ids  = allocator_->groupBlockPools()[group]->malloc(full ? blocks : 1);
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
    void fixedBoundary(const std::shared_ptr<KVCacheResource>& value, size_t index) {
        for (int group : {4, 5}) {
            const auto& current = value->blocks(group);
            const auto  valid   = std::find_if(current.begin(), current.end(), [](auto block) { return block > 0; });
            if (valid == current.end())
                throw std::runtime_error("missing active GPU ring");
            BlockIndicesType mapping(current.size(), NULL_BLOCK_IDX);
            mapping[index] = *valid;
            value->mutableBlockIds(group).assign(mapping);
        }
    }
    void fill(const std::shared_ptr<KVCacheResource>& value, uint32_t salt, bool fixed_only = false) {
        for (size_t group = fixed_only ? 4 : 0; group < 6; ++group) {
            for (int owner : config_.global_layer_ids[group]) {
                for (auto block : value->blocks(group)) {
                    if (block <= 0)
                        continue;
                    for (const auto& segment :
                         allocator_->convertIndexToBuffer(owner, config_.group_region_names[group], block)) {
                        std::vector<uint8_t> bytes(segment.size_bytes);
                        for (size_t i = 0; i < bytes.size(); ++i)
                            bytes[i] = (i * 37 + owner * 11 + group * 17 + block * 3 + salt) % 251;
                        cudaCheck(cudaMemcpy(segment.addr, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
                    }
                }
            }
        }
    }
    ByteMap bytes(const std::shared_ptr<KVCacheResource>& value, size_t count, bool reset_pair) {
        ByteMap result;
        for (size_t group = 0; group < 6; ++group) {
            for (int owner : config_.global_layer_ids[group]) {
                const size_t begin = group < 4 ? 0 : count - 1;
                for (size_t index = begin; index < count; ++index) {
                    auto&      data  = result[{group, owner, index}];
                    const auto block = value->blocks(group).at(index);
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
    bool stage(const std::shared_ptr<KVCacheResource>& value, size_t count) {
        const int64_t end   = count * connector_->dsv41ReuseUnit();
        auto          state = value->dsv41CacheState();
        state->advanceEncoder(end);
        state->completeDecoder(checkpoint(state->view().identity, end), connector_->dsv41ReuseUnit());
        return connector_->stageDsv41Checkpoint(value, [] { cudaCheck(cudaDeviceSynchronize()); });
    }
    void finishAndPublish(const std::shared_ptr<KVCacheResource>& value, int64_t end) {
        value->dsv41CacheState()->finish(end);
        auto result = connector_->asyncWrite(value, meta_);
        ASSERT_NE(result, nullptr);
        result->waitDone();
        ASSERT_TRUE(result->success());
    }
    bool restore(const std::shared_ptr<KVCacheResource>& destination, size_t expected_count) {
        auto match = connector_->asyncMatch(destination, meta_);
        if (!match || match->matchedBlockCount() != expected_count)
            return false;
        auto result = connector_->asyncRead(destination, meta_, match, 0, expected_count);
        result->waitDone();
        return result->success();
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

TEST_F(DSV41MemoryCheckpointGpuTest, ProtectedNRestoresExactBytesAfterLiveRingsAdvanceMoreThan128) {
    auto source = resource(1, 10000);
    fill(source, 1);
    const auto expected = bytes(source, 1, true);
    source->dsv41CacheState()->requireProtectedPrefix(128, 300);
    ASSERT_TRUE(stage(source, 1));
    EXPECT_TRUE(connector_->cacheKeys().empty());
    source->dsv41CacheState()->advanceEncoder(300);
    fill(source, 89, true);
    source->dsv41CacheState()->completeHandoff(300, true, true, true);
    finishAndPublish(source, 300);
    auto destination = resource(1, 10000);
    fill(destination, 199);
    ASSERT_TRUE(restore(destination, 1));
    EXPECT_EQ(bytes(destination, 1, false), expected);
    EXPECT_EQ(destination->cacheKeys(), source->cacheKeys());
    EXPECT_TRUE(*destination->dsv41CacheState()->view().completed == checkpoint(cacheIdentity(), 128));
    EXPECT_EQ(destination->memoryReuseBlockNum(), 1);
    EXPECT_GE(service_.copied_requests.load(), 2);
}

TEST_F(DSV41MemoryCheckpointGpuTest, IntermediateGlobalBlocksNeedNoSwaAndMatchLatestCompleteCheckpoint) {
    auto source = resource(3, 20000);
    fill(source, 7);
    fixedBoundary(source, 0);
    const auto at_n = bytes(source, 1, true);
    source->dsv41CacheState()->requireProtectedPrefix(128, 384);
    ASSERT_TRUE(stage(source, 1));
    fixedBoundary(source, 2);
    fill(source, 27, true);
    const auto at_t = bytes(source, 3, true);
    ASSERT_TRUE(stage(source, 3));
    finishAndPublish(source, 384);
    auto short_request = resource(1, 20000);
    ASSERT_TRUE(restore(short_request, 1));
    EXPECT_EQ(bytes(short_request, 1, false), at_n);
    auto long_request = resource(3, 20000);
    ASSERT_TRUE(restore(long_request, 3));
    EXPECT_EQ(bytes(long_request, 3, false), at_t);
    EXPECT_EQ(connector_->cacheKeys().size(), 2);
}

TEST_F(DSV41MemoryCheckpointGpuTest, CancelledPrivateCheckpointReleasesBothPoolsAndNeverMatches) {
    auto source = resource(1, 30000);
    fill(source, 3);
    const auto global_free = connector_->compressed_pool_->freeBlocksNum();
    const auto state_free  = connector_->state_swa_pool_->freeBlocksNum();
    ASSERT_TRUE(stage(source, 1));
    EXPECT_EQ(connector_->compressed_pool_->freeBlocksNum(), global_free - 1);
    EXPECT_EQ(connector_->state_swa_pool_->freeBlocksNum(), state_free - 1);
    source->dsv41CacheState()->cancel();
    EXPECT_EQ(connector_->compressed_pool_->freeBlocksNum(), global_free);
    EXPECT_EQ(connector_->state_swa_pool_->freeBlocksNum(), state_free);
    EXPECT_FALSE(connector_->asyncWrite(source, meta_)->success());
    EXPECT_TRUE(connector_->cacheKeys().empty());
}

TEST_F(DSV41MemoryCheckpointGpuTest, CopyFailureRollsBackWholePrivateTransaction) {
    auto source = resource(3, 40000);
    fill(source, 4);
    const auto global_free = connector_->compressed_pool_->freeBlocksNum();
    const auto state_free  = connector_->state_swa_pool_->freeBlocksNum();
    service_.fail_next     = true;
    EXPECT_FALSE(stage(source, 3));
    EXPECT_TRUE(source->dsv41CacheState()->view().snapshots.empty());
    EXPECT_TRUE(connector_->cacheKeys().empty());
    EXPECT_EQ(connector_->compressed_pool_->freeBlocksNum(), global_free);
    EXPECT_EQ(connector_->state_swa_pool_->freeBlocksNum(), state_free);
    EXPECT_EQ(service_.copied_requests.load(), 1);
}

TEST_F(DSV41MemoryCheckpointGpuTest, InFlightLeaseProtectsBothPoolsFromEitherKindPressure) {
    for (bool state_pressure : {false, true}) {
        const int64_t key    = state_pressure ? 51000 : 50000;
        auto          source = resource(1, key);
        fill(source, 5);
        ASSERT_TRUE(stage(source, 1));
        finishAndPublish(source, 128);
        auto destination = resource(1, key);
        auto lease       = connector_->asyncMatch(destination, meta_);
        ASSERT_NE(lease, nullptr);
        auto pressured = state_pressure ? connector_->state_swa_pool_ : connector_->compressed_pool_;
        auto held      = pressured->malloc(pressured->freeBlocksNum());
        auto next      = resource(1, key + 100);
        fill(next, 6);
        EXPECT_FALSE(stage(next, 1));
        EXPECT_EQ(connector_->cacheKeys(), (std::vector<CacheKeyType>{key}));
        lease.reset();
        ASSERT_TRUE(stage(next, 1));
        EXPECT_TRUE(connector_->cacheKeys().empty());
        next->dsv41CacheState()->cancel();
        pressured->requestFree(held);
    }
}

TEST_F(DSV41MemoryCheckpointGpuTest, FullAndBoundedUseIndependentNamespacesForIdenticalTokenKeys) {
    auto full = resource(1, 60000);
    fill(full, 61);
    const auto expected_full = bytes(full, 1, true);
    ASSERT_TRUE(stage(full, 1));
    finishAndPublish(full, 128);
    auto bounded = resource(1, 60000, DSV41ReplayMode::BOUNDED_CHECKPOINT_V1);
    fill(bounded, 62);
    const auto expected_bounded = bytes(bounded, 1, true);
    ASSERT_TRUE(stage(bounded, 1));
    finishAndPublish(bounded, 128);
    auto read_full    = resource(1, 60000);
    auto read_bounded = resource(1, 60000, DSV41ReplayMode::BOUNDED_CHECKPOINT_V1);
    ASSERT_TRUE(restore(read_full, 1));
    ASSERT_TRUE(restore(read_bounded, 1));
    EXPECT_EQ(bytes(read_full, 1, false), expected_full);
    EXPECT_EQ(bytes(read_bounded, 1, false), expected_bounded);
}

TEST_F(DSV41MemoryCheckpointGpuTest, BrokenHashChainOrChangedModelIdentityDoesNotMatch) {
    auto source = resource(3, 70000);
    fill(source, 7);
    ASSERT_TRUE(stage(source, 3));
    finishAndPublish(source, 384);
    auto destination = resource(3, 70000);
    destination->cacheKeys()[1] += 1234;
    destination->rebuildLinearBlockDependencies();
    EXPECT_EQ(connector_->asyncMatch(destination, meta_), nullptr);
    auto other_revision = resource(3, 70000);
    auto id             = cacheIdentity();
    id.model_revision += "-different";
    other_revision->setDsv41CacheState(std::make_shared<DSV41CacheState>(id));
    EXPECT_EQ(connector_->asyncMatch(other_revision, meta_), nullptr);
}

TEST_F(DSV41MemoryCheckpointGpuTest, MissingPhysicalDraftSlotCannotBeStagedAsComplete) {
    auto source = resource(1, 80000);
    fill(source, 8);
    source->mutableBlockIds(5).setAt(0, NULL_BLOCK_IDX);
    const auto before = connector_->state_swa_pool_->freeBlocksNum();
    EXPECT_FALSE(stage(source, 1));
    EXPECT_TRUE(connector_->cacheKeys().empty());
    EXPECT_EQ(connector_->state_swa_pool_->freeBlocksNum(), before);
}

TEST_F(DSV41MemoryCheckpointGpuTest, RestoreFailureKeepsProgressUnpublishedAndSameLeaseCanRetry) {
    auto source = resource(1, 90000);
    fill(source, 9);
    const auto expected = bytes(source, 1, true);
    ASSERT_TRUE(stage(source, 1));
    finishAndPublish(source, 128);
    auto destination = resource(1, 90000);
    auto match       = connector_->asyncMatch(destination, meta_);
    ASSERT_NE(match, nullptr);
    service_.fail_next = true;
    EXPECT_FALSE(connector_->asyncRead(destination, meta_, match, 0, 1)->success());
    EXPECT_EQ(destination->memoryReuseBlockNum(), 0);
    EXPECT_EQ(destination->dsv41CacheState()->view().decoder_checkpoint_end, 0);
    EXPECT_TRUE(connector_->asyncRead(destination, meta_, match, 0, 1)->success());
    EXPECT_EQ(bytes(destination, 1, false), expected);
    EXPECT_EQ(destination->dsv41CacheState()->view().encoder_materialized_end, 128);
}

TEST_F(DSV41MemoryCheckpointGpuTest, MissingTypedStateNeverUsesLegacyFallback) {
    auto source = resource(1, 100000);
    source->setDsv41CacheState(nullptr);
    EXPECT_THROW(connector_->asyncMatch(source, meta_), std::logic_error);
    EXPECT_THROW(connector_->asyncWrite(source, meta_), std::logic_error);
    EXPECT_FALSE(connector_->stageDsv41Checkpoint(source, [] {}));
}

TEST_F(DSV41MemoryCheckpointGpuTest, SameKeyByteConflictRejectsWholeTransactionAndKeepsOldCheckpoint) {
    auto first = resource(1, 110000);
    fill(first, 13);
    const auto expected = bytes(first, 1, true);
    ASSERT_TRUE(stage(first, 1));
    finishAndPublish(first, 128);
    auto conflicting = resource(2, 110000);
    fill(conflicting, 19);
    ASSERT_TRUE(stage(conflicting, 2));
    conflicting->dsv41CacheState()->finish(256);
    EXPECT_FALSE(connector_->asyncWrite(conflicting, meta_)->success());
    EXPECT_EQ(connector_->cacheKeys(), (std::vector<CacheKeyType>{110000}));
    auto destination = resource(1, 110000);
    ASSERT_TRUE(restore(destination, 1));
    EXPECT_EQ(bytes(destination, 1, false), expected);
    conflicting->dsv41CacheState()->cancel();
}

TEST_F(DSV41MemoryCheckpointGpuTest, ReadRefreshesDestinationBlockIdsAfterMatch) {
    auto source = resource(1, 120000);
    fill(source, 17);
    const auto expected = bytes(source, 1, true);
    ASSERT_TRUE(stage(source, 1));
    finishAndPublish(source, 128);
    auto destination = resource(1, 120000);
    fill(destination, 55);
    auto match = connector_->asyncMatch(destination, meta_);
    ASSERT_NE(match, nullptr);
    auto replacement = resource(1, 120000);
    fill(replacement, 75);
    for (int group = 0; group < 6; ++group)
        destination->mutableBlockIds(group).assign(replacement->blocks(group));
    EXPECT_TRUE(connector_->asyncRead(destination, meta_, match, 0, 1)->success());
    EXPECT_EQ(bytes(destination, 1, false), expected);
}

TEST_F(DSV41MemoryCheckpointGpuTest, AbandonedMatchDropsLeaseSoWholeTransactionCanEvict) {
    auto source = resource(3, 130000);
    fill(source, 29);
    ASSERT_TRUE(stage(source, 3));
    finishAndPublish(source, 384);
    auto destination = resource(3, 130000);
    auto match       = connector_->asyncMatch(destination, meta_);
    ASSERT_NE(match, nullptr);
    EXPECT_TRUE(connector_->prefix_block_cache_->popOldestDsv41JointEvictable().empty());
    match.reset();
    const auto evicted = connector_->prefix_block_cache_->popOldestDsv41JointEvictable();
    ASSERT_EQ(evicted.size(), 4);
    EXPECT_EQ(std::count_if(evicted.begin(),
                            evicted.end(),
                            [](const auto& item) { return item.kind == CacheBlockKind::STATE_SWA_KV; }),
              1);
    for (const auto& item : evicted)
        connector_->memoryPoolFor(item.kind)->blockCacheFree(item.block_index);
    EXPECT_TRUE(connector_->cacheKeys().empty());
}

TEST_F(DSV41MemoryCheckpointGpuTest, CommonEarlierBoundaryRestoresItsProtectedSwaAndNotTheLatestRing) {
    auto source = resource(3, 140000);
    fill(source, 31);
    fixedBoundary(source, 0);
    const auto at_n = bytes(source, 1, true);
    source->dsv41CacheState()->requireProtectedPrefix(128, 384);
    ASSERT_TRUE(stage(source, 1));
    fixedBoundary(source, 2);
    fill(source, 37, true);
    ASSERT_TRUE(stage(source, 3));
    finishAndPublish(source, 384);

    auto destination = resource(3, 140000);
    auto match       = connector_->asyncMatch(destination, meta_);
    ASSERT_NE(match, nullptr);
    EXPECT_EQ(match->matchedBlockCount(), 3);
    EXPECT_EQ(connector_->dsv41MatchedCheckpointEnds(match), (std::vector<int64_t>{128, 384}));
    const auto copied = service_.copied_requests.load();
    EXPECT_FALSE(connector_->asyncRead(destination, meta_, match, 0, 2)->success());
    EXPECT_EQ(service_.copied_requests.load(), copied);
    EXPECT_EQ(destination->reuseBlockNum(), 0);
    fixedBoundary(destination, 0);
    ASSERT_TRUE(connector_->asyncRead(destination, meta_, match, 0, 1)->success());
    EXPECT_EQ(bytes(destination, 1, false), at_n);
    EXPECT_EQ(destination->dsv41CacheState()->view().decoder_checkpoint_end, 128);
    EXPECT_EQ(destination->reuseBlockNum(), 1);
    EXPECT_TRUE(connector_->dsv41MatchedCheckpointEnds(match).empty());
    EXPECT_FALSE(connector_->asyncRead(destination, meta_, match, 0, 1)->success());
    EXPECT_EQ(destination->reuseBlockNum(), 1);
}

TEST_F(DSV41MemoryCheckpointGpuTest, MatchCanPrecedeDestinationPageAssignment) {
    auto source = resource(1, 150000);
    fill(source, 41);
    const auto expected = bytes(source, 1, true);
    ASSERT_TRUE(stage(source, 1));
    finishAndPublish(source, 128);
    auto destination = std::make_shared<KVCacheResource>();
    destination->initGroups(6, 43, config_.layer_to_group_id, 1, config_.group_types, config_.layer_region_to_group_id);
    destination->setCacheKeys(source->cacheKeys());
    destination->setDsv41CacheState(std::make_shared<DSV41CacheState>(cacheIdentity()));
    auto match = connector_->asyncMatch(destination, meta_);
    ASSERT_NE(match, nullptr);
    EXPECT_FALSE(connector_->asyncRead(destination, meta_, match, 0, 1)->success());
    auto allocated = resource(1, 150000);
    for (int group = 0; group < 6; ++group)
        destination->mutableBlockIds(group).assign(allocated->blocks(group));
    ASSERT_TRUE(connector_->asyncRead(destination, meta_, match, 0, 1)->success());
    EXPECT_EQ(bytes(destination, 1, false), expected);
}

TEST_F(DSV41MemoryCheckpointGpuTest, PhysicalCapacityMismatchCannotStageOrMatch) {
    auto source = resource(1, 160000);
    auto id     = cacheIdentity();
    id.physical_swa_entries++;
    source->setDsv41CacheState(std::make_shared<DSV41CacheState>(id));
    EXPECT_FALSE(stage(source, 1));
    EXPECT_EQ(connector_->asyncMatch(source, meta_), nullptr);
}

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

// This is one real local shard's byte-copy probe. The distributed common-hit
// and all-rank completion tests remain separate CP8 integration requirements.
TEST_P(DSV41MemoryCheckpointCp8GpuTest, ExactLocalShardCopyForCanonicalVirtualBoundary) {
    const int64_t boundary = blockSize() * 8;
    auto          source   = resource(1, 170000, DSV41ReplayMode::BOUNDED_CHECKPOINT_V1);
    fill(source, 43);
    const auto expected = bytes(source, 1, true);
    source->dsv41CacheState()->requireProtectedPrefix(boundary, boundary + 129);
    ASSERT_TRUE(stage(source, 1));
    source->dsv41CacheState()->advanceEncoder(boundary + 129);
    fill(source, 47, true);
    source->dsv41CacheState()->completeHandoff(boundary + 129, true, true, true);
    finishAndPublish(source, boundary + 129);
    auto destination = resource(1, 170000, DSV41ReplayMode::BOUNDED_CHECKPOINT_V1);
    destination->setCacheKeysAreCpCanonical(false);
    EXPECT_EQ(connector_->asyncMatch(destination, meta_), nullptr);
    destination->setCacheKeysAreCpCanonical(true);
    ASSERT_TRUE(restore(destination, 1));
    EXPECT_EQ(bytes(destination, 1, false), expected);
    EXPECT_EQ(destination->dsv41CacheState()->view().decoder_checkpoint_end, boundary);
    EXPECT_EQ(destination->dsv41CacheState()->view().identity.physical_swa_entries, 136);
}

INSTANTIATE_TEST_SUITE_P(PhysicalBlocks, DSV41MemoryCheckpointCp8GpuTest, ::testing::Values(128u, 256u));

}  // namespace rtp_llm::test
