#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <grpcpp/grpcpp.h>

#include <atomic>
#include <map>
#include <numeric>

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm::test {
namespace {

ModelConfig model(bool draft) {
    ModelConfig result;
    result.num_layers               = draft ? 3 : 40;
    result.hidden_size              = 5120;
    result.max_seq_len              = 1048576;
    auto& attn                      = result.attn_config;
    attn.dsv41_cache_layout_version = 1;
    attn.head_num                   = 64;
    attn.kv_head_num                = 1;
    attn.size_per_head              = 512;
    attn.sliding_window             = 128;
    attn.indexer_head_dim           = 128;
    attn.indexer_head_num           = 32;
    attn.indexer_topk               = 512;
    for (int layer = 0; layer < result.num_layers; ++layer)
        attn.layer_compress_ratios.push_back(draft || layer < 2 ? 0 : layer < 20 ? 2 : 1);
    return result;
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
    std::string          id_{"dsv41-gpu-checkpoint"};
    std::vector<int64_t> tokens_;
};

class CopyService final: public RpcService::Service {
public:
    KVCacheMemoryConnector* connector{nullptr};
    std::atomic<bool>       fail_next{false};
    grpc::Status
    ExecuteFunction(grpc::ServerContext*, const FunctionRequestPB* request, FunctionResponsePB* response) override {
        try {
            if (!connector || !request->has_mem_request())
                return {grpc::StatusCode::INVALID_ARGUMENT, "missing copy request"};
            connector->copyCache(request->mem_request(), *response->mutable_mem_response());
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

class DSV41GpuCacheAllocatorTest: public ::testing::Test {
protected:
    using Bytes = std::map<std::tuple<size_t, int, size_t>, std::vector<uint8_t>>;

    virtual int linearStep() const {
        return 1;
    }

    void SetUp() override {
        cudaDeviceProp properties{};
        ASSERT_EQ(cudaGetDeviceProperties(&properties, 0), cudaSuccess);
        ASSERT_EQ(properties.major, 10);
        initRuntime(0, false, false, MlaOpsType::AUTO);
        kv_.seq_size_per_block                      = 128;
        kv_.linear_step                             = linearStep();
        kv_.test_block_num                          = 16;
        kv_.dsv4_fixed_pool_blocks                  = 16;
        kv_.enable_memory_cache                     = true;
        kv_.memory_cache_size_mb                    = 64;
        kv_.memory_cache_sync_timeout_ms            = 30000;
        kv_.prefix_tree_memory_state_swa_pool_ratio = 50;
        SpeculativeExecutionConfig spec;
        spec.type              = SP_TYPE_DSPARK;
        spec.gen_num_per_cycle = 5;
        config_                = CacheConfigCreator::createSpConfig(
            model(false), model(true), ParallelismConfig(), RuntimeConfig(), kv_, spec, std::nullopt, true, false);
        allocator_ = std::make_shared<HybridPoolKVCacheAllocator>(config_, AllocationType::DEVICE);
        allocator_->setSharedBlockCache(std::make_shared<SharedBlockCache>());
        ASSERT_TRUE(allocator_->init());
        grpc::ServerBuilder builder;
        int                 port = 0;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(&service_);
        server_ = builder.BuildAndStart();
        ASSERT_TRUE(server_);
        memory_ = std::make_shared<KVCacheMemoryConnector>(
            config_, kv_, allocator_, std::vector<std::string>{"127.0.0.1:" + std::to_string(port)});
        ASSERT_TRUE(memory_->init());
        service_.connector = memory_.get();
        meta_              = std::make_shared<MemoryMeta>();
    }

    void TearDown() override {
        if (server_) {
            server_->Shutdown();
            server_->Wait();
        }
        service_.connector = nullptr;
        memory_.reset();
        allocator_.reset();
    }

    DSV41CacheIdentity identity(DSV41ReplayMode mode = DSV41ReplayMode::FULL) const {
        return memory_->dsv41CacheIdentity("gpu-component-revision", mode);
    }

    DSV41CheckpointMetadata metadata(const DSV41CacheIdentity& id, size_t count) const {
        DSV41CheckpointMetadata value;
        const int64_t           end = count * 128;
        value.identity              = id;
        value.materialized_end = value.encoder_materialized_end = value.decoder_checkpoint_end = end;
        value.aux_valid_start                                                                  = end - 128;
        value.aux_valid_end                                                                    = end;
        value.global_entries = value.index_entries = {end / 2, end / 2, end / 2, end};
        for (size_t layer = 0; layer < value.swa.size(); ++layer) {
            const auto floor = id.replay_mode == DSV41ReplayMode::BOUNDED_CHECKPOINT_V1 && layer > 20 ? end - 128 : 0;
            value.swa[layer] = {end - 128, end, floor};
        }
        value.history_token_ids = {1, 2, 3};
        value.history_ready = value.draft_committed = value.pair_empty = true;
        return value;
    }

    BatchKVCacheResourcePtr resource(size_t count, DSV41ReplayMode mode = DSV41ReplayMode::FULL, bool allocate = true) {
        auto batch = std::make_shared<BatchKVCacheResource>();
        batch->resetBatchSize(1);
        batch->initGroups(6, 43, config_.layer_to_group_id, 1, config_.group_types, config_.layer_region_to_group_id);
        auto&         result = batch->cacheResource(0);
        CacheKeysType keys(count + 1);
        std::iota(keys.begin(), keys.end(), 100);
        result.setCacheKeys(keys);
        result.setLastBlockAligned(false);
        result.setDsv41CacheState(std::make_shared<DSV41CacheState>(identity(mode)));
        if (allocate) {
            for (size_t group = 0; group < 6; ++group) {
                auto ids = allocator_->groupBlockPools()[group]->malloc(group < 4 ? count : 1);
                if (ids.size() != (group < 4 ? count : 1))
                    throw std::runtime_error("fixture GPU allocation failed");
                BlockIndicesType mapping(count + 1, NULL_BLOCK_IDX);
                if (group < 4)
                    std::copy(ids.begin(), ids.end(), mapping.begin());
                else
                    mapping[count - 1] = ids.front();
                result.mutableBlockIds(group).assign(std::move(mapping));
            }
        }
        return batch;
    }

    void ready(const BatchKVCacheResourcePtr& batch, size_t count) {
        auto& state = batch->cacheResource(0).dsv41CacheState();
        state->advanceEncoder(count * 128);
        state->completeDecoder(metadata(state->view().identity, count), 128);
        state->finish(count * 128);
    }

    void fill(const BatchKVCacheResourcePtr& batch, uint8_t salt) {
        for (size_t group = 0; group < 6; ++group) {
            for (int owner : config_.global_layer_ids[group]) {
                for (auto block : batch->blocks(0, group)) {
                    if (block <= 0)
                        continue;
                    for (const auto& segment :
                         allocator_->convertIndexToBuffer(owner, config_.group_region_names[group], block)) {
                        std::vector<uint8_t> bytes(segment.size_bytes);
                        for (size_t i = 0; i < bytes.size(); ++i)
                            bytes[i] = group == 4 ? 0 : (i * 37 + owner * 11 + group * 17 + salt) % 251;
                        cudaCheck(cudaMemcpy(segment.addr, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
                    }
                }
            }
        }
    }

    Bytes bytes(const KVCacheResource& resource, size_t count) {
        Bytes result;
        for (size_t group = 0; group < 6; ++group) {
            for (int owner : config_.global_layer_ids[group]) {
                for (size_t index = group < 4 ? 0 : count - 1; index < count; ++index) {
                    auto& value = result[{group, owner, index}];
                    for (const auto& segment : allocator_->convertIndexToBuffer(
                             owner, config_.group_region_names[group], resource.blocks(group).at(index))) {
                        const size_t offset = value.size();
                        value.resize(offset + segment.size_bytes);
                        cudaCheck(cudaMemcpy(
                            value.data() + offset, segment.addr, segment.size_bytes, cudaMemcpyDeviceToHost));
                    }
                }
            }
        }
        return result;
    }

    CompleteTokenIdsPtr tokens(int length) {
        auto result            = std::make_shared<CompleteTokenIds>(1, 1, length + 64, 128);
        auto input             = std::make_shared<GenerateInput>();
        input->input_ids       = torch::arange(1, length + 1, torch::kInt32);
        input->generate_config = std::make_shared<GenerateConfig>();
        result->init(input);
        return result;
    }

    void free(const BatchKVCacheResourcePtr& batch) {
        allocator_->free(FreeInfo{batch, tokens(129)});
    }

    std::array<size_t, 6> freeCounts() const {
        std::array<size_t, 6> result;
        for (size_t group = 0; group < result.size(); ++group)
            result[group] = allocator_->groupBlockPools()[group]->freeBlocksNum();
        return result;
    }

    KVCacheConfig                               kv_;
    CacheConfig                                 config_;
    std::shared_ptr<HybridPoolKVCacheAllocator> allocator_;
    std::shared_ptr<KVCacheMemoryConnector>     memory_;
    std::shared_ptr<MemoryMeta>                 meta_;
    CopyService                                 service_;
    std::unique_ptr<grpc::Server>               server_;
};

TEST_F(DSV41GpuCacheAllocatorTest, SameKeysKeepModeIdentityAndRestoreExactBytesIntoPrivateFixedPages) {
    for (auto mode : {DSV41ReplayMode::FULL, DSV41ReplayMode::BOUNDED_CHECKPOINT_V1}) {
        auto source = resource(1, mode);
        fill(source, mode == DSV41ReplayMode::FULL ? 19 : 67);
        const auto expected      = bytes(source->cacheResource(), 1);
        const auto source_global = source->blocks(0, 0)[0];
        const auto source_swa    = source->blocks(0, 5)[0];
        ready(source, 1);
        allocator_->insertIntoCache(InsertInfo{source, tokens(129), false});
        free(source);
        auto destination = resource(1, mode, false);
        auto result      = allocator_->malloc(MallocInfo{destination, tokens(129)});
        ASSERT_TRUE(result.success);
        EXPECT_EQ(result.reuse_len, 128);
        EXPECT_EQ(destination->blocks(0, 0)[0], source_global);
        EXPECT_NE(destination->blocks(0, 5)[0], source_swa);
        EXPECT_EQ(bytes(destination->cacheResource(), 1), expected);
        EXPECT_EQ(destination->cacheResource().dsv41CacheState()->view().decoder_checkpoint_end, 128);
        EXPECT_FALSE(destination->cacheResource().dsv41GpuRestorePending());
        const auto lease = destination->cacheResource().dsv41GpuLease();
        ASSERT_TRUE(lease);
        for (int owner : config_.global_layer_ids[5]) {
            for (const auto& segment :
                 allocator_->convertIndexToBuffer(owner, KVCacheRegionName::SWA_KV, destination->blocks(0, 5)[0]))
                cudaCheck(cudaMemset(segment.addr, 233, segment.size_bytes));
        }
        auto cached = allocator_->popBlocksFromCache(1);
        if (cached) {
            EXPECT_FALSE(cached->cacheResource().dsv41CacheState()->view().identity == identity(mode));
            allocator_->blockCacheFree(cached);
        }
        free(destination);
    }
    auto full    = allocator_->sharedBlockCache()->matchDsv41Checkpoint(identity(), {100}, 128, 1);
    auto bounded = allocator_->sharedBlockCache()->matchDsv41Checkpoint(
        identity(DSV41ReplayMode::BOUNDED_CHECKPOINT_V1), {100}, 128, 1);
    ASSERT_TRUE(full);
    ASSERT_TRUE(bounded);
    EXPECT_NE(full->data.blocks[5], bounded->data.blocks[5]);
}

TEST_F(DSV41GpuCacheAllocatorTest, IncompleteOrStaleProducerNeverPublishesAllocatedRegions) {
    auto incomplete = resource(1);
    fill(incomplete, 11);
    const auto before = allocator_->blockCacheRefBlocksNum();
    incomplete->cacheResource().dsv41CacheState()->advanceEncoder(128);
    EXPECT_THROW(allocator_->insertIntoCache(InsertInfo{incomplete, tokens(129), false}), std::invalid_argument);
    EXPECT_EQ(allocator_->blockCacheRefBlocksNum(), before);
    free(incomplete);

    auto stale = resource(2);
    auto state = stale->cacheResource().dsv41CacheState();
    state->requireProtectedPrefix(256, 385);
    state->advanceEncoder(256);
    auto meta = metadata(identity(), 2);
    state->completeDecoder(meta, 128);
    state->protect(std::make_shared<DSV41CheckpointSnapshot>(meta));
    state->advanceEncoder(385);
    state->completeHandoff(385, true, true, true);
    state->finish(385);
    EXPECT_THROW(allocator_->insertIntoCache(InsertInfo{stale, tokens(385), false}), std::invalid_argument);
    EXPECT_FALSE(allocator_->sharedBlockCache()->hasDsv41Checkpoints());
    free(stale);
}

TEST_F(DSV41GpuCacheAllocatorTest, FixedCopyAllocationFailureRollsBackRefsPagesAndBothProgressAxes) {
    auto source = resource(1);
    fill(source, 23);
    ready(source, 1);
    allocator_->insertIntoCache(InsertInfo{source, tokens(129), false});
    free(source);
    auto       pool        = allocator_->groupBlockPools()[5];
    const auto held        = pool->malloc(pool->freeBlocksNum() - 1);
    const auto before      = freeCounts();
    auto       destination = resource(1, DSV41ReplayMode::FULL, false);
    auto       result      = allocator_->malloc(MallocInfo{destination, tokens(129)});
    EXPECT_FALSE(result.success);
    EXPECT_EQ(destination->curBlocksNum(), 0);
    EXPECT_EQ(destination->cacheResource().deviceReuseBlockNum(), 0);
    EXPECT_EQ(destination->cacheResource().dsv41CacheState()->view().encoder_materialized_end, 0);
    EXPECT_EQ(destination->cacheResource().dsv41CacheState()->view().decoder_checkpoint_end, 0);
    EXPECT_EQ(freeCounts(), before);
    EXPECT_TRUE(allocator_->sharedBlockCache()->hasDsv41Checkpoints());
    pool->requestFree(held);
    ASSERT_TRUE(allocator_->malloc(MallocInfo{destination, tokens(129)}).success);
    EXPECT_EQ(destination->cacheResource().dsv41CacheState()->view().decoder_checkpoint_end, 128);
    free(destination);
}

TEST_F(DSV41GpuCacheAllocatorTest, UntypedLegacyEntryCannotSupplyTypedReuse) {
    auto                      source = resource(1);
    std::vector<BlockIdxType> slots(6, NULL_BLOCK_IDX);
    for (size_t group = 0; group < slots.size(); ++group)
        slots[group] = source->blocks(0, group)[0];
    allocator_->sharedBlockCache()->put(100, slots, false);
    free(source);
    auto destination = resource(1, DSV41ReplayMode::FULL, false);
    auto result      = allocator_->malloc(MallocInfo{destination, tokens(129)});
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 0);
    EXPECT_EQ(destination->cacheResource().dsv41CacheState()->view().decoder_checkpoint_end, 0);
    free(destination);
}

TEST_F(DSV41GpuCacheAllocatorTest, MemoryTransferKeepsGpuCheckpointOnFailureAndPreservesEveryByteOnRetry) {
    auto source = resource(2);
    fill(source, 31);
    const auto expected = bytes(source->cacheResource(), 2);
    ready(source, 2);
    allocator_->insertIntoCache(InsertInfo{source, tokens(257), false});
    free(source);
    auto evicted = allocator_->popBlocksFromCache(1);
    ASSERT_TRUE(evicted);
    ASSERT_TRUE(evicted->cacheResource().isDsv41GpuTransfer());
    auto       transfer = std::make_shared<KVCacheResource>(evicted->cacheResource());
    const auto before   = freeCounts();
    service_.fail_next  = true;
    EXPECT_FALSE(memory_->asyncWrite(transfer, meta_)->success());
    EXPECT_EQ(freeCounts(), before);
    EXPECT_TRUE(memory_->cacheKeys().empty());
    EXPECT_TRUE(allocator_->sharedBlockCache()->hasDsv41Checkpoints());
    ASSERT_TRUE(memory_->asyncWrite(transfer, meta_)->success());
    EXPECT_FALSE(allocator_->sharedBlockCache()->hasDsv41Checkpoints());
    allocator_->blockCacheFree(evicted);
    transfer.reset();
    evicted.reset();
    auto destination = resource(2);
    auto input       = std::make_shared<KVCacheResource>(destination->cacheResource());
    auto match       = memory_->asyncMatch(input, meta_);
    ASSERT_TRUE(match);
    ASSERT_TRUE(memory_->asyncRead(input, meta_, match, 0, 2)->success());
    EXPECT_EQ(bytes(*input, 2), expected);
    EXPECT_EQ(input->dsv41CacheState()->view().decoder_checkpoint_end, 256);
    free(destination);
}

class DSV41GpuLongSuffixTest: public DSV41GpuCacheAllocatorTest {
protected:
    int linearStep() const override {
        return 2;
    }
};

TEST_F(DSV41GpuLongSuffixTest, InitialAllocationRetainsExactCheckpointAcrossShortAndLongSuffixes) {
    for (auto mode : {DSV41ReplayMode::FULL, DSV41ReplayMode::BOUNDED_CHECKPOINT_V1}) {
        auto source = resource(1, mode);
        fill(source, 43);
        const auto expected = bytes(source->cacheResource(), 1);
        ready(source, 1);
        allocator_->insertIntoCache(InsertInfo{source, tokens(129), false});
        free(source);
        const auto before = freeCounts();
        for (const int suffix : {1, 3, 127, 128, 129, 257}) {
            SCOPED_TRACE(suffix);
            const int total       = 128 + suffix;
            auto      destination = resource((total - 1) / 128, mode, false);
            auto      result      = allocator_->malloc(MallocInfo{destination, tokens(total)});
            ASSERT_TRUE(result.success);
            ASSERT_EQ(result.reuse_len, 128);
            ASSERT_GT(destination->blocks(0, 4)[0], 0);
            ASSERT_GT(destination->blocks(0, 5)[0], 0);
            EXPECT_EQ(bytes(destination->cacheResource(), 1), expected);
            EXPECT_EQ(destination->cacheResource().dsv41CacheState()->view().decoder_checkpoint_end, 128);
            auto lease = destination->cacheResource().dsv41GpuLease();
            ASSERT_TRUE(lease);
            EXPECT_NE(destination->blocks(0, 5)[0], lease->data.blocks[5][0]);
            for (int owner : config_.global_layer_ids[5]) {
                for (const auto& segment :
                     allocator_->convertIndexToBuffer(owner, KVCacheRegionName::SWA_KV, destination->blocks(0, 5)[0]))
                    cudaCheck(cudaMemset(segment.addr, 219, segment.size_bytes));
            }
            KVCacheResource retained;
            retained.initGroups(
                6, 43, config_.layer_to_group_id, 1, config_.group_types, config_.layer_region_to_group_id);
            for (size_t group = 0; group < 6; ++group)
                retained.mutableBlockIds(group).assign(lease->data.blocks[group]);
            EXPECT_EQ(bytes(retained, 1), expected);
            free(destination);
            EXPECT_EQ(freeCounts(), before);
        }
    }
}

}  // namespace rtp_llm::test
