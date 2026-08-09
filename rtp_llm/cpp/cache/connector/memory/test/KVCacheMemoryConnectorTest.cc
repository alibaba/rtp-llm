// Copyright (c) RTP-LLM

#include <csignal>
#include <chrono>
#include <execinfo.h>
#include <thread>
#include <unistd.h>

#include <grpcpp/alarm.h>

#include "gtest/gtest.h"
#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.h"
#include "rtp_llm/cpp/cache/connector/memory/MemoryAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/memory/MemoryBlockCache.h"
#include "rtp_llm/cpp/cache/connector/memory/test/mock/TestRpcService.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/MLAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/EplbConfig.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm::test {

namespace {

// When bazel runs tests, a SIGSEGV in the test binary often ends up as a core dump of the bazel bash wrapper.
// Install a lightweight handler to print a backtrace to stderr so the real crash site is visible in test.log.
void crashBacktraceHandler(int sig) {
    void*  stack[128];
    size_t n = static_cast<size_t>(::backtrace(stack, 128));
    {
        constexpr char kHeader[] = "\n=== RTP-LLM unit test crash backtrace ===\n";
        const auto     rc        = ::write(STDERR_FILENO, kHeader, sizeof(kHeader) - 1);
        (void)rc;
    }
    ::backtrace_symbols_fd(stack, static_cast<int>(n), STDERR_FILENO);
    {
        constexpr char kFooter[] = "\n=== end backtrace ===\n";
        const auto     rc        = ::write(STDERR_FILENO, kFooter, sizeof(kFooter) - 1);
        (void)rc;
    }
    ::_exit(128 + sig);
}

struct CrashHandlerInstaller {
    CrashHandlerInstaller() {
        std::signal(SIGSEGV, crashBacktraceHandler);
        std::signal(SIGABRT, crashBacktraceHandler);
    }
};

template<typename Fn>
void expectUninitializedCacheGroups(Fn fn) {
    try {
        fn();
        FAIL() << "expected uninitialized cache groups to throw";
    } catch (const rtp_llm::RTPException& e) {
        EXPECT_NE(std::string(e.what()).find("KVCacheResource groups are not initialized"), std::string::npos);
    }
}

static CrashHandlerInstaller g_crash_handler_installer;

}  // namespace

// Test-local helper struct. Business code no longer exposes a LayerBlock type.
struct LayerBlock {
    int          layer_id{0};
    BlockIdxType block_id{NULL_BLOCK_IDX};
};

class TestReadMeta: public rtp_llm::Meta {
public:
    TestReadMeta(bool enable_memory_cache, bool enable_remote_cache = false, std::string trace_id = ""):
        enable_memory_cache_(enable_memory_cache), enable_remote_cache_(enable_remote_cache), trace_id_(trace_id) {}
    ~TestReadMeta() override = default;

    bool enableMemoryCache() const override {
        return enable_memory_cache_;
    }
    bool enableRemoteCache() const override {
        return enable_remote_cache_;
    }
    const std::string& trace_id() const override {
        return trace_id_;
    }
    const std::string& unique_id() const override {
        return unique_id_;
    }
    const std::vector<int64_t>& tokens() const override {
        return tokens_;
    }

private:
    bool                 enable_memory_cache_{false};
    bool                 enable_remote_cache_{false};
    std::string          trace_id_;
    std::string          unique_id_ = "";
    std::vector<int64_t> tokens_;  // TODO : get tokens (remote connector)
};

class KVCacheMemoryConnectorTest: public ::testing::Test {
protected:
    void SetUp() override {
        createDevice();

        cache_config_ = createMockCacheConfig();
        allocator_ =
            std::make_shared<HybridPoolKVCacheAllocator>(cache_config_, AllocationType::DEVICE);
        ASSERT_TRUE(allocator_->init());

        const int server_num = 4;
        startRpcServer(server_num);

        connector_ = std::make_shared<KVCacheMemoryConnector>(
            cache_config_, kv_cache_config_, allocator_, server_addrs_);
        ASSERT_TRUE(connector_->init());
    }

    void TearDown() override {}

    CacheConfig                                 cache_config_;
    KVCacheConfig                               kv_cache_config_;
    std::shared_ptr<KVCacheAllocator>  allocator_;
    std::shared_ptr<KVCacheMemoryConnector>     connector_;
    std::vector<std::unique_ptr<TestRpcServer>> servers_;
    std::vector<std::string>                    server_addrs_;

private:
    void createDevice() const {
        initRuntime(/*device_id=*/0,
                    /*trace_memory=*/false,
                    /*enable_comm_overlap=*/false,
                    MlaOpsType::AUTO);
    }
    CacheConfig createMockCacheConfig(int               layer_num          = 4,
                                      int               block_num          = 10,
                                      int               seq_size_per_block = 8,
                                      rtp_llm::DataType mha_dtype          = rtp_llm::DataType::TYPE_FP16) {
        constexpr int kTestMemoryCacheSizeMb          = 64;
        constexpr int kTestMemoryCacheSyncTimeout     = 1000;
        kv_cache_config_.memory_cache_size_mb         = kTestMemoryCacheSizeMb;
        kv_cache_config_.memory_cache_sync_timeout_ms = kTestMemoryCacheSyncTimeout;
        return makeSimpleMhaCacheConfig(layer_num,
                                        block_num,
                                        static_cast<size_t>(seq_size_per_block),
                                        mha_dtype,
                                        /*local_head_num_kv=*/8,
                                        /*size_per_head=*/128);
    }
    void startRpcServer(int server_num) {
        for (int i = 0; i < server_num; ++i) {
            auto service = std::make_unique<TestRpcService>();
            auto server  = std::make_unique<TestRpcServer>(std::move(service));
            ASSERT_TRUE(server->start());
            server_addrs_.push_back("127.0.0.1:" + std::to_string(server->listenPort()));
            servers_.push_back(std::move(server));
        }
    }

protected:
    void resetToHybridCacheConfig(int linear_step = 1) {
        cache_config_             = makeSimpleHybridMhaCacheConfig(/*layer_num=*/4,
                                                       /*block_num=*/10,
                                                       /*tokens_per_block=*/8,
                                                       DataType::TYPE_FP16,
                                                       /*group_layer_num=*/2,
                                                       /*local_head_num_kv=*/8,
                                                       /*size_per_head=*/128);
        cache_config_.linear_step = linear_step;
        allocator_ =
            std::make_shared<HybridPoolKVCacheAllocator>(cache_config_, AllocationType::DEVICE);
        ASSERT_TRUE(allocator_->init());
        connector_ = std::make_shared<KVCacheMemoryConnector>(
            cache_config_, kv_cache_config_, allocator_, server_addrs_);
        ASSERT_TRUE(connector_->init());
    }

private:
    // BlockInfo helpers: convertIndexToBuffer() now returns std::vector<BlockInfo>.
    size_t sumBlockInfosBytes(const std::vector<BlockInfo>& infos) const {
        size_t total = 0;
        for (const auto& b : infos) {
            if (b.addr && b.size_bytes > 0) {
                total += b.size_bytes;
            }
        }
        return total;
    }
    size_t memoryCacheBlockBytes(const CacheConfig& cfg) const {
        size_t total = 0;
        for (int layer = 0; layer < static_cast<int>(cfg.totalLayerNum()); ++layer) {
            for (const auto& group_ref : cfg.groupsForLayer(layer)) {
                const auto& group = group_ref.get();
                if (!group.policy.enable_prefix_reuse) {
                    continue;
                }
                total += group.kv_block_stride_bytes + group.kv_scale_stride_bytes;
            }
        }
        return total;
    }
    size_t memoryCacheBlockBytes() const {
        return memoryCacheBlockBytes(cache_config_);
    }
    void setGroupBlockBytes(CacheConfig& cfg, size_t stride_bytes) const {
        if (stride_bytes == 0) {
            auto spec = std::make_shared<MHAKVCacheSpec>(static_cast<uint32_t>(cfg.seq_size_per_block));
            spec->tag = "default";
            std::vector<int> layer_ids(cfg.totalLayerNum());
            std::iota(layer_ids.begin(), layer_ids.end(), 0);
            setTestTopology(cfg,
                            {makeTestGroupForConfig(cfg, spec, std::move(layer_ids), CacheGroupType::FULL, "default")});
            return;
        }
        const auto                 topology_groups = cfg.topology().groups();
        std::vector<GroupBase> groups(topology_groups.begin(), topology_groups.end());
        for (auto& group : groups) {
            group.kv_block_stride_bytes = stride_bytes;
            group.kv_scale_stride_bytes = 0;
        }
        cfg.setTopology(std::move(groups), cfg.topology().layers());
    }

    void setBlockBytes(const BlockInfo& b, size_t byte_offset, size_t byte_len, char c) const {
        ASSERT_NE(b.addr, nullptr);
        ASSERT_LE(byte_offset + byte_len, b.size_bytes);
        auto* addr = static_cast<char*>(b.addr) + byte_offset;
        if (b.is_cuda) {
            check_cuda_value(cudaMemset(addr, c, byte_len));
        } else {
            memset(addr, c, byte_len);
        }
    }

    void verifyBlockBytesEq(const BlockInfo& b, size_t byte_offset, size_t byte_len, char expected) const {
        ASSERT_NE(b.addr, nullptr);
        ASSERT_LE(byte_offset + byte_len, b.size_bytes);
        auto* addr = static_cast<const char*>(b.addr) + byte_offset;

        std::vector<unsigned char> data(byte_len, 0);
        if (b.is_cuda) {
            check_cuda_value(cudaMemcpy(data.data(), addr, byte_len, cudaMemcpyDeviceToHost));
        } else {
            memcpy(data.data(), addr, byte_len);
        }
        size_t mismatch = 0;
        for (; mismatch < byte_len; ++mismatch) {
            if (data[mismatch] != static_cast<unsigned char>(expected)) {
                break;
            }
        }
        ASSERT_EQ(mismatch, byte_len) << "mismatch at byte offset " << mismatch << " expect '" << expected << "' got 0x"
                                      << std::hex << static_cast<int>(data[mismatch]) << std::dec;
    }

    void setBlockInfosContent(const std::vector<BlockInfo>& infos, char c) const {
        for (const auto& b : infos) {
            if (!b.addr || b.size_bytes == 0) {
                continue;
            }
            setBlockBytes(b, /*byte_offset=*/0, b.size_bytes, c);
        }
    }

    void verifyBlockInfosContent(const std::vector<BlockInfo>& infos, char c) const {
        for (const auto& b : infos) {
            if (!b.addr || b.size_bytes == 0) {
                continue;
            }
            verifyBlockBytesEq(b, /*byte_offset=*/0, b.size_bytes, c);
        }
    }

    void addTaggedGpuBlocks(MemoryOperationRequestPB::CopyItem& item,
                            const std::vector<BlockIdxType>&    blocks_by_layer) const {
        const auto& slots = connector_->layerGroupSlots();
        for (const auto& slot : slots) {
            ASSERT_GE(slot.layer_id, 0);
            ASSERT_LT(static_cast<size_t>(slot.layer_id), blocks_by_layer.size());
            auto* tagged_block = item.add_tagged_gpu_blocks();
            tagged_block->set_layer_id(slot.layer_id);
            tagged_block->set_tag(slot.tag);
            tagged_block->set_block_id(blocks_by_layer[static_cast<size_t>(slot.layer_id)]);
        }
    }

    void verifyGpuBufferContent(const std::vector<LayerBlock>& gpu_layer_blocks) const {
        std::vector<BlockIdxType> blocks_by_layer(cache_config_.totalLayerNum(), NULL_BLOCK_IDX);
        for (const auto& layer_block : gpu_layer_blocks) {
            blocks_by_layer.at(static_cast<size_t>(layer_block.layer_id)) = layer_block.block_id;
        }
        for (const auto& slot : connector_->layerGroupSlots()) {
            const auto block_id = blocks_by_layer.at(static_cast<size_t>(slot.layer_id));
            if (isNullBlockIdx(block_id)) {
                continue;
            }
            const auto gpu_bufs = allocator_->convertIndexToBuffer(slot.layer_id, slot.tag, block_id);
            ASSERT_GT(sumBlockInfosBytes(gpu_bufs), 0u);
            verifyBlockInfosContent(gpu_bufs, static_cast<char>('k' + slot.layer_id));
        }
    }
    void verifyCpuBufferContent(const std::vector<LayerBlock>& gpu_layer_blocks,
                                BlockIdxType                   mem_block_index,
                                size_t                         mem_block_size) const {
        auto pool = requireExistingBlockPool(mem_block_size);
        ASSERT_NE(pool, nullptr);
        const auto mem_bufs = pool->convertIndexToBuffer(0, mem_block_index);
        ASSERT_EQ(mem_bufs.size(), 1u);
        const auto& mem_buffer = mem_bufs[0];
        ASSERT_NE(mem_buffer.addr, nullptr);
        ASSERT_GE(mem_buffer.size_bytes, mem_block_size);

        const size_t              layer_num = static_cast<size_t>(cache_config_.totalLayerNum());
        std::vector<BlockIdxType> layer_to_block(layer_num, NULL_BLOCK_IDX);
        for (const auto& lb : gpu_layer_blocks) {
            ASSERT_GE(lb.layer_id, 0);
            ASSERT_LT(static_cast<size_t>(lb.layer_id), layer_num);
            layer_to_block[static_cast<size_t>(lb.layer_id)] = lb.block_id;
        }

        size_t byte_off = 0;
        for (const auto& slot : connector_->layerGroupSlots()) {
            const auto block_id = layer_to_block[static_cast<size_t>(slot.layer_id)];
            if (isNullBlockIdx(block_id)) {
                byte_off += slot.stride_bytes;
                continue;
            }
            const auto gpu_bufs = allocator_->convertIndexToBuffer(slot.layer_id, slot.tag, block_id);
            const auto bytes    = sumBlockInfosBytes(gpu_bufs);
            ASSERT_GT(bytes, 0u);
            ASSERT_LE(bytes, slot.stride_bytes);

            const char expected_k = static_cast<char>('k' + slot.layer_id);
            verifyBlockBytesEq(mem_buffer, byte_off, bytes, expected_k);
            byte_off += slot.stride_bytes;
        }
    }
    void prepareBufferContent(const std::vector<LayerBlock>& gpu_layer_blocks,
                              BlockIdxType&                  mem_block_index,
                              size_t&                        mem_block_size,
                              bool                           fill_gpu,
                              bool                           fill_cpu) const {
        // std::vector<KVCacheMemoryConnector::LayerBlock> gpu_layer_blocks{
        //     {/*layer_id*/0, /*block_id*/1},
        //     {/*layer_id*/1, /*block_id*/2},
        //     {/*layer_id*/2, /*block_id*/2},
        // };
        const size_t              layer_num = static_cast<size_t>(cache_config_.totalLayerNum());
        std::vector<BlockIdxType> layer_to_block(layer_num, NULL_BLOCK_IDX);
        for (const auto& lb : gpu_layer_blocks) {
            ASSERT_GE(lb.layer_id, 0);
            ASSERT_LT(static_cast<size_t>(lb.layer_id), layer_num);
            layer_to_block[static_cast<size_t>(lb.layer_id)] = lb.block_id;
        }

        size_t total = 0;
        for (const auto& slot : connector_->layerGroupSlots()) {
            total += slot.stride_bytes;
        }

        for (const auto& slot : connector_->layerGroupSlots()) {
            const auto block_id = layer_to_block[static_cast<size_t>(slot.layer_id)];
            if (isNullBlockIdx(block_id)) {
                continue;
            }
            const auto gpu_bufs = allocator_->convertIndexToBuffer(slot.layer_id, slot.tag, block_id);
            const auto bytes    = sumBlockInfosBytes(gpu_bufs);
            ASSERT_GT(bytes, 0u);
            ASSERT_LE(bytes, slot.stride_bytes);
            if (fill_gpu) {
                setBlockInfosContent(gpu_bufs, static_cast<char>('k' + slot.layer_id));
            }
        }
        if (fill_gpu) {
            check_cuda_value(cudaDeviceSynchronize());
        }

        // 申请memory block
        auto pool = ensureBlockPool(total);
        ASSERT_NE(pool, nullptr);
        auto mem_blocks = pool->malloc(1);
        ASSERT_EQ(mem_blocks.size(), 1u);
        const BlockIdxType malloced_mem_block_index = static_cast<BlockIdxType>(mem_blocks[0]);
        const auto         mem_bufs                 = pool->convertIndexToBuffer(0, malloced_mem_block_index);
        ASSERT_EQ(mem_bufs.size(), 1u);
        const auto& mem_buffer = mem_bufs[0];
        ASSERT_NE(mem_buffer.addr, nullptr);
        EXPECT_GE(mem_buffer.size_bytes, total);

        // Fill memory buffer (merged layout: reserve per-layer stride even if block is null).
        if (fill_cpu) {
            size_t byte_off = 0;
            for (const auto& slot : connector_->layerGroupSlots()) {
                const auto block_id = layer_to_block[static_cast<size_t>(slot.layer_id)];
                if (isNullBlockIdx(block_id)) {
                    byte_off += slot.stride_bytes;
                    continue;
                }
                const auto gpu_bufs =
                    allocator_->convertIndexToBuffer(slot.layer_id, slot.tag, block_id);
                const auto bytes = sumBlockInfosBytes(gpu_bufs);
                ASSERT_GT(bytes, 0u);
                ASSERT_LE(bytes, slot.stride_bytes);
                setBlockBytes(mem_buffer, byte_off, bytes, static_cast<char>('k' + slot.layer_id));
                byte_off += slot.stride_bytes;
            }
        }

        mem_block_index = malloced_mem_block_index;
        // Use the actual pool block size as the mem-block-size key.
        mem_block_size = mem_buffer.size_bytes;
    }
    void addOneCopyItemToPb(MemoryOperationRequestPB&      req,
                            const std::vector<LayerBlock>& gpu_layer_blocks,
                            BlockIdxType                   mem_block_index) const {
        auto*                     item      = req.add_copy_items();
        const size_t              layer_num = static_cast<size_t>(cache_config_.totalLayerNum());
        std::vector<BlockIdxType> blocks(layer_num, NULL_BLOCK_IDX);
        for (const auto& layer_block : gpu_layer_blocks) {
            ASSERT_GE(layer_block.layer_id, 0);
            ASSERT_LT(static_cast<size_t>(layer_block.layer_id), layer_num);
            blocks[static_cast<size_t>(layer_block.layer_id)] = layer_block.block_id;
        }
        addTaggedGpuBlocks(*item, blocks);
        item->set_mem_block(static_cast<int>(mem_block_index));
    }
    std::shared_ptr<KVCacheResource>
    makeCacheResource(const CacheKeysType& cache_keys, BlockIndicesType block_indices, size_t reuse_len = 0) const {
        auto res = std::make_shared<KVCacheResource>();
        res->setCacheKeys(cache_keys);
        res->initGroups(cache_config_.topologyPtr());
        if (block_indices.size() < cache_keys.size()) {
            block_indices.resize(cache_keys.size(), NULL_BLOCK_IDX);
        }
        res->mutableBlockIds("default").assign(block_indices);
        // reuse_len in these tests means "GPU already-reused prefix length".
        // KVCacheResource::reuseBlockNum() is derived from (device + memory + remote),
        // so set device reuse here to make asyncMatch/asyncRead semantics consistent.
        res->setDeviceReuseBlockNum(reuse_len);
        // These unit tests want to include the whole cache_keys range by default.
        res->setLastBlockAligned(true);
        return res;
    }

    std::shared_ptr<KVCacheResource> makeHybridCacheResource(const CacheKeysType&             cache_keys,
                                                             const std::vector<BlockIdxType>& linear_blocks,
                                                             const std::vector<BlockIdxType>& full_blocks,
                                                             size_t                           reuse_len = 0) const {
        auto res = std::make_shared<KVCacheResource>();
        res->setCacheKeys(cache_keys);
        const size_t layer_num = static_cast<size_t>(cache_config_.totalLayerNum());
        RTP_LLM_CHECK_WITH_INFO(layer_num == 4, "test helper expects 4 layers, got %zu", layer_num);
        res->initGroups(cache_config_.topologyPtr());
        auto normalized_linear_blocks = linear_blocks;
        if (normalized_linear_blocks.size() < cache_keys.size()) {
            normalized_linear_blocks.resize(cache_keys.size(), NULL_BLOCK_IDX);
        }
        auto normalized_full_blocks = full_blocks;
        if (normalized_full_blocks.size() < cache_keys.size()) {
            normalized_full_blocks.resize(cache_keys.size(), NULL_BLOCK_IDX);
        }
        res->mutableBlockIds("linear").assign(normalized_linear_blocks);
        res->mutableBlockIds("full1").assign(normalized_full_blocks);
        res->setDeviceReuseBlockNum(reuse_len);
        res->setLastBlockAligned(true);
        return res;
    }

    // Put items into memory block cache.
    // If `is_complete_flags` is empty, all items are treated as "complete" by default.
    std::vector<BlockIdxType> putItemsToCache(const CacheKeysType&        keys,
                                              size_t                      mem_block_size,
                                              std::initializer_list<bool> is_complete_flags = {}) const {
        RTP_LLM_CHECK_WITH_INFO(
            is_complete_flags.size() == 0 || keys.size() == is_complete_flags.size(),
            "keys size must equal is_complete_flags size when flags are provided, keys=%zu flags=%zu",
            keys.size(),
            is_complete_flags.size());

        std::vector<BlockIdxType> block_indices;
        if (keys.empty()) {
            return block_indices;
        }

        auto pool = ensureBlockPool(mem_block_size);
        if (!pool) {
            return block_indices;
        }

        for (size_t i = 0; i < keys.size(); ++i) {
            auto blocks = pool->malloc(1);  // will increase request ref
            if (blocks.size() != 1u) {
                ADD_FAILURE() << "malloc memory block failed, block_size=" << mem_block_size;
                break;
            }
            const BlockIdxType block_idx = static_cast<BlockIdxType>(blocks[0]);
            block_indices.push_back(block_idx);

            MemoryBlockCache::CacheItem item;
            item.cache_key   = keys[i];
            item.block_index = block_idx;
            item.block_size  = mem_block_size;
            item.is_resident = false;
            item.is_complete = (is_complete_flags.size() == 0) ? true : *(is_complete_flags.begin() + i);
            connector_->block_cache_->put(item);

            pool->blockCacheReference({block_idx});

            // malloc会增加request ref, 所以这里需要requestFree减少request ref
            pool->requestFree({block_idx});
        }

        return block_indices;
    }
    std::shared_ptr<BlockPool> ensureBlockPool(size_t block_size) const {
        auto pool = connector_->block_pool_;
        if (!pool && !connector_->isDualPool()) {
            EXPECT_NO_THROW(connector_->initBlockPool());
            pool = connector_->block_pool_;
        }
        if (!pool && connector_->isDualPool()) {
            pool = block_size == connector_->incomplete_block_size_ ? connector_->incomplete_pool_ :
                                                                      connector_->complete_pool_;
        }
        if (!pool) {
            ADD_FAILURE() << "block pool is null";
            return nullptr;
        }
        if (block_size > 0) {
            // Pool block size should be >= requested mem_block_size.
            EXPECT_GE(memoryCacheBlockBytes(), block_size);
        }
        return pool;
    }
    std::shared_ptr<BlockPool> requireExistingBlockPool(size_t block_size) const {
        auto pool = connector_->block_pool_;
        if (!pool && connector_->isDualPool()) {
            pool = block_size == connector_->incomplete_block_size_ ? connector_->incomplete_pool_ :
                                                                      connector_->complete_pool_;
        }
        if (!pool) {
            ADD_FAILURE() << "expected block pool exists, block_size=" << block_size;
        }
        if (pool && block_size > 0) {
            EXPECT_GE(memoryCacheBlockBytes(), block_size);
        }
        return pool;
    }
    bool waitUntilDone(const std::shared_ptr<rtp_llm::AsyncContext>& ctx, int timeout_ms = 3000) const {
        if (!ctx) {
            return false;
        }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            if (ctx->done()) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return ctx->done();
    }
};

TEST_F(KVCacheMemoryConnectorTest, init_ReturnFalse_NoWorkerAddrs) {
    // 构造空的 worker 地址，BroadcastManager::init() 会失败；业务代码使用 RTP_LLM_CHECK，
    // 因此这里期望抛出 std::runtime_error。
    std::vector<std::string> empty_addrs;
    auto                     conn = std::make_shared<KVCacheMemoryConnector>(
        cache_config_, kv_cache_config_, allocator_, empty_addrs);
    EXPECT_THROW(conn->init(), std::runtime_error);
}

TEST_F(KVCacheMemoryConnectorTest, LayerGroupSlotsAreCachedDerivedBindings) {
    const auto& first  = connector_->layerGroupSlots();
    const auto& second = connector_->layerGroupSlots();

    EXPECT_EQ(&first, &second);
    ASSERT_FALSE(first.empty());
    for (const auto& slot : first) {
        EXPECT_EQ(slot.group_type, CacheGroupType::FULL);
        EXPECT_EQ(slot.block_kind, CacheBlockKind::COMPRESSED_KV);
        EXPECT_GT(slot.stride_bytes, 0u);
        EXPECT_FALSE(slot.tag.empty());
    }
}

TEST_F(KVCacheMemoryConnectorTest, init_ReturnFalse_WhenMemoryCacheSizeMbZero) {
    auto kv_cfg                         = kv_cache_config_;
    kv_cfg.memory_cache_size_mb         = 0;
    kv_cfg.memory_cache_sync_timeout_ms = 1000;

    auto conn =
        std::make_shared<KVCacheMemoryConnector>(cache_config_, kv_cfg, allocator_, server_addrs_);
    EXPECT_THROW(conn->init(), std::runtime_error);
    // Init fails early, nothing should be created.
    EXPECT_EQ(conn->block_cache_, nullptr);
    EXPECT_EQ(conn->broadcast_manager_, nullptr);
    EXPECT_EQ(conn->wait_done_thread_pool_, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, init_ReturnFalse_WhenMemoryCacheSyncTimeoutMsZero) {
    auto kv_cfg                         = kv_cache_config_;
    kv_cfg.memory_cache_size_mb         = 64;
    kv_cfg.memory_cache_sync_timeout_ms = 0;

    auto conn =
        std::make_shared<KVCacheMemoryConnector>(cache_config_, kv_cfg, allocator_, server_addrs_);
    EXPECT_THROW(conn->init(), std::runtime_error);
    // Init fails early, nothing should be created.
    EXPECT_EQ(conn->block_cache_, nullptr);
    EXPECT_EQ(conn->broadcast_manager_, nullptr);
    EXPECT_EQ(conn->wait_done_thread_pool_, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, init_ReturnFalse_WhenBlockSizeBytesZero) {
    auto cfg = cache_config_;
    setGroupBlockBytes(cfg, 0);

    auto kv_cfg                         = kv_cache_config_;
    kv_cfg.memory_cache_size_mb         = 64;
    kv_cfg.memory_cache_sync_timeout_ms = 1000;

    auto conn = std::make_shared<KVCacheMemoryConnector>(cfg, kv_cfg, allocator_, server_addrs_);
    EXPECT_THROW(conn->init(), std::runtime_error);
}

TEST_F(KVCacheMemoryConnectorTest, init_ReturnFalse_WhenPoolTooSmallForBlockSize) {
    // Use a valid physical MHA row larger than the pool; padded MHA rows are intentionally rejected.
    auto cfg = createMockCacheConfig(/*layer_num=*/4, /*block_num=*/10, /*seq_size_per_block=*/512);

    auto kv_cfg                         = kv_cache_config_;
    kv_cfg.memory_cache_size_mb         = 1;     // 1MB
    kv_cfg.memory_cache_sync_timeout_ms = 1000;  // valid

    auto conn = std::make_shared<KVCacheMemoryConnector>(cfg, kv_cfg, allocator_, server_addrs_);
    EXPECT_THROW(conn->init(), std::runtime_error);
}

TEST_F(KVCacheMemoryConnectorTest, init_ReturnTrue_WithWorkerAddrs) {
    // 使用有效的 worker 地址，init 应成功并正确设置 manager
    auto conn = std::make_shared<KVCacheMemoryConnector>(
        cache_config_, kv_cache_config_, allocator_, server_addrs_);
    auto ok = conn->init();
    EXPECT_TRUE(ok);
    ASSERT_NE(conn->block_cache_, nullptr);
    ASSERT_NE(conn->broadcast_manager_, nullptr);
    EXPECT_EQ(conn->broadcast_manager_->workerNum(), server_addrs_.size());
}

TEST_F(KVCacheMemoryConnectorTest, initBlockPool_Throw_WhenMemoryCacheSizeMbZero) {
    auto kv_cfg                         = kv_cache_config_;
    kv_cfg.memory_cache_size_mb         = 0;
    kv_cfg.memory_cache_sync_timeout_ms = 1000;

    auto conn =
        std::make_shared<KVCacheMemoryConnector>(cache_config_, kv_cfg, allocator_, server_addrs_);
    EXPECT_THROW(conn->initBlockPool(), std::runtime_error);
}

TEST_F(KVCacheMemoryConnectorTest, initBlockPool_Throw_WhenBlockSizeBytesZero) {
    auto cfg = cache_config_;
    setGroupBlockBytes(cfg, 0);

    auto kv_cfg                         = kv_cache_config_;
    kv_cfg.memory_cache_size_mb         = 64;
    kv_cfg.memory_cache_sync_timeout_ms = 1000;

    auto conn = std::make_shared<KVCacheMemoryConnector>(cfg, kv_cfg, allocator_, server_addrs_);
    EXPECT_THROW(conn->initBlockPool(), std::runtime_error);
}

TEST_F(KVCacheMemoryConnectorTest, initBlockPool_Throw_WhenCreateBlockPoolFails) {
    // Force block_num=0 with a valid physical MHA row larger than the memory pool.
    auto cfg = createMockCacheConfig(/*layer_num=*/4, /*block_num=*/10, /*seq_size_per_block=*/512);

    auto kv_cfg                         = kv_cache_config_;
    kv_cfg.memory_cache_size_mb         = 1;     // 1MB
    kv_cfg.memory_cache_sync_timeout_ms = 1000;  // not used by initBlockPool but keep valid

    auto conn = std::make_shared<KVCacheMemoryConnector>(cfg, kv_cfg, allocator_, server_addrs_);
    EXPECT_THROW(conn->initBlockPool(), std::runtime_error);
}

TEST_F(KVCacheMemoryConnectorTest, initBlockPool_ReturnTrue_AndRegistersPool) {
    auto kv_cfg                         = kv_cache_config_;
    kv_cfg.memory_cache_size_mb         = 64;
    kv_cfg.memory_cache_sync_timeout_ms = 1000;  // not used by initBlockPool but keep valid

    auto conn =
        std::make_shared<KVCacheMemoryConnector>(cache_config_, kv_cfg, allocator_, server_addrs_);
    EXPECT_NO_THROW(conn->initBlockPool());
    auto pool = conn->block_pool_;
    ASSERT_NE(pool, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, asyncMatch_ReturnNull_WhenGpuReuseLenGEKeysSize) {
    const size_t     N = 3;
    CacheKeysType    cache_keys{70001, 70002, 70003};
    BlockIndicesType lbs_vec{1, 1, 1};
    auto             res = makeCacheResource(cache_keys, lbs_vec, /*reuse_len=*/N);

    // Even if memory has matches, asyncMatch should skip when gpu reuse covers all keys.
    const size_t mem_size = memoryCacheBlockBytes();
    ASSERT_GT(mem_size, 0u);
    putItemsToCache(cache_keys, mem_size);

    auto meta      = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true, /*enable_remote_cache=*/false, "");
    auto match_ctx = connector_->asyncMatch(res, meta);
    EXPECT_EQ(match_ctx, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, asyncMatch_ReturnNull_WhenNoPrefixMatched) {
    CacheKeysType    cache_keys{71001, 71002};
    BlockIndicesType lbs_vec{1, 1};
    auto             res = makeCacheResource(cache_keys, lbs_vec, /*reuse_len=*/0);

    // No cache prefill => matched_num == 0
    auto meta      = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true, /*enable_remote_cache=*/false, "");
    auto match_ctx = connector_->asyncMatch(res, meta);
    EXPECT_EQ(match_ctx, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, asyncMatch_ReturnMatchedNum_WithHybridGroups) {
    resetToHybridCacheConfig();
    CacheKeysType cache_keys{71001, 71002, 71003};
    auto          res = makeHybridCacheResource(cache_keys,
                                       /*linear_blocks=*/{1, 2, 3},
                                       /*full_blocks=*/{4, 5, 6});
    for (int layer = 0; layer < static_cast<int>(cache_config_.totalLayerNum()); ++layer) {
        for (const auto& tag : res->groupTagsForLayer(layer)) {
            ASSERT_EQ(res->blockIdsForLayer(layer, tag).blocksNum(), 3u);
        }
    }
    putItemsToCache({cache_keys[0]}, memoryCacheBlockBytes());

    auto ctx = connector_->asyncMatch(res, std::make_shared<TestReadMeta>(true));
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->matchedBlockCount(), 1u);
}

TEST_F(KVCacheMemoryConnectorTest, asyncMatch_ReturnMatchedNum_WhenPrefixMatchedAndStopAtFirstMiss) {
    CacheKeysType    cache_keys{72001, 72002, 72003};
    BlockIndicesType lbs_vec{1, 1, 1};
    auto             res = makeCacheResource(cache_keys, lbs_vec, /*reuse_len=*/0);

    const size_t mem_size = memoryCacheBlockBytes();
    ASSERT_GT(mem_size, 0u);

    // Only prefill first 2 keys in cache; 3rd miss => matched_num should be 2.
    putItemsToCache({cache_keys[0], cache_keys[1]}, mem_size, /*is_complete_flags=*/{false, true});

    auto meta      = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true, /*enable_remote_cache=*/false, "");
    auto match_ctx = connector_->asyncMatch(res, meta);
    ASSERT_NE(match_ctx, nullptr);
    EXPECT_TRUE(match_ctx->done());
    EXPECT_TRUE(match_ctx->success());
    EXPECT_EQ(match_ctx->matchedBlockCount(), 2u);
}

TEST_F(KVCacheMemoryConnectorTest, asyncMatch_ReturnMatchedNum_MustEndAtBigKey_WhenSmallKeysAlsoHit) {
    // NOTE: asyncMatch always skips the last cache_key (see implementation comment),
    // so add a dummy tail key to keep the tested prefix length explicit.
    CacheKeysType    cache_keys{73001, 73002, 73003, 73004, 73999};
    BlockIndicesType lbs_vec{1, 1, 1, 1, 1};
    auto             res = makeCacheResource(cache_keys, lbs_vec, /*reuse_len=*/0);

    const size_t mem_size = memoryCacheBlockBytes();
    ASSERT_GT(mem_size, 0u);

    // Continuous prefix hits in cache:
    // - 73001: small
    // - 73002: small
    // - 73003: big   (last big => matched_num should end here)
    // - 73004: small (still hit, but must NOT extend matched_num beyond last big)
    putItemsToCache({cache_keys[0], cache_keys[1], cache_keys[2], cache_keys[3]},
                    mem_size,
                    /*is_complete_flags=*/{false, false, true, false});

    auto meta      = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true);
    auto match_ctx = connector_->asyncMatch(res, meta);
    ASSERT_NE(match_ctx, nullptr);
    EXPECT_TRUE(match_ctx->done());
    EXPECT_TRUE(match_ctx->success());
    EXPECT_EQ(match_ctx->matchedBlockCount(), 3u);
}

TEST_F(KVCacheMemoryConnectorTest, asyncMatch_AllowsContinuingWhenBigKeyHasInvalidGpuBlocks_UntilBigAndAllValid) {
    resetToHybridCacheConfig();
    // Hybrid-attn case: memory may have a "big" key, but the GPU blocks can still be partially invalid.
    // asyncMatch should keep scanning prefix hits, but ONLY count keys that are both:
    // - is_complete == true in memory cache
    // - all GPU blocks are valid (non-null) for that key
    //
    // NOTE: asyncMatch skips the last cache_key, so add a dummy tail key.
    CacheKeysType cache_keys{75001, 75002, 75999};

    // - key 75001: big in memory, but the linear group block is NULL
    // - key 75002: big in memory, both group blocks are valid => matched_num should become 2
    auto res = makeHybridCacheResource(cache_keys,
                                       /*linear_blocks=*/{NULL_BLOCK_IDX, 1, 1},
                                       /*full_blocks=*/{1, 1, 1},
                                       /*reuse_len=*/0);

    const size_t mem_size = memoryCacheBlockBytes();
    ASSERT_GT(mem_size, 0u);
    putItemsToCache({cache_keys[0], cache_keys[1]}, mem_size, /*is_complete_flags=*/{true, true});

    auto meta      = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true);
    auto match_ctx = connector_->asyncMatch(res, meta);
    ASSERT_NE(match_ctx, nullptr);
    EXPECT_TRUE(match_ctx->done());
    EXPECT_TRUE(match_ctx->success());
    EXPECT_EQ(match_ctx->matchedBlockCount(), 2u);
}

TEST_F(KVCacheMemoryConnectorTest, asyncMatch_StartsFromGpuReusePrefix_WhenTieredCacheOnlyStoresSuffixInMemory) {
    CacheKeysType    cache_keys{76001, 76002, 76003, 76004};
    BlockIndicesType lbs_vec{11, 12, 13, 14};
    auto             res = makeCacheResource(cache_keys, lbs_vec, /*reuse_len=*/2);

    const size_t mem_size = memoryCacheBlockBytes();
    ASSERT_GT(mem_size, 0u);
    putItemsToCache({cache_keys[2]}, mem_size, /*is_complete_flags=*/{true});

    auto meta      = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true);
    auto match_ctx = connector_->asyncMatch(res, meta);
    ASSERT_NE(match_ctx, nullptr);
    EXPECT_TRUE(match_ctx->done());
    EXPECT_TRUE(match_ctx->success());
    EXPECT_EQ(match_ctx->matchedBlockCount(), 3u);
}

TEST_F(KVCacheMemoryConnectorTest, asyncMatch_ReturnNull_WhenPrefixHitsButAllKeysAreSmall) {
    // Prefix keys hit (continuous), but none are big => matched_num stays 0 => asyncMatch returns nullptr.
    CacheKeysType    cache_keys{74001, 74002, 74999};
    BlockIndicesType lbs_vec{1, 1, 1};
    auto             res = makeCacheResource(cache_keys, lbs_vec, /*reuse_len=*/0);

    const size_t mem_size = memoryCacheBlockBytes();
    ASSERT_GT(mem_size, 0u);

    putItemsToCache({cache_keys[0], cache_keys[1]}, mem_size, /*is_complete_flags=*/{false, false});

    auto meta      = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true);
    auto match_ctx = connector_->asyncMatch(res, meta);
    EXPECT_EQ(match_ctx, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, asyncRead_InvalidInputs_ReturnNullOrThrow) {
    // resource is nullptr => RTP_LLM_CHECK triggers exception
    EXPECT_ANY_THROW(
        (void)connector_->asyncRead(nullptr, nullptr, nullptr, /*start_read_block_index=*/0, /*read_block_num=*/0));

    // empty cache_keys
    auto res_empty_keys = makeCacheResource({}, {1});
    auto ctx1 =
        connector_->asyncRead(res_empty_keys, nullptr, nullptr, /*start_read_block_index=*/0, /*read_block_num=*/0);
    EXPECT_EQ(ctx1, nullptr);

    // Missing group initialization must fail fast instead of falling back to positional routing.
    // NOTE: asyncRead always skips the last cache_key (cache_keys.size() - 1), so keep size >= 2 here.
    auto res_empty_lbs = std::make_shared<KVCacheResource>();
    res_empty_lbs->setCacheKeys({1, 2});
    expectUninitializedCacheGroups([&]() {
        (void)connector_->asyncRead(
            res_empty_lbs, nullptr, nullptr, /*start_read_block_index=*/0, /*read_block_num=*/1);
    });
}

TEST_F(KVCacheMemoryConnectorTest, asyncRead_ReturnNull_WhenReuseLenGEKeys) {
    const size_t     N = 3;
    CacheKeysType    cache_keys{10001, 10002, 10003};
    BlockIndicesType lbs_vec{1, 1, 1};
    auto             res = makeCacheResource(cache_keys, lbs_vec, N);

    // With reuse_len == keys size, asyncMatch should skip and there is nothing to read.
    auto meta      = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true, /*enable_remote_cache=*/false, "");
    auto match_ctx = connector_->asyncMatch(res, meta);
    EXPECT_EQ(match_ctx, nullptr);
    EXPECT_EQ(res->reuseBlockNum(), N);
}

TEST_F(KVCacheMemoryConnectorTest, asyncRead_ReturnNull_WhenPlanEmpty) {
    // Simulate mismatch between match result and current cache state:
    // asyncRead does NOT call asyncMatch any more, so it relies on match_context + meta.
    // Here cache has no items, so buildCopyPlanForRead should fail and asyncRead returns nullptr.
    CacheKeysType    cache_keys{20001, 20002};
    BlockIndicesType lbs_vec{3, 3};
    auto             res = makeCacheResource(cache_keys, lbs_vec);

    class TestMatchContext: public rtp_llm::AsyncMatchContext {
    public:
        explicit TestMatchContext(size_t matched): matched_(matched) {}
        void waitDone() override {
            return;
        }
        bool done() const override {
            return true;
        }
        bool success() const override {
            return true;
        }
        size_t matchedBlockCount() const override {
            return matched_;
        }

    private:
        size_t matched_{0};
    };

    auto match_ctx = std::make_shared<TestMatchContext>(/*matched=*/1);
    auto meta      = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true, /*enable_remote_cache=*/false, "");
    auto ctx       = connector_->asyncRead(res, meta, match_ctx, /*start_read_block_index=*/0, /*read_block_num=*/1);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, asyncRead_Success_IncrementsReuseLen_ByMatchedPrefix) {
    // 初始 reuse_len=1, 内存全部命中 => mem_match_len=3，最终 reuse_len=3
    CacheKeysType cache_keys{40001, 40002, 40003};

    const size_t mem_size = memoryCacheBlockBytes();
    ASSERT_GT(mem_size, 0u);

    auto block_indices = putItemsToCache(cache_keys, mem_size);
    ASSERT_EQ(block_indices.size(), cache_keys.size());

    BlockIndicesType lbs_vec{101, 102, 103};
    auto             res = makeCacheResource(cache_keys, lbs_vec, 1);

    auto meta      = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true, /*enable_remote_cache=*/false, "");
    auto match_ctx = connector_->asyncMatch(res, meta);
    ASSERT_NE(match_ctx, nullptr);
    const int reuse_num = static_cast<int>(res->reuseBlockNum());
    const int read_num  = static_cast<int>(match_ctx->matchedBlockCount()) - reuse_num;
    ASSERT_GT(read_num, 0);
    auto ctx = connector_->asyncRead(res, meta, match_ctx, reuse_num, read_num);
    ASSERT_NE(ctx, nullptr);
    auto mem_ctx = std::dynamic_pointer_cast<rtp_llm::MemoryAsyncContext>(ctx);
    ASSERT_NE(mem_ctx, nullptr);
    ASSERT_TRUE(waitUntilDone(ctx));
    EXPECT_TRUE(ctx->success());
    EXPECT_EQ(res->reuseBlockNum(), 2u);  // last cache key will not be read
}

TEST_F(KVCacheMemoryConnectorTest, asyncRead_Success_RemovesLoadedBlocksFromMemoryCache) {
    CacheKeysType cache_keys{41001, 41002, 41003};

    const size_t mem_size = memoryCacheBlockBytes();
    ASSERT_GT(mem_size, 0u);
    auto         pool        = ensureBlockPool(mem_size);
    const size_t free_before = pool->freeBlocksNum();

    auto block_indices = putItemsToCache(cache_keys, mem_size);
    ASSERT_EQ(block_indices.size(), cache_keys.size());
    ASSERT_LT(pool->freeBlocksNum(), free_before);

    BlockIndicesType lbs_vec{111, 112, 113};
    auto             res = makeCacheResource(cache_keys, lbs_vec, /*reuse_len=*/1);

    auto meta      = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true, /*enable_remote_cache=*/false, "");
    auto match_ctx = connector_->asyncMatch(res, meta);
    ASSERT_NE(match_ctx, nullptr);
    const int reuse_num = static_cast<int>(res->reuseBlockNum());
    const int read_num  = static_cast<int>(match_ctx->matchedBlockCount()) - reuse_num;
    ASSERT_GT(read_num, 0);
    auto ctx = connector_->asyncRead(res, meta, match_ctx, reuse_num, read_num);
    ASSERT_NE(ctx, nullptr);
    ASSERT_TRUE(waitUntilDone(ctx));
    ASSERT_TRUE(ctx->success());

    EXPECT_FALSE(connector_->block_cache_->contains(cache_keys[1]));
    EXPECT_EQ(pool->freeBlocksNum(), free_before - (block_indices.size() - static_cast<size_t>(read_num)));
}

TEST_F(KVCacheMemoryConnectorTest, asyncRead_Success_DoesNotRemoveUpgradedBlock) {
    // Simulate the race: after asyncRead builds its copy plan (capturing old block),
    // a concurrent write upgrades the same cache_key with a new block.
    // The read_done callback should NOT remove the upgraded entry because
    // removeIfMatch checks block_index equality.
    CacheKeysType cache_keys{42001, 42002, 42003};

    const size_t mem_size = memoryCacheBlockBytes();
    ASSERT_GT(mem_size, 0u);
    auto pool = ensureBlockPool(mem_size);

    auto block_indices = putItemsToCache(cache_keys, mem_size);
    ASSERT_EQ(block_indices.size(), cache_keys.size());

    BlockIndicesType lbs_vec{111, 112, 113};
    auto             res = makeCacheResource(cache_keys, lbs_vec, /*reuse_len=*/1);

    auto meta      = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true, /*enable_remote_cache=*/false, "");
    auto match_ctx = connector_->asyncMatch(res, meta);
    ASSERT_NE(match_ctx, nullptr);
    const int reuse_num = static_cast<int>(res->reuseBlockNum());
    const int read_num  = static_cast<int>(match_ctx->matchedBlockCount()) - reuse_num;
    ASSERT_GT(read_num, 0);

    // Start async read — buildCopyPlanForRead captures old block indices in the copy plan.
    auto ctx = connector_->asyncRead(res, meta, match_ctx, reuse_num, read_num);
    ASSERT_NE(ctx, nullptr);

    // Now replace cache_keys[1] with a new block AFTER the copy plan is built.
    // This simulates a concurrent write upgrading the entry while the read is in-flight.
    auto new_blocks = pool->malloc(1);
    ASSERT_EQ(new_blocks.size(), 1u);
    const BlockIdxType new_block_idx = static_cast<BlockIdxType>(new_blocks[0]);
    {
        MemoryBlockCache::CacheItem upgraded_item;
        upgraded_item.cache_key   = cache_keys[1];
        upgraded_item.block_index = new_block_idx;
        upgraded_item.block_size  = mem_size;
        upgraded_item.is_resident = false;
        upgraded_item.is_complete = true;
        connector_->block_cache_->remove(cache_keys[1]);
        pool->blockCacheFree({block_indices[1]});
        auto [ok, popped] = connector_->block_cache_->put(upgraded_item);
        ASSERT_TRUE(ok);
        pool->blockCacheReference({new_block_idx});
        pool->requestFree({new_block_idx});
    }

    ASSERT_TRUE(waitUntilDone(ctx));
    ASSERT_TRUE(ctx->success());

    // The upgraded block should still be in cache (removeIfMatch skips it
    // because block_index differs from the one captured in the copy plan).
    EXPECT_TRUE(connector_->block_cache_->contains(cache_keys[1]));
    auto match_result = connector_->block_cache_->match(cache_keys[1]);
    EXPECT_EQ(match_result.matched_index, new_block_idx);
}

TEST_F(KVCacheMemoryConnectorTest, asyncRead_FailureOnMemResponse_NoReuseLenIncrement) {
    // 构造部分 rank mem_response 失败，最终 AsyncContext->success() 应为 false，reuse_len 不增加
    std::vector<std::unique_ptr<TestRpcServer>> servers;
    std::vector<std::string>                    addrs;
    for (int i = 0; i < 4; ++i) {
        auto service = std::make_unique<TestRpcService>();
        service->setMemResponseSuccess(i % 2 == 0);  // 只有偶数 rank 成功
        auto server = std::make_unique<TestRpcServer>(std::move(service));
        ASSERT_TRUE(server->start());
        addrs.push_back("127.0.0.1:" + std::to_string(server->listenPort()));
        servers.push_back(std::move(server));
    }
    auto broadcast_manager = std::make_shared<BroadcastManager>(addrs);
    ASSERT_TRUE(broadcast_manager->init());
    connector_->broadcast_manager_ = broadcast_manager;

    CacheKeysType cache_keys{50001, 50002};

    const size_t mem_size = memoryCacheBlockBytes();
    ASSERT_GT(mem_size, 0u);

    auto block_indices = putItemsToCache(cache_keys, mem_size);
    ASSERT_EQ(block_indices.size(), cache_keys.size());

    BlockIndicesType lbs_vec{11, 12};
    auto             res = makeCacheResource(cache_keys, lbs_vec);

    auto meta      = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true, /*enable_remote_cache=*/false, "");
    auto match_ctx = connector_->asyncMatch(res, meta);
    ASSERT_NE(match_ctx, nullptr);
    const int start_read_block_index = 0;
    const int read_block_num         = static_cast<int>(match_ctx->matchedBlockCount());
    auto      ctx = connector_->asyncRead(res, meta, match_ctx, start_read_block_index, read_block_num);
    ASSERT_NE(ctx, nullptr);
    auto mem_ctx = std::dynamic_pointer_cast<rtp_llm::MemoryAsyncContext>(ctx);
    ASSERT_NE(mem_ctx, nullptr);
    ASSERT_TRUE(waitUntilDone(ctx));
    EXPECT_FALSE(ctx->success());
    EXPECT_EQ(res->reuseBlockNum(), 0u);

    connector_->broadcast_manager_.reset();
    for (auto& s : servers) {
        s->shutdown();
    }
    servers.clear();
    addrs.clear();
}

TEST_F(KVCacheMemoryConnectorTest, asyncRead_FailureOnRpcStatus_NoReuseLenIncrement) {
    // 构造部分 rank RPC 状态失败，最终 AsyncContext->success() 应为 false，reuse_len 不增加
    std::vector<std::unique_ptr<TestRpcServer>> servers;
    std::vector<std::string>                    addrs;
    for (int i = 0; i < 4; ++i) {
        auto service = std::make_unique<TestRpcService>();
        if (i % 2 == 0) {
            service->setRpcResponseStatus(::grpc::Status(::grpc::StatusCode::UNAVAILABLE, "down"));
        }
        auto server = std::make_unique<TestRpcServer>(std::move(service));
        ASSERT_TRUE(server->start());
        addrs.push_back("127.0.0.1:" + std::to_string(server->listenPort()));
        servers.push_back(std::move(server));
    }
    auto broadcast_manager = std::make_shared<BroadcastManager>(addrs);
    ASSERT_TRUE(broadcast_manager->init());
    connector_->broadcast_manager_ = broadcast_manager;

    CacheKeysType cache_keys{60001, 60002};

    const size_t mem_size = memoryCacheBlockBytes();
    ASSERT_GT(mem_size, 0u);

    auto block_indices = putItemsToCache(cache_keys, mem_size);
    ASSERT_EQ(block_indices.size(), cache_keys.size());

    BlockIndicesType lbs_vec{31, 32};
    auto             res = makeCacheResource(cache_keys, lbs_vec);

    auto meta      = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true, /*enable_remote_cache=*/false, "");
    auto match_ctx = connector_->asyncMatch(res, meta);
    ASSERT_NE(match_ctx, nullptr);
    const int start_read_block_index = 0;
    const int read_block_num         = static_cast<int>(match_ctx->matchedBlockCount());
    auto      ctx = connector_->asyncRead(res, meta, match_ctx, start_read_block_index, read_block_num);
    ASSERT_NE(ctx, nullptr);
    auto mem_ctx = std::dynamic_pointer_cast<rtp_llm::MemoryAsyncContext>(ctx);
    ASSERT_NE(mem_ctx, nullptr);
    ASSERT_TRUE(waitUntilDone(ctx));
    EXPECT_FALSE(ctx->success());
    EXPECT_EQ(res->reuseBlockNum(), 0u);

    connector_->broadcast_manager_.reset();
    for (auto& s : servers) {
        s->shutdown();
    }
    servers.clear();
    addrs.clear();
}

TEST_F(KVCacheMemoryConnectorTest, asyncRead_ReturnNull_WhenThreadPoolFull) {
    // 在单测里稳定模拟 startCopyAsync() 失败：把线程池替换成“未启动”的线程池，
    // 这样 pushTask 会返回非 ERROR_NONE，从而 asyncRead 返回 nullptr。
    auto old_pool = connector_->wait_done_thread_pool_;

    connector_->wait_done_thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(/*thread_num=*/1,
                                                                                     /*queue_size=*/1,
                                                                                     /*thread_init_func=*/nullptr,
                                                                                     /*name=*/"AsyncReadNotStartedTP");
    // 验证线程池未启动时 pushTask 会失败（避免平台/实现差异导致用例不稳定）。
    EXPECT_NE(connector_->wait_done_thread_pool_->pushTask([]() {}), autil::ThreadPoolBase::ERROR_NONE);

    CacheKeysType cache_keys{70001, 70002};
    const size_t  mem_size = memoryCacheBlockBytes();
    ASSERT_GT(mem_size, 0u);
    putItemsToCache(cache_keys, mem_size);
    auto res       = makeCacheResource(cache_keys, {1, 2});
    auto meta      = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true, /*enable_remote_cache=*/false, "");
    auto match_ctx = connector_->asyncMatch(res, meta);
    ASSERT_NE(match_ctx, nullptr);
    const int start_read_block_index = 0;
    const int read_block_num         = static_cast<int>(match_ctx->matchedBlockCount());

    auto ctx = connector_->asyncRead(res, meta, match_ctx, start_read_block_index, read_block_num);
    EXPECT_EQ(ctx, nullptr);

    connector_->wait_done_thread_pool_.reset();
    connector_->wait_done_thread_pool_ = old_pool;
}

TEST_F(KVCacheMemoryConnectorTest, asyncWrite_InvalidInputs_ReturnNullOrThrow) {
    auto meta = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true);

    // meta is nullptr => RTP_LLM_CHECK triggers exception
    EXPECT_ANY_THROW((void)connector_->asyncWrite(makeCacheResource(/*cache_keys=*/{1}, /*lbs=*/{1}), nullptr));

    // resource is nullptr => RTP_LLM_CHECK triggers exception
    EXPECT_ANY_THROW((void)connector_->asyncWrite(nullptr, meta));

    // empty cache_keys
    auto res_empty_keys = makeCacheResource({}, {1});
    auto ctx1           = connector_->asyncWrite(res_empty_keys, meta);
    EXPECT_EQ(ctx1, nullptr);

    // uninitialized legacy layer view
    auto res_empty_lbs = std::make_shared<KVCacheResource>();
    res_empty_lbs->setCacheKeys({1});
    res_empty_lbs->setLastBlockAligned(true);
    expectUninitializedCacheGroups([&]() { (void)connector_->asyncWrite(res_empty_lbs, meta); });
}

TEST_F(KVCacheMemoryConnectorTest, asyncWrite_ReturnNull_WhenAllKeysInCache) {
    // 两个 key 均已在内存缓存中
    const int        gpu_block_idx = 1;
    CacheKeysType    cache_keys{10, 11};
    BlockIndicesType lbs_vec{static_cast<BlockIdxType>(gpu_block_idx), static_cast<BlockIdxType>(gpu_block_idx)};
    auto             res = makeCacheResource(cache_keys, lbs_vec);

    const size_t mem_size = memoryCacheBlockBytes();
    ASSERT_GT(mem_size, 0u);

    // 预置到 cache
    auto block_indices = putItemsToCache(cache_keys, mem_size);
    ASSERT_EQ(block_indices.size(), cache_keys.size());

    const size_t cache_size_before = connector_->block_cache_->size();

    auto meta = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true, /*enable_remote_cache=*/false, "");
    auto ctx  = connector_->asyncWrite(res, meta);
    ASSERT_EQ(ctx, nullptr);
    EXPECT_EQ(connector_->block_cache_->size(), cache_size_before);
}

TEST_F(KVCacheMemoryConnectorTest, asyncWrite_ReturnSuccess_WhenPrefixInCacheOnlyWriteSuffix) {
    CacheKeysType    cache_keys{60001, 60002, 60003};
    BlockIndicesType lbs_vec{1, 2, 3};
    auto             res = makeCacheResource(cache_keys, lbs_vec);

    const size_t mem_size = memoryCacheBlockBytes();
    ASSERT_GT(mem_size, 0u);

    // Pre-insert only the first key, so cpu_matched_num should be 1 and only suffix gets written.
    auto pre_blocks = putItemsToCache({cache_keys[0]}, mem_size);
    ASSERT_EQ(pre_blocks.size(), 1u);
    ASSERT_TRUE(connector_->block_cache_->contains(cache_keys[0]));
    const size_t cache_before = connector_->block_cache_->size();

    auto meta = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true, /*enable_remote_cache=*/false, "");
    auto ctx  = connector_->asyncWrite(res, meta);
    ASSERT_NE(ctx, nullptr);
    auto mem_ctx = std::dynamic_pointer_cast<rtp_llm::MemoryAsyncContext>(ctx);
    ASSERT_NE(mem_ctx, nullptr);
    ASSERT_TRUE(waitUntilDone(ctx));
    EXPECT_TRUE(ctx->success());

    // Only 2 new items inserted.
    EXPECT_GE(connector_->block_cache_->size(), cache_before + 2);
    EXPECT_TRUE(connector_->block_cache_->contains(cache_keys[0]));
    EXPECT_TRUE(connector_->block_cache_->contains(cache_keys[1]));
    EXPECT_TRUE(connector_->block_cache_->contains(cache_keys[2]));
}

TEST_F(KVCacheMemoryConnectorTest, asyncWrite_ReturnSuccess_WhenKeyInsertedDuringWriteDone) {
    // Delay RPC so we can insert a key into block_cache_ while asyncWrite is in flight.
    std::vector<std::unique_ptr<TestRpcServer>> servers;
    std::vector<std::string>                    addrs;
    for (int i = 0; i < 2; ++i) {
        auto service = std::make_unique<TestRpcService>();
        service->setSleepMillis(200);
        auto server = std::make_unique<TestRpcServer>(std::move(service));
        ASSERT_TRUE(server->start());
        addrs.push_back("127.0.0.1:" + std::to_string(server->listenPort()));
        servers.push_back(std::move(server));
    }
    auto broadcast_manager = std::make_shared<BroadcastManager>(addrs);
    ASSERT_TRUE(broadcast_manager->init());
    connector_->broadcast_manager_ = broadcast_manager;

    CacheKeysType    cache_keys{61001, 61002};
    BlockIndicesType lbs_vec{1, 2};
    auto             res = makeCacheResource(cache_keys, lbs_vec);

    const size_t mem_size = memoryCacheBlockBytes();
    ASSERT_GT(mem_size, 0u);
    ASSERT_NE(ensureBlockPool(mem_size), nullptr);

    auto meta = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true, /*enable_remote_cache=*/false, "");
    auto ctx  = connector_->asyncWrite(res, meta);
    ASSERT_NE(ctx, nullptr);

    // While in flight, insert the first key so write_done should skip inserting it.
    auto pre_blocks = putItemsToCache({cache_keys[0]}, mem_size);
    ASSERT_EQ(pre_blocks.size(), 1u);
    ASSERT_TRUE(connector_->block_cache_->contains(cache_keys[0]));

    auto mem_ctx = std::dynamic_pointer_cast<rtp_llm::MemoryAsyncContext>(ctx);
    ASSERT_NE(mem_ctx, nullptr);
    ASSERT_TRUE(waitUntilDone(ctx));
    EXPECT_TRUE(ctx->success());

    EXPECT_TRUE(connector_->block_cache_->contains(cache_keys[0]));
    EXPECT_TRUE(connector_->block_cache_->contains(cache_keys[1]));

    connector_->broadcast_manager_.reset();
    for (auto& s : servers) {
        s->shutdown();
    }
    servers.clear();
    addrs.clear();
}

TEST_F(KVCacheMemoryConnectorTest, asyncWrite_ReturnNull_WhenBuildPlanEmpty) {
    // 所有 layer 对于第一个未命中 key 的 blockIdx 都为 NULL，导致 plan 为空
    CacheKeysType    cache_keys{100, 101};
    BlockIndicesType lbs_vec{NULL_BLOCK_IDX, NULL_BLOCK_IDX};
    auto             res = makeCacheResource(cache_keys, lbs_vec);

    auto meta = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true, /*enable_remote_cache=*/false, "");
    auto ctx  = connector_->asyncWrite(res, meta);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, asyncWrite_ReturnNull_WhenAllKeysAreSmall_NoNeedWrite) {
    resetToHybridCacheConfig();
    // Hybrid-attn: allow writing small keys for continuity, BUT if there is NO "big" key in the tail,
    // buildCopyPlanForWrite() should return nullptr and asyncWrite should be a no-op (return nullptr).
    CacheKeysType cache_keys{81001, 81002, 81003};
    auto          res = makeHybridCacheResource(cache_keys,
                                       /*linear_blocks=*/{NULL_BLOCK_IDX, 1, NULL_BLOCK_IDX},
                                       /*full_blocks=*/{1, NULL_BLOCK_IDX, 1});

    // linear_step is pinned to 1, so every block is complete and incomplete_pool_ is never created.
    auto pool = connector_->complete_pool_;
    ASSERT_NE(pool, nullptr);
    const size_t free_before = pool->freeBlocksNum();

    auto meta = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true);
    auto ctx  = connector_->asyncWrite(res, meta);
    EXPECT_EQ(ctx, nullptr);
    EXPECT_EQ(pool->freeBlocksNum(), free_before);
}

TEST_F(KVCacheMemoryConnectorTest, NonUnitLinearStepRetainedPathCreatesAndUsesIncompletePool) {
    resetToHybridCacheConfig(/*linear_step=*/2);
    ASSERT_NE(connector_->complete_pool_, nullptr);
    ASSERT_NE(connector_->incomplete_pool_, nullptr);
    EXPECT_EQ(connector_->memoryPoolFor(CacheBlockKind::COMPLETE), connector_->complete_pool_);
    EXPECT_EQ(connector_->memoryPoolFor(CacheBlockKind::INCOMPLETE), connector_->incomplete_pool_);

    CacheKeysType cache_keys{81101, 81102};
    auto          res = makeHybridCacheResource(cache_keys,
                                       /*linear_blocks=*/{NULL_BLOCK_IDX, 1},
                                       /*full_blocks=*/{1, 1});

    auto ctx = connector_->asyncWrite(res, std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true));
    ASSERT_NE(ctx, nullptr);
    ASSERT_TRUE(waitUntilDone(ctx));
    ASSERT_TRUE(ctx->success());

    const auto incomplete_match = connector_->block_cache_->match(cache_keys[0]);
    EXPECT_FALSE(incomplete_match.is_complete);
    EXPECT_EQ(incomplete_match.block_size, connector_->incomplete_block_size_);
    const auto complete_match = connector_->block_cache_->match(cache_keys[1]);
    EXPECT_TRUE(complete_match.is_complete);
    EXPECT_EQ(complete_match.block_size, connector_->complete_block_size_);
}

TEST_F(KVCacheMemoryConnectorTest, asyncWrite_DropsAllIncompleteKeysWithUnitLinearStep_InHybridAttn) {
    resetToHybridCacheConfig();
    // linear_step=1 has no incomplete pool, so incomplete keys are dropped whether they occur before or after the
    // last complete key.
    std::vector<std::unique_ptr<TestRpcServer>> servers;
    std::vector<std::string>                    addrs;
    for (int i = 0; i < 2; ++i) {
        auto service = std::make_unique<TestRpcService>();
        auto server  = std::make_unique<TestRpcServer>(std::move(service));
        ASSERT_TRUE(server->start());
        addrs.push_back("127.0.0.1:" + std::to_string(server->listenPort()));
        servers.push_back(std::move(server));
    }
    auto broadcast_manager = std::make_shared<BroadcastManager>(addrs);
    ASSERT_TRUE(broadcast_manager->init());
    connector_->broadcast_manager_ = broadcast_manager;

    CacheKeysType cache_keys{82001, 82002, 82003, 82004};
    // 4 layers, 4 keys:
    // - key0 complete (all valid)
    // - key1 incomplete (layer1 NULL) => dropped before the last complete key
    // - key2 complete (all valid) => last complete key
    // - key3 incomplete (layer3 NULL) => dropped after the last complete key
    auto res = makeHybridCacheResource(cache_keys,
                                       /*linear_blocks=*/{1, NULL_BLOCK_IDX, 1, 1},
                                       /*full_blocks=*/{1, 1, 1, NULL_BLOCK_IDX});

    const size_t cache_before = connector_->block_cache_->size();
    auto         meta         = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true);
    auto         ctx          = connector_->asyncWrite(res, meta);
    ASSERT_NE(ctx, nullptr);
    ASSERT_TRUE(waitUntilDone(ctx));
    EXPECT_TRUE(ctx->success());

    EXPECT_TRUE(connector_->block_cache_->contains(cache_keys[0]));
    // linear_step==1 disables the incomplete pool, so small (non-complete) keys are dropped too.
    EXPECT_FALSE(connector_->block_cache_->contains(cache_keys[1]));
    EXPECT_TRUE(connector_->block_cache_->contains(cache_keys[2]));
    EXPECT_FALSE(connector_->block_cache_->contains(cache_keys[3]));

    // Only the two complete keys are written (exact +2 if cache was empty and no evictions)
    EXPECT_GE(connector_->block_cache_->size(), cache_before + 2);

    connector_->broadcast_manager_.reset();
    for (auto& s : servers) {
        s->shutdown();
    }
    servers.clear();
    addrs.clear();
}

TEST_F(KVCacheMemoryConnectorTest, asyncWrite_Success_AddsToBlockCache_AndKeepsMemBlocks) {
    // 默认 RPC 服务均返回 OK + mem success
    const size_t     N = 2;
    CacheKeysType    cache_keys{200, 201};
    BlockIndicesType lbs_vec{2, 3};
    auto             res = makeCacheResource(cache_keys, lbs_vec);

    const size_t mem_size = memoryCacheBlockBytes();
    ASSERT_GT(mem_size, 0u);
    auto pool = ensureBlockPool(mem_size);
    ASSERT_NE(pool, nullptr);
    const size_t free_before  = pool->freeBlocksNum();
    const size_t cache_before = connector_->block_cache_->size();

    auto meta = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true, /*enable_remote_cache=*/false, "");
    auto ctx  = connector_->asyncWrite(res, meta);
    ASSERT_NE(ctx, nullptr);
    auto mem_ctx = std::dynamic_pointer_cast<rtp_llm::MemoryAsyncContext>(ctx);
    ASSERT_NE(mem_ctx, nullptr);
    ASSERT_TRUE(waitUntilDone(ctx));
    EXPECT_TRUE(ctx->success());

    // block_cache 中应新增 N 个条目
    EXPECT_EQ(connector_->block_cache_->size(), cache_before + N);
    for (auto key : cache_keys) {
        EXPECT_TRUE(connector_->block_cache_->contains(key));
    }
    // 对应大小的 pool 空闲块减少 N（分配后未释放，缓存驻留）
    EXPECT_EQ(pool->freeBlocksNum(), free_before - N);
}

TEST_F(KVCacheMemoryConnectorTest, asyncWrite_FailureOnMemResponse_FreesAllocatedBlocks_NoCacheInsert) {
    // 构造部分 rank mem_response 失败，最终 AsyncContext->success() 应为 false
    std::vector<std::unique_ptr<TestRpcServer>> servers;
    std::vector<std::string>                    addrs;
    for (int i = 0; i < 4; ++i) {
        auto service = std::make_unique<TestRpcService>();
        service->setMemResponseSuccess(i % 2 == 0);  // 只有偶数 rank 成功
        auto server = std::make_unique<TestRpcServer>(std::move(service));
        ASSERT_TRUE(server->start());
        addrs.push_back("127.0.0.1:" + std::to_string(server->listenPort()));
        servers.push_back(std::move(server));
    }
    auto broadcast_manager = std::make_shared<BroadcastManager>(addrs);
    ASSERT_TRUE(broadcast_manager->init());
    connector_->broadcast_manager_ = broadcast_manager;

    CacheKeysType    cache_keys{301, 302};
    BlockIndicesType lbs_vec{1, 2};
    auto             res = makeCacheResource(cache_keys, lbs_vec);

    const size_t mem_size = memoryCacheBlockBytes();
    ASSERT_GT(mem_size, 0u);
    auto pool = ensureBlockPool(mem_size);
    ASSERT_NE(pool, nullptr);
    const size_t free_before  = pool->freeBlocksNum();
    const size_t cache_before = connector_->block_cache_->size();

    auto meta = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true, /*enable_remote_cache=*/false, "");
    auto ctx  = connector_->asyncWrite(res, meta);
    ASSERT_NE(ctx, nullptr);
    auto mem_ctx = std::dynamic_pointer_cast<rtp_llm::MemoryAsyncContext>(ctx);
    ASSERT_NE(mem_ctx, nullptr);
    ASSERT_TRUE(waitUntilDone(ctx));
    EXPECT_FALSE(ctx->success());
    // 应未插入缓存
    EXPECT_EQ(connector_->block_cache_->size(), cache_before);
    // 分配的块应被回收
    EXPECT_EQ(pool->freeBlocksNum(), free_before);

    connector_->broadcast_manager_.reset();
    for (auto& s : servers) {
        s->shutdown();
    }
    servers.clear();
    addrs.clear();
}

TEST_F(KVCacheMemoryConnectorTest, asyncWrite_ReturnNull_WhenThreadPoolFull) {
    // 在单测里稳定模拟 startCopyAsync() 失败：把线程池替换成“未启动”的线程池，
    // 这样 pushTask 会返回非 ERROR_NONE，从而 asyncWrite 返回 nullptr。
    auto old_pool = connector_->wait_done_thread_pool_;

    connector_->wait_done_thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(/*thread_num=*/1,
                                                                                     /*queue_size=*/1,
                                                                                     /*thread_init_func=*/nullptr,
                                                                                     /*name=*/"AsyncWriteNotStartedTP");
    // 验证线程池未启动时 pushTask 会失败（避免平台/实现差异导致用例不稳定）。
    EXPECT_NE(connector_->wait_done_thread_pool_->pushTask([]() {}), autil::ThreadPoolBase::ERROR_NONE);

    const int        layer0 = 0;
    CacheKeysType    cache_keys{71001, 71002, 71003};
    BlockIndicesType lbs_vec{1, 2, 3};
    auto             res = makeCacheResource(cache_keys, lbs_vec);

    // Pre-insert one key so cpu_matched_num < cache_keys.size() and it reaches the thread-pool-full check.
    const auto   bufs  = allocator_->convertIndexToBuffer(layer0, "default", /*block_id=*/1);
    const size_t total = sumBlockInfosBytes(bufs);
    ASSERT_GT(total, 0u);
    ASSERT_NE(ensureBlockPool(total), nullptr);
    (void)putItemsToCache({cache_keys[0]}, total);

    auto meta = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true, /*enable_remote_cache=*/false, "");
    auto ctx  = connector_->asyncWrite(res, meta);
    EXPECT_EQ(ctx, nullptr);

    connector_->wait_done_thread_pool_.reset();
    connector_->wait_done_thread_pool_ = old_pool;
}

TEST_F(KVCacheMemoryConnectorTest, sendCopyPlan_ReturnContext_WhenNoWorkers_NoOp) {
    // BroadcastManager treats "0 workers" as a no-op success (requests.size()==workerNum()==0).
    connector_->broadcast_manager_->worker_addrs_.clear();

    std::vector<KVCacheMemoryConnector::CopyInfoPerKey> infos;
    auto                                                plan = std::make_shared<KVCacheMemoryConnector::CopyPlan>();
    plan->copy_infos                                         = std::move(infos);
    plan->direction                                          = KVCacheMemoryConnector::CopyDirection::H2D;
    auto result                                              = connector_->sendCopyPlan(plan);
    ASSERT_NE(result, nullptr);
    result->waitDone();
    EXPECT_TRUE(result->success());
    EXPECT_TRUE(result->responses().empty());
}

TEST_F(KVCacheMemoryConnectorTest, sendCopyPlan_ReturnContext_AllRanksSuccess) {
    const int    layer_id      = 0;
    const int    gpu_block_idx = 2;
    const auto   gpu_bufs      = allocator_->convertIndexToBuffer(layer_id, "default", gpu_block_idx);
    const size_t total         = sumBlockInfosBytes(gpu_bufs);
    ASSERT_GT(total, 0u);

    KVCacheMemoryConnector::CopyInfoPerKey info;
    info.cache_key = 1;
    info.mem_block = static_cast<BlockIdxType>(1);
    info.gpu_blocks.assign(static_cast<size_t>(cache_config_.totalLayerNum()), NULL_BLOCK_IDX);
    info.gpu_blocks[static_cast<size_t>(layer_id)] = static_cast<BlockIdxType>(gpu_block_idx);
    std::vector<KVCacheMemoryConnector::CopyInfoPerKey> infos{info};

    auto plan        = std::make_shared<KVCacheMemoryConnector::CopyPlan>();
    plan->copy_infos = std::move(infos);
    plan->direction  = KVCacheMemoryConnector::CopyDirection::H2D;
    auto result      = connector_->sendCopyPlan(plan);
    ASSERT_NE(result, nullptr);
    result->waitDone();
    EXPECT_TRUE(result->success());
    const auto responses = result->responses();
    EXPECT_EQ(responses.size(), server_addrs_.size());
    for (const auto& response : responses) {
        EXPECT_TRUE(response.has_mem_response());
        EXPECT_TRUE(response.mem_response().success());
    }
}

TEST_F(KVCacheMemoryConnectorTest, sendCopyPlan_ReturnContext_PartialRanksFail) {
    std::vector<std::unique_ptr<TestRpcServer>> servers;
    std::vector<std::string>                    addrs;
    for (int i = 0; i < 4; ++i) {
        auto service = std::make_unique<TestRpcService>();
        // 只有偶数rank response成功
        service->setMemResponseSuccess(i % 2 == 0);
        auto server = std::make_unique<TestRpcServer>(std::move(service));
        ASSERT_TRUE(server->start());
        addrs.push_back("127.0.0.1:" + std::to_string(server->listenPort()));
        servers.push_back(std::move(server));
    }
    auto broadcast_manager = std::make_shared<BroadcastManager>(addrs);
    ASSERT_TRUE(broadcast_manager->init());
    connector_->broadcast_manager_ = broadcast_manager;

    const int    layer_id      = 0;
    const int    gpu_block_idx = 2;
    const auto   gpu_bufs      = allocator_->convertIndexToBuffer(layer_id, "default", gpu_block_idx);
    const size_t total         = sumBlockInfosBytes(gpu_bufs);
    ASSERT_GT(total, 0u);

    KVCacheMemoryConnector::CopyInfoPerKey info;
    info.cache_key = 2;
    info.mem_block = static_cast<BlockIdxType>(1);
    info.gpu_blocks.assign(static_cast<size_t>(cache_config_.totalLayerNum()), NULL_BLOCK_IDX);
    info.gpu_blocks[static_cast<size_t>(layer_id)] = static_cast<BlockIdxType>(gpu_block_idx);
    std::vector<KVCacheMemoryConnector::CopyInfoPerKey> infos{info};

    auto plan        = std::make_shared<KVCacheMemoryConnector::CopyPlan>();
    plan->copy_infos = std::move(infos);
    plan->direction  = KVCacheMemoryConnector::CopyDirection::H2D;
    auto result      = connector_->sendCopyPlan(plan);
    ASSERT_NE(result, nullptr);

    result->waitDone();
    EXPECT_TRUE(result->success());
    const auto responses = result->responses();
    EXPECT_EQ(responses.size(), addrs.size());
    for (size_t i = 0; i < responses.size(); ++i) {
        EXPECT_TRUE(responses[i].has_mem_response());
        if (i % 2 == 0) {
            EXPECT_TRUE(responses[i].mem_response().success());
        } else {
            EXPECT_FALSE(responses[i].mem_response().success());
        }
    }
    connector_->broadcast_manager_.reset();
    for (auto& server : servers) {
        server->shutdown();
    }
    servers.clear();
    addrs.clear();
}

TEST_F(KVCacheMemoryConnectorTest, sendCopyPlan_ReturnContext_RpcStatusError) {
    std::vector<std::unique_ptr<TestRpcServer>> servers;
    std::vector<std::string>                    addrs;
    for (int i = 0; i < 4; ++i) {
        auto service = std::make_unique<TestRpcService>();
        // 只有偶数rank返回rpc状态失败
        if (i % 2 == 0) {
            service->setRpcResponseStatus(::grpc::Status(::grpc::StatusCode::UNAVAILABLE, "down"));
        }
        auto server = std::make_unique<TestRpcServer>(std::move(service));
        ASSERT_TRUE(server->start());
        addrs.push_back("127.0.0.1:" + std::to_string(server->listenPort()));
        servers.push_back(std::move(server));
    }
    auto broadcast_manager = std::make_shared<BroadcastManager>(addrs);
    ASSERT_TRUE(broadcast_manager->init());
    connector_->broadcast_manager_ = broadcast_manager;

    const int    layer_id      = 0;
    const int    gpu_block_idx = 2;
    const auto   gpu_bufs      = allocator_->convertIndexToBuffer(layer_id, "default", gpu_block_idx);
    const size_t total         = sumBlockInfosBytes(gpu_bufs);
    ASSERT_GT(total, 0u);

    KVCacheMemoryConnector::CopyInfoPerKey info;
    info.cache_key = 3;
    info.mem_block = static_cast<BlockIdxType>(1);
    info.gpu_blocks.assign(static_cast<size_t>(cache_config_.totalLayerNum()), NULL_BLOCK_IDX);
    info.gpu_blocks[static_cast<size_t>(layer_id)] = static_cast<BlockIdxType>(gpu_block_idx);
    std::vector<KVCacheMemoryConnector::CopyInfoPerKey> infos{info};

    auto plan        = std::make_shared<KVCacheMemoryConnector::CopyPlan>();
    plan->copy_infos = std::move(infos);
    plan->direction  = KVCacheMemoryConnector::CopyDirection::H2D;
    auto result      = connector_->sendCopyPlan(plan);
    ASSERT_NE(result, nullptr);
    result->waitDone();
    EXPECT_FALSE(result->success());
}

TEST_F(KVCacheMemoryConnectorTest, copyCache_RejectsEmptyTaggedBlocks) {
    MemoryOperationRequestPB req;
    req.add_copy_items();
    // An empty tagged set cannot be normalized against the local topology.
    req.set_copy_direction(MemoryOperationRequestPB::H2D);

    MemoryOperationResponsePB resp;
    EXPECT_THROW((void)connector_->copyCache(req, resp), rtp_llm::RTPException);
}

TEST_F(KVCacheMemoryConnectorTest, copyCache_ReturnFalse_InvalidMemBlock) {
    const int    layer_id      = 0;
    const int    gpu_block_idx = 1;
    const auto   gpu_bufs      = allocator_->convertIndexToBuffer(layer_id, "default", gpu_block_idx);
    const size_t total         = sumBlockInfosBytes(gpu_bufs);
    ASSERT_GT(total, 0u);

    MemoryOperationRequestPB req;
    auto*                    item = req.add_copy_items();
    addTaggedGpuBlocks(*item,
                       std::vector<BlockIdxType>(static_cast<size_t>(cache_config_.totalLayerNum()), gpu_block_idx));
    // invalid mem_block index for block_pool_
    item->set_mem_block(NULL_BLOCK_IDX);
    req.set_copy_direction(MemoryOperationRequestPB::H2D);

    MemoryOperationResponsePB resp;
    EXPECT_THROW(connector_->copyCache(req, resp), rtp_llm::RTPException);
}

TEST_F(KVCacheMemoryConnectorTest, copyCache_ReturnFalse_InvalidLayerId_BuildCopyPlanFailed) {
    const int    valid_layer   = 0;
    const int    gpu_block_idx = 1;
    const auto   gpu_bufs = allocator_->convertIndexToBuffer(valid_layer, "default", gpu_block_idx);
    const size_t total    = sumBlockInfosBytes(gpu_bufs);
    ASSERT_GT(total, 0u);

    auto pool = ensureBlockPool(total);
    ASSERT_NE(pool, nullptr);
    auto mem_blocks = pool->malloc(1);
    ASSERT_EQ(mem_blocks.size(), 1u);
    const BlockIdxType mem_block_index = static_cast<BlockIdxType>(mem_blocks[0]);

    MemoryOperationRequestPB req;
    auto*                    item = req.add_copy_items();
    addTaggedGpuBlocks(*item,
                       std::vector<BlockIdxType>(static_cast<size_t>(cache_config_.totalLayerNum()), gpu_block_idx));
    const auto& slots = connector_->layerGroupSlots();
    ASSERT_FALSE(slots.empty());
    auto* invalid_block = item->add_tagged_gpu_blocks();
    invalid_block->set_layer_id(cache_config_.layer_num);
    invalid_block->set_tag(slots.front().tag);
    invalid_block->set_block_id(gpu_block_idx);
    item->set_mem_block(mem_block_index);
    req.set_copy_direction(MemoryOperationRequestPB::H2D);

    MemoryOperationResponsePB resp;
    EXPECT_ANY_THROW((void)connector_->copyCache(req, resp));
}

TEST_F(KVCacheMemoryConnectorTest, copyCache_ReturnTrue_H2D_SingleLayer) {
    const int    layer_id      = 0;
    const int    gpu_block_idx = 2;
    const auto   gpu_bufs      = allocator_->convertIndexToBuffer(layer_id, "default", gpu_block_idx);
    const size_t total         = sumBlockInfosBytes(gpu_bufs);
    ASSERT_GT(total, 0u);

    // H2D 路径需要预先存在 mem pool 与有效 block
    auto pool = ensureBlockPool(total);
    ASSERT_NE(pool, nullptr);
    auto mem_blocks = pool->malloc(1);
    ASSERT_EQ(mem_blocks.size(), 1u);
    const BlockIdxType mem_block_index = static_cast<BlockIdxType>(mem_blocks[0]);
    const auto         mem_bufs        = pool->convertIndexToBuffer(0, mem_block_index);
    ASSERT_EQ(mem_bufs.size(), 1u);
    const auto& mem_buffer = mem_bufs[0];
    ASSERT_NE(mem_buffer.addr, nullptr);
    EXPECT_GE(mem_buffer.size_bytes, total);

    // 给mem_buffer填充数据
    setBlockBytes(mem_buffer, /*byte_offset=*/0, total, 'a');

    MemoryOperationRequestPB  req;
    auto*                     item = req.add_copy_items();
    std::vector<BlockIdxType> blocks_by_layer(cache_config_.layer_num, NULL_BLOCK_IDX);
    blocks_by_layer[static_cast<size_t>(layer_id)] = gpu_block_idx;
    addTaggedGpuBlocks(*item, blocks_by_layer);
    item->set_mem_block(mem_block_index);
    req.set_copy_direction(MemoryOperationRequestPB::H2D);

    MemoryOperationResponsePB resp;
    auto                      ok = connector_->copyCache(req, resp);
    EXPECT_TRUE(ok);
    EXPECT_TRUE(resp.success());

    // H2D, 验证数据是否拷贝成功
    verifyBlockInfosContent(gpu_bufs, 'a');
}

// MLA FP8 online-style: separate kv + kv-scale blobs per layer (656 + 132 bytes/token at seq_size_per_block=512).
// copyCache uses execNoBlockCopy split KV path (sm_copy scatter/gather) when eligible.
// ~40k prompt tokens => 79 full blocks in one request.
TEST_F(KVCacheMemoryConnectorTest, copyCache_ReturnTrue_H2D_SplitKvScale_NoBlockCopyOpt) {
    constexpr int      kLayerNum    = 78;
    constexpr uint32_t kSeqPerBlock = 512;
    constexpr int      kCopySeqLen  = 40000;
    constexpr int kCopyBlockCount = (kCopySeqLen + static_cast<int>(kSeqPerBlock) - 1) / static_cast<int>(kSeqPerBlock);
    constexpr int kGpuBlockBase   = 2;
    // Block pool must include GPU indices [kGpuBlockBase, kGpuBlockBase + kCopyBlockCount - 1] (from kCopySeqLen).
    constexpr int    kBlockNum         = kGpuBlockBase + kCopyBlockCount;
    constexpr size_t kKvBytesPerTok    = 656;
    constexpr size_t kScaleBytesPerTok = 132;

    AttentionConfigs attn_config;
    attn_config.kv_lora_rank            = 512;
    attn_config.rope_head_dim           = 64;
    attn_config.tokens_per_block        = kSeqPerBlock;
    attn_config.kernel_tokens_per_block = kSeqPerBlock;
    KVCacheSpecDesc desc;
    desc.tag                       = "default";
    desc.cache_type                = rtp_llm::KVCacheSpecType::MultiHeadLatentAttention;
    desc.dtype                     = rtp_llm::DataType::TYPE_FP8_E4M3;
    desc.kernel_seq_size_per_block = kSeqPerBlock;
    SpecBuildContext ctx;
    ctx.dtype              = rtp_llm::DataType::TYPE_FP8_E4M3;
    ctx.seq_size_per_block = kSeqPerBlock;
    ctx.attn_config        = &attn_config;
    auto mla_spec          = SpecBuilder::build(desc, ctx).first;

    cache_config_.layer_num                = static_cast<uint32_t>(kLayerNum);
    cache_config_.seq_size_per_block       = kSeqPerBlock;
    const size_t     kv_block_stride_bytes = kKvBytesPerTok * kSeqPerBlock;
    const size_t     kv_scale_stride_bytes = kScaleBytesPerTok * kSeqPerBlock;
    const size_t     kPerLayerStrideBytes  = kv_block_stride_bytes + kv_scale_stride_bytes;
    std::vector<int> layer_ids(kLayerNum);
    for (int i = 0; i < kLayerNum; ++i) {
        layer_ids[i] = i;
    }
    setTestTopology(
        cache_config_,
        {makeTestGroupForConfig(cache_config_, mla_spec, std::move(layer_ids), CacheGroupType::FULL, "default")});
    const auto                 topology_groups = cache_config_.topology().groups();
    std::vector<GroupBase> groups(topology_groups.begin(), topology_groups.end());
    ASSERT_EQ(groups.size(), 1u);
    groups[0].policy.explicit_block_num = static_cast<uint32_t>(kBlockNum);
    groups[0].kv_block_stride_bytes     = kv_block_stride_bytes;
    groups[0].kv_scale_stride_bytes     = kv_scale_stride_bytes;
    cache_config_.setTopology(std::move(groups), cache_config_.topology().layers());
    ASSERT_EQ(mla_spec->block_size_bytes(), cache_config_.kvBlockStrideBytesForGroup("default"));

    const size_t merged_one_key = memoryCacheBlockBytes(cache_config_);
    ASSERT_EQ(merged_one_key, static_cast<size_t>(kLayerNum) * kPerLayerStrideBytes);
    const int pool_mb =
        static_cast<int>((merged_one_key * static_cast<size_t>(kCopyBlockCount) + (1024ULL * 1024 - 1)) / (1024 * 1024))
        + 256;
    kv_cache_config_.memory_cache_size_mb = std::max(pool_mb, 512);

    allocator_ =
        std::make_shared<HybridPoolKVCacheAllocator>(cache_config_, AllocationType::DEVICE);
    ASSERT_TRUE(allocator_->init());
    connector_ = std::make_shared<KVCacheMemoryConnector>(
        cache_config_, kv_cache_config_, allocator_, server_addrs_);
    ASSERT_TRUE(connector_->init());

    auto pool = ensureBlockPool(merged_one_key);
    ASSERT_NE(pool, nullptr);
    std::vector<BlockIdxType> mem_block_indices;
    mem_block_indices.reserve(static_cast<size_t>(kCopyBlockCount));
    for (int i = 0; i < kCopyBlockCount; ++i) {
        auto blocks = pool->malloc(1);
        ASSERT_EQ(blocks.size(), 1u);
        mem_block_indices.push_back(static_cast<BlockIdxType>(blocks[0]));
        const auto mem_bufs = pool->convertIndexToBuffer(0, mem_block_indices.back());
        ASSERT_EQ(mem_bufs.size(), 1u);
        const auto& mem_buffer = mem_bufs[0];
        ASSERT_NE(mem_buffer.addr, nullptr);
        EXPECT_GE(mem_buffer.size_bytes, merged_one_key);
        const char tag = static_cast<char>(0x21 + (i % 94));
        setBlockBytes(mem_buffer, /*byte_offset=*/0, merged_one_key, tag);
    }

    MemoryOperationRequestPB req;
    req.set_copy_direction(MemoryOperationRequestPB::H2D);
    for (int i = 0; i < kCopyBlockCount; ++i) {
        auto* item = req.add_copy_items();
        addTaggedGpuBlocks(*item, std::vector<BlockIdxType>(kLayerNum, kGpuBlockBase + i));
        item->set_mem_block(mem_block_indices[static_cast<size_t>(i)]);
    }
    MemoryOperationResponsePB resp;
    ASSERT_TRUE(connector_->copyCache(req, resp));
    EXPECT_TRUE(resp.success());

    for (int i = 0; i < kCopyBlockCount; ++i) {
        const char tag = static_cast<char>(0x21 + (i % 94));
        for (int l = 0; l < kLayerNum; ++l) {
            const auto gpu_bufs = allocator_->convertIndexToBuffer(l, "default", kGpuBlockBase + i);
            ASSERT_GE(gpu_bufs.size(), 2u);
            verifyBlockInfosContent(gpu_bufs, tag);
        }
    }
}

TEST_F(KVCacheMemoryConnectorTest, copyCache_ReturnTrue_H2D_MultiLayer) {
    // 创建两个block_size不同的memory buffer
    std::vector<LayerBlock> gpu_layer_blocks1{
        {/*layer_id*/ 0, /*block_id*/ 1},
        {/*layer_id*/ 1, /*block_id*/ 2},
    };
    BlockIdxType mem_block_index1 = NULL_BLOCK_IDX;
    size_t       mem_block_size1  = 0;
    prepareBufferContent(gpu_layer_blocks1, mem_block_index1, mem_block_size1, /*fill_gpu=*/false, /*fill_cpu=*/true);
    ASSERT_FALSE(isNullBlockIdx(mem_block_index1));
    ASSERT_NE(mem_block_size1, 0);

    std::vector<LayerBlock> gpu_layer_blocks2{
        {/*layer_id*/ 0, /*block_id*/ 1},
        {/*layer_id*/ 1, /*block_id*/ 2},
        {/*layer_id*/ 2, /*block_id*/ 2},
    };
    BlockIdxType mem_block_index2 = NULL_BLOCK_IDX;
    size_t       mem_block_size2  = 0;
    prepareBufferContent(gpu_layer_blocks2, mem_block_index2, mem_block_size2, /*fill_gpu=*/false, /*fill_cpu=*/true);
    ASSERT_FALSE(isNullBlockIdx(mem_block_index2));
    ASSERT_NE(mem_block_size2, 0);

    MemoryOperationRequestPB req;
    req.set_copy_direction(MemoryOperationRequestPB::H2D);
    addOneCopyItemToPb(req, gpu_layer_blocks1, mem_block_index1);
    addOneCopyItemToPb(req, gpu_layer_blocks2, mem_block_index2);

    MemoryOperationResponsePB resp;
    auto                      ok = connector_->copyCache(req, resp);
    EXPECT_TRUE(ok);
    EXPECT_TRUE(resp.success());

    // H2D, 验证数据是否拷贝成功
    verifyGpuBufferContent(gpu_layer_blocks1);
    verifyGpuBufferContent(gpu_layer_blocks2);
}

TEST_F(KVCacheMemoryConnectorTest, copyCache_ReturnTrue_D2H_SingleLayer) {
    const int    layer_id      = 0;
    const int    gpu_block_idx = 3;
    const auto   gpu_bufs      = allocator_->convertIndexToBuffer(layer_id, "default", gpu_block_idx);
    const size_t total         = sumBlockInfosBytes(gpu_bufs);
    ASSERT_GT(total, 0u);

    // 给gpu_buf填充数据
    setBlockInfosContent(gpu_bufs, 'a');
    check_cuda_value(cudaDeviceSynchronize());

    // 为确保索引有效，仍然预先创建并分配一个块
    auto pool = ensureBlockPool(total);
    ASSERT_NE(pool, nullptr);
    auto mem_blocks = pool->malloc(1);
    ASSERT_EQ(mem_blocks.size(), 1u);
    const BlockIdxType mem_block_index = static_cast<BlockIdxType>(mem_blocks[0]);
    const auto         mem_bufs        = pool->convertIndexToBuffer(0, mem_block_index);
    ASSERT_EQ(mem_bufs.size(), 1u);
    const auto& mem_buffer = mem_bufs[0];
    ASSERT_NE(mem_buffer.addr, nullptr);
    EXPECT_GE(mem_buffer.size_bytes, total);

    MemoryOperationRequestPB  req;
    auto*                     item = req.add_copy_items();
    std::vector<BlockIdxType> blocks_by_layer(cache_config_.layer_num, NULL_BLOCK_IDX);
    blocks_by_layer[static_cast<size_t>(layer_id)] = gpu_block_idx;
    addTaggedGpuBlocks(*item, blocks_by_layer);
    item->set_mem_block(mem_block_index);
    req.set_copy_direction(MemoryOperationRequestPB::D2H);

    MemoryOperationResponsePB resp;
    auto                      ok = connector_->copyCache(req, resp);
    EXPECT_TRUE(ok);
    EXPECT_TRUE(resp.success());

    // D2H, 验证数据是否拷贝成功
    verifyBlockBytesEq(mem_buffer, /*byte_offset=*/0, total, 'a');
}

TEST_F(KVCacheMemoryConnectorTest, copyCache_ReturnTrue_D2H_MultiLayer) {
    std::vector<LayerBlock> gpu_layer_blocks1{
        {/*layer_id*/ 0, /*block_id*/ 1},
        {/*layer_id*/ 1, /*block_id*/ 2},
    };
    BlockIdxType mem_block_index1 = NULL_BLOCK_IDX;
    size_t       mem_block_size1  = 0;
    prepareBufferContent(gpu_layer_blocks1, mem_block_index1, mem_block_size1, /*fill_gpu=*/true, /*fill_cpu=*/false);
    ASSERT_FALSE(isNullBlockIdx(mem_block_index1));
    ASSERT_NE(mem_block_size1, 0);

    std::vector<LayerBlock> gpu_layer_blocks2{
        {/*layer_id*/ 0, /*block_id*/ 1},
        {/*layer_id*/ 1, /*block_id*/ 2},
        {/*layer_id*/ 2, /*block_id*/ 2},
    };
    BlockIdxType mem_block_index2 = NULL_BLOCK_IDX;
    size_t       mem_block_size2  = 0;
    prepareBufferContent(gpu_layer_blocks2, mem_block_index2, mem_block_size2, /*fill_gpu=*/true, /*fill_cpu=*/false);
    ASSERT_FALSE(isNullBlockIdx(mem_block_index2));
    ASSERT_NE(mem_block_size2, 0);

    MemoryOperationRequestPB req;
    req.set_copy_direction(MemoryOperationRequestPB::D2H);
    addOneCopyItemToPb(req, gpu_layer_blocks1, mem_block_index1);
    addOneCopyItemToPb(req, gpu_layer_blocks2, mem_block_index2);

    MemoryOperationResponsePB resp;
    auto                      ok = connector_->copyCache(req, resp);
    EXPECT_TRUE(ok);
    EXPECT_TRUE(resp.success());

    // D2H, 验证数据是否拷贝成功
    verifyCpuBufferContent(gpu_layer_blocks1, mem_block_index1, mem_block_size1);
    verifyCpuBufferContent(gpu_layer_blocks2, mem_block_index2, mem_block_size2);
}

// Regression test: verifies multi-layer D2H copy uses correct byte offsets even when memory buffer dtype size != 1.
// This would fail if prepareCopyBuffers mistakenly treats bytes as elements in Buffer::slice().
TEST_F(KVCacheMemoryConnectorTest, copyCache_D2H_MultiLayer_ValidatesByteOffsets) {
    std::vector<LayerBlock> gpu_layer_blocks{
        {/*layer_id*/ 0, /*block_id*/ 1},
        {/*layer_id*/ 1, /*block_id*/ 2},
    };

    // Fill GPU source with distinct byte patterns.
    for (const auto& lb : gpu_layer_blocks) {
        const auto gpu_bufs = allocator_->convertIndexToBuffer(lb.layer_id, "default", lb.block_id);
        ASSERT_GT(sumBlockInfosBytes(gpu_bufs), 0u);
        setBlockInfosContent(gpu_bufs, static_cast<char>('k' + lb.layer_id));
    }
    check_cuda_value(cudaDeviceSynchronize());

    // Allocate one memory block for the merged layout (one cache-key across all layers).
    size_t total_bytes = 0;
    for (const auto& slot : connector_->layerGroupSlots()) {
        total_bytes += slot.stride_bytes;
    }
    ASSERT_GT(total_bytes, 0u);

    auto pool = ensureBlockPool(total_bytes);
    ASSERT_NE(pool, nullptr);
    auto mem_blocks = pool->malloc(1);
    ASSERT_EQ(mem_blocks.size(), 1u);
    const BlockIdxType mem_block_index = static_cast<BlockIdxType>(mem_blocks[0]);

    const auto mem_bufs = pool->convertIndexToBuffer(0, mem_block_index);
    ASSERT_EQ(mem_bufs.size(), 1u);
    const auto& mem_buffer = mem_bufs[0];
    ASSERT_NE(mem_buffer.addr, nullptr);
    EXPECT_GE(mem_buffer.size_bytes, total_bytes);
    setBlockBytes(mem_buffer, /*byte_offset=*/0, total_bytes, 0);

    MemoryOperationRequestPB req;
    req.set_copy_direction(MemoryOperationRequestPB::D2H);
    addOneCopyItemToPb(req, gpu_layer_blocks, mem_block_index);

    MemoryOperationResponsePB resp;
    const auto                ok = connector_->copyCache(req, resp);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(resp.success());

    // Validate segments land at correct per-layer stride offsets.
    const size_t              layer_num = static_cast<size_t>(cache_config_.totalLayerNum());
    std::vector<BlockIdxType> layer_to_block(layer_num, NULL_BLOCK_IDX);
    for (const auto& lb : gpu_layer_blocks) {
        layer_to_block[static_cast<size_t>(lb.layer_id)] = lb.block_id;
    }
    size_t byte_off = 0;
    for (const auto& slot : connector_->layerGroupSlots()) {
        const auto block_id = layer_to_block[static_cast<size_t>(slot.layer_id)];
        if (isNullBlockIdx(block_id)) {
            byte_off += slot.stride_bytes;
            continue;
        }
        const auto gpu_bufs = allocator_->convertIndexToBuffer(slot.layer_id, slot.tag, block_id);
        const auto bytes    = sumBlockInfosBytes(gpu_bufs);
        ASSERT_GT(bytes, 0u);
        ASSERT_LE(bytes, slot.stride_bytes);
        verifyBlockBytesEq(mem_buffer, byte_off, bytes, static_cast<char>('k' + slot.layer_id));
        byte_off += slot.stride_bytes;
    }
}

}  // namespace rtp_llm::test

int main(int argc, char** argv) {
    rtp_llm::initLogger();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
