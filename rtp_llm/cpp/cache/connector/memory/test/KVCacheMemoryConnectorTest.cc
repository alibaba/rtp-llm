// Copyright (c) RTP-LLM

#include <csignal>
#include <chrono>
#include <cstdio>
#include <dirent.h>
#include <execinfo.h>
#include <string>
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
#include "rtp_llm/cpp/cache/SingleTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/HybridTypeKVCacheAllocator.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
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

static CrashHandlerInstaller g_crash_handler_installer;

class DiskTempDir {
public:
    DiskTempDir() {
        char tmpl[] = "/tmp/rtp_memory_connector_disk_test_XXXXXX";
        auto path   = ::mkdtemp(tmpl);
        EXPECT_NE(path, nullptr);
        if (path != nullptr) {
            path_ = path;
        }
    }
    ~DiskTempDir() {
        if (path_.empty()) {
            return;
        }
        const auto work_dir = path_ + "/rtp_llm_disk_kv";
        if (auto* dir = ::opendir(work_dir.c_str())) {
            while (auto* entry = ::readdir(dir)) {
                const std::string name(entry->d_name);
                if (name == "." || name == "..") {
                    continue;
                }
                ::unlink((work_dir + "/" + name).c_str());
            }
            ::closedir(dir);
        }
        ::rmdir(work_dir.c_str());
        ::rmdir(path_.c_str());
    }

    const std::string& path() const {
        return path_;
    }

private:
    std::string path_;
};

std::string joinPaths(const std::vector<std::string>& paths) {
    std::string joined;
    for (const auto& path : paths) {
        if (!joined.empty()) {
            joined += ",";
        }
        joined += path;
    }
    return joined;
}

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
        allocator_    = std::make_shared<SingleTypeKVCacheAllocator>(cache_config_, AllocationType::DEVICE);
        ASSERT_TRUE(allocator_->init());

        const int server_num = 4;
        startRpcServer(server_num);

        connector_ =
            std::make_shared<KVCacheMemoryConnector>(cache_config_, kv_cache_config_, allocator_, server_addrs_);
        ASSERT_TRUE(connector_->init());
    }

    void TearDown() override {}

    CacheConfig                                 cache_config_;
    KVCacheConfig                               kv_cache_config_;
    std::shared_ptr<KVCacheAllocator>           allocator_;
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
        constexpr int kTestMemoryCacheSizeMb      = 64;
        constexpr int kTestMemoryCacheSyncTimeout = 1000;

        CacheConfig config;
        config.layer_num                              = layer_num;
        config.layer_all_num                          = layer_num;
        config.block_num                              = block_num;
        config.seq_size_per_block                     = seq_size_per_block;
        kv_cache_config_.memory_cache_size_mb         = kTestMemoryCacheSizeMb;
        kv_cache_config_.memory_cache_sync_timeout_ms = kTestMemoryCacheSyncTimeout;

        auto mha_spec       = std::make_shared<MHAKVCacheSpec>();
        mha_spec->layer_num = layer_num;
        // mha_spec->block_nums         = block_num;
        mha_spec->local_head_num_kv  = 8;
        mha_spec->size_per_head      = 128;
        mha_spec->seq_size_per_block = seq_size_per_block;
        mha_spec->dtype              = mha_dtype;
        mha_spec->type               = KVCacheSpecType::MultiHeadAttention;
        config.cache_specs.push_back(mha_spec);
        // Keep CacheConfig sizes consistent with current business definition (see CacheConfig.h):
        // - kv_block_stride_bytes / kv_scale_stride_bytes are "per-layer" strides for one logical block
        // - kv_block_size_bytes / kv_scale_size_bytes are "all layers" totals for one logical block
        // - block_size_bytes = kv + scales together for one logical block (all layers)
        config.dtype                 = mha_spec->dtype;
        config.kv_block_stride_bytes = mha_spec->block_size_bytes();
        config.kv_scale_stride_bytes = mha_spec->scale_block_size_bytes();
        config.kv_block_size_bytes   = static_cast<size_t>(layer_num) * config.kv_block_stride_bytes;
        config.kv_scale_size_bytes   = static_cast<size_t>(layer_num) * config.kv_scale_stride_bytes;
        config.block_size_bytes      = config.kv_block_size_bytes + config.kv_scale_size_bytes;
        // Per-layer stride used by MemoryConnector merged layout.
        const size_t per_layer_stride_bytes = config.kv_block_stride_bytes + config.kv_scale_stride_bytes;
        config.layer_to_block_stride_bytes.assign(static_cast<size_t>(config.layer_all_num),
                                                  static_cast<int>(per_layer_stride_bytes));

        std::vector<int> layer_ids(layer_num);
        for (int i = 0; i < layer_num; ++i) {
            layer_ids[i] = i;
        }
        config.layer_ids.push_back(layer_ids);
        // SingleTypeKVCacheAllocator::init() expects global_layer_ids[0] to exist.
        // In these unit tests we only have one "model group", so keep it consistent with layer_ids.
        config.global_layer_ids.push_back(layer_ids);

        return config;
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
        for (const auto& stride : cfg.layer_to_block_stride_bytes) {
            if (stride > 0) {
                total += static_cast<size_t>(stride);
            }
        }
        return total;
    }
    size_t memoryCacheBlockBytes() const {
        return memoryCacheBlockBytes(cache_config_);
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

    void verifyGpuBufferContent(const std::vector<LayerBlock>& gpu_layer_blocks) const {
        for (const auto& layer_block : gpu_layer_blocks) {
            if (isNullBlockIdx(layer_block.block_id)) {
                continue;
            }
            const auto gpu_bufs = allocator_->convertIndexToBuffer(layer_block.layer_id, layer_block.block_id);
            ASSERT_GT(sumBlockInfosBytes(gpu_bufs), 0u);
            verifyBlockInfosContent(gpu_bufs, static_cast<char>('k' + layer_block.layer_id));
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

        const size_t              layer_num = static_cast<size_t>(cache_config_.layer_all_num);
        std::vector<BlockIdxType> layer_to_block(layer_num, NULL_BLOCK_IDX);
        for (const auto& lb : gpu_layer_blocks) {
            ASSERT_GE(lb.layer_id, 0);
            ASSERT_LT(static_cast<size_t>(lb.layer_id), layer_num);
            layer_to_block[static_cast<size_t>(lb.layer_id)] = lb.block_id;
        }

        size_t byte_off = 0;
        for (size_t layer = 0; layer < layer_num; ++layer) {
            const size_t layer_stride = static_cast<size_t>(cache_config_.layer_to_block_stride_bytes[layer]);
            const auto   block_id     = layer_to_block[layer];
            if (isNullBlockIdx(block_id)) {
                byte_off += layer_stride;
                continue;
            }
            const auto gpu_bufs = allocator_->convertIndexToBuffer(static_cast<int>(layer), block_id);
            const auto bytes    = sumBlockInfosBytes(gpu_bufs);
            ASSERT_GT(bytes, 0u);
            ASSERT_LE(bytes, layer_stride);

            const char expected_k = static_cast<char>('k' + static_cast<int>(layer));
            verifyBlockBytesEq(mem_buffer, byte_off, bytes, expected_k);
            byte_off += layer_stride;
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
        const size_t              layer_num = static_cast<size_t>(cache_config_.layer_all_num);
        std::vector<BlockIdxType> layer_to_block(layer_num, NULL_BLOCK_IDX);
        for (const auto& lb : gpu_layer_blocks) {
            ASSERT_GE(lb.layer_id, 0);
            ASSERT_LT(static_cast<size_t>(lb.layer_id), layer_num);
            layer_to_block[static_cast<size_t>(lb.layer_id)] = lb.block_id;
        }

        size_t total = 0;
        for (size_t layer = 0; layer < layer_num; ++layer) {
            total += static_cast<size_t>(cache_config_.layer_to_block_stride_bytes[layer]);
        }

        for (size_t layer = 0; layer < layer_num; ++layer) {
            const auto block_id = layer_to_block[layer];
            if (isNullBlockIdx(block_id)) {
                continue;
            }
            const auto gpu_bufs = allocator_->convertIndexToBuffer(static_cast<int>(layer), block_id);
            const auto bytes    = sumBlockInfosBytes(gpu_bufs);
            ASSERT_GT(bytes, 0u);
            ASSERT_LE(bytes, static_cast<size_t>(cache_config_.layer_to_block_stride_bytes[layer]));
            if (fill_gpu) {
                setBlockInfosContent(gpu_bufs, static_cast<char>('k' + static_cast<int>(layer)));
            }
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
            for (size_t layer = 0; layer < layer_num; ++layer) {
                const size_t layer_stride = static_cast<size_t>(cache_config_.layer_to_block_stride_bytes[layer]);
                const auto   block_id     = layer_to_block[layer];
                if (isNullBlockIdx(block_id)) {
                    byte_off += layer_stride;
                    continue;
                }
                const auto gpu_bufs = allocator_->convertIndexToBuffer(static_cast<int>(layer), block_id);
                const auto bytes    = sumBlockInfosBytes(gpu_bufs);
                ASSERT_GT(bytes, 0u);
                ASSERT_LE(bytes, layer_stride);
                setBlockBytes(mem_buffer, byte_off, bytes, static_cast<char>('k' + static_cast<int>(layer)));
                byte_off += layer_stride;
            }
        }

        mem_block_index = malloced_mem_block_index;
        // Use the actual pool block size as the mem-block-size key.
        mem_block_size = mem_buffer.size_bytes;
    }
    void addOneCopyItemToPb(MemoryOperationRequestPB&      req,
                            const std::vector<LayerBlock>& gpu_layer_blocks,
                            BlockIdxType                   mem_block_index) const {
        auto*            item      = req.add_copy_items();
        const size_t     layer_num = static_cast<size_t>(cache_config_.layer_all_num);
        std::vector<int> blocks(layer_num, static_cast<int>(NULL_BLOCK_IDX));
        for (const auto& layer_block : gpu_layer_blocks) {
            ASSERT_GE(layer_block.layer_id, 0);
            ASSERT_LT(static_cast<size_t>(layer_block.layer_id), layer_num);
            blocks[static_cast<size_t>(layer_block.layer_id)] = static_cast<int>(layer_block.block_id);
        }
        for (size_t layer = 0; layer < layer_num; ++layer) {
            item->add_gpu_blocks(blocks[layer]);
        }
        item->set_mem_block(static_cast<int>(mem_block_index));
        item->set_is_complete(true);
        item->set_backing_type(MemoryOperationRequestPB::MEMORY);
    }
    LayerBlockIds makeLayerBlockIds(const std::vector<std::vector<BlockIdxType>>& per_layer_block_indices,
                                    size_t                                        cache_keys_num) const {
        LayerBlockIds lbs;
        const size_t  layer_num = cache_config_.layer_num;
        lbs.reserve(layer_num);

        for (size_t layer = 0; layer < layer_num; ++layer) {
            auto ptr = std::make_shared<BlockIds>();
            if (layer < per_layer_block_indices.size()) {
                ptr->assign(per_layer_block_indices[layer]);
            }
            if (ptr->blocks().size() < cache_keys_num) {
                ptr->resize(cache_keys_num, NULL_BLOCK_IDX);
            }
            lbs.emplace_back(std::move(ptr));
        }
        return lbs;
    }
    std::shared_ptr<KVCacheResource>
    makeCacheResource(const CacheKeysType&                          cache_keys,
                      const std::vector<std::vector<BlockIdxType>>& per_layer_block_indices,
                      size_t                                        reuse_len = 0) const {
        auto res                  = std::make_shared<KVCacheResource>();
        res->cache_keys           = cache_keys;
        res->layer_block_ids      = makeLayerBlockIds(per_layer_block_indices, cache_keys.size());
        const size_t region_count = static_cast<size_t>(KVCacheRegionName::REGION_COUNT);
        res->layer_region_block_ids.resize(res->layer_block_ids.size());
        for (size_t layer = 0; layer < res->layer_block_ids.size(); ++layer) {
            res->layer_region_block_ids[layer].assign(region_count, nullptr);
            res->layer_region_block_ids[layer][static_cast<size_t>(KVCacheRegionName::DEFAULT)] =
                res->layer_block_ids[layer];
        }
        // reuse_len in these tests means "GPU already-reused prefix length".
        // KVCacheResource::reuseBlockNum() is derived from (device + memory + remote),
        // so set device reuse here to make asyncMatch/asyncRead semantics consistent.
        res->setDeviceReuseBlockNum(reuse_len);
        // These unit tests want to include the whole cache_keys range by default.
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
        // Business implementation uses a single `block_pool_` with fixed block_size_bytes
        // ("one cache-key across all layers" total bytes). Smaller mem_block_size values
        // (e.g. when some layers are NULL for a key) should still be served by the same pool.
        auto pool = connector_->block_pool_;
        if (!pool) {
            // initBlockPool uses cache_config_.block_size_bytes and kv_cache_config_.memory_cache_size_mb.
            EXPECT_NO_THROW(connector_->initBlockPool());
            pool = connector_->block_pool_;
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
    KVCacheConfig makeDiskKvConfig(const std::vector<std::string>& paths, int64_t disk_size_mb = 1) const {
        auto kv_cfg                              = kv_cache_config_;
        kv_cfg.enable_memory_cache               = true;
        kv_cfg.enable_memory_cache_disk          = true;
        kv_cfg.memory_cache_disk_paths           = joinPaths(paths);
        kv_cfg.memory_cache_disk_size_mb         = disk_size_mb;
        kv_cfg.memory_cache_disk_buffered_io     = true;
        kv_cfg.memory_cache_disk_sync_timeout_ms = 1000;
        kv_cfg.enable_tiered_memory_cache        = false;
        return kv_cfg;
    }
    ParallelismConfig
    makeParallelismConfig(int64_t local_rank = 0, int64_t local_world_size = 1, int64_t world_rank = 0) const {
        ParallelismConfig parallelism_config;
        parallelism_config.world_size       = local_world_size;
        parallelism_config.world_rank       = world_rank;
        parallelism_config.local_world_size = local_world_size;
        parallelism_config.local_rank       = local_rank;
        return parallelism_config;
    }
};

TEST_F(KVCacheMemoryConnectorTest, init_ReturnFalse_NoWorkerAddrs) {
    // 构造空的 worker 地址，BroadcastManager::init() 会失败；业务代码使用 RTP_LLM_CHECK，
    // 因此这里期望抛出 std::runtime_error。
    std::vector<std::string> empty_addrs;
    auto conn = std::make_shared<KVCacheMemoryConnector>(cache_config_, kv_cache_config_, allocator_, empty_addrs);
    EXPECT_THROW(conn->init(), std::runtime_error);
}

TEST_F(KVCacheMemoryConnectorTest, init_ReturnFalse_WhenMemoryCacheSizeMbZero) {
    auto kv_cfg                         = kv_cache_config_;
    kv_cfg.memory_cache_size_mb         = 0;
    kv_cfg.memory_cache_sync_timeout_ms = 1000;

    auto conn = std::make_shared<KVCacheMemoryConnector>(cache_config_, kv_cfg, allocator_, server_addrs_);
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

    auto conn = std::make_shared<KVCacheMemoryConnector>(cache_config_, kv_cfg, allocator_, server_addrs_);
    EXPECT_THROW(conn->init(), std::runtime_error);
    // Init fails early, nothing should be created.
    EXPECT_EQ(conn->block_cache_, nullptr);
    EXPECT_EQ(conn->broadcast_manager_, nullptr);
    EXPECT_EQ(conn->wait_done_thread_pool_, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, init_ReturnFalse_WhenBlockSizeBytesZero) {
    auto cfg = cache_config_;
    cfg.layer_to_block_stride_bytes.clear();
    cfg.group_kv_block_stride_bytes.clear();
    cfg.group_kv_scale_stride_bytes.clear();
    cfg.kv_block_stride_bytes = 0;
    cfg.kv_scale_stride_bytes = 0;
    cfg.kv_block_size_bytes   = 0;
    cfg.kv_scale_size_bytes   = 0;
    cfg.block_size_bytes      = 0;

    auto kv_cfg                         = kv_cache_config_;
    kv_cfg.memory_cache_size_mb         = 64;
    kv_cfg.memory_cache_sync_timeout_ms = 1000;

    auto conn = std::make_shared<KVCacheMemoryConnector>(cfg, kv_cfg, allocator_, server_addrs_);
    EXPECT_THROW(conn->init(), std::runtime_error);
}

TEST_F(KVCacheMemoryConnectorTest, init_ReturnFalse_WhenPoolTooSmallForBlockSize) {
    auto cfg = cache_config_;
    // Make sure pool_size_mb * 1MB / total_stride_bytes == 0 -> createBlockPool() should fail with CHECK.
    cfg.layer_to_block_stride_bytes.assign(static_cast<size_t>(cfg.layer_num), 1024 * 1024);  // 1MB per layer

    auto kv_cfg                         = kv_cache_config_;
    kv_cfg.memory_cache_size_mb         = 1;     // 1MB
    kv_cfg.memory_cache_sync_timeout_ms = 1000;  // valid

    auto conn = std::make_shared<KVCacheMemoryConnector>(cfg, kv_cfg, allocator_, server_addrs_);
    EXPECT_THROW(conn->init(), std::runtime_error);
}

TEST_F(KVCacheMemoryConnectorTest, init_ReturnTrue_WithWorkerAddrs) {
    // 使用有效的 worker 地址，init 应成功并正确设置 manager
    auto conn = std::make_shared<KVCacheMemoryConnector>(cache_config_, kv_cache_config_, allocator_, server_addrs_);
    auto ok   = conn->init();
    EXPECT_TRUE(ok);
    ASSERT_NE(conn->block_cache_, nullptr);
    ASSERT_NE(conn->broadcast_manager_, nullptr);
    EXPECT_EQ(conn->broadcast_manager_->workerNum(), server_addrs_.size());
}

TEST_F(KVCacheMemoryConnectorTest, initDiskBlockPool_UsesLocalRankPathAndPreallocatesFile) {
    DiskTempDir disk0;
    DiskTempDir disk1;
    ASSERT_FALSE(disk0.path().empty());
    ASSERT_FALSE(disk1.path().empty());

    auto kv_cfg = makeDiskKvConfig({disk0.path(), disk1.path()});
    auto conn   = std::make_shared<KVCacheMemoryConnector>(cache_config_,
                                                         kv_cfg,
                                                         makeParallelismConfig(/*local_rank=*/1,
                                                                               /*local_world_size=*/2,
                                                                               /*world_rank=*/5),
                                                         allocator_,
                                                         server_addrs_,
                                                         nullptr);
    ASSERT_TRUE(conn->init());
    ASSERT_NE(conn->complete_disk_pool_, nullptr);
    ASSERT_EQ(conn->incomplete_disk_pool_, nullptr);
    EXPECT_NE(conn->complete_disk_pool_->filePath().find(disk1.path()), std::string::npos);
    EXPECT_NE(conn->complete_disk_pool_->filePath().find("rank_1_world_5_complete.kv"), std::string::npos);
    EXPECT_GT(conn->complete_disk_pool_->totalSlots(), 0u);
    EXPECT_EQ(conn->complete_disk_pool_->freeSlots(), conn->complete_disk_pool_->totalSlots());
}

TEST_F(KVCacheMemoryConnectorTest, initDiskBlockPool_RejectsInvalidDiskConfig) {
    DiskTempDir disk0;
    ASSERT_FALSE(disk0.path().empty());

    {
        auto kv_cfg                = makeDiskKvConfig({disk0.path()});
        kv_cfg.enable_memory_cache = false;
        auto conn                  = std::make_shared<KVCacheMemoryConnector>(
            cache_config_, kv_cfg, makeParallelismConfig(), allocator_, server_addrs_, nullptr);
        EXPECT_THROW(conn->init(), std::runtime_error);
    }
    {
        auto kv_cfg                       = makeDiskKvConfig({disk0.path()});
        kv_cfg.enable_tiered_memory_cache = true;
        auto conn                         = std::make_shared<KVCacheMemoryConnector>(
            cache_config_, kv_cfg, makeParallelismConfig(), allocator_, server_addrs_, nullptr);
        EXPECT_THROW(conn->init(), std::runtime_error);
    }
    {
        auto kv_cfg = makeDiskKvConfig({disk0.path()}, /*disk_size_mb=*/0);
        auto conn   = std::make_shared<KVCacheMemoryConnector>(
            cache_config_, kv_cfg, makeParallelismConfig(), allocator_, server_addrs_, nullptr);
        EXPECT_THROW(conn->init(), std::runtime_error);
    }
    {
        auto kv_cfg = makeDiskKvConfig({disk0.path()});
        auto conn   = std::make_shared<KVCacheMemoryConnector>(cache_config_,
                                                             kv_cfg,
                                                             makeParallelismConfig(/*local_rank=*/0,
                                                                                   /*local_world_size=*/2,
                                                                                   /*world_rank=*/0),
                                                             allocator_,
                                                             server_addrs_,
                                                             nullptr);
        EXPECT_THROW(conn->init(), std::runtime_error);
    }
    {
        auto kv_cfg = makeDiskKvConfig({disk0.path()});
        auto conn   = std::make_shared<KVCacheMemoryConnector>(cache_config_,
                                                             kv_cfg,
                                                             makeParallelismConfig(/*local_rank=*/1,
                                                                                   /*local_world_size=*/1,
                                                                                   /*world_rank=*/1),
                                                             allocator_,
                                                             server_addrs_,
                                                             nullptr);
        EXPECT_THROW(conn->init(), std::runtime_error);
    }
}

TEST_F(KVCacheMemoryConnectorTest, allocateOneBacking_FallsBackToDiskWhenMemoryPoolFull) {
    DiskTempDir disk0;
    ASSERT_FALSE(disk0.path().empty());

    auto kv_cfg = makeDiskKvConfig({disk0.path()});
    auto conn   = std::make_shared<KVCacheMemoryConnector>(
        cache_config_, kv_cfg, makeParallelismConfig(), allocator_, server_addrs_, nullptr);
    ASSERT_TRUE(conn->init());
    ASSERT_NE(conn->block_pool_, nullptr);
    ASSERT_NE(conn->complete_disk_pool_, nullptr);

    const auto free_memory_blocks = conn->block_pool_->freeBlocksNum();
    ASSERT_GT(free_memory_blocks, 0u);
    auto held_memory_blocks = conn->block_pool_->malloc(static_cast<int>(free_memory_blocks));
    ASSERT_EQ(held_memory_blocks.size(), free_memory_blocks);
    EXPECT_EQ(conn->block_pool_->freeBlocksNum(), 0u);

    KVCacheMemoryConnector::CopyInfoPerKey copy_info;
    ASSERT_TRUE(conn->allocateOneBacking(copy_info));
    EXPECT_EQ(copy_info.backing_type, CacheBackingType::DISK);
    EXPECT_TRUE(isNullBlockIdx(copy_info.mem_block));
    EXPECT_GE(copy_info.disk_slot, 0);
    EXPECT_EQ(conn->complete_disk_pool_->freeSlots(), conn->complete_disk_pool_->totalSlots() - 1);

    conn->releaseRequestBacking(copy_info);
    conn->block_pool_->requestFree(held_memory_blocks);
    EXPECT_EQ(conn->complete_disk_pool_->freeSlots(), conn->complete_disk_pool_->totalSlots());
}

TEST_F(KVCacheMemoryConnectorTest, putToCache_DiskBackingTransfersRequestRefToCacheRef) {
    DiskTempDir disk0;
    ASSERT_FALSE(disk0.path().empty());

    auto kv_cfg = makeDiskKvConfig({disk0.path()});
    auto conn   = std::make_shared<KVCacheMemoryConnector>(
        cache_config_, kv_cfg, makeParallelismConfig(), allocator_, server_addrs_, nullptr);
    ASSERT_TRUE(conn->init());

    const auto free_memory_blocks = conn->block_pool_->freeBlocksNum();
    auto       held_memory_blocks = conn->block_pool_->malloc(static_cast<int>(free_memory_blocks));
    ASSERT_EQ(held_memory_blocks.size(), free_memory_blocks);

    KVCacheMemoryConnector::CopyInfoPerKey copy_info;
    ASSERT_TRUE(conn->allocateOneBacking(copy_info));
    ASSERT_EQ(copy_info.backing_type, CacheBackingType::DISK);
    copy_info.cache_key   = 1001;
    copy_info.is_complete = true;
    conn->putToCache(copy_info);
    EXPECT_TRUE(copy_info.request_released);

    EXPECT_EQ(conn->complete_disk_pool_->freeSlots(), conn->complete_disk_pool_->totalSlots() - 1);
    EXPECT_EQ(conn->complete_disk_pool_->availableSlots(), conn->complete_disk_pool_->totalSlots());

    auto evicted = conn->block_cache_->popOldestEvictable();
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->disk_slot, copy_info.disk_slot);
    conn->releaseCacheBacking(*evicted);
    EXPECT_EQ(conn->complete_disk_pool_->freeSlots(), conn->complete_disk_pool_->totalSlots());

    conn->block_pool_->requestFree(held_memory_blocks);
}

TEST_F(KVCacheMemoryConnectorTest, putToCache_DuplicateDiskItemRollsBackCacheRef) {
    DiskTempDir disk0;
    ASSERT_FALSE(disk0.path().empty());

    auto kv_cfg = makeDiskKvConfig({disk0.path()});
    auto conn   = std::make_shared<KVCacheMemoryConnector>(
        cache_config_, kv_cfg, makeParallelismConfig(), allocator_, server_addrs_, nullptr);
    ASSERT_TRUE(conn->init());
    ASSERT_GT(conn->complete_disk_pool_->totalSlots(), 1u);

    const auto free_memory_blocks = conn->block_pool_->freeBlocksNum();
    auto       held_memory_blocks = conn->block_pool_->malloc(static_cast<int>(free_memory_blocks));
    ASSERT_EQ(held_memory_blocks.size(), free_memory_blocks);

    KVCacheMemoryConnector::CopyInfoPerKey first;
    ASSERT_TRUE(conn->allocateOneBacking(first));
    ASSERT_EQ(first.backing_type, CacheBackingType::DISK);
    first.cache_key   = 1002;
    first.is_complete = true;
    conn->putToCache(first);
    EXPECT_TRUE(first.request_released);
    EXPECT_EQ(conn->complete_disk_pool_->freeSlots(), conn->complete_disk_pool_->totalSlots() - 1);

    KVCacheMemoryConnector::CopyInfoPerKey duplicate;
    ASSERT_TRUE(conn->allocateOneBacking(duplicate));
    ASSERT_EQ(duplicate.backing_type, CacheBackingType::DISK);
    duplicate.cache_key   = first.cache_key;
    duplicate.is_complete = true;
    conn->putToCache(duplicate);
    EXPECT_TRUE(duplicate.request_released);
    EXPECT_EQ(conn->complete_disk_pool_->freeSlots(), conn->complete_disk_pool_->totalSlots() - 1);

    auto evicted = conn->block_cache_->popOldestEvictable();
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->disk_slot, first.disk_slot);
    conn->releaseCacheBacking(*evicted);
    EXPECT_EQ(conn->complete_disk_pool_->freeSlots(), conn->complete_disk_pool_->totalSlots());

    conn->block_pool_->requestFree(held_memory_blocks);
}

TEST_F(KVCacheMemoryConnectorTest, validateCopyItemBacking_AcceptsUnsetDiskSlotForMemoryOnly) {
    MemoryOperationRequestPB::CopyItem item;
    item.set_backing_type(MemoryOperationRequestPB::MEMORY);
    item.set_mem_block(1);
    EXPECT_TRUE(connector_->validateCopyItemBacking(item));

    item.set_disk_slot(-1);
    EXPECT_TRUE(connector_->validateCopyItemBacking(item));

    item.set_disk_slot(0);
    EXPECT_FALSE(connector_->validateCopyItemBacking(item));

    MemoryOperationRequestPB::CopyItem disk_item;
    disk_item.set_backing_type(MemoryOperationRequestPB::DISK);
    disk_item.set_mem_block(NULL_BLOCK_IDX);
    EXPECT_FALSE(connector_->validateCopyItemBacking(disk_item));
    disk_item.set_disk_slot(0);
    EXPECT_FALSE(connector_->validateCopyItemBacking(disk_item));

    DiskTempDir disk0;
    auto        kv_cfg    = makeDiskKvConfig({disk0.path()});
    auto        disk_conn = std::make_shared<KVCacheMemoryConnector>(
        cache_config_, kv_cfg, makeParallelismConfig(), allocator_, server_addrs_, nullptr);
    ASSERT_TRUE(disk_conn->init());
    EXPECT_TRUE(disk_conn->validateCopyItemBacking(disk_item));
}

TEST_F(KVCacheMemoryConnectorTest, initBlockPool_Throw_WhenMemoryCacheSizeMbZero) {
    auto kv_cfg                         = kv_cache_config_;
    kv_cfg.memory_cache_size_mb         = 0;
    kv_cfg.memory_cache_sync_timeout_ms = 1000;

    auto conn = std::make_shared<KVCacheMemoryConnector>(cache_config_, kv_cfg, allocator_, server_addrs_);
    EXPECT_THROW(conn->initBlockPool(), std::runtime_error);
}

TEST_F(KVCacheMemoryConnectorTest, initBlockPool_Throw_WhenBlockSizeBytesZero) {
    auto cfg = cache_config_;
    cfg.layer_to_block_stride_bytes.clear();
    cfg.group_kv_block_stride_bytes.clear();
    cfg.group_kv_scale_stride_bytes.clear();
    cfg.kv_block_stride_bytes = 0;
    cfg.kv_scale_stride_bytes = 0;
    cfg.kv_block_size_bytes   = 0;
    cfg.kv_scale_size_bytes   = 0;
    cfg.block_size_bytes      = 0;

    auto kv_cfg                         = kv_cache_config_;
    kv_cfg.memory_cache_size_mb         = 64;
    kv_cfg.memory_cache_sync_timeout_ms = 1000;

    auto conn = std::make_shared<KVCacheMemoryConnector>(cfg, kv_cfg, allocator_, server_addrs_);
    EXPECT_THROW(conn->initBlockPool(), std::runtime_error);
}

TEST_F(KVCacheMemoryConnectorTest, initBlockPool_Throw_WhenCreateBlockPoolFails) {
    auto cfg = cache_config_;
    // Force createBlockPool() to compute block_num=0:
    // block_num = pool_size_mb * 1MB / total_stride_bytes.
    cfg.layer_to_block_stride_bytes.assign(static_cast<size_t>(cfg.layer_num), 1024 * 1024);  // 1MB per layer

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

    auto conn = std::make_shared<KVCacheMemoryConnector>(cache_config_, kv_cfg, allocator_, server_addrs_);
    EXPECT_NO_THROW(conn->initBlockPool());
    auto pool = conn->block_pool_;
    ASSERT_NE(pool, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, buildCopyPlanForWrite_UsesLayerAndRegionSlots) {
    auto cfg          = cache_config_;
    cfg.layer_num     = 1;
    cfg.layer_all_num = 1;
    cfg.layer_to_group_id.assign(1, 0);
    cfg.layer_to_group_ids.assign(1, std::vector<int>{0, 1});
    cfg.layer_region_to_group_id.assign(1, std::vector<int>(static_cast<size_t>(KVCacheRegionName::REGION_COUNT), -1));
    cfg.layer_region_to_group_id[0][static_cast<size_t>(KVCacheRegionName::CSA_KV)] = 0;
    cfg.layer_region_to_group_id[0][static_cast<size_t>(KVCacheRegionName::SWA_KV)] = 1;
    cfg.group_types                 = {CacheGroupType::FULL, CacheGroupType::FULL};
    cfg.group_kv_block_stride_bytes = {16, 32};
    cfg.group_kv_scale_stride_bytes = {0, 0};
    cfg.layer_to_block_stride_bytes = {999};

    auto kv_cfg                         = kv_cache_config_;
    kv_cfg.memory_cache_size_mb         = 64;
    kv_cfg.memory_cache_sync_timeout_ms = 1000;
    auto conn          = std::make_shared<KVCacheMemoryConnector>(cfg, kv_cfg, allocator_, server_addrs_);
    conn->block_cache_ = std::make_shared<MemoryDiskBlockCache>();
    ASSERT_NO_THROW(conn->initBlockPool());

    auto slots = conn->layerRegionSlots();
    ASSERT_EQ(slots.size(), 2u);
    EXPECT_EQ(slots[0].layer_id, 0);
    EXPECT_EQ(slots[0].region_name, KVCacheRegionName::CSA_KV);
    EXPECT_EQ(slots[0].group_id, 0);
    EXPECT_EQ(slots[0].stride_bytes, 16u);
    EXPECT_EQ(slots[1].layer_id, 0);
    EXPECT_EQ(slots[1].region_name, KVCacheRegionName::SWA_KV);
    EXPECT_EQ(slots[1].group_id, 1);
    EXPECT_EQ(slots[1].stride_bytes, 32u);

    auto resource         = std::make_shared<KVCacheResource>();
    resource->cacheKeys() = {101, 102, 103};
    resource->initGroups(/*group_num=*/2,
                         /*layer_num=*/1,
                         cfg.layer_to_group_id,
                         /*kernel_blocks_per_kv_block=*/1,
                         cfg.group_types,
                         cfg.layer_region_to_group_id);
    resource->mutableBlockIds(/*group_id=*/0).assign({11, 12, 13});
    resource->mutableBlockIds(/*group_id=*/1).assign({21, NULL_BLOCK_IDX, 23});

    bool no_need_write = true;
    auto plan          = conn->buildCopyPlanForWrite(
        resource->cacheKeys(), resource->layerAttnBlocks(), slots, /*start_index=*/0, /*write_num=*/3, no_need_write);

    ASSERT_NE(plan, nullptr);
    EXPECT_FALSE(no_need_write);
    ASSERT_EQ(plan->copy_infos.size(), 3u);
    EXPECT_TRUE(plan->copy_infos[0].is_complete);
    EXPECT_FALSE(plan->copy_infos[1].is_complete);
    EXPECT_TRUE(plan->copy_infos[2].is_complete);
    EXPECT_EQ(plan->copy_infos[0].gpu_blocks, (std::vector<BlockIdxType>{11, 21}));
    EXPECT_EQ(plan->copy_infos[1].gpu_blocks, (std::vector<BlockIdxType>{12, NULL_BLOCK_IDX}));
    EXPECT_EQ(plan->copy_infos[2].gpu_blocks, (std::vector<BlockIdxType>{13, 23}));
}

TEST_F(KVCacheMemoryConnectorTest, asyncMatch_ReturnNull_WhenGpuReuseLenGEKeysSize) {
    const size_t                           N = 3;
    CacheKeysType                          cache_keys{70001, 70002, 70003};
    std::vector<std::vector<BlockIdxType>> lbs_vec{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}};
    auto                                   res = makeCacheResource(cache_keys, lbs_vec, /*reuse_len=*/N);

    // Even if memory has matches, asyncMatch should skip when gpu reuse covers all keys.
    const size_t mem_size = memoryCacheBlockBytes();
    ASSERT_GT(mem_size, 0u);
    putItemsToCache(cache_keys, mem_size);

    auto meta      = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true, /*enable_remote_cache=*/false, "");
    auto match_ctx = connector_->asyncMatch(res, meta);
    EXPECT_EQ(match_ctx, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, asyncMatch_ReturnNull_WhenNoPrefixMatched) {
    CacheKeysType                          cache_keys{71001, 71002};
    std::vector<std::vector<BlockIdxType>> lbs_vec{{1, 1}, {2, 2}, {3, 3}, {4, 4}};
    auto                                   res = makeCacheResource(cache_keys, lbs_vec, /*reuse_len=*/0);

    // No cache prefill => matched_num == 0
    auto meta      = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true, /*enable_remote_cache=*/false, "");
    auto match_ctx = connector_->asyncMatch(res, meta);
    EXPECT_EQ(match_ctx, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, asyncMatch_ReturnMatchedNum_WhenPrefixMatchedAndStopAtFirstMiss) {
    CacheKeysType                          cache_keys{72001, 72002, 72003};
    std::vector<std::vector<BlockIdxType>> lbs_vec{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}};
    auto                                   res = makeCacheResource(cache_keys, lbs_vec, /*reuse_len=*/0);

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
    CacheKeysType                          cache_keys{73001, 73002, 73003, 73004, 73999};
    std::vector<std::vector<BlockIdxType>> lbs_vec{
        {1, 1, 1, 1, 1},
        {2, 2, 2, 2, 2},
        {3, 3, 3, 3, 3},
        {4, 4, 4, 4, 4},
    };
    auto res = makeCacheResource(cache_keys, lbs_vec, /*reuse_len=*/0);

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
    // Hybrid-attn case: memory may have a "big" key, but the GPU blocks can still be partially invalid.
    // asyncMatch should keep scanning prefix hits, but ONLY count keys that are both:
    // - is_complete == true in memory cache
    // - all GPU blocks are valid (non-null) for that key
    //
    // NOTE: asyncMatch skips the last cache_key, so add a dummy tail key.
    CacheKeysType cache_keys{75001, 75002, 75999};

    // 4 layers, 3 keys:
    // - key 75001: big in memory, but GPU blocks are NOT all valid (layer1 is NULL)
    // - key 75002: big in memory, GPU blocks are all valid => matched_num should become 2
    std::vector<std::vector<BlockIdxType>> lbs_vec{
        /*layer0*/ {1, 1, 1},
        /*layer1*/ {NULL_BLOCK_IDX, 1, 1},
        /*layer2*/ {1, 1, 1},
        /*layer3*/ {1, 1, 1},
    };
    auto res = makeCacheResource(cache_keys, lbs_vec, /*reuse_len=*/0);

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
    CacheKeysType                          cache_keys{76001, 76002, 76003, 76004};
    std::vector<std::vector<BlockIdxType>> lbs_vec{
        {11, 12, 13, 14},
        {21, 22, 23, 24},
        {31, 32, 33, 34},
        {41, 42, 43, 44},
    };
    auto res = makeCacheResource(cache_keys, lbs_vec, /*reuse_len=*/2);

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
    CacheKeysType                          cache_keys{74001, 74002, 74999};
    std::vector<std::vector<BlockIdxType>> lbs_vec{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}};
    auto                                   res = makeCacheResource(cache_keys, lbs_vec, /*reuse_len=*/0);

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
    auto res_empty_keys = makeCacheResource({}, {{1}});
    auto ctx1 =
        connector_->asyncRead(res_empty_keys, nullptr, nullptr, /*start_read_block_index=*/0, /*read_block_num=*/0);
    EXPECT_EQ(ctx1, nullptr);

    // empty layer_block_ids
    // NOTE: asyncRead always skips the last cache_key (cache_keys.size() - 1), so keep size >= 2 here.
    auto res_empty_lbs = makeCacheResource(/*cache_keys=*/{1, 2}, /*per_layer_block_indices=*/{{1, 2}});
    res_empty_lbs->layer_block_ids.clear();
    auto ctx2 =
        connector_->asyncRead(res_empty_lbs, nullptr, nullptr, /*start_read_block_index=*/0, /*read_block_num=*/1);
    EXPECT_EQ(ctx2, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, asyncRead_ReturnNull_WhenReuseLenGEKeys) {
    const size_t                           N = 3;
    CacheKeysType                          cache_keys{10001, 10002, 10003};
    std::vector<std::vector<BlockIdxType>> lbs_vec{{1, 1, 1}, {2, 2, 2}};
    auto                                   res = makeCacheResource(cache_keys, lbs_vec, N);

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
    CacheKeysType                          cache_keys{20001, 20002};
    std::vector<std::vector<BlockIdxType>> lbs_vec{{3, 3}, {4, 4}};
    auto                                   res = makeCacheResource(cache_keys, lbs_vec);

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

TEST_F(KVCacheMemoryConnectorTest, asyncRead_Success_WhenCacheEntryRemovedAfterMatch) {
    // asyncMatch should pin the matched memory blocks so asyncRead can still use them
    // even if another request consumes and removes the cache entries before read starts.
    CacheKeysType cache_keys{21001, 21002, 21003};

    const size_t mem_size = memoryCacheBlockBytes();
    ASSERT_GT(mem_size, 0u);
    auto pool = ensureBlockPool(mem_size);
    ASSERT_NE(pool, nullptr);

    auto block_indices = putItemsToCache(cache_keys, mem_size);
    ASSERT_EQ(block_indices.size(), cache_keys.size());

    std::vector<std::vector<BlockIdxType>> lbs_vec{
        {1, 2, 3},
        {1, 2, 3},
        {1, 2, 3},
        {1, 2, 3},
    };
    auto res  = makeCacheResource(cache_keys, lbs_vec);
    auto meta = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true, /*enable_remote_cache=*/false, "");

    auto match_ctx = connector_->asyncMatch(res, meta);
    ASSERT_NE(match_ctx, nullptr);
    const int start_read_block_index = static_cast<int>(res->reuseBlockNum());
    const int read_block_num         = static_cast<int>(match_ctx->matchedBlockCount()) - start_read_block_index;
    ASSERT_GT(read_block_num, 0);

    for (int i = start_read_block_index; i < start_read_block_index + read_block_num; ++i) {
        auto removed = connector_->block_cache_->remove(cache_keys[i]);
        ASSERT_TRUE(removed.has_value());
        pool->blockCacheFree({removed->block_index});
    }

    auto ctx = connector_->asyncRead(res, meta, match_ctx, start_read_block_index, read_block_num);
    ASSERT_NE(ctx, nullptr);
    ASSERT_TRUE(waitUntilDone(ctx));
    EXPECT_TRUE(ctx->success());
    EXPECT_EQ(res->memoryReuseBlockNum(), static_cast<size_t>(read_block_num));
}

TEST_F(KVCacheMemoryConnectorTest, asyncRead_Success_IncrementsReuseLen_ByMatchedPrefix) {
    // 初始 reuse_len=1, 内存全部命中 => mem_match_len=3，最终 reuse_len=3
    CacheKeysType cache_keys{40001, 40002, 40003};

    const size_t mem_size = memoryCacheBlockBytes();
    ASSERT_GT(mem_size, 0u);

    auto block_indices = putItemsToCache(cache_keys, mem_size);
    ASSERT_EQ(block_indices.size(), cache_keys.size());

    std::vector<std::vector<BlockIdxType>> lbs_vec{
        {1, 2, 3},  // layer0
        {1, 2, 3},  // layer1
        {1, 2, 3},  // layer2
        {1, 2, 3},  // layer3
    };
    auto res = makeCacheResource(cache_keys, lbs_vec, 1);

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

    std::vector<std::vector<BlockIdxType>> lbs_vec{
        {1, 2, 3},
        {1, 2, 3},
        {1, 2, 3},
        {1, 2, 3},
    };
    auto res = makeCacheResource(cache_keys, lbs_vec, /*reuse_len=*/1);

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

    std::vector<std::vector<BlockIdxType>> lbs_vec{
        {1, 2, 3},
        {1, 2, 3},
        {1, 2, 3},
        {1, 2, 3},
    };
    auto res = makeCacheResource(cache_keys, lbs_vec, /*reuse_len=*/1);

    auto meta      = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true, /*enable_remote_cache=*/false, "");
    auto match_ctx = connector_->asyncMatch(res, meta);
    ASSERT_NE(match_ctx, nullptr);
    const int reuse_num = static_cast<int>(res->reuseBlockNum());
    const int read_num  = static_cast<int>(match_ctx->matchedBlockCount()) - reuse_num;
    ASSERT_GT(read_num, 0);

    // asyncMatch captured old block indices in the pinned copy plan; asyncRead consumes that plan.
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

    std::vector<std::vector<BlockIdxType>> lbs_vec{{1, 2}, {3, 4}, {5, 6}, {7, 8}};
    auto                                   res = makeCacheResource(cache_keys, lbs_vec);

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

    std::vector<std::vector<BlockIdxType>> lbs_vec{{1, 2}, {3, 4}, {5, 6}, {7, 8}};
    auto                                   res = makeCacheResource(cache_keys, lbs_vec);

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
    auto res       = makeCacheResource(cache_keys, {{1, 2}, {3, 4}, {5, 6}, {7, 8}});
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
    EXPECT_ANY_THROW((void)connector_->asyncWrite(makeCacheResource(/*cache_keys=*/{1}, /*lbs=*/{{1}}), nullptr));

    // resource is nullptr => RTP_LLM_CHECK triggers exception
    EXPECT_ANY_THROW((void)connector_->asyncWrite(nullptr, meta));

    // empty cache_keys
    auto res_empty_keys = makeCacheResource({}, {{1}});
    auto ctx1           = connector_->asyncWrite(res_empty_keys, meta);
    EXPECT_EQ(ctx1, nullptr);

    // empty layer_block_ids
    auto res_empty_lbs = makeCacheResource(/*cache_keys=*/{1}, /*lbs=*/{{1}});
    res_empty_lbs->layer_block_ids.clear();
    auto ctx2 = connector_->asyncWrite(res_empty_lbs, meta);
    EXPECT_EQ(ctx2, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, asyncWrite_ReturnNull_WhenAllKeysInCache) {
    // 两个 key 均已在内存缓存中
    const int                              gpu_block_idx = 1;
    CacheKeysType                          cache_keys{10, 11};
    std::vector<std::vector<BlockIdxType>> lbs_vec{
        {static_cast<BlockIdxType>(gpu_block_idx), static_cast<BlockIdxType>(gpu_block_idx)},
        {static_cast<BlockIdxType>(gpu_block_idx), static_cast<BlockIdxType>(gpu_block_idx)},
        {static_cast<BlockIdxType>(gpu_block_idx), static_cast<BlockIdxType>(gpu_block_idx)},
        {static_cast<BlockIdxType>(gpu_block_idx), static_cast<BlockIdxType>(gpu_block_idx)},
    };
    auto res = makeCacheResource(cache_keys, lbs_vec);

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
    CacheKeysType                          cache_keys{60001, 60002, 60003};
    std::vector<std::vector<BlockIdxType>> lbs_vec{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
    auto                                   res = makeCacheResource(cache_keys, lbs_vec);

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

    CacheKeysType                          cache_keys{61001, 61002};
    std::vector<std::vector<BlockIdxType>> lbs_vec{{1, 2}, {3, 4}, {5, 6}, {7, 8}};
    auto                                   res = makeCacheResource(cache_keys, lbs_vec);

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
    CacheKeysType cache_keys{100, 101};
    // 4 层，全部 NULL
    std::vector<std::vector<BlockIdxType>> lbs_vec{
        {NULL_BLOCK_IDX, NULL_BLOCK_IDX},
        {NULL_BLOCK_IDX, NULL_BLOCK_IDX},
        {NULL_BLOCK_IDX, NULL_BLOCK_IDX},
        {NULL_BLOCK_IDX, NULL_BLOCK_IDX},
    };
    auto res = makeCacheResource(cache_keys, lbs_vec);

    auto meta = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true, /*enable_remote_cache=*/false, "");
    auto ctx  = connector_->asyncWrite(res, meta);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, asyncWrite_ReturnNull_WhenAllKeysAreSmall_NoNeedWrite) {
    // Hybrid-attn: allow writing small keys for continuity, BUT if there is NO "big" key in the tail,
    // buildCopyPlanForWrite() should return nullptr and asyncWrite should be a no-op (return nullptr).
    CacheKeysType                          cache_keys{81001, 81002, 81003};
    std::vector<std::vector<BlockIdxType>> lbs_vec{
        /*layer0*/ {1, 1, 1},
        /*layer1*/ {NULL_BLOCK_IDX, 1, 1},  // key0 small
        /*layer2*/ {1, NULL_BLOCK_IDX, 1},  // key1 small
        /*layer3*/ {1, 1, NULL_BLOCK_IDX},  // key2 small
    };
    auto res = makeCacheResource(cache_keys, lbs_vec);

    auto pool = connector_->block_pool_;
    ASSERT_NE(pool, nullptr);
    const size_t free_before = pool->freeBlocksNum();

    auto meta = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true);
    auto ctx  = connector_->asyncWrite(res, meta);
    EXPECT_EQ(ctx, nullptr);
    EXPECT_EQ(pool->freeBlocksNum(), free_before);
}

TEST_F(KVCacheMemoryConnectorTest, asyncWrite_DropsTailAfterLastBigKey_InHybridAttn) {
    // Hybrid-attn: write small keys for continuity, but ensure the final written key is big.
    // Keys after the last big key must be dropped.
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
    // - key0 big (all valid)
    // - key1 small (layer1 NULL)
    // - key2 big (all valid)  => last big
    // - key3 small (layer3 NULL) => should be DROPPED (not written, not inserted)
    std::vector<std::vector<BlockIdxType>> lbs_vec{
        /*layer0*/ {1, 1, 1, 1},
        /*layer1*/ {1, NULL_BLOCK_IDX, 1, 1},
        /*layer2*/ {1, 1, 1, 1},
        /*layer3*/ {1, 1, 1, NULL_BLOCK_IDX},
    };
    auto res = makeCacheResource(cache_keys, lbs_vec);

    const size_t cache_before = connector_->block_cache_->size();
    auto         meta         = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true);
    auto         ctx          = connector_->asyncWrite(res, meta);
    ASSERT_NE(ctx, nullptr);
    ASSERT_TRUE(waitUntilDone(ctx));
    EXPECT_TRUE(ctx->success());

    EXPECT_TRUE(connector_->block_cache_->contains(cache_keys[0]));
    EXPECT_TRUE(connector_->block_cache_->contains(cache_keys[1]));
    EXPECT_TRUE(connector_->block_cache_->contains(cache_keys[2]));
    EXPECT_FALSE(connector_->block_cache_->contains(cache_keys[3]));

    // Written count should be >= 3 (exact +3 if cache was empty and no evictions)
    EXPECT_GE(connector_->block_cache_->size(), cache_before + 3);

    connector_->broadcast_manager_.reset();
    for (auto& s : servers) {
        s->shutdown();
    }
    servers.clear();
    addrs.clear();
}

TEST_F(KVCacheMemoryConnectorTest, asyncWrite_Success_AddsToBlockCache_AndKeepsMemBlocks) {
    // 默认 RPC 服务均返回 OK + mem success
    const size_t                           N = 2;
    CacheKeysType                          cache_keys{200, 201};
    std::vector<std::vector<BlockIdxType>> lbs_vec{
        {2, 3},
        {4, 5},
        {6, 7},
        {8, 9},
    };
    auto res = makeCacheResource(cache_keys, lbs_vec);

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

    CacheKeysType                          cache_keys{301, 302};
    std::vector<std::vector<BlockIdxType>> lbs_vec{
        {1, 2},
        {3, 4},
        {5, 6},
        {7, 8},
    };
    auto res = makeCacheResource(cache_keys, lbs_vec);

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

    const int                              layer0 = 0;
    CacheKeysType                          cache_keys{71001, 71002, 71003};
    std::vector<std::vector<BlockIdxType>> lbs_vec{{1, 2, 3}};
    auto                                   res = makeCacheResource(cache_keys, lbs_vec);

    // Pre-insert one key so cpu_matched_num < cache_keys.size() and it reaches the thread-pool-full check.
    const auto   bufs  = allocator_->convertIndexToBuffer(layer0, /*block_id=*/1);
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
    const auto   gpu_bufs      = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
    const size_t total         = sumBlockInfosBytes(gpu_bufs);
    ASSERT_GT(total, 0u);

    KVCacheMemoryConnector::CopyInfoPerKey info;
    info.cache_key = 1;
    info.mem_block = static_cast<BlockIdxType>(1);
    info.gpu_blocks.assign(static_cast<size_t>(cache_config_.layer_all_num), NULL_BLOCK_IDX);
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
    const auto   gpu_bufs      = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
    const size_t total         = sumBlockInfosBytes(gpu_bufs);
    ASSERT_GT(total, 0u);

    KVCacheMemoryConnector::CopyInfoPerKey info;
    info.cache_key = 2;
    info.mem_block = static_cast<BlockIdxType>(1);
    info.gpu_blocks.assign(static_cast<size_t>(cache_config_.layer_all_num), NULL_BLOCK_IDX);
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
    const auto   gpu_bufs      = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
    const size_t total         = sumBlockInfosBytes(gpu_bufs);
    ASSERT_GT(total, 0u);

    KVCacheMemoryConnector::CopyInfoPerKey info;
    info.cache_key = 3;
    info.mem_block = static_cast<BlockIdxType>(1);
    info.gpu_blocks.assign(static_cast<size_t>(cache_config_.layer_all_num), NULL_BLOCK_IDX);
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

TEST_F(KVCacheMemoryConnectorTest, copyCache_ReturnFalse_CountMismatch) {
    MemoryOperationRequestPB req;
    auto*                    item = req.add_copy_items();
    item->add_gpu_blocks(1);
    // Intentionally do not set mem_block to trigger mismatch
    req.set_copy_direction(MemoryOperationRequestPB::H2D);

    MemoryOperationResponsePB resp;
    EXPECT_THROW((void)connector_->copyCache(req, resp), rtp_llm::RTPException);
}

TEST_F(KVCacheMemoryConnectorTest, copyCache_ReturnFalse_InvalidMemBlock) {
    const int    layer_id      = 0;
    const int    gpu_block_idx = 1;
    const auto   gpu_bufs      = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
    const size_t total         = sumBlockInfosBytes(gpu_bufs);
    ASSERT_GT(total, 0u);

    MemoryOperationRequestPB req;
    auto*                    item = req.add_copy_items();
    item->add_gpu_blocks(gpu_block_idx);
    // invalid mem_block index for block_pool_
    item->set_mem_block(NULL_BLOCK_IDX);
    req.set_copy_direction(MemoryOperationRequestPB::H2D);

    MemoryOperationResponsePB resp;
    EXPECT_THROW(connector_->copyCache(req, resp), rtp_llm::RTPException);
}

TEST_F(KVCacheMemoryConnectorTest, copyCache_ReturnFalse_InvalidLayerId_BuildCopyPlanFailed) {
    const int    valid_layer   = 0;
    const int    gpu_block_idx = 1;
    const auto   gpu_bufs      = allocator_->convertIndexToBuffer(valid_layer, gpu_block_idx);
    const size_t total         = sumBlockInfosBytes(gpu_bufs);
    ASSERT_GT(total, 0u);

    auto pool = ensureBlockPool(total);
    ASSERT_NE(pool, nullptr);
    auto mem_blocks = pool->malloc(1);
    ASSERT_EQ(mem_blocks.size(), 1u);
    const BlockIdxType mem_block_index = static_cast<BlockIdxType>(mem_blocks[0]);

    MemoryOperationRequestPB req;
    auto*                    item = req.add_copy_items();
    // gpu_blocks size > layer_num => invalid "layer index"
    for (int l = 0; l < cache_config_.layer_num + 1; ++l) {
        item->add_gpu_blocks(gpu_block_idx);
    }
    item->set_mem_block(mem_block_index);
    req.set_copy_direction(MemoryOperationRequestPB::H2D);

    MemoryOperationResponsePB resp;
    EXPECT_ANY_THROW((void)connector_->copyCache(req, resp));
}

TEST_F(KVCacheMemoryConnectorTest, copyCache_ReturnTrue_H2D_SingleLayer) {
    const int    layer_id      = 0;
    const int    gpu_block_idx = 2;
    const auto   gpu_bufs      = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
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

    MemoryOperationRequestPB req;
    auto*                    item = req.add_copy_items();
    for (int l = 0; l < cache_config_.layer_num; ++l) {
        item->add_gpu_blocks(l == layer_id ? gpu_block_idx : NULL_BLOCK_IDX);
    }
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

    auto mla_spec                = std::make_shared<rtp_llm::MLAKVCacheSpec>();
    mla_spec->type               = rtp_llm::KVCacheSpecType::MultiHeadLatentAttention;
    mla_spec->layer_num          = static_cast<uint32_t>(kLayerNum);
    mla_spec->local_head_num_kv  = 1;
    mla_spec->seq_size_per_block = kSeqPerBlock;
    mla_spec->kv_lora_rank       = 512;
    mla_spec->rope_head_dim      = 64;
    mla_spec->dtype              = rtp_llm::DataType::TYPE_FP8_E4M3;

    cache_config_.layer_num             = static_cast<uint32_t>(kLayerNum);
    cache_config_.layer_all_num         = static_cast<uint32_t>(kLayerNum);
    cache_config_.block_num             = static_cast<uint32_t>(kBlockNum);
    cache_config_.seq_size_per_block    = kSeqPerBlock;
    cache_config_.use_mla               = true;
    cache_config_.is_sparse             = false;
    cache_config_.dtype                 = mla_spec->dtype;
    cache_config_.cache_specs           = {mla_spec};
    cache_config_.kv_block_stride_bytes = kKvBytesPerTok * kSeqPerBlock;
    cache_config_.kv_scale_stride_bytes = kScaleBytesPerTok * kSeqPerBlock;
    cache_config_.kv_block_size_bytes   = static_cast<size_t>(kLayerNum) * cache_config_.kv_block_stride_bytes;
    cache_config_.kv_scale_size_bytes   = static_cast<size_t>(kLayerNum) * cache_config_.kv_scale_stride_bytes;
    cache_config_.block_size_bytes      = cache_config_.kv_block_size_bytes + cache_config_.kv_scale_size_bytes;
    const size_t kPerLayerStrideBytes   = cache_config_.kv_block_stride_bytes + cache_config_.kv_scale_stride_bytes;
    cache_config_.layer_to_block_stride_bytes.assign(static_cast<size_t>(kLayerNum),
                                                     static_cast<int>(kPerLayerStrideBytes));
    std::vector<int> layer_ids(kLayerNum);
    for (int i = 0; i < kLayerNum; ++i) {
        layer_ids[i] = i;
    }
    cache_config_.layer_ids.clear();
    cache_config_.global_layer_ids.clear();
    cache_config_.layer_ids.push_back(layer_ids);
    cache_config_.global_layer_ids.push_back(layer_ids);

    ASSERT_EQ(mla_spec->block_size_bytes(), cache_config_.kv_block_stride_bytes);

    const size_t merged_one_key = memoryCacheBlockBytes(cache_config_);
    ASSERT_EQ(merged_one_key, static_cast<size_t>(kLayerNum) * kPerLayerStrideBytes);
    const int pool_mb =
        static_cast<int>((merged_one_key * static_cast<size_t>(kCopyBlockCount) + (1024ULL * 1024 - 1)) / (1024 * 1024))
        + 256;
    kv_cache_config_.memory_cache_size_mb = std::max(pool_mb, 512);

    allocator_ = std::make_shared<SingleTypeKVCacheAllocator>(cache_config_, AllocationType::DEVICE);
    ASSERT_TRUE(allocator_->init());
    connector_ = std::make_shared<KVCacheMemoryConnector>(cache_config_, kv_cache_config_, allocator_, server_addrs_);
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
        for (int l = 0; l < kLayerNum; ++l) {
            item->add_gpu_blocks(kGpuBlockBase + i);
        }
        item->set_mem_block(mem_block_indices[static_cast<size_t>(i)]);
    }
    MemoryOperationResponsePB resp;
    ASSERT_TRUE(connector_->copyCache(req, resp));
    EXPECT_TRUE(resp.success());

    for (int i = 0; i < kCopyBlockCount; ++i) {
        const char tag = static_cast<char>(0x21 + (i % 94));
        for (int l = 0; l < kLayerNum; ++l) {
            const auto gpu_bufs = allocator_->convertIndexToBuffer(l, kGpuBlockBase + i);
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
    const auto   gpu_bufs      = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
    const size_t total         = sumBlockInfosBytes(gpu_bufs);
    ASSERT_GT(total, 0u);

    // 给gpu_buf填充数据
    setBlockInfosContent(gpu_bufs, 'a');

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

    MemoryOperationRequestPB req;
    auto*                    item = req.add_copy_items();
    for (int l = 0; l < cache_config_.layer_num; ++l) {
        item->add_gpu_blocks(l == layer_id ? gpu_block_idx : NULL_BLOCK_IDX);
    }
    item->set_mem_block(mem_block_index);
    item->set_is_complete(true);
    item->set_backing_type(MemoryOperationRequestPB::MEMORY);
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
        const auto gpu_bufs = allocator_->convertIndexToBuffer(lb.layer_id, lb.block_id);
        ASSERT_GT(sumBlockInfosBytes(gpu_bufs), 0u);
        setBlockInfosContent(gpu_bufs, static_cast<char>('k' + lb.layer_id));
    }

    // Allocate one memory block for the merged layout (one cache-key across all layers).
    size_t total_bytes = 0;
    for (int layer = 0; layer < cache_config_.layer_all_num; ++layer) {
        total_bytes += static_cast<size_t>(cache_config_.layer_to_block_stride_bytes[static_cast<size_t>(layer)]);
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
    const size_t              layer_num = static_cast<size_t>(cache_config_.layer_all_num);
    std::vector<BlockIdxType> layer_to_block(layer_num, NULL_BLOCK_IDX);
    for (const auto& lb : gpu_layer_blocks) {
        layer_to_block[static_cast<size_t>(lb.layer_id)] = lb.block_id;
    }
    size_t byte_off = 0;
    for (size_t layer = 0; layer < layer_num; ++layer) {
        const size_t layer_stride = static_cast<size_t>(cache_config_.layer_to_block_stride_bytes[layer]);
        const auto   block_id     = layer_to_block[layer];
        if (isNullBlockIdx(block_id)) {
            byte_off += layer_stride;
            continue;
        }
        const auto gpu_bufs = allocator_->convertIndexToBuffer(static_cast<int>(layer), block_id);
        const auto bytes    = sumBlockInfosBytes(gpu_bufs);
        ASSERT_GT(bytes, 0u);
        ASSERT_LE(bytes, layer_stride);
        verifyBlockBytesEq(mem_buffer, byte_off, bytes, static_cast<char>('k' + static_cast<int>(layer)));
        byte_off += layer_stride;
    }
}

// ============================== Dual-pool tests ==============================

class KVCacheMemoryConnectorDualPoolTest: public ::testing::Test {
protected:
    void SetUp() override {
        createDevice();
        startRpcServer(4);
    }

    void TearDown() override {}

    CacheConfig
    createHybridCacheConfig(int layer_num = 4, int block_num = 10, int seq_size_per_block = 8, int linear_step = 4) {
        constexpr int kTestMemoryCacheSizeMb      = 64;
        constexpr int kTestMemoryCacheSyncTimeout = 1000;

        CacheConfig config;
        config.layer_num                              = layer_num;
        config.layer_all_num                          = layer_num;
        config.block_num                              = block_num;
        config.seq_size_per_block                     = seq_size_per_block;
        config.linear_step                            = linear_step;
        config.group_layer_num                        = layer_num;
        config.full_group_num                         = 1;
        kv_cache_config_.memory_cache_size_mb         = kTestMemoryCacheSizeMb;
        kv_cache_config_.memory_cache_sync_timeout_ms = kTestMemoryCacheSyncTimeout;

        auto full_spec                = std::make_shared<MHAKVCacheSpec>();
        full_spec->layer_num          = layer_num;
        full_spec->local_head_num_kv  = 4;
        full_spec->size_per_head      = 64;
        full_spec->seq_size_per_block = seq_size_per_block;
        full_spec->dtype              = rtp_llm::DataType::TYPE_FP16;

        auto swa_spec                = std::make_shared<MHAKVCacheSpec>();
        swa_spec->layer_num          = layer_num;
        swa_spec->local_head_num_kv  = 4;
        swa_spec->size_per_head      = 64;
        swa_spec->seq_size_per_block = seq_size_per_block;
        swa_spec->dtype              = rtp_llm::DataType::TYPE_FP16;

        config.cache_specs        = {full_spec, swa_spec};
        config.group_types        = {CacheGroupType::FULL, CacheGroupType::SWA};
        config.group_region_names = {KVCacheRegionName::DEFAULT, KVCacheRegionName::SWA_KV};

        const size_t full_stride           = full_spec->block_size_bytes();
        const size_t swa_stride            = swa_spec->block_size_bytes();
        config.group_kv_block_stride_bytes = {full_stride, swa_stride};
        config.group_kv_scale_stride_bytes = {0, 0};

        config.dtype                 = full_spec->dtype;
        config.kv_block_stride_bytes = std::max(full_stride, swa_stride);
        config.kv_scale_stride_bytes = 0;
        config.kv_block_size_bytes   = static_cast<size_t>(layer_num) * full_stride;
        config.kv_scale_size_bytes   = 0;
        config.swa_block_size_bytes  = static_cast<size_t>(layer_num) * swa_stride;
        config.block_size_bytes      = config.kv_block_size_bytes;

        std::vector<int> full_layer_ids(layer_num);
        std::vector<int> swa_layer_ids(layer_num);
        for (int i = 0; i < layer_num; ++i) {
            full_layer_ids[i] = i;
            swa_layer_ids[i]  = i;
        }
        config.layer_ids        = {full_layer_ids, swa_layer_ids};
        config.global_layer_ids = {full_layer_ids, swa_layer_ids};

        config.layer_to_group_id.assign(layer_num, 0);
        config.layer_to_group_ids.resize(layer_num);
        const size_t region_count = static_cast<size_t>(KVCacheRegionName::REGION_COUNT);
        config.layer_region_to_group_id.assign(layer_num, std::vector<int>(region_count, -1));
        for (int i = 0; i < layer_num; ++i) {
            config.layer_to_group_ids[i]                                                        = {0, 1};
            config.layer_region_to_group_id[i][static_cast<size_t>(KVCacheRegionName::DEFAULT)] = 0;
            config.layer_region_to_group_id[i][static_cast<size_t>(KVCacheRegionName::SWA_KV)]  = 1;
        }
        config.layer_group_types.assign(layer_num, CacheGroupType::FULL);
        config.layer_to_block_stride_bytes.assign(layer_num, static_cast<int>(full_stride));

        config.use_independent_block_pools = true;

        return config;
    }

    std::shared_ptr<KVCacheMemoryConnector> createConnector(const CacheConfig& cfg) {
        auto conn = std::make_shared<KVCacheMemoryConnector>(cfg, kv_cache_config_, allocator_, server_addrs_);
        EXPECT_TRUE(conn->init());
        return conn;
    }

    KVCacheConfig makeDiskKvConfig(const std::vector<std::string>& paths, int64_t disk_size_mb = 1) const {
        auto kv_cfg                              = kv_cache_config_;
        kv_cfg.enable_memory_cache               = true;
        kv_cfg.enable_memory_cache_disk          = true;
        kv_cfg.memory_cache_disk_paths           = joinPaths(paths);
        kv_cfg.memory_cache_disk_size_mb         = disk_size_mb;
        kv_cfg.memory_cache_disk_buffered_io     = true;
        kv_cfg.memory_cache_disk_sync_timeout_ms = 1000;
        kv_cfg.enable_tiered_memory_cache        = false;
        return kv_cfg;
    }

    std::shared_ptr<KVCacheMemoryConnector> createConnectorWithKvConfig(const CacheConfig&   cfg,
                                                                        const KVCacheConfig& kv_cfg) {
        auto conn = std::make_shared<KVCacheMemoryConnector>(cfg, kv_cfg, allocator_, server_addrs_);
        EXPECT_TRUE(conn->init());
        return conn;
    }

    std::shared_ptr<KVCacheResource> makeHybridResource(const CacheConfig&                            cfg,
                                                        const CacheKeysType&                          cache_keys,
                                                        const std::vector<std::vector<BlockIdxType>>& full_blocks,
                                                        const std::vector<std::vector<BlockIdxType>>& swa_blocks,
                                                        size_t reuse_len = 0) const {
        auto         res       = std::make_shared<KVCacheResource>();
        const size_t layer_num = static_cast<size_t>(cfg.layer_all_num);

        res->initGroups(
            /*group_num=*/2,
            /*layer_num=*/cfg.layer_all_num,
            cfg.layer_to_group_id,
            /*kernel_blocks_per_kv_block=*/1,
            cfg.group_types,
            cfg.layer_region_to_group_id);

        res->resizeBlocks(static_cast<int>(cache_keys.size()), NULL_BLOCK_IDX);

        for (size_t l = 0; l < layer_num; ++l) {
            if (l < full_blocks.size()) {
                BlockIndicesType padded(cache_keys.size(), NULL_BLOCK_IDX);
                for (size_t k = 0; k < std::min(cache_keys.size(), full_blocks[l].size()); ++k) {
                    padded[k] = full_blocks[l][k];
                }
                res->mutableBlockIds(static_cast<int>(l), KVCacheRegionName::DEFAULT).assign(padded);
            }
            if (l < swa_blocks.size()) {
                BlockIndicesType padded(cache_keys.size(), NULL_BLOCK_IDX);
                for (size_t k = 0; k < std::min(cache_keys.size(), swa_blocks[l].size()); ++k) {
                    padded[k] = swa_blocks[l][k];
                }
                res->mutableBlockIds(static_cast<int>(l), KVCacheRegionName::SWA_KV).assign(padded);
            }
        }

        res->cacheKeys() = cache_keys;
        res->setDeviceReuseBlockNum(reuse_len);
        res->setLastBlockAligned(true);
        return res;
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

    KVCacheConfig                               kv_cache_config_;
    std::shared_ptr<KVCacheAllocator>           allocator_;
    std::vector<std::unique_ptr<TestRpcServer>> servers_;
    std::vector<std::string>                    server_addrs_;

private:
    void createDevice() const {
        initRuntime(/*device_id=*/0,
                    /*trace_memory=*/false,
                    /*enable_comm_overlap=*/false,
                    MlaOpsType::AUTO);
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
};

TEST_F(KVCacheMemoryConnectorDualPoolTest, Init_CreatesDualPools) {
    auto cfg   = createHybridCacheConfig();
    allocator_ = std::make_shared<HybridTypeKVCacheAllocator>(cfg, AllocationType::DEVICE);
    ASSERT_TRUE(allocator_->init());
    auto conn = createConnector(cfg);

    EXPECT_TRUE(conn->isDualPool());
    EXPECT_NE(conn->complete_pool_, nullptr);
    EXPECT_NE(conn->incomplete_pool_, nullptr);
    EXPECT_EQ(conn->block_pool_, nullptr);
    EXPECT_NE(conn->block_cache_, nullptr);
    EXPECT_GT(conn->complete_block_size_, 0u);
    EXPECT_GT(conn->incomplete_block_size_, 0u);
    EXPECT_GT(conn->complete_block_size_, conn->incomplete_block_size_);
}

TEST_F(KVCacheMemoryConnectorDualPoolTest, Init_PureFullUsesSinglePool) {
    // Pure FULL config: no typed slots, should use single pool
    CacheConfig config;
    config.layer_num                              = 4;
    config.layer_all_num                          = 4;
    config.block_num                              = 10;
    config.seq_size_per_block                     = 8;
    kv_cache_config_.memory_cache_size_mb         = 64;
    kv_cache_config_.memory_cache_sync_timeout_ms = 1000;

    auto spec                    = std::make_shared<MHAKVCacheSpec>();
    spec->layer_num              = 4;
    spec->local_head_num_kv      = 8;
    spec->size_per_head          = 128;
    spec->seq_size_per_block     = 8;
    spec->dtype                  = rtp_llm::DataType::TYPE_FP16;
    config.cache_specs           = {spec};
    config.dtype                 = spec->dtype;
    config.kv_block_stride_bytes = spec->block_size_bytes();
    config.kv_scale_stride_bytes = spec->scale_block_size_bytes();
    config.kv_block_size_bytes   = 4UL * config.kv_block_stride_bytes;
    config.kv_scale_size_bytes   = 4UL * config.kv_scale_stride_bytes;
    config.block_size_bytes      = config.kv_block_size_bytes + config.kv_scale_size_bytes;
    const size_t per_layer       = config.kv_block_stride_bytes + config.kv_scale_stride_bytes;
    config.layer_to_block_stride_bytes.assign(4, static_cast<int>(per_layer));
    std::vector<int> ids = {0, 1, 2, 3};
    config.layer_ids.push_back(ids);
    config.global_layer_ids.push_back(ids);

    allocator_ = std::make_shared<SingleTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator_->init());
    auto conn = createConnector(config);

    EXPECT_FALSE(conn->isDualPool());
    EXPECT_NE(conn->block_pool_, nullptr);
    EXPECT_NE(conn->block_cache_, nullptr);
    EXPECT_EQ(conn->complete_pool_, nullptr);
    EXPECT_EQ(conn->incomplete_pool_, nullptr);
}

TEST_F(KVCacheMemoryConnectorDualPoolTest, AsyncMatch_AdvancesOnlyOnCompleteHit) {
    auto cfg = createHybridCacheConfig(/*layer_num=*/2, /*block_num=*/10, /*seq_size_per_block=*/8, /*linear_step=*/4);
    allocator_ = std::make_shared<HybridTypeKVCacheAllocator>(cfg, AllocationType::DEVICE);
    ASSERT_TRUE(allocator_->init());
    auto conn = createConnector(cfg);
    ASSERT_TRUE(conn->isDualPool());

    // 4 cache keys (3 real + 1 tail dummy that won't be matched)
    CacheKeysType cache_keys{90001, 90002, 90003, 90999};
    // FULL blocks: all valid (non-null) for both layers
    std::vector<std::vector<BlockIdxType>> full_blocks{{1, 1, 1, 1}, {2, 2, 2, 2}};
    // SWA blocks: key0 NULL (incomplete), key1 valid (complete), key2 NULL (incomplete)
    std::vector<std::vector<BlockIdxType>> swa_blocks{{NULL_BLOCK_IDX, 1, NULL_BLOCK_IDX, 1},
                                                      {NULL_BLOCK_IDX, 2, NULL_BLOCK_IDX, 2}};
    auto                                   res = makeHybridResource(cfg, cache_keys, full_blocks, swa_blocks);

    // Populate caches directly
    {
        // key0: incomplete (SWA NULL) → incomplete cache
        auto inc_blks = conn->incomplete_pool_->malloc(2);
        ASSERT_EQ(inc_blks.size(), 2u);
        MemoryDiskBlockCache::CacheItem item0;
        item0.cache_key    = cache_keys[0];
        item0.backing_type = CacheBackingType::MEMORY;
        item0.block_index  = static_cast<BlockIdxType>(inc_blks[0]);
        item0.is_complete  = false;
        conn->block_cache_->putCommitted(item0);
        conn->incomplete_pool_->blockCacheReference({static_cast<BlockIdxType>(inc_blks[0])});
        conn->incomplete_pool_->requestFree({inc_blks[0]});

        // key2: incomplete → incomplete cache
        MemoryDiskBlockCache::CacheItem item2;
        item2.cache_key    = cache_keys[2];
        item2.backing_type = CacheBackingType::MEMORY;
        item2.block_index  = static_cast<BlockIdxType>(inc_blks[1]);
        item2.is_complete  = false;
        conn->block_cache_->putCommitted(item2);
        conn->incomplete_pool_->blockCacheReference({static_cast<BlockIdxType>(inc_blks[1])});
        conn->incomplete_pool_->requestFree({inc_blks[1]});

        // key1: complete → complete cache
        auto comp_blks = conn->complete_pool_->malloc(1);
        ASSERT_EQ(comp_blks.size(), 1u);
        MemoryDiskBlockCache::CacheItem item1;
        item1.cache_key    = cache_keys[1];
        item1.backing_type = CacheBackingType::MEMORY;
        item1.block_index  = static_cast<BlockIdxType>(comp_blks[0]);
        item1.is_complete  = true;
        conn->block_cache_->putCommitted(item1);
        conn->complete_pool_->blockCacheReference({static_cast<BlockIdxType>(comp_blks[0])});
        conn->complete_pool_->requestFree({comp_blks[0]});
    }

    auto meta      = std::make_shared<TestReadMeta>(true);
    auto match_ctx = conn->asyncMatch(res, meta);
    ASSERT_NE(match_ctx, nullptr);
    // key0: incomplete hit → scan continues, matched_num stays 0
    // key1: complete hit + all GPU valid → matched_num = 2
    // key2: incomplete hit → scan continues, matched_num stays 2
    // Result: matched_num = 2
    EXPECT_EQ(match_ctx->matchedBlockCount(), 2u);
}

TEST_F(KVCacheMemoryConnectorDualPoolTest, AsyncMatch_StopsOnDoubleMiss) {
    auto cfg = createHybridCacheConfig(/*layer_num=*/2, /*block_num=*/10, /*seq_size_per_block=*/8, /*linear_step=*/4);
    allocator_ = std::make_shared<HybridTypeKVCacheAllocator>(cfg, AllocationType::DEVICE);
    ASSERT_TRUE(allocator_->init());
    auto conn = createConnector(cfg);
    ASSERT_TRUE(conn->isDualPool());

    CacheKeysType                          cache_keys{92001, 92002, 92003, 92999};
    std::vector<std::vector<BlockIdxType>> full_blocks{{1, 1, 1, 1}, {2, 2, 2, 2}};
    std::vector<std::vector<BlockIdxType>> swa_blocks{{1, 1, 1, 1}, {2, 2, 2, 2}};
    auto                                   res = makeHybridResource(cfg, cache_keys, full_blocks, swa_blocks);

    // Put key0 as complete, skip key1 (gap), key2 as complete
    auto blks = conn->complete_pool_->malloc(2);
    ASSERT_EQ(blks.size(), 2u);
    MemoryDiskBlockCache::CacheItem item0;
    item0.cache_key    = cache_keys[0];
    item0.backing_type = CacheBackingType::MEMORY;
    item0.block_index  = static_cast<BlockIdxType>(blks[0]);
    item0.is_complete  = true;
    conn->block_cache_->putCommitted(item0);
    conn->complete_pool_->blockCacheReference({static_cast<BlockIdxType>(blks[0])});
    conn->complete_pool_->requestFree({blks[0]});

    MemoryDiskBlockCache::CacheItem item2;
    item2.cache_key    = cache_keys[2];
    item2.backing_type = CacheBackingType::MEMORY;
    item2.block_index  = static_cast<BlockIdxType>(blks[1]);
    item2.is_complete  = true;
    conn->block_cache_->putCommitted(item2);
    conn->complete_pool_->blockCacheReference({static_cast<BlockIdxType>(blks[1])});
    conn->complete_pool_->requestFree({blks[1]});

    auto meta      = std::make_shared<TestReadMeta>(true);
    auto match_ctx = conn->asyncMatch(res, meta);
    ASSERT_NE(match_ctx, nullptr);
    // key0 hit → matched=1, key1 miss in both caches → break
    EXPECT_EQ(match_ctx->matchedBlockCount(), 1u);
}

TEST_F(KVCacheMemoryConnectorDualPoolTest, PoolSizing_JointCalculation) {
    auto cfg = createHybridCacheConfig(/*layer_num=*/2, /*block_num=*/10, /*seq_size_per_block=*/8, /*linear_step=*/4);
    allocator_ = std::make_shared<HybridTypeKVCacheAllocator>(cfg, AllocationType::DEVICE);
    ASSERT_TRUE(allocator_->init());
    auto conn = createConnector(cfg);
    ASSERT_TRUE(conn->isDualPool());

    // With linear_step=4: incomplete_num = complete_num * 3.
    // totalBlocksNum() reports allocatable blocks and excludes reserved block 0.
    const auto complete_total   = conn->complete_pool_->totalBlocksNum();
    const auto incomplete_total = conn->incomplete_pool_->totalBlocksNum();
    EXPECT_GT(complete_total, 0u);
    EXPECT_GT(incomplete_total, 0u);
    EXPECT_EQ(incomplete_total, (complete_total + 1) * 3 - 1);
}

TEST_F(KVCacheMemoryConnectorDualPoolTest, InitDiskDualPools_MirrorsMemoryPoolRatioOnSameMount) {
    DiskTempDir disk0;
    auto cfg = createHybridCacheConfig(/*layer_num=*/2, /*block_num=*/10, /*seq_size_per_block=*/8, /*linear_step=*/4);
    allocator_ = std::make_shared<HybridTypeKVCacheAllocator>(cfg, AllocationType::DEVICE);
    ASSERT_TRUE(allocator_->init());

    auto kv_cfg = makeDiskKvConfig({disk0.path()}, /*disk_size_mb=*/1);
    auto conn   = createConnectorWithKvConfig(cfg, kv_cfg);
    ASSERT_TRUE(conn->isDualPool());
    ASSERT_NE(conn->complete_disk_pool_, nullptr);
    ASSERT_NE(conn->incomplete_disk_pool_, nullptr);

    EXPECT_NE(conn->complete_disk_pool_->filePath().find(disk0.path()), std::string::npos);
    EXPECT_NE(conn->incomplete_disk_pool_->filePath().find(disk0.path()), std::string::npos);
    EXPECT_NE(conn->complete_disk_pool_->filePath(), conn->incomplete_disk_pool_->filePath());
    EXPECT_NE(conn->complete_disk_pool_->filePath().find("complete.kv"), std::string::npos);
    EXPECT_NE(conn->incomplete_disk_pool_->filePath().find("incomplete.kv"), std::string::npos);

    const auto complete_slots   = conn->complete_disk_pool_->totalSlots();
    const auto incomplete_slots = conn->incomplete_disk_pool_->totalSlots();
    EXPECT_GT(complete_slots, 0u);
    EXPECT_EQ(incomplete_slots, complete_slots * 3);
}

TEST_F(KVCacheMemoryConnectorDualPoolTest, AllocateOneBackingUsesMatchingDiskPoolWhenMemoryKindIsFull) {
    DiskTempDir disk0;
    auto cfg = createHybridCacheConfig(/*layer_num=*/2, /*block_num=*/10, /*seq_size_per_block=*/8, /*linear_step=*/4);
    allocator_ = std::make_shared<HybridTypeKVCacheAllocator>(cfg, AllocationType::DEVICE);
    ASSERT_TRUE(allocator_->init());

    auto kv_cfg = makeDiskKvConfig({disk0.path()}, /*disk_size_mb=*/1);
    auto conn   = createConnectorWithKvConfig(cfg, kv_cfg);
    ASSERT_TRUE(conn->isDualPool());
    ASSERT_NE(conn->complete_pool_, nullptr);
    ASSERT_NE(conn->incomplete_pool_, nullptr);
    ASSERT_NE(conn->complete_disk_pool_, nullptr);
    ASSERT_NE(conn->incomplete_disk_pool_, nullptr);

    const auto held_complete = conn->complete_pool_->malloc(static_cast<int>(conn->complete_pool_->freeBlocksNum()));
    ASSERT_GT(held_complete.size(), 0u);
    KVCacheMemoryConnector::CopyInfoPerKey complete_info;
    complete_info.is_complete = true;
    ASSERT_TRUE(conn->allocateOneBacking(complete_info));
    EXPECT_EQ(complete_info.backing_type, CacheBackingType::DISK);
    EXPECT_GE(complete_info.disk_slot, 0);
    EXPECT_EQ(conn->complete_disk_pool_->freeSlots(), conn->complete_disk_pool_->totalSlots() - 1);
    EXPECT_EQ(conn->incomplete_disk_pool_->freeSlots(), conn->incomplete_disk_pool_->totalSlots());
    conn->releaseRequestBacking(complete_info);
    conn->complete_pool_->requestFree(held_complete);

    const auto held_incomplete =
        conn->incomplete_pool_->malloc(static_cast<int>(conn->incomplete_pool_->freeBlocksNum()));
    ASSERT_GT(held_incomplete.size(), 0u);
    KVCacheMemoryConnector::CopyInfoPerKey incomplete_info;
    incomplete_info.is_complete = false;
    ASSERT_TRUE(conn->allocateOneBacking(incomplete_info));
    EXPECT_EQ(incomplete_info.backing_type, CacheBackingType::DISK);
    EXPECT_GE(incomplete_info.disk_slot, 0);
    EXPECT_EQ(conn->complete_disk_pool_->freeSlots(), conn->complete_disk_pool_->totalSlots());
    EXPECT_EQ(conn->incomplete_disk_pool_->freeSlots(), conn->incomplete_disk_pool_->totalSlots() - 1);
    conn->releaseRequestBacking(incomplete_info);
    conn->incomplete_pool_->requestFree(held_incomplete);
}

TEST_F(KVCacheMemoryConnectorDualPoolTest, BuildCopyPlanForWrite_SkipsIncompleteWhenIncompletePoolDisabled) {
    auto cfg = createHybridCacheConfig(/*layer_num=*/2, /*block_num=*/10, /*seq_size_per_block=*/8, /*linear_step=*/1);
    allocator_ = std::make_shared<HybridTypeKVCacheAllocator>(cfg, AllocationType::DEVICE);
    ASSERT_TRUE(allocator_->init());
    auto conn = createConnector(cfg);
    ASSERT_TRUE(conn->isDualPool());
    ASSERT_NE(conn->complete_pool_, nullptr);
    ASSERT_EQ(conn->incomplete_pool_, nullptr);

    CacheKeysType                          cache_keys{93001, 93002, 93003};
    std::vector<std::vector<BlockIdxType>> full_blocks{{1, 1, 1}, {2, 2, 2}};
    std::vector<std::vector<BlockIdxType>> swa_blocks{{NULL_BLOCK_IDX, 1, NULL_BLOCK_IDX},
                                                      {NULL_BLOCK_IDX, 2, NULL_BLOCK_IDX}};
    auto                                   res   = makeHybridResource(cfg, cache_keys, full_blocks, swa_blocks);
    auto                                   slots = conn->layerRegionSlots();

    bool no_need_write = true;
    auto plan          = conn->buildCopyPlanForWrite(
        res->cacheKeys(), res->layerAttnBlocks(), slots, /*start_index=*/0, /*write_num=*/3, no_need_write);

    ASSERT_NE(plan, nullptr);
    EXPECT_FALSE(no_need_write);
    ASSERT_EQ(plan->copy_infos.size(), 1u);
    EXPECT_EQ(plan->copy_infos[0].cache_key, cache_keys[1]);
    EXPECT_TRUE(plan->copy_infos[0].is_complete);
}

TEST_F(KVCacheMemoryConnectorDualPoolTest, CacheKeys_MergesBothCaches) {
    auto cfg   = createHybridCacheConfig(/*layer_num=*/2);
    allocator_ = std::make_shared<HybridTypeKVCacheAllocator>(cfg, AllocationType::DEVICE);
    ASSERT_TRUE(allocator_->init());
    auto conn = createConnector(cfg);
    ASSERT_TRUE(conn->isDualPool());

    // Put complete/incomplete items into the unified backing cache.
    auto comp_blks = conn->complete_pool_->malloc(1);
    ASSERT_EQ(comp_blks.size(), 1u);
    MemoryDiskBlockCache::CacheItem item1;
    item1.cache_key    = 100;
    item1.backing_type = CacheBackingType::MEMORY;
    item1.block_index  = static_cast<BlockIdxType>(comp_blks[0]);
    item1.is_complete  = true;
    conn->block_cache_->putCommitted(item1);
    conn->complete_pool_->blockCacheReference({static_cast<BlockIdxType>(comp_blks[0])});
    conn->complete_pool_->requestFree({comp_blks[0]});

    auto inc_blks = conn->incomplete_pool_->malloc(1);
    ASSERT_EQ(inc_blks.size(), 1u);
    MemoryDiskBlockCache::CacheItem item2;
    item2.cache_key    = 200;
    item2.backing_type = CacheBackingType::MEMORY;
    item2.block_index  = static_cast<BlockIdxType>(inc_blks[0]);
    item2.is_complete  = false;
    conn->block_cache_->putCommitted(item2);
    conn->incomplete_pool_->blockCacheReference({static_cast<BlockIdxType>(inc_blks[0])});
    conn->incomplete_pool_->requestFree({inc_blks[0]});

    auto keys = conn->cacheKeys();
    EXPECT_EQ(keys.size(), 2u);
    bool has_100 = std::find(keys.begin(), keys.end(), 100) != keys.end();
    bool has_200 = std::find(keys.begin(), keys.end(), 200) != keys.end();
    EXPECT_TRUE(has_100);
    EXPECT_TRUE(has_200);
}

TEST_F(KVCacheMemoryConnectorDualPoolTest, Init_IncompletePoolTracksCompletePoolByStep) {
    const int linear_step = 4;
    const int layer_num   = 4;
    const int block_num   = 10;
    const int spb         = 8;

    auto cfg = createHybridCacheConfig(layer_num, block_num, spb, linear_step);
    allocator_ = std::make_shared<HybridTypeKVCacheAllocator>(cfg, AllocationType::DEVICE);
    ASSERT_TRUE(allocator_->init());
    auto conn = createConnector(cfg);
    ASSERT_TRUE(conn->isDualPool());

    const size_t incomplete = conn->incomplete_pool_->totalBlocksNum();
    const size_t complete   = conn->complete_pool_->totalBlocksNum();
    // BlockPool reserves block 0 in each pool, while initBlockPool sizes the
    // incomplete pool from the complete pool's configured block_num.
    EXPECT_EQ(incomplete, (complete + 1) * static_cast<size_t>(linear_step - 1) - 1);
}

}  // namespace rtp_llm::test

int main(int argc, char** argv) {
    rtp_llm::initLogger();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
