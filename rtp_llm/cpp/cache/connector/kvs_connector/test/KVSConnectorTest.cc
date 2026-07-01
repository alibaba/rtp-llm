#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/allocator/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/cache/connector/kvs_connector/KVSConnector.h"
#include "rtp_llm/cpp/cache/spec/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"

namespace rtp_llm {

void execBatchCopy(const BatchCopyParams&) {}

}  // namespace rtp_llm

namespace rtp_llm::kvs {
namespace {

class TestMeta final: public Meta {
public:
    TestMeta(bool enable_remote_cache, std::string trace_id):
        enable_remote_cache_(enable_remote_cache), trace_id_(std::move(trace_id)) {}

    bool enableMemoryCache() const override {
        return false;
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
    bool                 enable_remote_cache_;
    std::string          trace_id_;
    std::string          unique_id_ = "test_unique_id";
    std::vector<int64_t> tokens_;
};

class FakeKVCacheAllocator final: public KVCacheAllocator {
public:
    explicit FakeKVCacheAllocator(const CacheConfig& config): KVCacheAllocator(config), data_(4096, 0) {}

    void free(const FreeInfo&) override {}
    void insertIntoCache(const InsertInfo&) override {}

    BlockAddrInfo convertIndexToAddr(int, int) const override {
        return {};
    }

    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, int block_id) const override {
        return convertIndexToBuffer(layer_id, 0, block_id);
    }

    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, int block_id, int, int) const override {
        return convertIndexToBuffer(layer_id, block_id);
    }

    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, int group_id, int block_id) const override {
        if (return_empty_buffers) {
            return {};
        }
        if (return_null_buffers) {
            return {BlockInfo{
                false,
                0,
                0,
                nullptr,
                kBlockBytes,
            }};
        }
        const size_t offset = static_cast<size_t>((layer_id * 128) + (group_id * 32) + block_id);
        return {BlockInfo{
            false,
            0,
            0,
            const_cast<char*>(data_.data()) + (offset % (data_.size() - kBlockBytes)),
            kBlockBytes,
        }};
    }

    std::shared_ptr<KVCacheResource> incrKVCacheRef(const KVCacheResource&, const CacheKeysType&, bool) override {
        return nullptr;
    }

    CacheLayerLayout allLayerCacheBase() const override {
        return {};
    }

    bool
    updateKVBlock(const BatchKVCacheResourcePtr&, const std::vector<int>&, bool, std::vector<BlockIdPair>&) override {
        return false;
    }

    int seqSizePerBlock() const override {
        return 1;
    }

    int singleBatchNeedBlocks(const BatchKVCacheResourcePtr&, int, int) const override {
        return 0;
    }

    bool return_empty_buffers = false;
    bool return_null_buffers  = false;

protected:
    bool doInit() override {
        return true;
    }

    MallocResult incrMalloc(const MallocInfo&) override {
        return {};
    }

    MallocResult initMallocForCommonLen(const MallocInfo&) override {
        return {};
    }

    int getNeedBlocks(const MallocInfo&) const override {
        return 0;
    }

    void decrKVCacheRef(const KVCacheResource&, bool) override {}

private:
    static constexpr size_t kBlockBytes = 16;
    std::vector<char>       data_;
};

class FakeKVSClient final: public KVSClient {
public:
    bool init(const KVSClientConfig& config) override {
        init_config = config;
        return init_ok;
    }

    std::optional<KVSReadSession> acquireForRead(const std::vector<std::string>& object_keys,
                                                 const std::string&              trace_id) override {
        acquired_keys = object_keys;
        acquire_trace = trace_id;
        if (!acquire_ok) {
            return std::nullopt;
        }
        KVSReadSession session;
        session.lease_id = lease_id;
        for (const auto& object_key : object_keys) {
            if (std::find(missing_keys.begin(), missing_keys.end(), object_key) != missing_keys.end()) {
                continue;
            }
            session.handles.emplace(object_key, KVSObjectHandle{object_key, 16, 0});
        }
        return session;
    }

    bool load(const KVSReadSession& session, const std::vector<KVSObjectBuffer>& dst_buffers) override {
        load_lease_id = session.lease_id;
        loaded_keys.clear();
        for (const auto& dst : dst_buffers) {
            loaded_keys.push_back(dst.object_key);
        }
        return load_ok;
    }

    bool store(const std::vector<KVSObjectBuffer>& src_buffers, const std::string& trace_id) override {
        store_trace = trace_id;
        stored_keys.clear();
        stored_iov_counts.clear();
        for (const auto& src : src_buffers) {
            stored_keys.push_back(src.object_key);
            stored_iov_counts.push_back(src.iovs.size());
        }
        return store_ok;
    }

    void release(const std::string& lease) override {
        released_leases.push_back(lease);
    }

    bool                     init_ok    = true;
    bool                     acquire_ok = true;
    bool                     load_ok    = true;
    bool                     store_ok   = true;
    std::string              lease_id   = "lease-1";
    KVSClientConfig          init_config;
    std::vector<std::string> acquired_keys;
    std::vector<std::string> missing_keys;
    std::vector<std::string> loaded_keys;
    std::vector<std::string> stored_keys;
    std::vector<size_t>      stored_iov_counts;
    std::vector<std::string> released_leases;
    std::string              acquire_trace;
    std::string              load_lease_id;
    std::string              store_trace;
};

CacheConfig makeCacheConfig() {
    CacheConfig config;
    config.dtype                     = DataType::TYPE_FP16;
    config.layer_num                 = 2;
    config.layer_all_num             = 2;
    config.block_num                 = 8;
    config.seq_size_per_block        = 1;
    config.kernel_seq_size_per_block = 1;

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->type               = KVCacheSpecType::MultiHeadAttention;
    spec->dtype              = config.dtype;
    spec->local_head_num_kv  = 1;
    spec->size_per_head      = 1;
    spec->seq_size_per_block = 1;
    config.fromGroupedSpecs({spec}, {{0, 1}}, {CacheGroupType::FULL}, {"default"});
    return config;
}

KVCacheConfig makeKVCacheConfig() {
    KVCacheConfig config;
    config.reuse_cache                  = true;
    config.enable_kvs_cache             = true;
    config.kvs_object_namespace         = "rtp llm/test";
    config.kvs_cache_key_version        = "test/v";
    config.reco_asyncwrapper_thread_num = 1;
    config.reco_asyncwrapper_queue_size = 8;
    return config;
}

std::shared_ptr<KVCacheResource> makeResource(bool last_block_aligned = true) {
    auto resource = std::make_shared<KVCacheResource>();
    resource->initGroups(1, 2, {{0}, {0}}, 1, {CacheGroupType::FULL});
    resource->mutableBlockIds(0).assign({10, 11, 12});
    resource->setCacheKeys({100, 101, 102});
    resource->setLastBlockAligned(last_block_aligned);
    return resource;
}

template<typename ContextPtr>
void waitDone(const ContextPtr& context) {
    ASSERT_NE(context, nullptr);
    for (int i = 0; i < 200 && !context->done(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(context->done());
}

bool containsSuffix(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

class KVSConnectorTest: public ::testing::Test {
public:
    void SetUp() override {
        initLogger();
        cache_config_               = makeCacheConfig();
        kv_config_                  = makeKVCacheConfig();
        runtime_config_.model_name  = "test/model";
        parallelism_config_.tp_size = 1;
        parallelism_config_.tp_rank = 0;
        allocator_                  = std::make_shared<FakeKVCacheAllocator>(cache_config_);
        client_                     = std::make_shared<FakeKVSClient>();
    }

    std::unique_ptr<KVSConnector> makeConnector() {
        auto connector = std::make_unique<KVSConnector>(
            cache_config_, kv_config_, runtime_config_, parallelism_config_, allocator_, nullptr, client_);
        EXPECT_TRUE(connector->init());
        return connector;
    }

protected:
    CacheConfig                           cache_config_;
    KVCacheConfig                         kv_config_;
    RuntimeConfig                         runtime_config_;
    ParallelismConfig                     parallelism_config_;
    std::shared_ptr<FakeKVCacheAllocator> allocator_;
    std::shared_ptr<FakeKVSClient>        client_;
};

TEST_F(KVSConnectorTest, AsyncWriteUsesStableCacheKeysAndSkipsUnalignedTail) {
    auto connector = makeConnector();
    auto resource  = makeResource(false);
    auto meta      = std::make_shared<TestMeta>(true, "trace-write");

    auto context = connector->asyncWrite(resource, meta);
    waitDone(context);

    ASSERT_TRUE(context->success());
    ASSERT_EQ(client_->stored_keys.size(), 2);
    EXPECT_TRUE(containsSuffix(client_->stored_keys[0], "/tp-0/group-0/block-100"));
    EXPECT_TRUE(containsSuffix(client_->stored_keys[1], "/tp-0/group-0/block-101"));
    EXPECT_FALSE(containsSuffix(client_->stored_keys[0], "/tp-0/group-0/block-10"));
    EXPECT_FALSE(containsSuffix(client_->stored_keys[1], "/tp-0/group-0/block-11"));
    EXPECT_EQ(client_->stored_iov_counts, std::vector<size_t>({2, 2}));
    EXPECT_EQ(client_->store_trace, "kvs_store_trace-write");
}

TEST_F(KVSConnectorTest, AsyncMatchDegradesAcquireFailureToMiss) {
    client_->acquire_ok = false;
    auto connector      = makeConnector();
    auto resource       = makeResource();
    auto meta           = std::make_shared<TestMeta>(true, "trace-miss");

    auto match_context = connector->asyncMatch(resource, meta);
    waitDone(match_context);

    EXPECT_TRUE(match_context->success());
    EXPECT_EQ(match_context->matchedBlockCount(), 0);
    ASSERT_EQ(client_->acquired_keys.size(), 2);
    EXPECT_TRUE(containsSuffix(client_->acquired_keys[0], "/tp-0/group-0/block-100"));
    EXPECT_TRUE(containsSuffix(client_->acquired_keys[1], "/tp-0/group-0/block-101"));
}

TEST_F(KVSConnectorTest, AsyncMatchDoesNotReportHitWhenBlockObjectsAreEmpty) {
    allocator_->return_empty_buffers = true;
    auto connector                   = makeConnector();
    auto resource                    = makeResource();
    auto meta                        = std::make_shared<TestMeta>(true, "trace-empty");

    auto match_context = connector->asyncMatch(resource, meta);
    waitDone(match_context);

    EXPECT_TRUE(match_context->success());
    EXPECT_EQ(match_context->matchedBlockCount(), 0);
    EXPECT_TRUE(client_->acquired_keys.empty());
}

TEST_F(KVSConnectorTest, AsyncMatchDoesNotReportHitWhenBlockObjectsAreInvalid) {
    allocator_->return_null_buffers = true;
    auto connector                  = makeConnector();
    auto resource                   = makeResource();
    auto meta                       = std::make_shared<TestMeta>(true, "trace-invalid");

    auto match_context = connector->asyncMatch(resource, meta);
    waitDone(match_context);

    EXPECT_TRUE(match_context->success());
    EXPECT_EQ(match_context->matchedBlockCount(), 0);
    EXPECT_TRUE(client_->acquired_keys.empty());
}

TEST_F(KVSConnectorTest, AsyncReadLoadsMatchedBlocksAndReleasesLease) {
    auto connector = makeConnector();
    auto resource  = makeResource();
    auto meta      = std::make_shared<TestMeta>(true, "trace-hit");

    auto match_context = connector->asyncMatch(resource, meta);
    waitDone(match_context);
    ASSERT_TRUE(match_context->success());
    ASSERT_EQ(match_context->matchedBlockCount(), 2);

    auto read_context = connector->asyncRead(resource, meta, match_context, 0, 2);
    waitDone(read_context);

    EXPECT_TRUE(read_context->success());
    EXPECT_EQ(resource->remoteReuseBlockNum(), 2);
    EXPECT_EQ(client_->load_lease_id, "lease-1");
    EXPECT_EQ(client_->loaded_keys.size(), 2);
    EXPECT_EQ(client_->released_leases, std::vector<std::string>({"lease-1"}));
}

}  // namespace rtp_llm::kvs
