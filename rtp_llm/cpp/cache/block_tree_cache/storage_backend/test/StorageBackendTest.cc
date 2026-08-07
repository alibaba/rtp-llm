#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackend.h"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"

namespace rtp_llm {
namespace {

class TestBlockPool: public IBlockPool {
public:
    TestBlockPool(): IBlockPool(makeConfig()) {
        markInitialized();
    }

    size_t blockSizeBytes() const override {
        return 16;
    }

private:
    static std::shared_ptr<const BlockPoolConfigBase> makeConfig() {
        auto config                  = std::make_shared<BlockPoolConfigBase>();
        config->pool_type            = BlockPoolType::DEVICE;
        config->pool_name            = "storage_backend_test";
        config->physical_block_count = 8;
        return config;
    }
};

class TestBackend: public StorageBackend {
public:
    explicit TestBackend(bool init_result = true): init_result_(init_result) {}

    ~TestBackend() override {
        shutdown();
    }

    void releaseMatch() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            match_released_ = true;
        }
        cv_.notify_all();
    }

    void waitForMatch() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return match_started_; });
    }

    void waitForRead() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return static_cast<bool>(read_done_); });
    }

    void waitForWrite() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return static_cast<bool>(write_done_); });
    }

    void retainReadCompletionCopy() {
        std::lock_guard<std::mutex> lock(mutex_);
        retained_read_done_ = read_done_;
    }

    void retainWriteCompletionCopy() {
        std::lock_guard<std::mutex> lock(mutex_);
        retained_write_done_ = write_done_;
    }

    void finishRead() {
        finish(read_done_);
    }

    void finishWrite() {
        finish(write_done_);
    }

    size_t writeHandleCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return write_handle_count_;
    }

    std::vector<BlockInfo> resolve(int layer_id, int group_id, int block_id) const {
        return convertIndexToBuffer(layer_id, group_id, block_id);
    }

    std::string groupTag(size_t group_id) const {
        return topology().groupById(group_id).tag;
    }

    size_t initCalls() const {
        return init_calls_;
    }

    void* initResolvedAddress() const {
        return init_resolved_address_;
    }

    void shutdown() override {
        releaseMatch();
        finishRead();
        finishWrite();
        if (match_thread_.joinable()) {
            match_thread_.join();
        }
    }

protected:
    bool initImpl() override {
        ++init_calls_;
        init_resolved_address_ = convertIndexToBuffer(0, 0, 0).front().addr;
        return init_result_;
    }

    void matchImpl(StorageRequest request, MatchDone done) override {
        match_thread_ = std::thread([this, count = request.handles.size(), done = std::move(done)]() mutable {
            std::unique_lock<std::mutex> lock(mutex_);
            match_started_ = true;
            cv_.notify_all();
            cv_.wait(lock, [this] { return match_released_; });
            lock.unlock();
            done(count, nullptr);
        });
    }

    void readImpl(StorageRequest, std::shared_ptr<StorageBackendMatchMeta>, Done done) override {
        std::lock_guard<std::mutex> lock(mutex_);
        read_done_ = std::move(done);
        cv_.notify_all();
    }

    void writeImpl(StorageRequest request, Done done) override {
        std::lock_guard<std::mutex> lock(mutex_);
        write_handle_count_ = 0;
        for (const auto& key_handles : request.handles) {
            write_handle_count_ += key_handles.size();
        }
        write_done_         = std::move(done);
        cv_.notify_all();
    }

private:
    void finish(Done& pending) {
        Done done;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            done = std::move(pending);
        }
        if (done) {
            done();
        }
    }

    mutable std::mutex      mutex_;
    std::condition_variable cv_;
    bool                    match_started_{false};
    bool                    match_released_{false};
    size_t                  write_handle_count_{0};
    Done                    read_done_;
    Done                    write_done_;
    Done                    retained_read_done_;
    Done                    retained_write_done_;
    std::thread             match_thread_;
    bool                    init_result_{true};
    size_t                  init_calls_{0};
    void*                   init_resolved_address_{nullptr};
};

std::shared_ptr<const CacheTopology> makeTopology() {
    auto spec = std::make_shared<MHAKVCacheSpec>();
    spec->tag = "default";
    GroupBase group;
    group.tag                       = spec->tag;
    group.spec                      = std::move(spec);
    group.policy                    = defaultCacheGroupPolicy(CacheGroupType::FULL);
    group.layer_ids                 = {0};
    group.seq_size_per_block        = 1;
    group.kernel_seq_size_per_block = 1;
    return CacheTopology::create({std::move(group)}, {{0, {"default"}}});
}

bool initBackend(TestBackend& backend, const std::shared_ptr<IBlockPool>& pool) {
    return backend.init(
        makeTopology(),
        {pool},
        [](int layer_id, int group_id, int block_id) {
            auto address = reinterpret_cast<void*>(static_cast<uintptr_t>(block_id + 1));
            return std::vector<BlockInfo>{{false, layer_id, group_id, address, 16}};
        },
        [](const auto&) {});
}

StorageRequest makeRequest(BlockIdxType block, size_t key_count = 1) {
    CacheKeysType keys;
    keys.reserve(key_count);
    std::vector<std::vector<StorageBlockHandle>> handles;
    handles.reserve(key_count);
    for (size_t i = 0; i < key_count; ++i) {
        keys.push_back(i + 1);
        handles.push_back({{0, block}});
    }
    return {std::make_shared<CacheKeysType>(std::move(keys)), std::move(handles)};
}

TEST(StorageBackendTest, MatchReturnsBeforeAsynchronousCompletion) {
    auto        pool      = std::make_shared<TestBlockPool>();
    TestBackend backend;
    ASSERT_TRUE(initBackend(backend, pool));
    bool completed = false;
    backend.match(makeRequest(NULL_BLOCK_IDX, 2), [&](size_t count, std::shared_ptr<StorageBackendMatchMeta>) {
        EXPECT_EQ(count, 2u);
        completed = true;
    });
    backend.waitForMatch();
    EXPECT_FALSE(completed);
    backend.releaseMatch();
    backend.shutdown();
    EXPECT_TRUE(completed);
}

TEST(StorageBackendTest, EmptyPerKeyRowsDoNotCreateWriteTask) {
    auto        pool = std::make_shared<TestBlockPool>();
    TestBackend backend;
    ASSERT_TRUE(initBackend(backend, pool));
    StorageRequest request{std::make_shared<CacheKeysType>(CacheKeysType{1}), {{}}};
    EXPECT_FALSE(backend.prepareWrite(std::move(request)));
}

TEST(StorageBackendTest, InitPopulatesBaseResourcesBeforeCallingDerivedImplementation) {
    auto        pool = std::make_shared<TestBlockPool>();
    TestBackend backend;

    EXPECT_TRUE(initBackend(backend, pool));
    EXPECT_EQ(backend.initCalls(), 1u);
    EXPECT_EQ(backend.groupTag(0), "default");
    EXPECT_EQ(backend.initResolvedAddress(), reinterpret_cast<void*>(1));
}

TEST(StorageBackendTest, InitPropagatesDerivedFailure) {
    auto        pool = std::make_shared<TestBlockPool>();
    TestBackend backend(/*init_result=*/false);

    EXPECT_FALSE(initBackend(backend, pool));
    EXPECT_EQ(backend.initCalls(), 1u);
}

TEST(StorageBackendTest, WriteIsFireAndForgetAndPinsEachPhysicalBlockOnce) {
    auto pool  = std::make_shared<TestBlockPool>();
    auto block = pool->malloc().value();
    pool->incRef(block, BlockRefType::REQUEST);
    TestBackend backend;
    ASSERT_TRUE(initBackend(backend, pool));

    auto task = backend.prepareWrite(makeRequest(block, 2));
    EXPECT_EQ(pool->referencedBlocksNum(BlockRefType::STORAGE_BACKEND), 1u);
    backend.write(std::move(task));
    backend.waitForWrite();
    backend.retainWriteCompletionCopy();
    EXPECT_EQ(backend.writeHandleCount(), 2u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockRefType::STORAGE_BACKEND), 1u);

    backend.finishWrite();
    EXPECT_EQ(pool->referencedBlocksNum(BlockRefType::STORAGE_BACKEND), 0u);
    pool->decRef(block, BlockRefType::REQUEST);
}

TEST(StorageBackendTest, UninitializedBackendRejectsTaskAndReleasesPinnedSource) {
    auto pool  = std::make_shared<TestBlockPool>();
    auto block = pool->malloc().value();
    pool->incRef(block, BlockRefType::REQUEST);
    TestBackend initialized_backend;
    ASSERT_TRUE(initBackend(initialized_backend, pool));
    auto task = initialized_backend.prepareWrite(makeRequest(block));
    EXPECT_EQ(pool->referencedBlocksNum(BlockRefType::STORAGE_BACKEND), 1u);

    TestBackend uninitialized_backend;
    EXPECT_ANY_THROW(uninitialized_backend.write(std::move(task)));
    EXPECT_EQ(pool->referencedBlocksNum(BlockRefType::STORAGE_BACKEND), 0u);
    pool->decRef(block, BlockRefType::REQUEST);
}

TEST(StorageBackendTest, ReadPinsTargetsUntilCompletion) {
    auto pool  = std::make_shared<TestBlockPool>();
    auto block = pool->malloc().value();
    pool->incRef(block, BlockRefType::REQUEST);
    TestBackend backend;
    ASSERT_TRUE(initBackend(backend, pool));
    bool completed = false;

    backend.read(makeRequest(block), nullptr, [&] { completed = true; });
    backend.waitForRead();
    backend.retainReadCompletionCopy();
    EXPECT_FALSE(completed);
    EXPECT_EQ(pool->referencedBlocksNum(BlockRefType::STORAGE_BACKEND), 1u);
    backend.finishRead();
    EXPECT_TRUE(completed);
    EXPECT_EQ(pool->referencedBlocksNum(BlockRefType::STORAGE_BACKEND), 0u);
    pool->decRef(block, BlockRefType::REQUEST);
}

TEST(StorageBackendTest, ResolvesGpuBufferAndGroupMetadataFromBoundResources) {
    auto        pool = std::make_shared<TestBlockPool>();
    TestBackend backend;
    ASSERT_TRUE(initBackend(backend, pool));

    EXPECT_EQ(backend.groupTag(0), "default");
    const auto buffers = backend.resolve(0, 0, 3);
    ASSERT_EQ(buffers.size(), 1u);
    EXPECT_EQ(buffers.front().addr, reinterpret_cast<void*>(4));
}

}  // namespace
}  // namespace rtp_llm
