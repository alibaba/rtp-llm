#include "rtp_llm/cpp/cache/connector/kvs/KVSConnector.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/spec/MHAKVCacheSpec.h"

namespace rtp_llm {
namespace {

class FakeKVSObjectBackend: public KVSObjectBackend {
public:
    std::optional<KVSReadHandle> get(const std::vector<std::string>& object_keys,
                                      const std::string&              trace_id) override {
        ++get_count;
        get_trace = trace_id;
        KVSReadHandle handle;
        handle.handle_id = "read-lease";
        for (const auto& key : object_keys) {
            if (objects.count(key) != 0) {
                handle.object_keys.insert(key);
            }
        }
        return handle;
    }

    std::optional<KVSReadHandle> getLocal(const std::vector<std::string>& object_keys,
                                          const std::string&              trace_id) override {
        return get(object_keys, trace_id);
    }

    std::optional<KVSWriteHandle> getMutableLocal(const std::vector<std::string>& object_keys,
                                                  const std::string&              trace_id) override {
        mutable_get_trace = trace_id;
        KVSWriteHandle handle;
        handle.handle_id = "write-lease";
        for (const auto& key : object_keys) {
            if (objects.count(key) != 0) {
                handle.object_keys.insert(key);
            }
        }
        return handle;
    }

    std::optional<KVSWriteHandle> create(const std::vector<std::string>& object_keys,
                                          const std::vector<size_t>&      object_sizes,
                                          const std::string&              trace_id) override {
        create_trace  = trace_id;
        created_keys  = object_keys;
        created_sizes = object_sizes;
        KVSWriteHandle handle;
        for (size_t i = 0; i < object_keys.size(); ++i) {
            const auto& key = object_keys[i];
            if (objects.count(key) != 0) {
                continue;
            }
            handle.object_keys.insert(key);
            objects[key].assign(object_sizes[i], '\0');
        }
        if (!handle.object_keys.empty()) {
            handle.handle_id = "create-lease";
        }
        return handle;
    }

    bool fetch(const KVSReadHandle& handle,
               const std::vector<std::string>& object_keys,
               const std::string&              trace_id) override {
        ++fetch_count;
        fetch_trace = trace_id;
        fetched_keys.insert(fetched_keys.end(), object_keys.begin(), object_keys.end());
        for (const auto& key : object_keys) {
            if (fetch_fail_keys.count(key) != 0) {
                return false;
            }
        }
        return handle.containsAll(object_keys);
    }

    bool load(const KVSReadHandle& handle, const std::vector<KVSObjectBuffer>& dst_buffers) override {
        ++load_count;
        for (const auto& dst : dst_buffers) {
            if (!handle.contains(dst.object_key) || objects.count(dst.object_key) == 0) {
                return false;
            }
            size_t copied = 0;
            for (const auto& buffer : dst.buffers) {
                const size_t object_offset = dst.partial ? buffer.object_offset : copied;
                if (object_offset + buffer.size > objects.at(dst.object_key).size()) {
                    return false;
                }
                auto* dst_ptr = reinterpret_cast<char*>(buffer.addr);
                std::memcpy(dst_ptr, objects.at(dst.object_key).data() + object_offset, buffer.size);
                copied += buffer.size;
            }
        }
        return true;
    }

    bool store(const KVSWriteHandle& handle, const std::vector<KVSObjectBuffer>& src_buffers) override {
        for (const auto& src : src_buffers) {
            if (!handle.contains(src.object_key)) {
                return false;
            }
            std::string data = src.partial ? objects[src.object_key] : std::string(src.totalBytes(), '\0');
            size_t      copied = 0;
            for (const auto& buffer : src.buffers) {
                const size_t object_offset = src.partial ? buffer.object_offset : copied;
                if (object_offset + buffer.size > data.size()) {
                    return false;
                }
                const auto* src_ptr = reinterpret_cast<const char*>(buffer.addr);
                std::memcpy(data.data() + object_offset, src_ptr, buffer.size);
                copied += buffer.size;
            }
            objects[src.object_key] = std::move(data);
        }
        return true;
    }

    bool complete(const KVSReadHandle& handle,
                  const std::vector<std::string>& object_keys,
                  const std::string&              trace_id) override {
        ++complete_count;
        complete_trace = trace_id;
        return handle.containsAll(object_keys);
    }

    bool complete(const KVSWriteHandle& handle,
                  const std::vector<std::string>& object_keys,
                  const std::string&              trace_id) override {
        ++complete_count;
        complete_trace = trace_id;
        return handle.containsAll(object_keys);
    }

    void release(const KVSReadHandle& handle, const std::string& trace_id) override {
        released.emplace_back(handle.handle_id, trace_id);
    }

    void release(const KVSWriteHandle& handle, const std::string& trace_id) override {
        released.emplace_back(handle.handle_id, trace_id);
    }

    void discard(const KVSWriteHandle& handle, const std::string& trace_id) override {
        discarded.emplace_back(handle.handle_id, trace_id);
    }

    int get_count{0};
    int fetch_count{0};
    int complete_count{0};
    int load_count{0};
    std::string get_trace;
    std::string mutable_get_trace;
    std::string create_trace;
    std::string fetch_trace;
    std::string complete_trace;
    std::vector<std::string> fetched_keys;
    std::vector<std::string> created_keys;
    std::vector<size_t> created_sizes;
    std::vector<std::pair<std::string, std::string>> released;
    std::vector<std::pair<std::string, std::string>> discarded;
    std::unordered_set<std::string> fetch_fail_keys;
    std::unordered_map<std::string, std::string> objects;
};

CacheConfig makeCacheConfig() {
    CacheConfig config;
    config.layer_num     = 1;
    config.layer_all_num = 1;
    auto spec            = std::make_shared<MHAKVCacheSpec>();
    spec->tag                = "default";
    spec->dtype              = TYPE_FP16;
    spec->layers             = {0};
    spec->local_head_num_kv  = 1;
    spec->size_per_head      = 1;
    spec->seq_size_per_block = 1;

    GroupBase group;
    group.spec      = spec;
    group.policy    = defaultCacheGroupPolicy(CacheGroupType::FULL);
    group.layer_ids = {0};

    std::vector<LayerBase> layers(1);
    layers[0].group_ids             = {0};
    layers[0].tag_to_gid[spec->tag] = 0;
    config.setTopology({group}, std::move(layers));
    return config;
}

CacheConfig makeTwoLayerCacheConfig() {
    CacheConfig config;
    config.layer_num     = 2;
    config.layer_all_num = 2;
    auto spec            = std::make_shared<MHAKVCacheSpec>();
    spec->tag                = "default";
    spec->dtype              = TYPE_FP16;
    spec->layers             = {0, 1};
    spec->local_head_num_kv  = 1;
    spec->size_per_head      = 1;
    spec->seq_size_per_block = 1;

    GroupBase group;
    group.spec      = spec;
    group.policy    = defaultCacheGroupPolicy(CacheGroupType::FULL);
    group.layer_ids = {0, 1};

    std::vector<LayerBase> layers(2);
    for (auto& layer : layers) {
        layer.group_ids             = {0};
        layer.tag_to_gid[spec->tag] = 0;
    }
    config.setTopology({group}, std::move(layers));
    return config;
}

std::shared_ptr<KVCacheResource> makeResource(const std::vector<KVSCacheKey>& cache_keys,
                                              const std::vector<BlockIdxType>& block_ids,
                                              bool                            last_block_aligned = true) {
    auto resource = std::make_shared<KVCacheResource>();
    resource->initGroups(1, 1, {{0}});
    resource->setCacheKeys(CacheKeysType(cache_keys.begin(), cache_keys.end()));
    resource->mutableBlockIds(0).assign(block_ids);
    resource->setLastBlockAligned(last_block_aligned);
    return resource;
}

std::shared_ptr<KVCacheResource> makeTwoLayerResource(const std::vector<KVSCacheKey>& cache_keys,
                                                      const std::vector<BlockIdxType>& block_ids) {
    auto resource = std::make_shared<KVCacheResource>();
    resource->initGroups(1, 2, {{0}, {0}});
    resource->setCacheKeys(CacheKeysType(cache_keys.begin(), cache_keys.end()));
    resource->mutableBlockIds(0).assign(block_ids);
    resource->setLastBlockAligned(true);
    return resource;
}

std::string objectKey(KVSCacheKey cache_key) {
    return "rtp/v1/" + std::to_string(cache_key) + "/g0";
}

KVSConnectorConfig makeConnectorConfig() {
    KVSConnectorConfig config;
    config.object_namespace  = "rtp";
    config.cache_key_version = "v1";
    config.worker_thread_num = 1;
    config.worker_queue_size = 16;
    config.inline_execute    = true;
    return config;
}

std::shared_ptr<KVSObjectStore> makeObjectStore(const std::shared_ptr<KVSObjectBackend>& backend) {
    KVSObjectStoreConfig config;
    config.object_namespace  = "rtp";
    config.cache_key_version = "v1";
    return std::make_shared<KVSObjectStore>(std::move(config), backend);
}

KVSConnector::BlockBufferResolver
makeBlockBufferResolver(const std::shared_ptr<std::unordered_map<int, std::vector<char>>>& blocks) {
    return [blocks](int layer_id, int group_id, BlockIdxType block_id) {
        (void)layer_id;
        (void)group_id;
        auto iter = blocks->find(block_id);
        if (iter == blocks->end()) {
            return std::vector<BlockInfo>{};
        }
        BlockInfo info;
        info.addr       = iter->second.data();
        info.size_bytes = iter->second.size();
        return std::vector<BlockInfo>{info};
    };
}

KVSConnector::BlockBufferResolver
makeLayerBlockBufferResolver(const std::shared_ptr<std::unordered_map<int, std::vector<char>>>& layer_blocks) {
    return [layer_blocks](int layer_id, int group_id, BlockIdxType block_id) {
        (void)group_id;
        auto iter = layer_blocks->find(layer_id);
        if (iter == layer_blocks->end() || block_id <= 0) {
            return std::vector<BlockInfo>{};
        }
        BlockInfo info;
        info.addr       = iter->second.data();
        info.size_bytes = iter->second.size();
        return std::vector<BlockInfo>{info};
    };
}

TEST(KVSAsyncContextTest, SuccessCompletesWithMatchedBlockCount) {
    KVSAsyncContext context;
    EXPECT_FALSE(context.done());
    EXPECT_FALSE(context.success());

    context.markRunning();
    EXPECT_FALSE(context.done());

    context.markSuccess(3);
    EXPECT_TRUE(context.done());
    EXPECT_TRUE(context.success());
    EXPECT_EQ(context.matchedBlockCount(), 3);
    EXPECT_TRUE(context.errorInfo().ok());
}

TEST(KVSAsyncContextTest, FailureCompletesWithErrorInfo) {
    KVSAsyncContext context;
    context.markFailed("thread pool full");

    EXPECT_TRUE(context.done());
    EXPECT_FALSE(context.success());
    EXPECT_TRUE(context.errorInfo().hasError());
    EXPECT_EQ(context.errorInfo().ToString(), "thread pool full");
}

TEST(KVSAsyncContextTest, WaitDoneBlocksUntilCompletion) {
    KVSAsyncContext context;
    std::atomic<bool> wait_returned{false};

    std::thread waiter([&]() {
        context.waitDone();
        wait_returned = true;
    });

    context.markSuccess(1);
    waiter.join();
    EXPECT_TRUE(wait_returned.load());
}

TEST(KVSConnectorTest, AsyncWriteWritesCompleteBlocks) {
    auto cache_config = makeCacheConfig();
    auto backend      = std::make_shared<FakeKVSObjectBackend>();
    auto blocks       = std::make_shared<std::unordered_map<int, std::vector<char>>>();
    (*blocks)[1]      = {'a', 'b', 'c'};
    (*blocks)[2]      = {'d', 'e', 'f'};

    KVSConnector connector(
        cache_config, makeConnectorConfig(), makeObjectStore(backend), makeBlockBufferResolver(blocks));
    ASSERT_TRUE(connector.init());
    auto context = connector.asyncWrite(makeResource({101, 102}, {1, 2}), nullptr);
    context->waitDone();

    EXPECT_TRUE(context->success()) << context->errorInfo().ToString();
    EXPECT_EQ(backend->objects[objectKey(101)], "abc");
    EXPECT_EQ(backend->objects[objectKey(102)], "def");
    auto kvs_context = std::dynamic_pointer_cast<KVSAsyncContext>(context);
    ASSERT_TRUE(kvs_context);
    EXPECT_EQ(kvs_context->matchedBlockCount(), 2);
}

TEST(KVSConnectorTest, AsyncWriteAggregatesRequiredLayersIntoOneGroupObject) {
    auto cache_config = makeTwoLayerCacheConfig();
    auto backend      = std::make_shared<FakeKVSObjectBackend>();
    auto layer_blocks = std::make_shared<std::unordered_map<int, std::vector<char>>>();
    (*layer_blocks)[0] = {'a', 'b'};
    (*layer_blocks)[1] = {'c', 'd'};

    KVSConnector connector(
        cache_config, makeConnectorConfig(), makeObjectStore(backend), makeLayerBlockBufferResolver(layer_blocks));
    ASSERT_TRUE(connector.init());
    auto context = connector.asyncWrite(makeTwoLayerResource({101}, {1}), nullptr);
    context->waitDone();

    EXPECT_TRUE(context->success()) << context->errorInfo().ToString();
    EXPECT_EQ(backend->created_keys, std::vector<std::string>{objectKey(101)});
    EXPECT_EQ(backend->created_sizes, std::vector<size_t>{4});
    EXPECT_EQ(backend->objects[objectKey(101)], "abcd");
}

TEST(KVSConnectorTest, AsyncWriteStoresOneRankSlicePerTpWorker) {
    auto cache_config = makeCacheConfig();
    auto backend      = std::make_shared<FakeKVSObjectBackend>();
    auto rank_blocks  = std::make_shared<std::vector<std::vector<char>>>(
        std::initializer_list<std::vector<char>>{{'a', 'b', 'c'}, {'X', 'Y', 'Z'}});
    auto active_rank = std::make_shared<size_t>(0);
    KVSConnector::BlockBufferResolver resolver = [rank_blocks, active_rank](int, int, BlockIdxType block_id) {
        if (block_id != 1 || *active_rank >= rank_blocks->size()) {
            return std::vector<BlockInfo>{};
        }
        auto& block = rank_blocks->at(*active_rank);
        BlockInfo info;
        info.addr       = block.data();
        info.size_bytes = block.size();
        return std::vector<BlockInfo>{info};
    };

    KVSConnector* connector_ptr = nullptr;
    KVSConnector::KVSPlanSender sender =
        [&connector_ptr, active_rank](KVSConnector::Operation                       operation,
                                      const std::vector<KVSConnector::KVSObjectPlan>& objects,
                                      const std::string&                              trace_id) {
        if (operation != KVSConnector::Operation::WRITE || connector_ptr == nullptr) {
            return false;
        }
        for (size_t rank = 0; rank < 2; ++rank) {
            *active_rank = rank;
            KVSOperationRequestPB request;
            request.set_operation(KVSOperationRequestPB::WRITE);
            request.set_trace_id(trace_id);
            for (const auto& object : objects) {
                auto* item = request.add_items();
                item->set_object_key(object.object_key);
                for (const auto& buffer : object.buffers) {
                    auto* spec = item->add_buffers();
                    spec->set_layer_id(buffer.layer_id);
                    spec->set_group_id(buffer.group_id);
                    spec->set_block_id(buffer.block_id);
                    spec->set_object_offset(rank * object.rank_bytes + buffer.object_offset);
                }
            }
            KVSOperationResponsePB response;
            if (!connector_ptr->executeWorkerPlan(request, response) || !response.success()) {
                return false;
            }
        }
        return true;
    };

    KVSConnector connector(cache_config,
                           makeConnectorConfig(),
                           makeObjectStore(backend),
                           std::move(resolver),
                           {},
                           2,
                           std::move(sender));
    connector_ptr = &connector;
    ASSERT_TRUE(connector.init());

    auto context = connector.asyncWrite(makeResource({101}, {1}), nullptr);
    context->waitDone();

    EXPECT_TRUE(context->success()) << context->errorInfo().ToString();
    EXPECT_EQ(backend->created_sizes, std::vector<size_t>{6});
    EXPECT_EQ(backend->objects[objectKey(101)], "abcXYZ");
}

TEST(KVSConnectorTest, AsyncWriteSendsOnlyObjectsCreatedByThisWrite) {
    auto cache_config = makeCacheConfig();
    auto backend      = std::make_shared<FakeKVSObjectBackend>();
    backend->objects[objectKey(101)] = "existing";
    auto blocks       = std::make_shared<std::unordered_map<int, std::vector<char>>>();
    (*blocks)[1]      = {'a', 'b', 'c'};
    (*blocks)[2]      = {'d', 'e', 'f'};

    std::vector<std::string> sent_keys;
    KVSConnector::KVSPlanSender sender = [&sent_keys](KVSConnector::Operation operation,
                                                       const std::vector<KVSConnector::KVSObjectPlan>& objects,
                                                       const std::string&) {
        if (operation != KVSConnector::Operation::WRITE) {
            return false;
        }
        for (const auto& object : objects) {
            sent_keys.push_back(object.object_key);
        }
        return true;
    };
    KVSConnector connector(cache_config,
                           makeConnectorConfig(),
                           makeObjectStore(backend),
                           makeBlockBufferResolver(blocks),
                           {},
                           1,
                           std::move(sender));
    ASSERT_TRUE(connector.init());

    auto context = connector.asyncWrite(makeResource({101, 102}, {1, 2}), nullptr);
    context->waitDone();

    EXPECT_TRUE(context->success()) << context->errorInfo().ToString();
    EXPECT_EQ(sent_keys, std::vector<std::string>{objectKey(102)});
}

TEST(KVSConnectorTest, AsyncWriteIsNoopWhenAllObjectsExist) {
    auto cache_config = makeCacheConfig();
    auto backend      = std::make_shared<FakeKVSObjectBackend>();
    backend->objects[objectKey(101)] = "existing";
    auto blocks       = std::make_shared<std::unordered_map<int, std::vector<char>>>();
    (*blocks)[1]      = {'a', 'b', 'c'};

    bool sender_called = false;
    KVSConnector::KVSPlanSender sender = [&sender_called](KVSConnector::Operation,
                                                          const std::vector<KVSConnector::KVSObjectPlan>&,
                                                          const std::string&) {
        sender_called = true;
        return false;
    };
    KVSConnector connector(cache_config,
                           makeConnectorConfig(),
                           makeObjectStore(backend),
                           makeBlockBufferResolver(blocks),
                           {},
                           1,
                           std::move(sender));
    ASSERT_TRUE(connector.init());

    auto context = connector.asyncWrite(makeResource({101}, {1}), nullptr);
    context->waitDone();

    EXPECT_TRUE(context->success()) << context->errorInfo().ToString();
    EXPECT_FALSE(sender_called);
}

TEST(KVSConnectorTest, AsyncMatchStopsAtIncompleteBlockPlan) {
    auto cache_config = makeCacheConfig();
    auto backend      = std::make_shared<FakeKVSObjectBackend>();
    auto blocks       = std::make_shared<std::unordered_map<int, std::vector<char>>>();
    (*blocks)[1]      = {'a', 'b', 'c'};
    backend->objects[objectKey(101)] = "abc";
    backend->objects[objectKey(102)] = "def";

    KVSConnector connector(
        cache_config, makeConnectorConfig(), makeObjectStore(backend), makeBlockBufferResolver(blocks));
    ASSERT_TRUE(connector.init());
    auto context = connector.asyncMatch(makeResource({101, 102}, {1, NULL_BLOCK_IDX}), nullptr);
    context->waitDone();

    EXPECT_TRUE(context->success()) << context->errorInfo().ToString();
    EXPECT_EQ(context->matchedBlockCount(), 1);
}

TEST(KVSConnectorTest, AsyncMatchGetsLeaseButDoesNotFetchOrResolveCoordinatorBuffers) {
    auto cache_config = makeCacheConfig();
    auto backend      = std::make_shared<FakeKVSObjectBackend>();
    backend->objects[objectKey(101)] = "abc";

    std::atomic<int> resolve_count{0};
    KVSConnector::BlockBufferResolver resolver = [&resolve_count](int layer_id, int group_id, BlockIdxType block_id) {
        (void)layer_id;
        (void)group_id;
        (void)block_id;
        ++resolve_count;
        return std::vector<BlockInfo>{};
    };

    KVSConnector connector(cache_config, makeConnectorConfig(), makeObjectStore(backend), std::move(resolver));
    ASSERT_TRUE(connector.init());
    auto context = connector.asyncMatch(makeResource({101}, {1}), nullptr);
    context->waitDone();

    EXPECT_TRUE(context->success()) << context->errorInfo().ToString();
    EXPECT_EQ(context->matchedBlockCount(), 1);
    EXPECT_EQ(resolve_count.load(), 0);
    EXPECT_EQ(backend->get_count, 1);
    EXPECT_EQ(backend->fetch_count, 0);
    EXPECT_EQ(backend->complete_count, 0);
    EXPECT_EQ(backend->load_count, 0);
    EXPECT_TRUE(backend->released.empty());

    context.reset();
    ASSERT_EQ(backend->released.size(), 1);
    EXPECT_EQ(backend->released[0].first, "read-lease");
}

TEST(KVSConnectorTest, AsyncReadFetchesBeforeSendingReadPlan) {
    auto cache_config = makeCacheConfig();
    auto backend      = std::make_shared<FakeKVSObjectBackend>();
    backend->objects[objectKey(101)] = "abc";

    auto blocks  = std::make_shared<std::unordered_map<int, std::vector<char>>>();
    (*blocks)[1] = {0, 0, 0};

    std::vector<KVSConnector::KVSObjectPlan> captured_objects;
    KVSConnector::KVSPlanSender sender =
        [&captured_objects](KVSConnector::Operation                 operation,
                            const std::vector<KVSConnector::KVSObjectPlan>& objects,
                            const std::string&                       trace_id) {
        (void)operation;
        (void)trace_id;
        captured_objects = objects;
        return true;
    };

    KVSConnector connector(cache_config,
                           makeConnectorConfig(),
                           makeObjectStore(backend),
                           makeBlockBufferResolver(blocks),
                           {},
                           1,
                           std::move(sender));
    ASSERT_TRUE(connector.init());
    auto match_context = connector.asyncMatch(makeResource({101}, {1}), nullptr);
    match_context->waitDone();
    ASSERT_TRUE(match_context->success());

    auto read_context = connector.asyncRead(makeResource({101}, {1}), nullptr, match_context, 0, 1);
    read_context->waitDone();

    EXPECT_TRUE(read_context->success()) << read_context->errorInfo().ToString();
    auto kvs_read_context = std::dynamic_pointer_cast<KVSAsyncContext>(read_context);
    ASSERT_TRUE(kvs_read_context);
    EXPECT_EQ(kvs_read_context->matchedBlockCount(), 1);
    EXPECT_EQ(backend->fetch_count, 1);
    EXPECT_EQ(backend->complete_count, 1);
    ASSERT_EQ(backend->released.size(), 1);
    EXPECT_EQ(backend->released[0].first, "read-lease");
    ASSERT_EQ(captured_objects.size(), 1);
    EXPECT_EQ(captured_objects[0].object_key, objectKey(101));
    ASSERT_EQ(captured_objects[0].buffers.size(), 1);
    EXPECT_EQ(captured_objects[0].buffers[0].layer_id, 0);
    EXPECT_EQ(captured_objects[0].buffers[0].group_id, 0);
    EXPECT_EQ(captured_objects[0].buffers[0].block_id, 1);
}

TEST(KVSConnectorTest, AsyncReadTreatsFetchFailureAsPrefixMiss) {
    auto cache_config = makeCacheConfig();
    auto backend      = std::make_shared<FakeKVSObjectBackend>();
    backend->objects[objectKey(101)] = "abc";
    backend->objects[objectKey(102)] = "def";
    backend->fetch_fail_keys.insert(objectKey(102));

    auto blocks  = std::make_shared<std::unordered_map<int, std::vector<char>>>();
    (*blocks)[1] = {'\0', '\0', '\0'};
    (*blocks)[2] = {'\0', '\0', '\0'};

    std::vector<KVSConnector::KVSObjectPlan> captured_objects;
    KVSConnector::KVSPlanSender sender =
        [&captured_objects](KVSConnector::Operation                 operation,
                            const std::vector<KVSConnector::KVSObjectPlan>& objects,
                            const std::string&                       trace_id) {
        (void)operation;
        (void)trace_id;
        captured_objects = objects;
        return true;
    };

    KVSConnector connector(cache_config,
                           makeConnectorConfig(),
                           makeObjectStore(backend),
                           makeBlockBufferResolver(blocks),
                           {},
                           1,
                           std::move(sender));
    ASSERT_TRUE(connector.init());
    auto resource      = makeResource({101, 102}, {1, 2});
    auto match_context = connector.asyncMatch(resource, nullptr);
    match_context->waitDone();
    ASSERT_TRUE(match_context->success());
    ASSERT_EQ(match_context->matchedBlockCount(), 2);

    auto read_context = connector.asyncRead(resource, nullptr, match_context, 0, 2);
    read_context->waitDone();

    EXPECT_TRUE(read_context->success()) << read_context->errorInfo().ToString();
    auto kvs_read_context = std::dynamic_pointer_cast<KVSAsyncContext>(read_context);
    ASSERT_TRUE(kvs_read_context);
    EXPECT_EQ(kvs_read_context->matchedBlockCount(), 1);
    ASSERT_EQ(captured_objects.size(), 1);
    EXPECT_EQ(captured_objects[0].object_key, objectKey(101));
    EXPECT_EQ(resource->remoteReuseBlockNum(), 1);
}

TEST(KVSConnectorTest, AsyncReadPlanCarriesObjectOffsetsForRequiredLayers) {
    auto cache_config = makeTwoLayerCacheConfig();
    auto backend      = std::make_shared<FakeKVSObjectBackend>();
    backend->objects[objectKey(101)] = "abcd";

    auto layer_blocks = std::make_shared<std::unordered_map<int, std::vector<char>>>();
    (*layer_blocks)[0] = {0, 0};
    (*layer_blocks)[1] = {0, 0};

    std::vector<KVSConnector::KVSObjectPlan> captured_objects;
    KVSConnector::KVSPlanSender sender =
        [&captured_objects](KVSConnector::Operation                 operation,
                            const std::vector<KVSConnector::KVSObjectPlan>& objects,
                            const std::string&                       trace_id) {
        (void)operation;
        (void)trace_id;
        captured_objects = objects;
        return true;
    };

    KVSConnector connector(cache_config,
                           makeConnectorConfig(),
                           makeObjectStore(backend),
                           makeLayerBlockBufferResolver(layer_blocks),
                           {},
                           1,
                           std::move(sender));
    ASSERT_TRUE(connector.init());
    auto resource      = makeTwoLayerResource({101}, {1});
    auto match_context = connector.asyncMatch(resource, nullptr);
    match_context->waitDone();
    ASSERT_TRUE(match_context->success());

    auto read_context = connector.asyncRead(resource, nullptr, match_context, 0, 1);
    read_context->waitDone();

    EXPECT_TRUE(read_context->success()) << read_context->errorInfo().ToString();
    ASSERT_EQ(captured_objects.size(), 1);
    ASSERT_EQ(captured_objects[0].buffers.size(), 2);
    EXPECT_EQ(captured_objects[0].buffers[0].object_offset, 0);
    EXPECT_EQ(captured_objects[0].buffers[1].object_offset, 2);
}

TEST(KVSConnectorTest, AsyncReadFromNonZeroBlockFetchesSuffixOnly) {
    auto cache_config = makeCacheConfig();
    auto backend      = std::make_shared<FakeKVSObjectBackend>();
    backend->objects[objectKey(101)] = "abc";
    backend->objects[objectKey(102)] = "def";
    backend->objects[objectKey(103)] = "ghi";

    auto blocks  = std::make_shared<std::unordered_map<int, std::vector<char>>>();
    (*blocks)[1] = {0, 0, 0};
    (*blocks)[2] = {0, 0, 0};
    (*blocks)[3] = {0, 0, 0};

    std::vector<KVSConnector::KVSObjectPlan> captured_objects;
    KVSConnector::KVSPlanSender sender =
        [&captured_objects](KVSConnector::Operation                 operation,
                            const std::vector<KVSConnector::KVSObjectPlan>& objects,
                            const std::string&                       trace_id) {
        (void)operation;
        (void)trace_id;
        captured_objects = objects;
        return true;
    };

    KVSConnector connector(cache_config,
                           makeConnectorConfig(),
                           makeObjectStore(backend),
                           makeBlockBufferResolver(blocks),
                           {},
                           1,
                           std::move(sender));
    ASSERT_TRUE(connector.init());
    auto resource      = makeResource({101, 102, 103}, {1, 2, 3});
    auto match_context = connector.asyncMatch(resource, nullptr);
    match_context->waitDone();
    ASSERT_TRUE(match_context->success());
    ASSERT_EQ(match_context->matchedBlockCount(), 3);

    auto read_context = connector.asyncRead(resource, nullptr, match_context, 1, 2);
    read_context->waitDone();

    EXPECT_TRUE(read_context->success()) << read_context->errorInfo().ToString();
    auto kvs_read_context = std::dynamic_pointer_cast<KVSAsyncContext>(read_context);
    ASSERT_TRUE(kvs_read_context);
    EXPECT_EQ(kvs_read_context->matchedBlockCount(), 2);
    EXPECT_EQ(backend->fetched_keys, (std::vector<std::string>{objectKey(102), objectKey(103)}));
    ASSERT_EQ(captured_objects.size(), 2);
    EXPECT_EQ(captured_objects[0].object_key, objectKey(102));
    EXPECT_EQ(captured_objects[1].object_key, objectKey(103));
}

TEST(KVSConnectorTest, CopyCacheUsesExplicitObjectOffsets) {
    auto cache_config = makeCacheConfig();
    auto backend      = std::make_shared<FakeKVSObjectBackend>();
    backend->objects[objectKey(101)] = "xxabczz";
    auto blocks  = std::make_shared<std::unordered_map<int, std::vector<char>>>();
    (*blocks)[1] = {0, 0, 0};

    KVSConnector connector(
        cache_config, makeConnectorConfig(), makeObjectStore(backend), makeBlockBufferResolver(blocks));
    ASSERT_TRUE(connector.init());

    KVSOperationRequestPB request;
    request.set_trace_id("trace");
    auto* item = request.add_items();
    item->set_object_key(objectKey(101));
    auto* buffer = item->add_buffers();
    buffer->set_layer_id(0);
    buffer->set_group_id(0);
    buffer->set_block_id(1);
    buffer->set_object_offset(2);

    KVSOperationResponsePB response;
    EXPECT_TRUE(connector.executeWorkerPlan(request, response));
    EXPECT_TRUE(response.success());
    EXPECT_EQ((*blocks)[1], (std::vector<char>{'a', 'b', 'c'}));
}

TEST(KVSConnectorTest, CopyCacheLoadsLocalWithoutFetchOrComplete) {
    auto cache_config = makeCacheConfig();
    auto backend      = std::make_shared<FakeKVSObjectBackend>();
    backend->objects[objectKey(101)] = "abc";
    auto blocks  = std::make_shared<std::unordered_map<int, std::vector<char>>>();
    (*blocks)[1] = {'\0', '\0', '\0'};

    KVSConnector connector(
        cache_config, makeConnectorConfig(), makeObjectStore(backend), makeBlockBufferResolver(blocks));
    ASSERT_TRUE(connector.init());

    KVSOperationRequestPB request;
    request.set_trace_id("trace");
    auto* item = request.add_items();
    item->set_object_key(objectKey(101));
    auto* buffer = item->add_buffers();
    buffer->set_layer_id(0);
    buffer->set_group_id(0);
    buffer->set_block_id(1);

    KVSOperationResponsePB response;
    EXPECT_TRUE(connector.executeWorkerPlan(request, response));
    EXPECT_TRUE(response.success());
    EXPECT_EQ((*blocks)[1], (std::vector<char>{'a', 'b', 'c'}));
    EXPECT_EQ(backend->fetch_count, 0);
    EXPECT_EQ(backend->complete_count, 0);
    EXPECT_EQ(backend->load_count, 1);
}

}  // namespace
}  // namespace rtp_llm
