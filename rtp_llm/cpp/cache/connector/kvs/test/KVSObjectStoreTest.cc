#include "rtp_llm/cpp/cache/connector/kvs/KVSObjectStore.h"

#include <gtest/gtest.h>

#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rtp_llm {
namespace {

class FakeKVSObjectBackend: public KVSObjectBackend {
public:
    std::optional<KVSReadHandle> get(const std::vector<std::string>& object_keys,
                                      const std::string&              trace_id) override {
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
        local_get_trace = trace_id;
        KVSReadHandle handle;
        handle.handle_id = "read-lease";
        for (const auto& key : object_keys) {
            if (objects.count(key) != 0) {
                handle.object_keys.insert(key);
            }
        }
        return handle;
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
        create_trace = trace_id;
        created_keys = object_keys;
        created_sizes = object_sizes;
        KVSWriteHandle handle;
        handle.handle_id = "create-lease";
        for (const auto& key : object_keys) {
            handle.object_keys.insert(key);
        }
        return handle;
    }

    bool fetch(const KVSReadHandle& handle,
               const std::vector<std::string>& object_keys,
               const std::string& trace_id) override {
        fetch_trace = trace_id;
        fetched_keys = object_keys;
        return fetch_result && handle.containsAll(object_keys);
    }

    bool load(const KVSReadHandle& handle, const std::vector<KVSObjectBuffer>& dst_buffers) override {
        if (throw_on_load) {
            throw std::runtime_error("load failed");
        }
        loaded_keys.clear();
        for (const auto& dst : dst_buffers) {
            if (!handle.contains(dst.object_key) || objects.count(dst.object_key) == 0) {
                return false;
            }
            loaded_keys.push_back(dst.object_key);
            size_t copied = 0;
            for (const auto& buffer : dst.buffers) {
                auto* dst_ptr = reinterpret_cast<char*>(buffer.addr);
                std::memcpy(dst_ptr, objects.at(dst.object_key).data() + copied, buffer.size);
                copied += buffer.size;
            }
        }
        return true;
    }

    bool store(const KVSWriteHandle& handle, const std::vector<KVSObjectBuffer>& src_buffers) override {
        if (throw_on_store) {
            throw std::runtime_error("store failed");
        }
        stored_keys.clear();
        for (const auto& src : src_buffers) {
            if (!handle.contains(src.object_key)) {
                return false;
            }
            stored_keys.push_back(src.object_key);
            std::string data = src.partial ? objects[src.object_key] : std::string(src.totalBytes(), '\0');
            size_t copied = 0;
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
                  const std::string& trace_id) override {
        complete_trace = trace_id;
        completed_keys = object_keys;
        return handle.containsAll(object_keys);
    }

    bool complete(const KVSWriteHandle& handle,
                  const std::vector<std::string>& object_keys,
                  const std::string& trace_id) override {
        complete_trace = trace_id;
        completed_keys = object_keys;
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

    std::string get_trace;
    std::string local_get_trace;
    std::string mutable_get_trace;
    std::string create_trace;
    std::string fetch_trace;
    std::string complete_trace;
    std::vector<std::string> created_keys;
    std::vector<size_t> created_sizes;
    std::vector<std::string> fetched_keys;
    std::vector<std::string> loaded_keys;
    std::vector<std::string> stored_keys;
    std::vector<std::string> completed_keys;
    std::vector<std::pair<std::string, std::string>> released;
    std::vector<std::pair<std::string, std::string>> discarded;
    std::unordered_map<std::string, std::string> objects;
    bool throw_on_load = false;
    bool throw_on_store = false;
    bool fetch_result = true;
};

KVSObjectBuffer makeBuffer(const std::string& key, std::string& data) {
    KVSObjectBuffer buffer;
    buffer.object_key = key;
    KVSBuffer kvs_buffer;
    kvs_buffer.addr = reinterpret_cast<uint64_t>(data.data());
    kvs_buffer.size = data.size();
    buffer.buffers.push_back(kvs_buffer);
    return buffer;
}

KVSObjectStoreConfig makeConfig(const std::string& object_namespace, const std::string& cache_key_version) {
    KVSObjectStoreConfig config;
    config.object_namespace = object_namespace;
    config.cache_key_version = cache_key_version;
    return config;
}

KVSBlockIdentity makeIdentity(KVSCacheKey cache_key, int group_id) {
    KVSBlockIdentity identity;
    identity.cache_key = cache_key;
    identity.group_id = group_id;
    return identity;
}

}  // namespace

TEST(KVSObjectStoreTest, BuildsStableObjectKey) {
    KVSObjectStore store(makeConfig("rtp_llm/test", "v2"), nullptr);

    EXPECT_EQ(store.makeKey(makeIdentity(123, 1)),
              "rtp_llm/test/v2/123/g1");
}

TEST(KVSObjectStoreTest, WriteCreatesStoresCompletesAndReleases) {
    auto backend = std::make_shared<FakeKVSObjectBackend>();
    KVSObjectStore store(makeConfig("rtp", "v1"), backend);

    std::string data = "abcdef";
    auto object_key = store.makeKey(makeIdentity(7, 0));
    ASSERT_TRUE(store.write({makeBuffer(object_key, data)}, "write-trace"));

    EXPECT_EQ(backend->created_keys, std::vector<std::string>{object_key});
    EXPECT_EQ(backend->created_sizes, std::vector<size_t>{data.size()});
    EXPECT_EQ(backend->stored_keys, std::vector<std::string>{object_key});
    EXPECT_EQ(backend->completed_keys, std::vector<std::string>{object_key});
    ASSERT_EQ(backend->released.size(), 1);
    EXPECT_EQ(backend->released[0], std::make_pair(std::string("create-lease"), std::string("write-trace")));
}

TEST(KVSObjectStoreTest, AcquireFetchesCompletesAndReleases) {
    auto backend = std::make_shared<FakeKVSObjectBackend>();
    KVSObjectStore store(makeConfig("rtp", "v1"), backend);

    std::string expected = "abcdef";
    auto object_key = store.makeKey(makeIdentity(7, 0));
    backend->objects[object_key] = expected;

    std::string actual(expected.size(), '\0');
    auto buffer = makeBuffer(object_key, actual);
    auto handle = store.acquire({buffer}, "read-trace");
    ASSERT_TRUE(handle.has_value());
    ASSERT_TRUE(store.fetch(*handle, {buffer}, "read-trace"));
    store.release(*handle, "read-trace");

    EXPECT_EQ(backend->fetched_keys, std::vector<std::string>{object_key});
    EXPECT_EQ(backend->completed_keys, std::vector<std::string>{object_key});
    ASSERT_EQ(backend->released.size(), 1);
    EXPECT_EQ(backend->released[0], std::make_pair(std::string("read-lease"), std::string("read-trace")));
}

TEST(KVSObjectStoreTest, FetchFailureDoesNotComplete) {
    auto backend = std::make_shared<FakeKVSObjectBackend>();
    KVSObjectStore store(makeConfig("rtp", "v1"), backend);

    std::string data = "abcdef";
    auto object_key = store.makeKey(makeIdentity(7, 0));
    backend->objects[object_key] = data;
    backend->fetch_result = false;

    auto buffer = makeBuffer(object_key, data);
    auto handle = store.acquire({buffer}, "fetch-failure");
    ASSERT_TRUE(handle.has_value());
    EXPECT_FALSE(store.fetch(*handle, {buffer}, "fetch-failure"));
    EXPECT_TRUE(backend->completed_keys.empty());
    store.release(*handle, "fetch-failure");
}

TEST(KVSObjectStoreTest, LoadLocalAcquiresLoadsAndReleases) {
    auto backend = std::make_shared<FakeKVSObjectBackend>();
    KVSObjectStore store(makeConfig("rtp", "v1"), backend);

    std::string expected = "abcdef";
    auto object_key = store.makeKey(makeIdentity(7, 0));
    backend->objects[object_key] = expected;

    std::string actual(expected.size(), '\0');
    ASSERT_TRUE(store.loadLocal({makeBuffer(object_key, actual)}, "load-trace"));

    EXPECT_EQ(actual, expected);
    EXPECT_EQ(backend->local_get_trace, "load-trace");
    EXPECT_TRUE(backend->get_trace.empty());
    EXPECT_EQ(backend->loaded_keys, std::vector<std::string>{object_key});
    ASSERT_EQ(backend->released.size(), 1);
    EXPECT_EQ(backend->released[0], std::make_pair(std::string("read-lease"), std::string("load-trace")));
}

TEST(KVSObjectStoreTest, WriteLocalUsesMutableLeaseAndExplicitOffsets) {
    auto backend = std::make_shared<FakeKVSObjectBackend>();
    KVSObjectStore store(makeConfig("rtp", "v1"), backend);

    const auto object_key = store.makeKey(makeIdentity(7, 0));
    backend->objects[object_key] = "abcdef";
    std::string data = "XYZ";
    auto        object = makeBuffer(object_key, data);
    object.partial = true;
    object.buffers[0].object_offset = 3;

    ASSERT_TRUE(store.writeLocal({object}, "write-local-trace"));

    EXPECT_EQ(backend->objects[object_key], "abcXYZ");
    EXPECT_EQ(backend->mutable_get_trace, "write-local-trace");
    ASSERT_EQ(backend->released.size(), 1);
    EXPECT_EQ(backend->released[0], std::make_pair(std::string("write-lease"), std::string("write-local-trace")));
}

TEST(KVSObjectStoreTest, LoadMissReleasesLeaseAndReturnsFalse) {
    auto backend = std::make_shared<FakeKVSObjectBackend>();
    KVSObjectStore store(makeConfig("rtp", "v1"), backend);

    std::string actual(6, 0);
    auto object_key = store.makeKey(makeIdentity(7, 0));
    EXPECT_FALSE(store.loadLocal({makeBuffer(object_key, actual)}, "local-miss"));

    EXPECT_TRUE(backend->loaded_keys.empty());
    ASSERT_EQ(backend->released.size(), 1);
    EXPECT_EQ(backend->released[0], std::make_pair(std::string("read-lease"), std::string("local-miss")));
}

TEST(KVSObjectStoreTest, LoadExceptionReleasesLease) {
    auto backend = std::make_shared<FakeKVSObjectBackend>();
    KVSObjectStore store(makeConfig("rtp", "v1"), backend);

    std::string actual(6, '\0');
    auto object_key = store.makeKey(makeIdentity(7, 0));
    backend->objects[object_key] = "abcdef";
    backend->throw_on_load = true;

    EXPECT_THROW(store.loadLocal({makeBuffer(object_key, actual)}, "read-throw"), std::runtime_error);
    ASSERT_EQ(backend->released.size(), 1);
    EXPECT_EQ(backend->released[0], std::make_pair(std::string("read-lease"), std::string("read-throw")));
}

TEST(KVSObjectStoreTest, WriteExceptionDiscardsLease) {
    auto backend = std::make_shared<FakeKVSObjectBackend>();
    KVSObjectStore store(makeConfig("rtp", "v1"), backend);

    std::string data = "abcdef";
    auto object_key = store.makeKey(makeIdentity(7, 0));
    backend->throw_on_store = true;

    EXPECT_THROW(store.write({makeBuffer(object_key, data)}, "write-throw"), std::runtime_error);
    ASSERT_EQ(backend->discarded.size(), 1);
    EXPECT_EQ(backend->discarded[0], std::make_pair(std::string("create-lease"), std::string("write-throw")));
    EXPECT_TRUE(backend->released.empty());
}

}  // namespace rtp_llm
