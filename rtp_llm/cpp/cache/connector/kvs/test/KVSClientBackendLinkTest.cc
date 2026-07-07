#include "rtp_llm/cpp/cache/connector/kvs/KVSClientBackend.h"

#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

class FakeKVSNativeClient: public KVSNativeClient {
public:
    vineyard::Status init(const v6d::kvs::KVSClientConfig& config) override {
        init_called = true;
        endpoint_url = config.endpoint_url;
        socket_path  = config.socket_path;
        return vineyard::Status::OK();
    }

    std::optional<v6d::kvs::LeaseHandle> get(const std::vector<std::string>& object_keys,
                                             const std::string&              peer,
                                             bool                            unsafe,
                                             const std::string&              trace_id) override {
        get_peer = peer;
        get_trace = trace_id;
        get_unsafe = unsafe;
        v6d::kvs::LeaseHandle lease_handle;
        lease_handle.lease_id = unsafe ? "write-lease" : "read-lease";
        lease_handle.scope    = unsafe ? "write" : "read";
        auto& objects = unsafe ? writable_objects : readable_objects;
        for (const auto& key : object_keys) {
            auto iter = objects.find(key);
            if (iter != objects.end()) {
                lease_handle.object_handles[key] = makeHandle(key, iter->second);
            }
        }
        return lease_handle;
    }

    std::optional<v6d::kvs::LeaseHandle> create(const std::vector<std::string>& object_keys,
                                                const std::vector<size_t>&      object_sizes,
                                                const std::string&              trace_id) override {
        create_trace = trace_id;
        created_keys = object_keys;
        created_sizes = object_sizes;
        v6d::kvs::LeaseHandle lease_handle;
        if (create_noop) {
            lease_handle.scope = "CREATE";
            return lease_handle;
        }
        lease_handle.lease_id = "create-lease";
        lease_handle.scope = "create";
        lease_handle.seal_target_count = object_keys.size();
        for (size_t i = 0; i < object_keys.size(); ++i) {
            writable_objects[object_keys[i]].assign(object_sizes[i], '\0');
            lease_handle.object_handles[object_keys[i]] =
                makeHandle(object_keys[i], writable_objects[object_keys[i]]);
        }
        return lease_handle;
    }

    vineyard::Status fetch(v6d::kvs::LeaseHandle&         lease_handle,
                           const std::vector<std::string>& object_keys,
                           const std::string&              trace_id) override {
        fetch_trace = trace_id;
        fetched_keys = object_keys;
        for (const auto& key : object_keys) {
            if (lease_handle.object_handles.count(key) == 0) {
                return vineyard::Status::ObjectNotExists(key);
            }
        }
        return vineyard::Status::OK();
    }

    vineyard::Status complete(v6d::kvs::LeaseHandle&         lease_handle,
                              const std::vector<std::string>& object_keys,
                              const std::string&              trace_id) override {
        complete_trace = trace_id;
        completed_keys = object_keys;
        for (const auto& key : object_keys) {
            lease_handle.completed_object_keys.insert(key);
        }
        return vineyard::Status::OK();
    }

    vineyard::Status release(const std::string& lease_id, const std::string& trace_id) override {
        released.emplace_back(lease_id, trace_id);
        return vineyard::Status::OK();
    }

    vineyard::Status discard(const std::string& lease_id, const std::string& trace_id) override {
        discarded.emplace_back(lease_id, trace_id);
        return vineyard::Status::OK();
    }

    std::optional<v6d::kvs::BufferView> localBuffer(const v6d::kvs::ObjectHandle& handle) const override {
        v6d::kvs::BufferView view;
        view.addr = handle.data_offset;
        view.size = handle.bytes;
        return view;
    }

    static v6d::kvs::ObjectHandle makeHandle(const std::string& key, std::string& data) {
        v6d::kvs::ObjectHandle handle;
        handle.object_key  = key;
        handle.bytes       = data.size();
        handle.data_offset = reinterpret_cast<uint64_t>(data.data());
        return handle;
    }

    bool init_called = false;
    std::string endpoint_url;
    std::string socket_path;
    std::string get_peer;
    std::string get_trace;
    bool get_unsafe{false};
    std::string create_trace;
    std::string fetch_trace;
    std::string complete_trace;
    std::vector<std::string> created_keys;
    std::vector<size_t> created_sizes;
    std::vector<std::string> fetched_keys;
    std::vector<std::string> completed_keys;
    std::vector<std::pair<std::string, std::string>> released;
    std::vector<std::pair<std::string, std::string>> discarded;
    bool create_noop{false};
    std::unordered_map<std::string, std::string> readable_objects;
    std::unordered_map<std::string, std::string> writable_objects;
};

KVSObjectBuffer makeBuffer(const std::string& key, std::string& data) {
    KVSObjectBuffer object_buffer;
    object_buffer.object_key = key;
    KVSBuffer buffer;
    buffer.addr = reinterpret_cast<uint64_t>(data.data());
    buffer.size = data.size();
    object_buffer.buffers.push_back(buffer);
    return object_buffer;
}

}  // namespace

TEST(KVSClientBackendLinkTest, ReadPathCallsNativeClientAndCopiesData) {
    auto* fake = new FakeKVSNativeClient();
    fake->readable_objects["object-a"] = "abcdef";

    KVSConnectorConfig config;
    config.endpoint_url = "http://127.0.0.1:8080";
    config.socket_path  = "/tmp/kvs.sock";
    config.read_peer    = "remote.example:9600";
    KVSClientBackend backend(config, std::unique_ptr<KVSNativeClient>(fake));

    auto handle = backend.get({"object-a"}, "read-trace");
    ASSERT_TRUE(handle.has_value());
    ASSERT_TRUE(fake->init_called);
    EXPECT_EQ(fake->endpoint_url, config.endpoint_url);
    EXPECT_EQ(fake->socket_path, config.socket_path);
    EXPECT_EQ(fake->get_peer, config.read_peer);
    EXPECT_EQ(fake->get_trace, "read-trace");

    ASSERT_TRUE(backend.fetch(*handle, {"object-a"}, "fetch-trace"));
    std::string actual(6, '\0');
    ASSERT_TRUE(backend.load(*handle, {makeBuffer("object-a", actual)}));
    ASSERT_TRUE(backend.complete(*handle, {"object-a"}, "complete-trace"));
    backend.release(*handle, "release-trace");

    EXPECT_EQ(actual, "abcdef");
    EXPECT_EQ(fake->fetched_keys, std::vector<std::string>{"object-a"});
    EXPECT_EQ(fake->completed_keys, std::vector<std::string>{"object-a"});
    ASSERT_EQ(fake->released.size(), 1);
    EXPECT_EQ(fake->released[0], std::make_pair(std::string("read-lease"), std::string("release-trace")));
}

TEST(KVSClientBackendLinkTest, LocalReadOverridesConfiguredPeer) {
    auto* fake = new FakeKVSNativeClient();
    fake->readable_objects["object-local"] = "local";

    KVSConnectorConfig config;
    config.read_peer = "remote.example:9600";
    KVSClientBackend backend(config, std::unique_ptr<KVSNativeClient>(fake));

    auto handle = backend.getLocal({"object-local"}, "local-read");
    ASSERT_TRUE(handle.has_value());
    EXPECT_EQ(fake->get_peer, "local");
    EXPECT_EQ(fake->get_trace, "local-read");
    backend.release(*handle, "local-release");
}

TEST(KVSClientBackendLinkTest, WritePathCallsNativeClientAndCopiesData) {
    auto* fake = new FakeKVSNativeClient();

    KVSConnectorConfig config;
    KVSClientBackend backend(config, std::unique_ptr<KVSNativeClient>(fake));

    auto handle = backend.create({"object-b"}, {6}, "create-trace");
    ASSERT_TRUE(handle.has_value());
    std::string data = "uvwxyz";
    ASSERT_TRUE(backend.store(*handle, {makeBuffer("object-b", data)}));
    ASSERT_TRUE(backend.complete(*handle, {"object-b"}, "complete-trace"));
    backend.release(*handle, "release-trace");

    EXPECT_EQ(fake->created_keys, std::vector<std::string>{"object-b"});
    EXPECT_EQ(fake->created_sizes, std::vector<size_t>{6});
    EXPECT_EQ(fake->writable_objects["object-b"], "uvwxyz");
    EXPECT_EQ(fake->completed_keys, std::vector<std::string>{"object-b"});
    ASSERT_EQ(fake->released.size(), 1);
    EXPECT_EQ(fake->released[0], std::make_pair(std::string("create-lease"), std::string("release-trace")));
}

TEST(KVSClientBackendLinkTest, CreateReturnsNoopHandleWhenAllObjectsExist) {
    auto* fake = new FakeKVSNativeClient();
    fake->create_noop = true;

    KVSConnectorConfig config;
    KVSClientBackend backend(config, std::unique_ptr<KVSNativeClient>(fake));

    auto handle = backend.create({"object-b"}, {6}, "create-noop");

    ASSERT_TRUE(handle.has_value());
    EXPECT_TRUE(handle->object_keys.empty());
    EXPECT_FALSE(handle->valid());
}

TEST(KVSClientBackendLinkTest, MutableLocalWriteUsesUnsafeGet) {
    auto* fake = new FakeKVSNativeClient();
    fake->writable_objects["object-b"] = "abcdef";

    KVSConnectorConfig config;
    KVSClientBackend backend(config, std::unique_ptr<KVSNativeClient>(fake));

    auto handle = backend.getMutableLocal({"object-b"}, "mutable-get");
    ASSERT_TRUE(handle.has_value());
    EXPECT_EQ(fake->get_peer, "local");
    EXPECT_TRUE(fake->get_unsafe);

    std::string data = "XYZ";
    auto        object = makeBuffer("object-b", data);
    object.partial = true;
    object.buffers[0].object_offset = 3;
    ASSERT_TRUE(backend.store(*handle, {object}));
    backend.release(*handle, "mutable-release");

    EXPECT_EQ(fake->writable_objects["object-b"], "abcXYZ");
}

TEST(KVSClientBackendLinkTest, DiscardReleasesLeaseState) {
    auto* fake = new FakeKVSNativeClient();

    KVSConnectorConfig config;
    KVSClientBackend backend(config, std::unique_ptr<KVSNativeClient>(fake));

    auto handle = backend.create({"object-c"}, {4}, "create-trace");
    ASSERT_TRUE(handle.has_value());
    backend.discard(*handle, "discard-trace");

    ASSERT_EQ(fake->discarded.size(), 1);
    EXPECT_EQ(fake->discarded[0], std::make_pair(std::string("create-lease"), std::string("discard-trace")));

    std::string data = "zzzz";
    EXPECT_FALSE(backend.store(*handle, {makeBuffer("object-c", data)}));
}

}  // namespace rtp_llm
