#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

#include "rtp_llm/cpp/cache/connector/kvs_connector/KVSClient.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm::kvs {
namespace {

const char* envOrNull(const char* name) {
    const char* value = std::getenv(name);
    return value && value[0] != '\0' ? value : nullptr;
}

BlockInfo cpuBlock(void* addr, size_t bytes) {
    BlockInfo info;
    info.is_cuda    = false;
    info.addr       = addr;
    info.size_bytes = bytes;
    return info;
}

TEST(KVSClientIntegrationTest, StoreAcquireLoadRelease) {
    const char* url    = envOrNull("KVS_INTEGRATION_URL");
    const char* socket = envOrNull("KVS_INTEGRATION_SOCKET");
    if (!url || !socket) {
        GTEST_SKIP() << "set KVS_INTEGRATION_URL and KVS_INTEGRATION_SOCKET to run real KVS integration";
    }

    KVSClientConfig config(url, socket, 5000, 20);

    KVSClient client;
    initLogger();
    ASSERT_TRUE(client.init(config));

    std::vector<char> src_a(32);
    std::vector<char> src_b(17);
    for (size_t i = 0; i < src_a.size(); ++i) {
        src_a[i] = static_cast<char>('a' + (i % 26));
    }
    for (size_t i = 0; i < src_b.size(); ++i) {
        src_b[i] = static_cast<char>('A' + (i % 26));
    }

    const std::string object_key = "rtp_llm/test/kvs_client/" + std::to_string(getpid()) + "/object";
    std::cerr << "object_key=" << object_key << std::endl;
    KVSObjectBuffer src;
    src.object_key = object_key;
    src.iovs       = {cpuBlock(src_a.data(), src_a.size()), cpuBlock(src_b.data(), src_b.size())};

    ASSERT_TRUE(client.store({src}, "kvs-client-integration-store"));

    auto session = client.acquireForRead({object_key}, "kvs-client-integration-read");
    ASSERT_TRUE(session.has_value());
    ASSERT_EQ(session->handles.size(), 1);
    ASSERT_NE(session->handles.find(object_key), session->handles.end());

    std::vector<char> dst_a(src_a.size(), 0);
    std::vector<char> dst_b(src_b.size(), 0);
    KVSObjectBuffer   dst;
    dst.object_key = object_key;
    dst.iovs       = {cpuBlock(dst_a.data(), dst_a.size()), cpuBlock(dst_b.data(), dst_b.size())};

    EXPECT_TRUE(client.load(*session, {dst}));
    EXPECT_EQ(dst_a, src_a);
    EXPECT_EQ(dst_b, src_b);
    client.release(session->lease_id);
}

TEST(KVSClientIntegrationTest, StoreThenReloadWithNewClient) {
    const char* url    = envOrNull("KVS_INTEGRATION_URL");
    const char* socket = envOrNull("KVS_INTEGRATION_SOCKET");
    if (!url || !socket) {
        GTEST_SKIP() << "set KVS_INTEGRATION_URL and KVS_INTEGRATION_SOCKET to run real KVS integration";
    }

    KVSClientConfig config(url, socket, 5000, 300);

    initLogger();

    std::vector<char> src(64);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<char>((i * 17) & 0xff);
    }

    const std::string object_key = "rtp_llm/test/kvs_client/reload/" + std::to_string(getpid()) + "/object";
    std::cerr << "reload_object_key=" << object_key << std::endl;

    {
        KVSClient client;
        ASSERT_TRUE(client.init(config));

        KVSObjectBuffer src_buffer;
        src_buffer.object_key = object_key;
        src_buffer.iovs       = {cpuBlock(src.data(), src.size())};
        ASSERT_TRUE(client.store({src_buffer}, "kvs-client-integration-store-reload"));
        ASSERT_TRUE(client.store({src_buffer}, "kvs-client-integration-store-reload-duplicate"));
    }

    KVSClient reload_client;
    ASSERT_TRUE(reload_client.init(config));
    auto session = reload_client.acquireForRead({object_key}, "kvs-client-integration-read-reload");
    ASSERT_TRUE(session.has_value());
    ASSERT_EQ(session->handles.size(), 1);

    std::vector<char> dst(src.size(), 0);
    KVSObjectBuffer   dst_buffer;
    dst_buffer.object_key = object_key;
    dst_buffer.iovs       = {cpuBlock(dst.data(), dst.size())};

    EXPECT_TRUE(reload_client.load(*session, {dst_buffer}));
    EXPECT_EQ(dst, src);
    reload_client.release(session->lease_id);
}

TEST(KVSClientIntegrationTest, MissingObjectReturnsEmptyHandles) {
    const char* url    = envOrNull("KVS_INTEGRATION_URL");
    const char* socket = envOrNull("KVS_INTEGRATION_SOCKET");
    if (!url || !socket) {
        GTEST_SKIP() << "set KVS_INTEGRATION_URL and KVS_INTEGRATION_SOCKET to run real KVS integration";
    }

    KVSClientConfig config(url, socket, 5000, 20);

    KVSClient client;
    initLogger();
    ASSERT_TRUE(client.init(config));

    const std::string object_key = "rtp_llm/test/kvs_client/missing/" + std::to_string(getpid()) + "/object";
    auto              session    = client.acquireForRead({object_key}, "kvs-client-integration-missing");

    ASSERT_TRUE(session.has_value());
    EXPECT_TRUE(session->handles.empty());
    client.release(session->lease_id);
}

}  // namespace
}  // namespace rtp_llm::kvs
