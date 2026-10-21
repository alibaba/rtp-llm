#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/model_rpc/RPCPool.h"

namespace rtp_llm::test {
namespace {

using namespace std::chrono_literals;

class TcpBlackhole {
public:
    TcpBlackhole() {
        listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        EXPECT_GE(listen_fd_, 0);

        int reuse_addr = 1;
        EXPECT_EQ(::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse_addr, sizeof(reuse_addr)), 0);

        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port        = 0;
        EXPECT_EQ(::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
        EXPECT_EQ(::listen(listen_fd_, 1), 0);

        socklen_t addr_len = sizeof(addr);
        EXPECT_EQ(::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &addr_len), 0);
        port_ = ntohs(addr.sin_port);
        saturateAcceptQueue(addr);
    }

    ~TcpBlackhole() {
        for (int fd : filler_fds_) {
            ::close(fd);
        }
        ::close(listen_fd_);
    }

    std::string address() const {
        return "127.0.0.1:" + std::to_string(port_);
    }

private:
    void saturateAcceptQueue(const sockaddr_in& addr) {
        for (int i = 0; i < 64; ++i) {
            int fd = ::socket(AF_INET, SOCK_STREAM, 0);
            ASSERT_GE(fd, 0);
            ASSERT_NE(::fcntl(fd, F_SETFL, O_NONBLOCK), -1);
            int rc = ::connect(fd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr));
            ASSERT_TRUE(rc == 0 || errno == EINPROGRESS);
            filler_fds_.push_back(fd);
        }
        std::this_thread::sleep_for(50ms);
    }

private:
    int              listen_fd_{-1};
    int              port_{0};
    std::vector<int> filler_fds_;
};

class TestGrpcServer {
public:
    TestGrpcServer() {
        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port_);
        builder.RegisterService(&service_);
        server_ = builder.BuildAndStart();
        EXPECT_NE(server_, nullptr);
        EXPECT_GT(port_, 0);
    }

    ~TestGrpcServer() {
        if (server_) {
            server_->Shutdown();
            server_->Wait();
        }
    }

    std::string address() const {
        return "127.0.0.1:" + std::to_string(port_);
    }

private:
    RpcService::Service           service_;
    std::unique_ptr<grpc::Server> server_;
    int                           port_{0};
};

int64_t elapsedMs(std::chrono::steady_clock::time_point begin) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count();
}

TEST(RPCPoolTest, UnreadyRemoteGenerateBlocksUntilClientDeadline) {
    TcpBlackhole blackhole;
    RPCPool      pool;
    auto         connection = pool.getConnection(blackhole.address());
    ASSERT_TRUE(connection.ok()) << connection.status();
    ASSERT_EQ(connection->channel->GetState(false), GRPC_CHANNEL_IDLE);

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + 400ms);
    auto begin      = std::chrono::steady_clock::now();
    auto stream     = connection->stub->RemoteGenerate(&context);
    auto elapsed_ms = elapsedMs(begin);

    ASSERT_NE(stream, nullptr);
    auto status = stream->Finish();
    EXPECT_EQ(status.error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);
    EXPECT_GE(elapsed_ms, 300);
    EXPECT_LT(elapsed_ms, 1500);
}

TEST(RPCPoolTest, CachedLookupDoesNotStartConnectingAnIdleChannel) {
    TcpBlackhole blackhole;
    RPCPool pool;
    auto original = pool.getConnection(blackhole.address());
    ASSERT_TRUE(original.ok());
    ASSERT_EQ(original->channel->GetState(false), GRPC_CHANNEL_IDLE);
    auto reused = pool.getConnection(blackhole.address());
    ASSERT_TRUE(reused.ok());
    EXPECT_EQ(reused->channel, original->channel);
    EXPECT_EQ(reused->stub, original->stub);
    EXPECT_EQ(reused->channel->GetState(false), GRPC_CHANNEL_IDLE);
}

TEST(RPCPoolTest, ReadyPreflightBoundsUnreadyChannelAndEvictsItsGeneration) {
    TcpBlackhole blackhole;
    RPCPool      pool;
    auto         original = pool.getConnection(blackhole.address());
    ASSERT_TRUE(original.ok()) << original.status();
    ASSERT_EQ(original->channel->GetState(false), GRPC_CHANNEL_IDLE);

    auto begin      = std::chrono::steady_clock::now();
    auto result     = pool.getReadyConnection(blackhole.address(), 100ms);
    auto elapsed_ms = elapsedMs(begin);

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), absl::StatusCode::kUnavailable);
    EXPECT_GE(elapsed_ms, 75);
    EXPECT_LT(elapsed_ms, 1000);

    auto replacement = pool.getConnection(blackhole.address());
    ASSERT_TRUE(replacement.ok()) << replacement.status();
    EXPECT_NE(replacement->channel, original->channel);
}

TEST(RPCPoolTest, ReadyPreflightCancellationStopsWaitBeforeDeadline) {
    TcpBlackhole      blackhole;
    RPCPool           pool;
    std::atomic<bool> cancelled{false};
    std::thread cancel_thread([&cancelled]() {
        std::this_thread::sleep_for(30ms);
        cancelled.store(true, std::memory_order_release);
    });

    const auto begin  = std::chrono::steady_clock::now();
    auto       result = pool.getReadyConnection(blackhole.address(),
                                          std::chrono::system_clock::now() + 2s,
                                          [&cancelled]() { return cancelled.load(std::memory_order_acquire); });
    const auto elapsed_ms = elapsedMs(begin);
    cancel_thread.join();

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), absl::StatusCode::kCancelled);
    EXPECT_LT(elapsed_ms, 500);
}

TEST(RPCPoolTest, ReadyPreflightReusesReadyChannel) {
    TestGrpcServer server;
    RPCPool        pool;

    auto first = pool.getReadyConnection(server.address(), 2s);
    ASSERT_TRUE(first.ok()) << first.status();
    auto second = pool.getReadyConnection(server.address(), 2s);
    ASSERT_TRUE(second.ok()) << second.status();
    EXPECT_EQ(second->channel, first->channel);
}

TEST(RPCPoolTest, PeerMutexIsReclaimedAfterNewAndCachedConnectionLookups) {
    TestGrpcServer server;
    RPCPool        pool;
    auto first = pool.getReadyConnection(server.address(), 2s);
    ASSERT_TRUE(first.ok()) << first.status();
    EXPECT_TRUE(pool.peer_mutexes_.empty());

    auto second = pool.getConnection(server.address());
    ASSERT_TRUE(second.ok()) << second.status();
    EXPECT_EQ(first->channel, second->channel);
    EXPECT_TRUE(pool.peer_mutexes_.empty());
}

TEST(RPCPoolTest, PeerMutexRemainsSharedUntilLastConcurrentHolderExits) {
    RPCPool pool;
    const std::string peer = "127.0.0.1:12345";
    std::shared_ptr<std::mutex> first;
    std::shared_ptr<std::mutex> second;
    {
        std::lock_guard<std::mutex> guard(pool.mutex_);
        first = pool.getPeerMutexLocked(peer);
        second = pool.getPeerMutexLocked(peer);
    }
    ASSERT_EQ(first, second);
    pool.releasePeerMutex(peer, first);
    EXPECT_EQ(first, nullptr);
    ASSERT_EQ(pool.peer_mutexes_.size(), 1u);

    std::shared_ptr<std::mutex> third;
    {
        std::lock_guard<std::mutex> guard(pool.mutex_);
        third = pool.getPeerMutexLocked(peer);
    }
    EXPECT_EQ(second, third);
    pool.releasePeerMutex(peer, second);
    EXPECT_EQ(second, nullptr);
    EXPECT_EQ(pool.peer_mutexes_.size(), 1u);
    pool.releasePeerMutex(peer, third);
    EXPECT_EQ(third, nullptr);
    EXPECT_TRUE(pool.peer_mutexes_.empty());
}

TEST(RPCPoolTest, SimultaneousExitsReclaimPeerMutexRegistryEntry) {
    RPCPool pool;
    const std::string peer = "127.0.0.1:12345";
    std::vector<std::shared_ptr<std::mutex>> holders(8);
    {
        std::lock_guard<std::mutex> guard(pool.mutex_);
        for (auto& holder : holders) {
            holder = pool.getPeerMutexLocked(peer);
        }
    }
    std::promise<void> release;
    auto ready = release.get_future().share();
    std::vector<std::thread> threads;
    for (size_t i = 0; i < holders.size(); ++i) {
        threads.emplace_back([&, i, ready] {
            ready.wait();
            pool.releasePeerMutex(peer, holders[i]);
        });
    }
    release.set_value();
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_TRUE(pool.peer_mutexes_.empty());
    for (const auto& holder : holders) {
        EXPECT_EQ(holder, nullptr);
    }
}

struct FailingStubService {
    struct Stub {};
    static std::unique_ptr<Stub> NewStub(const std::shared_ptr<grpc::Channel>&) {
        return nullptr;
    }
};

TEST(RPCPoolTest, FailedStubCreationReclaimsPeerMutex) {
    Pool<FailingStubService> pool;
    auto result = pool.getConnection("127.0.0.1:12345");
    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(pool.connection_pool_.empty());
    EXPECT_TRUE(pool.peer_mutexes_.empty());
}

TEST(RPCPoolTest, ConnectionCacheIsBoundedAndKeepsRecentlyUsedPeer) {
    TestGrpcServer server;
    RPCPool pool;
    auto live = pool.getReadyConnection(server.address(), 2s);
    ASSERT_TRUE(live.ok()) << live.status();
    {
        std::lock_guard<std::mutex> guard(pool.mutex_);
        for (size_t i = 0; i < RPCPool::kMaxCachedConnections - 1; ++i) {
            pool.cacheConnectionLocked("cache-peer-" + std::to_string(i), live.value());
        }
    }
    ASSERT_EQ(pool.connection_pool_.size(), RPCPool::kMaxCachedConnections);
    auto recent = pool.getConnection(server.address());
    ASSERT_TRUE(recent.ok()) << recent.status();
    EXPECT_EQ(recent->channel, live->channel);
    {
        std::lock_guard<std::mutex> guard(pool.mutex_);
        pool.cacheConnectionLocked("new-cache-peer", live.value());
    }
    EXPECT_EQ(pool.connection_pool_.size(), RPCPool::kMaxCachedConnections);
    EXPECT_EQ(pool.connection_lru_.size(), pool.connection_pool_.size());
    EXPECT_EQ(pool.connection_pool_.count("cache-peer-0"), 0u);
    EXPECT_EQ(pool.connection_pool_.count(server.address()), 1u);
}

TEST(RPCPoolTest, CacheEvictionPreservesOutstandingChannelAndStub) {
    TestGrpcServer server;
    TestGrpcServer other_server;
    RPCPool pool;
    auto outstanding = pool.getReadyConnection(server.address(), 2s);
    ASSERT_TRUE(outstanding.ok()) << outstanding.status();
    auto filler = pool.getReadyConnection(other_server.address(), 2s);
    ASSERT_TRUE(filler.ok()) << filler.status();
    {
        std::lock_guard<std::mutex> guard(pool.mutex_);
        for (size_t i = 0; i < RPCPool::kMaxCachedConnections; ++i) {
            pool.cacheConnectionLocked("cache-peer-" + std::to_string(i), filler.value());
        }
    }
    ASSERT_EQ(pool.connection_pool_.count(server.address()), 0u);
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + 2s);
    GetPeerInfoRequestPB request;
    GetPeerInfoResponsePB response;
    auto status = outstanding->stub->GetPeerInfo(&context, request, &response);
    // The test service does not implement GetPeerInfo. Receiving its status
    // proves the externally held stub/channel still works after cache eviction.
    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNIMPLEMENTED);
}

TEST(RPCPoolTest, OldGenerationCannotEraseReadyReplacement) {
    TestGrpcServer server;
    RPCPool        pool;
    auto           first = pool.getReadyConnection(server.address(), 2s);
    ASSERT_TRUE(first.ok()) << first.status();

    pool.removeConnection(server.address(), first->channel);
    auto replacement = pool.getReadyConnection(server.address(), 2s);
    ASSERT_TRUE(replacement.ok()) << replacement.status();
    ASSERT_NE(replacement->channel, first->channel);

    pool.removeConnection(server.address(), first->channel);
    auto current = pool.getReadyConnection(server.address(), 2s);
    ASSERT_TRUE(current.ok()) << current.status();
    EXPECT_EQ(current->channel, replacement->channel);
    EXPECT_EQ(pool.connection_lru_.size(), pool.connection_pool_.size());
}

}  // namespace
}  // namespace rtp_llm::test
