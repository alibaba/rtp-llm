#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
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

TEST(RPCPoolTest, ReadyPreflightReusesReadyChannel) {
    TestGrpcServer server;
    RPCPool        pool;

    auto first = pool.getReadyConnection(server.address(), 2s);
    ASSERT_TRUE(first.ok()) << first.status();
    auto second = pool.getReadyConnection(server.address(), 2s);
    ASSERT_TRUE(second.ok()) << second.status();
    EXPECT_EQ(second->channel, first->channel);
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
}

}  // namespace
}  // namespace rtp_llm::test
