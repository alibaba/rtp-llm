#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <utility>

#include "gtest/gtest.h"
#include "aios/network/anet/tcpcomponent.h"
#include "aios/network/anet/timeutil.h"
#include "aios/network/arpc/arpc/anet/ANetRPCServerClosure.h"
#include "autil/NetUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpClient.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpServer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/proto/cache_store_service.pb.h"

namespace rtp_llm {
namespace {

struct LoadResult {
    bool                        failed;
    arpc::ErrorCode              error_code;
    KvCacheStoreServiceErrorCode load_error;
    bool                        direct_write_response;
};

class LoadDone: public google::protobuf::Closure {
public:
    void Run() override {
        std::unique_ptr<LoadDone> self(this);
        result.set_value({controller.Failed(),
                          controller.GetErrorCode(),
                          response.error_code(),
                          response.direct_write_response()});
    }

    arpc::ANetRPCController  controller;
    CacheLoadRequest        request;
    CacheLoadResponse       response;
    std::promise<LoadResult> result;
};

class DeferredLoadService: public KvCacheStoreService {
public:
    void load(google::protobuf::RpcController*,
              const CacheLoadRequest*,
              CacheLoadResponse*         response,
              google::protobuf::Closure* done) override {
        auto* rpc_done = dynamic_cast<arpc::ANetRPCServerClosure*>(done);
        bool  complete_now;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            complete_now = released_;
            if (!complete_now) {
                response_ = response;
                done_     = done;
            }
            received.set_value(rpc_done ? rpc_done->GetConnection() : nullptr);
        }
        if (complete_now) {
            reply(response, done);
        }
    }

    void release() {
        CacheLoadResponse*         response;
        google::protobuf::Closure* done;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            released_ = true;
            response  = std::exchange(response_, nullptr);
            done      = std::exchange(done_, nullptr);
        }
        if (done) {
            reply(response, done);
        }
    }

    std::promise<anet::Connection*> received;

private:
    static void reply(CacheLoadResponse* response, google::protobuf::Closure* done) {
        response->set_error_code(KvCacheStoreServiceErrorCode::EC_SUCCESS);
        response->set_direct_write_response(true);
        done->Run();
    }

    std::mutex                mutex_;
    bool                      released_{false};
    CacheLoadResponse*         response_{nullptr};
    google::protobuf::Closure* done_{nullptr};
};

}  // namespace

class TcpServerTest: public ::testing::Test {
protected:
    void SetUp() override {
        service_ = std::make_unique<DeferredLoadService>();
        server_  = std::make_unique<TcpServer>();
        ASSERT_TRUE(server_->init(1, 1, false));
        ASSERT_TRUE(server_->registerService(service_.get()));
        const auto port = autil::NetUtil::randomPort();
        ASSERT_TRUE(server_->start(port));

        client_ = std::make_unique<TcpClient>();
        ASSERT_TRUE(client_->init(1));
        channel_ = client_->getChannel("127.0.0.1", port);
        ASSERT_NE(channel_, nullptr);
    }

    void TearDown() override {
        service_->release();
        channel_.reset();
        client_.reset();
        server_.reset();
        service_.reset();
    }

    std::future<LoadResult> load(uint32_t timeout_ms) {
        auto call   = std::make_unique<LoadDone>();
        auto result = call->result.get_future();
        call->controller.SetExpireTime(timeout_ms);
        call->request.set_timeout_ms(timeout_ms);
        call->request.set_requestid("deferred-load");
        KvCacheStoreService_Stub stub(channel_.get());
        auto* done = call.release();
        stub.load(&done->controller, &done->request, &done->response, done);
        return result;
    }

    std::unique_ptr<DeferredLoadService> service_;
    std::unique_ptr<TcpServer>           server_;
    std::unique_ptr<TcpClient>           client_;
    std::shared_ptr<arpc::RPCChannelBase> channel_;
};

TEST_F(TcpServerTest, DeferredLoadSurvivesTransportIdleTimeout) {
    auto received  = service_->received.get_future();
    auto completed = load(30000);
    ASSERT_EQ(received.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    auto* connection = received.get();
    ASSERT_NE(connection, nullptr);
    auto* component = dynamic_cast<anet::TCPComponent*>(connection->getIOComponent());
    ASSERT_NE(component, nullptr);

    // Advance only the server's idle check while the real async RPC is pending.
    const auto after_idle_limit = anet::TimeUtil::getTime() + anet::MAX_IDLE_TIME_IN_MICROSECONDS + 1000000;
    EXPECT_TRUE(component->checkTimeout(after_idle_limit));
    EXPECT_FALSE(connection->isClosed());
    EXPECT_EQ(completed.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
    service_->release();

    ASSERT_EQ(completed.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    const auto result = completed.get();
    EXPECT_FALSE(result.failed);
    EXPECT_EQ(result.error_code, arpc::ARPC_ERROR_NONE);
    EXPECT_EQ(result.load_error, KvCacheStoreServiceErrorCode::EC_SUCCESS);
    EXPECT_TRUE(result.direct_write_response);
}

TEST_F(TcpServerTest, DeferredLoadHonorsRequestDeadline) {
    auto received  = service_->received.get_future();
    auto completed = load(1000);
    ASSERT_EQ(received.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    ASSERT_NE(received.get(), nullptr);

    ASSERT_EQ(completed.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    const auto result = completed.get();
    EXPECT_TRUE(result.failed);
    EXPECT_EQ(result.error_code, arpc::ARPC_ERROR_TIMEOUT);
    service_->release();
}

}  // namespace rtp_llm
