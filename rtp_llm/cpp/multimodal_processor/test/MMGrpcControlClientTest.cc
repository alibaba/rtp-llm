#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/multimodal_processor/transport/grpc/MMGrpcTransport.h"

namespace rtp_llm {
namespace {

// Real loopback gRPC exercises the production control client and generated stub.
class StatusService final: public MultimodalRpcService::Service {
public:
    explicit StatusService(grpc::Status status): status_(std::move(status)) {}

    grpc::Status RemoteMultimodalEmbedding(grpc::ServerContext*,
                                           const MultimodalInputsPB*,
                                           MultimodalOutputPB* response) override {
        if (status_.ok()) {
            response->add_split_size(17);
        }
        return status_;
    }

private:
    grpc::Status status_;
};

class StatusServer {
public:
    explicit StatusServer(grpc::Status status): service_(std::move(status)) {
        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port_);
        builder.RegisterService(&service_);
        server_ = builder.BuildAndStart();
    }

    ~StatusServer() {
        if (server_) {
            server_->Shutdown();
            server_->Wait();
        }
    }

    bool ready() const { return server_ != nullptr && port_ > 0; }
    std::string endpoint() const { return "127.0.0.1:" + std::to_string(port_); }

private:
    StatusService service_;
    int port_ = 0;
    std::unique_ptr<grpc::Server> server_;
};

TEST(MMGrpcControlClientTest, TransportFailuresTakePriorityOverStructuredMessage) {
    const std::string message = "[MM_WRONG_FORMAT_ERROR] structured failure";
    for (auto code : {grpc::StatusCode::UNAVAILABLE, grpc::StatusCode::DEADLINE_EXCEEDED}) {
        SCOPED_TRACE(static_cast<int>(code));
        StatusServer server(grpc::Status(code, message));
        ASSERT_TRUE(server.ready());
        auto client = createGrpcMMControlClient(std::make_shared<MMTransportMetrics>(nullptr), 100);
        DeadlineBudget budget(5000);
        MultimodalInputsPB request;
        auto result = client->request(server.endpoint(), request, budget);
        ASSERT_FALSE(result.ok());
        EXPECT_EQ(result.status().code(), ErrorCode::MM_REMOTE_RPC_FAILED);
        EXPECT_EQ(result.status().ToString(), message);
    }
}

TEST(MMGrpcControlClientTest, OtherStatusPreservesStructuredMessage) {
    for (auto code : {grpc::StatusCode::INVALID_ARGUMENT,
                      grpc::StatusCode::CANCELLED,
                      grpc::StatusCode::RESOURCE_EXHAUSTED}) {
        SCOPED_TRACE(static_cast<int>(code));
        StatusServer server(grpc::Status(code, "[MM_WRONG_FORMAT_ERROR] structured failure"));
        ASSERT_TRUE(server.ready());
        auto client = createGrpcMMControlClient(std::make_shared<MMTransportMetrics>(nullptr), 100);
        DeadlineBudget budget(5000);
        MultimodalInputsPB request;
        auto result = client->request(server.endpoint(), request, budget);
        ASSERT_FALSE(result.ok());
        EXPECT_EQ(result.status().code(), ErrorCode::MM_WRONG_FORMAT_ERROR);
        EXPECT_EQ(result.status().ToString(), "structured failure");
    }
}

TEST(MMGrpcControlClientTest, UnstructuredCancellationCapacityAndUnknownStayDistinct) {
    const std::vector<std::pair<grpc::StatusCode, ErrorCode>> cases = {
        {grpc::StatusCode::CANCELLED, ErrorCode::CANCELLED},
        {grpc::StatusCode::RESOURCE_EXHAUSTED, ErrorCode::CONCURRENCY_LIMIT_ERROR},
        {grpc::StatusCode::INTERNAL, ErrorCode::UNKNOWN_ERROR},
    };
    for (const auto& [code, expected] : cases) {
        SCOPED_TRACE(static_cast<int>(code));
        StatusServer server(grpc::Status(code, "unstructured failure"));
        ASSERT_TRUE(server.ready());
        auto client = createGrpcMMControlClient(std::make_shared<MMTransportMetrics>(nullptr), 100);
        DeadlineBudget budget(5000);
        MultimodalInputsPB request;
        auto result = client->request(server.endpoint(), request, budget);
        ASSERT_FALSE(result.ok());
        EXPECT_EQ(result.status().code(), expected);
        EXPECT_EQ(result.status().ToString(), "unstructured failure");
    }
}

TEST(MMGrpcControlClientTest, SuccessfulReceiptIsUnchanged) {
    StatusServer server(grpc::Status::OK);
    ASSERT_TRUE(server.ready());
    auto client = createGrpcMMControlClient(std::make_shared<MMTransportMetrics>(nullptr), 100);
    DeadlineBudget budget(5000);
    MultimodalInputsPB request;
    auto result = client->request(server.endpoint(), request, budget);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result.value().split_size_size(), 1);
    EXPECT_EQ(result.value().split_size(0), 17);
}

}  // namespace
}  // namespace rtp_llm
