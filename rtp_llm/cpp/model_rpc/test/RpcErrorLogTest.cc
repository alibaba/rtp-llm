#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <grpc/grpc.h>
#include <grpc/support/log.h>
#include <gtest/gtest.h>
#include "rtp_llm/cpp/model_rpc/RpcErrorLog.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"

namespace rtp_llm {
namespace {
class RpcErrorLogTest: public ::testing::Test {
protected:
    static std::string logPath() {
        return std::string(std::getenv("TEST_TMPDIR")) + "/rpc-error.log";
    }
    static void SetUpTestSuite() {
        const auto config_path = logPath() + ".conf";
        {
            std::ofstream config(config_path);
            config << "alog.rootLogger=INFO, rpcErrorTest\n"
                   << "alog.appender.rpcErrorTest=FileAppender\n"
                   << "alog.appender.rpcErrorTest.fileName=" << logPath() << "\n"
                   << "alog.appender.rpcErrorTest.async_flush=false\n"
                   << "alog.appender.rpcErrorTest.flush=true\n";
        }
        ASSERT_TRUE(initLogger(config_path));
        grpc_init();
        ASSERT_EQ(gpr_should_log(GPR_LOG_SEVERITY_INFO), 0);
    }
    static void TearDownTestSuite() { grpc_shutdown(); }
    static std::string logs() {
        Logger::getEngineLogger().flush();
        std::ifstream file(logPath());
        return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    }
};

class ErrorService final: public RpcService::Service {
public:
    explicit ErrorService(std::string message): message_(std::move(message)) {}
    grpc::Status RemoteLoad(grpc::ServerContext*, const BroadcastLoadRequestPB*,
                            BroadcastLoadResponsePB*) override {
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, message_);
    }
private:
    std::string message_;
};

// Exercise returned details over a real RPC. The server supplies the text;
// this tests logging, not reproduction of an HTTP/2 GOAWAY or socket race.
void logReturnedError(const std::string& message, const std::string& attempt) {
    ErrorService service(message);
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    auto server = builder.BuildAndStart();
    ASSERT_NE(server, nullptr);
    auto channel = grpc::CreateChannel("127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials());
    auto stub = RpcService::NewStub(channel);
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
    BroadcastLoadRequestPB request;
    BroadcastLoadResponsePB response;
    const auto status = stub->RemoteLoad(&context, request, &response);
    logKvRpcFailure("request-test", attempt, 0, status, context);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
    EXPECT_EQ(status.error_message(), message);
    EXPECT_EQ(gpr_should_log(GPR_LOG_SEVERITY_INFO), 0);
    server->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(2));
    server->Wait();
}

TEST_F(RpcErrorLogTest, ReturnedGoawayDetailsAreLoggedWithNativeInfoDisabled) {
    logReturnedError("GOAWAY received; Error code: 2; Debug Text: enter idle", "goaway-test");
    const auto output = logs();
    EXPECT_NE(output.find("[KV_RPC] RPC_FAILED request=[request-test] attempt=goaway-test worker_index=0 grpc_code=14"),
              std::string::npos);
    EXPECT_NE(output.find("status=[GOAWAY received; Error code: 2; Debug Text: enter idle]"), std::string::npos);
    EXPECT_NE(output.find("grpc_debug=["), std::string::npos);
    EXPECT_NE(output.find("INFO"), std::string::npos);
}

TEST_F(RpcErrorLogTest, ReturnedSocketClosedIsNotRewrittenAsGoaway) {
    logReturnedError("Socket closed", "socket-test");
    const auto output = logs();
    const auto begin = output.find("attempt=socket-test");
    ASSERT_NE(begin, std::string::npos);
    const auto line = output.substr(begin, output.find('\n', begin) - begin);
    EXPECT_NE(line.find("status=[Socket closed]"), std::string::npos);
    EXPECT_EQ(line.find("GOAWAY"), std::string::npos);
}

TEST_F(RpcErrorLogTest, ErrorDetailsAreRedactedAndSingleLine) {
    logReturnedError("OSS_ACCESS_ID=test-id&OSS_ACCESS_KEY=test-secret Bearer test-bearer sk-test-key\nforged-line",
                     "redaction-test");
    const auto output = logs();
    for (const char* secret : {"test-id", "test-secret", "test-bearer", "test-key"}) {
        EXPECT_EQ(output.find(secret), std::string::npos);
    }
    EXPECT_NE(output.find("<redacted> forged-line"), std::string::npos);
    EXPECT_EQ(output.find("\nforged-line"), std::string::npos);
}

TEST_F(RpcErrorLogTest, SuccessfulRpcDoesNotProduceFailureLog) {
    grpc::ClientContext context;
    logKvRpcFailure("request-test", "successful-test", 0, grpc::Status::OK, context);
    EXPECT_EQ(logs().find("successful-test"), std::string::npos);
}

TEST_F(RpcErrorLogTest, LongErrorIsBoundedAndNullIsSafe) {
    EXPECT_EQ(sanitizeRpcError(nullptr), "");
    EXPECT_EQ(sanitizeRpcError(std::string(8192, 'x').c_str()).size(), 4096u);
}
}  // namespace
}  // namespace rtp_llm
