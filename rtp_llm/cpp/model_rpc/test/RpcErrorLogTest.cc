#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <grpc/grpc.h>
#include <grpc/support/log.h>
#include <gtest/gtest.h>
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"

namespace rtp_llm {
namespace {
class TestService final: public RpcService::Service {
public:
    grpc::Status status;
    grpc::Status ExecuteFunction(grpc::ServerContext*, const FunctionRequestPB*, FunctionResponsePB*) override {
        return status;
    }
};

class RpcErrorLogTest: public ::testing::Test {
protected:
    static std::string logPath() {
        return std::string(std::getenv("TEST_TMPDIR")) + "/rpc-error.log";
    }
    static void SetUpTestSuite() {
        const auto config_path = logPath() + ".conf";
        {
            std::ofstream config(config_path);
            config << "alog.max_msg_len=16384\n"
                   << "alog.rootLogger=INFO, rpcErrorTest\n"
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

    void broadcast(const grpc::Status& status) {
        TestService service;
        service.status = status;
        grpc::ServerBuilder builder;
        int port = 0;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(&service);
        auto server = builder.BuildAndStart();
        ASSERT_NE(server, nullptr);
        BroadcastManager manager({"127.0.0.1:" + std::to_string(port)});
        ASSERT_TRUE(manager.init());

        FunctionRequestPB request;
        auto*             mem = request.mutable_mem_request();
        mem->set_transfer_trace_id(31);
        mem->set_has_transfer_request_id(true);
        mem->set_transfer_request_id(42);
        auto call = [](const auto& stub, const auto& context, const auto& payload, grpc::CompletionQueue* queue) {
            return stub->AsyncExecuteFunction(context.get(), payload, queue);
        };
        const auto offset = logs().size();
        auto       result = manager.broadcast<FunctionRequestPB, FunctionResponsePB>(request, 5000, call, 84, "REMOTE2D");
        ASSERT_NE(result, nullptr);
        ASSERT_TRUE(result->waitDone(10000, /*fatal_on_error=*/false));
        EXPECT_EQ(result->success(), status.ok());
        const auto ranks = result->rankSnapshots();
        ASSERT_EQ(ranks.size(), 1u);
        EXPECT_EQ(ranks[0].grpc_code.value_or(-1), status.error_code());
        EXPECT_EQ(ranks[0].grpc_message, status.error_message());

        output            = logs().substr(offset);
        const auto prefix = "transfer_lifecycle event=rpc_cq_received trace=31 request=42 rank=0 ";
        const auto begin  = output.find(prefix);
        ASSERT_NE(begin, std::string::npos);
        EXPECT_EQ(output.find(prefix, begin + 1), std::string::npos);
        completion = output.substr(begin, output.find('\n', begin) - begin);
    }

    std::string output;
    std::string completion;
};

TEST_F(RpcErrorLogTest, FailureDetailsShareTheTransferCompletionRecord) {
    for (const char* message : {"GOAWAY received; Error code: 2; Debug Text: enter idle", "Socket closed"}) {
        SCOPED_TRACE(message);
        ASSERT_NO_FATAL_FAILURE(broadcast(grpc::Status(grpc::StatusCode::UNAVAILABLE, message)));
        EXPECT_NE(completion.find("cq_received=1 cq_ok=1 grpc_code=14"), std::string::npos);
        EXPECT_NE(completion.find(std::string("grpc_message=\"") + message + "\""), std::string::npos);
        EXPECT_NE(completion.find("request_id=84 direction=REMOTE2D"), std::string::npos);
        EXPECT_NE(completion.find("grpc_debug=["), std::string::npos);
        EXPECT_EQ(completion.find("grpc_debug=[]"), std::string::npos);
        EXPECT_EQ(output.find("[KV_RPC] RPC_FAILED"), std::string::npos);
    }
}

TEST_F(RpcErrorLogTest, ErrorTextIsRedactedInCompletionAndWarning) {
    const auto message = "OSS_ACCESS_ID=test-id&OSS_ACCESS_KEY=test-secret Bearer test-bearer sk-test-key\nforged-line";
    ASSERT_NO_FATAL_FAILURE(broadcast(grpc::Status(grpc::StatusCode::UNAVAILABLE, message)));
    EXPECT_NE(completion.find("<redacted> forged-line"), std::string::npos);
    EXPECT_NE(output.find("broadcast rpc failed, rank=0"), std::string::npos);
    for (const char* secret : {"test-id", "test-secret", "test-bearer", "test-key", "\nforged-line"}) {
        EXPECT_EQ(output.find(secret), std::string::npos) << secret;
    }
}

TEST_F(RpcErrorLogTest, SuccessfulCompletionOmitsFailureDetails) {
    ASSERT_NO_FATAL_FAILURE(broadcast(grpc::Status::OK));
    EXPECT_NE(completion.find("cq_received=1 cq_ok=1 grpc_code=0"), std::string::npos);
    EXPECT_EQ(output.find("grpc_debug="), std::string::npos);
}

TEST_F(RpcErrorLogTest, LongPeerMessageIsBoundedInLogs) {
    ASSERT_NO_FATAL_FAILURE(broadcast(grpc::Status(grpc::StatusCode::UNAVAILABLE,
                                                   std::string(4096, 'x') + "beyond-log-limit")));
    EXPECT_NE(completion.find("grpc_message=\"" + std::string(4096, 'x') + "\""), std::string::npos);
    EXPECT_EQ(output.find("beyond-log-limit"), std::string::npos);
}

}  // namespace
}  // namespace rtp_llm
