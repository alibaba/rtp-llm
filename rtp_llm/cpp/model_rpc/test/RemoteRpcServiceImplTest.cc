#include "gtest/gtest.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServiceImpl.h"

namespace rtp_llm {

TEST(RemoteRpcServiceImplTest, RetiredPDMethodsReturnUnimplemented) {
    RemoteRpcServiceImpl service;
    grpc::ServerContext  context;
    EXPECT_EQ(service.RemoteGenerate(&context, nullptr).error_code(), grpc::StatusCode::UNIMPLEMENTED);
    EXPECT_EQ(service.RemoteLoad(&context, nullptr, nullptr).error_code(), grpc::StatusCode::UNIMPLEMENTED);
    EXPECT_EQ(service.RemoteFinish(&context, nullptr, nullptr).error_code(), grpc::StatusCode::UNIMPLEMENTED);
    EXPECT_EQ(service.EnqueueGroup(&context, nullptr, nullptr).error_code(), grpc::StatusCode::UNIMPLEMENTED);
    EXPECT_EQ(service.FetchResponse(&context, nullptr, nullptr).error_code(), grpc::StatusCode::UNIMPLEMENTED);
    EXPECT_EQ(service.Cancel(&context, nullptr, nullptr).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(RemoteRpcServiceImplTest, RejectsUnsupportedPDRoleBeforeEngineInitialization) {
    RemoteRpcServiceImpl service;
    EngineInitParams     params;
    params.pd_sep_config.role_type = RoleType::PDFUSION;
    EXPECT_EQ(service.init(params, nullptr, py::object{}).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

}  // namespace rtp_llm

namespace rtp_llm {

TEST(RemoteRpcServiceImplTest, CancelIsImplementedButRequiresAnInitializedPDRole) {
    RemoteRpcServiceImpl service;
    grpc::ServerContext  context;
    CancelRequestPB      request;
    request.set_request_id(1);
    CancelResponsePB response;
    EXPECT_EQ(service.Cancel(&context, &request, &response).error_code(), grpc::StatusCode::UNAVAILABLE);
}

}  // namespace rtp_llm

namespace rtp_llm {
namespace {
class DispatchTestServer: public LocalRpcServer {
public:
    grpc::Status
    GenerateStreamCall(grpc::ServerContext*, const GenerateInputPB*, grpc::ServerWriter<GenerateOutputsPB>*) override {
        return grpc::Status(grpc::StatusCode::ABORTED, "selected role handler");
    }
};
}  // namespace

TEST(RemoteRpcServiceImplTest, GenerateUsesSelectedLocalServerVirtualDispatch) {
    RemoteRpcServiceImpl service;
    service.local_server_ = std::make_shared<DispatchTestServer>();
    grpc::ServerContext context;
    GenerateInputPB     request;
    const auto          status = service.GenerateStreamCall(&context, &request, nullptr);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::ABORTED);
    EXPECT_EQ(status.error_message(), "selected role handler");
}
}  // namespace rtp_llm
