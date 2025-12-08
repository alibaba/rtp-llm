#pragma once

#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PrefillServerCaller.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorDecode.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

class DecodeRpcServerNew2: public RemoteRpcServer {
public:
    DecodeRpcServerNew2()  = default;
    ~DecodeRpcServerNew2() = default;

public:
    grpc::Status init(const EngineInitParams&                                maga_init_params,
                      py::object                                             mm_process_engine,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params);

    grpc::Status GenerateStreamCall(grpc::ServerContext*                   server_context,
                                    const GenerateInputPB*                 request,
                                    grpc::ServerWriter<GenerateOutputsPB>* response_writer);

private:
    RoleAddr                                    getPrefillRoleAddr(const GenerateInputPB* request);
    std::shared_ptr<GenerateStream>             initStream(const GenerateInputPB* request);
    std::shared_ptr<PrefillServerCallerContext> callPrefill(const GenerateInputPB* request,
                                                            const std::string&     prefill_ip,
                                                            uint32_t               prefill_port,
                                                            const std::string&     unique_key,
                                                            int64_t                deadline_us);
    bool                                        loadKVCacheFromPrefill(const std::shared_ptr<GenerateStream>& stream,
                                                                       const std::string&                     unique_key,
                                                                       const std::string&                     prefill_ip,
                                                                       uint32_t                               prefill_port);
    ErrorInfo localGenerate(const std::shared_ptr<PrefillServerCallerContext>& prefill_rpc_context,
                            std::shared_ptr<GenerateStream>&                   stream,
                            grpc::ServerContext*                               server_context,
                            grpc::ServerWriter<GenerateOutputsPB>*             response_writer);

private:
    std::shared_ptr<P2PConnectorDecode>  p2p_connector_decode_;
    std::shared_ptr<PrefillServerCaller> prefill_server_caller_;
};

}  // namespace rtp_llm
