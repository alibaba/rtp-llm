#pragma once

#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServer.h"
#include "rtp_llm/cpp/model_rpc/DecodeGenerateContextNew.h"

namespace rtp_llm {

class DecodeRpcServerNew: public RemoteRpcServer {
public:
    DecodeRpcServerNew()  = default;
    ~DecodeRpcServerNew() = default;

public:
    grpc::Status init(const EngineInitParams&                                maga_init_params,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params);

    grpc::Status GenerateStreamCall(grpc::ServerContext*                   server_context,
                                    const GenerateInputPB*                 request,
                                    grpc::ServerWriter<GenerateOutputsPB>* response_writer);

private:
    ErrorInfo loadCacheFromPrefill(DecodeGenerateContextNew& decode_context);
    void      makeRemoteGenerateRequest(DecodeGenerateContextNew& decode_context);
    ErrorInfo callPrefill(DecodeGenerateContextNew& decode_context);

    grpc::Status localGenerate(DecodeGenerateContextNew& decode_context);
    ErrorInfo    writeAppendFirstToken(DecodeGenerateContextNew& decode_context);

private:
    std::string prefill_cluster_name_;
};

}  // namespace rtp_llm