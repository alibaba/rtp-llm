#pragma once

#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PrefillGenerateContextNew.h"

namespace rtp_llm {

class PrefillRpcServerNew2: public RemoteRpcServer {
public:
    PrefillRpcServerNew2() {}
    ~PrefillRpcServerNew2() {}

    grpc::Status init(const EngineInitParams&                                maga_init_params,
                      py::object                                             mm_process_engine,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) override;

    grpc::Status GenerateStreamCall(grpc::ServerContext*                   context,
                                    const GenerateInputPB*                 request,
                                    grpc::ServerWriter<GenerateOutputsPB>* writer);

    ::grpc::Status StartLoad(::grpc::ServerContext*                context,
                             const P2PConnectorStartLoadRequestPB* request,
                             P2PConnectorStartLoadResponsePB*      response);
};

}  // namespace rtp_llm