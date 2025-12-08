#pragma once

#include "rtp_llm/cpp/model_rpc/RemoteRpcServer.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorPrefill.h"

namespace rtp_llm {

class PrefillRpcServerNew2: public RemoteRpcServer {
public:
    PrefillRpcServerNew2()           = default;
    ~PrefillRpcServerNew2() override = default;

public:
    grpc::Status init(const EngineInitParams&                                maga_init_params,
                      py::object                                             mm_process_engine,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params);

    virtual grpc::Status GenerateStreamCall(grpc::ServerContext*                   server_context,
                                            const GenerateInputPB*                 request,
                                            grpc::ServerWriter<GenerateOutputsPB>* response_writer);

    virtual grpc::Status StartLoad(grpc::ServerContext*                  context,
                                   const P2PConnectorStartLoadRequestPB* request,
                                   P2PConnectorStartLoadResponsePB*      response);

private:
    std::shared_ptr<GenerateStream> initStream(const GenerateInputPB* request);

private:
    std::shared_ptr<P2PConnectorPrefill> p2p_connector_prefill_;
};
}  // namespace rtp_llm