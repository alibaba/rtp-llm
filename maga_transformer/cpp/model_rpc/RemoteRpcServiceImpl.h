#pragma once
#include <memory>
#include "grpc++/grpc++.h"
#include "maga_transformer/cpp/model_rpc/LocalRpcServiceImpl.h"
#include "maga_transformer/cpp/model_rpc/PrefillRpcServer.h"
#include "maga_transformer/cpp/model_rpc/DecodeRpcServer.h"

namespace rtp_llm {

class RemoteRpcServiceImpl: public LocalRpcServiceImpl {
public:
    RemoteRpcServiceImpl() {}
    ~RemoteRpcServiceImpl() {}
    grpc::Status init(const EngineInitParams& maga_init_params, py::object mm_process_engine,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) override;

    grpc::Status remote_generate(grpc::ServerContext*                                            context,
                                 grpc::ServerReaderWriter<GenerateOutputsPB, GenerateRequestPB>* stream) override {
        return decode_server_->remote_generate(context, stream);
    }

    grpc::Status generate_stream(grpc::ServerContext*                   context,
                                 const GenerateInputPB*                 request,
                                 grpc::ServerWriter<GenerateOutputsPB>* writer) override {
        return prefill_server_->generate_stream(context, request, writer);
    }

    grpc::Status remote_load(grpc::ServerContext* context,
                             const RemoteLoadRequestPB* request, EmptyPB* response) override {
        return decode_server_->remote_load(context, request, response);
    }

    grpc::Status remote_finish(grpc::ServerContext* context,
                               const RemoteFinishRequestPB* request, EmptyPB* response) override {
        return prefill_server_->remote_finish(context, request, response);
    }

    bool ready() override {
        if (prefill_server_) {
            return prefill_server_->ready();
        } else {
            return decode_server_->ready();
        }
    }

private:
    std::shared_ptr<PrefillRpcServer> prefill_server_;
    std::shared_ptr<DecodeRpcServer> decode_server_;
};

}