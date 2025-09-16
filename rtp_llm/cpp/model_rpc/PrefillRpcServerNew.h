#pragma once

#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PrefillGenerateContextNew.h"
#include <atomic>

namespace rtp_llm {

class PrefillRpcServerNew: public RemoteRpcServer {
public:
    PrefillRpcServerNew()  = default;
    ~PrefillRpcServerNew() = default;

public:
    grpc::Status init(const EngineInitParams&                                maga_init_params,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params);

    grpc::Status RemoteGenerateNew(grpc::ServerContext*              context,
                                   const RemoteGenerateRequestPBNew* request,
                                   RemoteGenerateResponsePBNew*      response);

    grpc::Status
    RemoteStore(grpc::ServerContext* context, const RemoteStoreRequestPB* request, RemoteStoreResponsePB* response);

    grpc::Status RemoteFinish(grpc::ServerContext* context, const RemoteFinishRequestPB* request, EmptyPB* response);

private:
    bool validRequest(PrefillGenerateContextNew& prefill_context);

    ErrorInfo notifyStoreCacheForAllRank(PrefillGenerateContextNew& prefill_context);
    ErrorInfo notifyStoreCache(PrefillGenerateContextNew& prefill_context, int index);
    void      constructRemoteLoadRequest(PrefillGenerateContextNew& prefill_context, int index);

    ErrorInfo generateFirstToken(PrefillGenerateContextNew& prefill_context);
    ErrorInfo waitStoreCacheForAllRankDone(PrefillGenerateContextNew& prefill_context);
};

}  // namespace rtp_llm