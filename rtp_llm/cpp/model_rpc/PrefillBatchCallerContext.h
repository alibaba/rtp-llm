#pragma once

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include <memory>
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "grpc++/grpc++.h"

namespace rtp_llm {

// Owned and polled by the batch RPC handler. Finish storage outlives cancellation.
class PrefillBatchCallerContext {
public:
    ~PrefillBatchCallerContext();
    bool                 done();
    FirstError::Snapshot firstError() {
        done();
        return first_error_.snapshot();
    }
    void                cancel();
    const grpc::Status& status() const {
        return status_;
    }
    const BatchGenerateOutputsPB& response() const {
        return response_;
    }

private:
    friend class PrefillServerCaller;
    std::string                                                              address_;
    FirstError                                                               first_error_;
    grpc::ClientContext                                                      client_context_;
    grpc::CompletionQueue                                                    queue_;
    std::shared_ptr<RpcService::Stub>                                        stub_;
    BatchGenerateInputPB                                                     request_;
    BatchGenerateOutputsPB                                                   response_;
    grpc::Status                                                             status_;
    std::unique_ptr<grpc::ClientAsyncResponseReader<BatchGenerateOutputsPB>> reader_;
    bool                                                                     started_  = false;
    bool                                                                     finished_ = false;
};

}  // namespace rtp_llm
