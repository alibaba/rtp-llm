#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/model_rpc/RpcServerRuntimeMeta.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PrefillGenerateContext.h"

namespace rtp_llm {

enum class PriorityCancelResult : uint8_t {
    ACCEPTED,
    TOMBSTONED,
    NOT_FOUND,
};

// Prefill-side gRPC server for PD (prefill/decode) separation — the single-request path.
//
//   GenerateStreamCall
//     → syncPrefix    (prepareAllocateResource with retry + enqueueRequest)
//     → finishStream  (remoteLoadCacheStart → pollLocalOutput → remoteLoadCacheEnd
//                       → remoteGenerate → pollRemoteOutput)
//
// The batch-enqueue path (EnqueueBatch / EnqueueGroup / FetchResponse and the thread pools,
// response registry and pool metrics behind it) lives entirely in the derived PrefillBatchRpcServer,
// which reuses finishStream / prepareAllocateResource from this class. This base is never mutated by
// the batch path, keeping the single-request behavior isolated.
class PrefillRpcServer: public RemoteRpcServer {
public:
    PrefillRpcServer() {}
    ~PrefillRpcServer() override;
    grpc::Status init(const EngineInitParams&                                maga_init_params,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                      py::object                                             mm_process_engine) override;

    grpc::Status GenerateStreamCall(grpc::ServerContext*                   context,
                                    const GenerateInputPB*                 request,
                                    grpc::ServerWriter<GenerateOutputsPB>* writer);

    grpc::Status RemoteFinish(grpc::ServerContext* context, const RemoteFinishRequestPB* request, EmptyPB* response);

    // AutoTPM Cancel targets an active batch request. ACCEPTED is a weak ACK:
    // the priority first-cause latch is installed and P-to-D cancellation is
    // triggered, while completion is reported later through WorkerStatus.
    grpc::Status Cancel(grpc::ServerContext* context, const CancelRequestPB* request, CancelResponsePB* response);

protected:
    // Shared with the derived batch server (each batch slot reuses these).
    grpc::Status prepareAllocateResource(PrefillGenerateContext& prefill_context);
    grpc::Status finishStream(PrefillGenerateContext& prefill_context);
    grpc::Status preferPriorityPreemption(PrefillGenerateContext& prefill_context, const grpc::Status& fallback);
    void         setContextError(PrefillGenerateContext& prefill_context, const ErrorInfo& error_info);
    void         setContextError(PrefillGenerateContext& prefill_context,
                                 const ErrorInfo&        error_info,
                                 const grpc::Status&     error_status);
    virtual PriorityCancelResult onCancelRequest(int64_t request_id) {
        return PriorityCancelResult::NOT_FOUND;
    }

private:
    grpc::Status      syncPrefix(PrefillGenerateContext& prefill_context);
    ErrorInfo         waitStreamBeforeRun(std::shared_ptr<GenerateStream> stream);
    void              prepareGenerateInput(PrefillGenerateContext& prefill_context);
    void              getRpcConnection(PrefillGenerateContext& prefill_context);
    void              multimodalProcess(PrefillGenerateContext& prefill_context);
    void              remoteAllocateResource(PrefillGenerateContext& prefill_context);
    GenerateRequestPB buildAllocateRequest(PrefillGenerateContext& prefill_context);
    void              enqueueRequest(PrefillGenerateContext& prefill_context);
    void              remoteLoadCacheStart(PrefillGenerateContext& prefill_context);
    void              pollLocalOutput(PrefillGenerateContext& prefill_context);
    void              remoteLoadCacheEnd(PrefillGenerateContext& prefill_context);
    void              remoteGenerate(PrefillGenerateContext& prefill_context);
    void              pollRemoteOutput(PrefillGenerateContext& prefill_context);
    static void       mergeMultimodalLengths(GenerateOutputsPB& response, const std::map<int, int>& multimodal_lengths);
    static void       mergeCacheReuseInfo(AuxInfoPB& aux_info,
                                          int        prefill_total_reuse_len,
                                          int        prefill_local_reuse_len,
                                          int        prefill_remote_reuse_len,
                                          int        prefill_memory_reuse_len,
                                          int        prefill_disk_reuse_len,
                                          bool       use_independent_block_pools);

private:
    std::string decode_cluster_name_;
};

}  // namespace rtp_llm
