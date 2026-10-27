#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcCallerContext.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"

namespace rtp_llm {

namespace test {
class DecodeRpcCallerTest;
}

struct PrefillPeerInfo {
    int                               tp_size = -1;
    int                               cp_size = -1;
    std::optional<P2PPreparedInputPB> prepared_input;
};

class DecodeRpcCaller {
public:
    DecodeRpcCaller();

    ErrorResult<std::shared_ptr<DecodeRpcCallerContext>>
    startPrefill(const GenerateInputPB*                                              request,
                 const std::string&                                                  ip,
                 uint32_t                                                            port,
                 const std::string&                                                  unique_key,
                 int64_t                                                             request_deadline_ms,
                 const opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>& request_span);

    grpc::Status forwardGenerate(grpc::ServerContext*                   server_context,
                                 const GenerateInputPB*                 request,
                                 grpc::ServerWriter<GenerateOutputsPB>* response_writer);

    grpc::Status enqueueBatch(const EnqueueBatchRequestPB& request,
                              EnqueueBatchResponsePB*      response,
                              const std::string&           prefill_address,
                              int64_t                      deadline_ms,
                              const std::function<bool()>& is_cancelled);

    grpc::Status forwardBatchGenerate(const BatchGenerateInputPB&  request,
                                      BatchGenerateOutputsPB*      response,
                                      const std::string&           prefill_address,
                                      int64_t                      deadline_ms,
                                      const std::function<bool()>& is_cancelled);

    // Fetch once for each PD request; only the RPC connection is reused.
    ErrorResult<PrefillPeerInfo> getPrefillPeerInfo(const std::string&           ip,
                                                    uint32_t                     port,
                                                    int32_t                      request_timeout_ms,
                                                    const GenerateInputPB&       input,
                                                    const std::function<bool()>& is_cancelled);

private:
    grpc::Status forwardGenerate(grpc::ServerContext*                   server_context,
                                 const GenerateInputPB*                 request,
                                 grpc::ServerWriter<GenerateOutputsPB>* response_writer,
                                 const std::function<bool()>&           is_cancelled);

    grpc::Status forwardGenerateToAddr(grpc::ServerContext*                   server_context,
                                       const GenerateInputPB*                 request,
                                       grpc::ServerWriter<GenerateOutputsPB>* response_writer,
                                       const std::string&                     prefill_addr,
                                       const std::function<bool()>&           is_cancelled);

private:
    friend class test::DecodeRpcCallerTest;

    using AsyncReaderFactory = std::function<std::unique_ptr<grpc::ClientAsyncReader<GenerateOutputsPB>>(
        const std::shared_ptr<RpcService::Stub>&,
        grpc::ClientContext*,
        const GenerateInputPB&,
        grpc::CompletionQueue*)>;

    std::shared_ptr<RPCPool> rpc_pool_;
    AsyncReaderFactory       async_reader_factory_;
};

}  // namespace rtp_llm
