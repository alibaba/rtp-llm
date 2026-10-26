#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/model_rpc/PrefillServerCallerContext.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"

namespace rtp_llm {

namespace test {
class PrefillServerCallerTest;
}

struct PrefillPeerInfo {
    int                               tp_size = -1;
    int                               cp_size = -1;
    std::optional<P2PPreparedInputPB> prepared_input;
};

class PrefillServerCaller {
public:
    PrefillServerCaller();

    ErrorResult<std::shared_ptr<PrefillServerCallerContext>>
    callPrefill(const GenerateInputPB*                                              request,
                const std::string&                                                  ip,
                uint32_t                                                            port,
                const std::string&                                                  unique_key,
                int64_t                                                             request_deadline_ms,
                const opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>& request_span);

    grpc::Status callPrefill(grpc::ServerContext*                   server_context,
                             const GenerateInputPB*                 request,
                             grpc::ServerWriter<GenerateOutputsPB>* response_writer);

    // Fetch once for each PD request; only the RPC connection is reused.
    ErrorResult<PrefillPeerInfo> getPrefillPeerInfo(const std::string&           ip,
                                                    uint32_t                     port,
                                                    int32_t                      request_timeout_ms,
                                                    const GenerateInputPB&       input,
                                                    const std::function<bool()>& is_cancelled);

private:
    grpc::Status callPrefill(grpc::ServerContext*                   server_context,
                             const GenerateInputPB*                 request,
                             grpc::ServerWriter<GenerateOutputsPB>* response_writer,
                             const std::function<bool()>&           is_cancelled);

    grpc::Status callPrefillToAddr(grpc::ServerContext*                   server_context,
                                   const GenerateInputPB*                 request,
                                   grpc::ServerWriter<GenerateOutputsPB>* response_writer,
                                   const std::string&                     prefill_addr,
                                   const std::function<bool()>&           is_cancelled);

private:
    friend class test::PrefillServerCallerTest;

    using AsyncReaderFactory = std::function<std::unique_ptr<grpc::ClientAsyncReader<GenerateOutputsPB>>(
        const std::shared_ptr<RpcService::Stub>&,
        grpc::ClientContext*,
        const GenerateInputPB&,
        grpc::CompletionQueue*)>;

    std::shared_ptr<RPCPool> rpc_pool_;
    AsyncReaderFactory       async_reader_factory_;
};

}  // namespace rtp_llm
