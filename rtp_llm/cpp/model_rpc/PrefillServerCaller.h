#pragma once

#include <functional>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
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
    int                      tp_size = -1;
    std::vector<std::string> dp_addrs;
    int64_t                  cached_at_ms{0};
};

class PrefillServerCaller {
public:
    explicit PrefillServerCaller(const std::string& process_id);

    std::shared_ptr<PrefillServerCallerContext> callPrefill(const GenerateInputPB* request,
                                                            const std::string&     ip,
                                                            uint32_t               port,
                                                            const std::string&     unique_key,
                                                            int64_t                deadline_us);

    grpc::Status callPrefill(grpc::ServerContext*                   server_context,
                             const GenerateInputPB*                 request,
                             grpc::ServerWriter<GenerateOutputsPB>* response_writer);

    PrefillPeerInfo getPrefillPeerInfo(const std::string& ip, uint32_t port, int32_t request_timeout_ms);

    void invalidatePrefillPeerInfo(const std::string& ip, uint32_t port);

    int getPrefillTpSize(const std::string& ip, uint32_t port, int32_t request_timeout_ms);

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
    std::string              process_id_;
    AsyncReaderFactory       async_reader_factory_;

    mutable std::shared_mutex                              prefill_peer_cache_mutex_;
    std::unordered_map<std::string, PrefillPeerInfo>       prefill_peer_cache_;
};

}  // namespace rtp_llm
