#pragma once

#include "grpc++/grpc++.h"
#include "maga_transformer/cpp/model_rpc/RemoteRpcServer.h"
#include "maga_transformer/cpp/model_rpc/DecodeGenerateContext.h"

namespace rtp_llm {

class DecodeRpcServer: public RemoteRpcServer {
public:
    DecodeRpcServer() {}
    ~DecodeRpcServer() {}
    grpc::Status init(const EngineInitParams& maga_init_params, py::object mm_process_engine,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params);
    
    grpc::Status RemoteGenerate(grpc::ServerContext* server_context, ServerStream* stream);

    grpc::Status RemoteLoad(grpc::ServerContext* server_context,
                            const BroadcastLoadRequestPB* request, BroadcastLoadResponsePB* response);

    bool ready() {
        return true;
    }

    class LoadKVCacheContext {
    public:
        LoadKVCacheContext(int64_t request_id, const std::string& request_key, const std::string& peer_ip,
                            const std::vector<int64_t>& cache_keys, const std::vector<int32_t>& block_ids,
                            int64_t reuse_block_size, int64_t timeout_ms, grpc::ServerContext* server_context) :
                            request_id(request_id), request_key(request_key), peer_ip(peer_ip),
                            cache_keys(cache_keys), block_ids(block_ids), reuse_block_size(reuse_block_size),
                            timeout_ms(timeout_ms), server_context(server_context) {}
        int64_t request_id;
        const std::string& request_key;
        const std::string& peer_ip;
        const std::vector<int64_t>& cache_keys;
        const std::vector<int32_t>& block_ids;
        int64_t reuse_block_size;
        int64_t timeout_ms;

        grpc::ServerContext* server_context;
    };

private:
    void prepareGenerateContext(DecodeGenerateContext& decode_context);
    void allocateResource(DecodeGenerateContext& decode_context);
    grpc::Status allocateResourceFunc(DecodeGenerateContext& decode_context);
    void loadCacheFromPrefill(DecodeGenerateContext& decode_context);
    void localGenerate(DecodeGenerateContext& decode_context);
    void writeTime(DecodeGenerateContext& decode_context);

    ErrorInfo loadCache(const LoadKVCacheContext& load_context);
    ErrorInfo loadCacheForAllRank(DecodeGenerateContext& decode_context);
    BroadcastLoadRequestPB constructRemoteLoadRequest(const LoadKVCacheContext& load_context) const;

private:
    autil::ThreadPoolBasePtr thread_pool_;
    std::atomic<size_t> onflight_load_cache_requests_{0};
};

}
