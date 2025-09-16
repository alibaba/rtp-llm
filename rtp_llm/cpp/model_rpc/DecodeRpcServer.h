#pragma once

#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServer.h"
#include "rtp_llm/cpp/model_rpc/DecodeGenerateContext.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"

namespace rtp_llm {

class DecodeRpcServer: public RemoteRpcServer {
public:
    DecodeRpcServer() {}
    ~DecodeRpcServer();
    grpc::Status init(const EngineInitParams&                                maga_init_params,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params);

    grpc::Status RemoteGenerate(grpc::ServerContext* server_context, ServerStream* stream);

    grpc::Status RemoteLoad(grpc::ServerContext*          server_context,
                            const BroadcastLoadRequestPB* request,
                            BroadcastLoadResponsePB*      response);

    class LoadKVCacheContext {
    public:
        LoadKVCacheContext(int64_t                          request_id,
                           const std::string&               request_key,
                           const std::vector<std::string>&  peer_addrs,
                           const std::vector<CacheKeyType>& cache_keys,
                           const GroupBlockIds&             block_ids_by_group,
                           int64_t                          reuse_block_size,
                           int64_t                          timeout_ms,
                           int                              partition_count,
                           int                              partition_id,
                           grpc::ServerContext*             server_context):
            request_id(request_id),
            request_key(request_key),
            peer_addrs(peer_addrs),
            cache_keys(cache_keys),
            block_ids_by_group(block_ids_by_group),
            reuse_block_size(reuse_block_size),
            timeout_ms(timeout_ms),
            partition_count(partition_count),
            partition_id(partition_id),
            server_context(server_context) {}
        int64_t                          request_id;
        const std::string&               request_key;
        const std::vector<std::string>&  peer_addrs;
        const std::vector<CacheKeyType>& cache_keys;
        const GroupBlockIds&             block_ids_by_group;
        int64_t                          reuse_block_size;
        int64_t                          timeout_ms;
        int                              partition_count;
        int                              partition_id;

        grpc::ServerContext* server_context;
    };

private:
    void         initThreadPool();
    void         prepareGenerateContext(DecodeGenerateContext& decode_context);
    void         allocateResource(DecodeGenerateContext& decode_context);
    grpc::Status allocateResourceFunc(DecodeGenerateContext& decode_context);
    void         loadCacheFromPrefill(DecodeGenerateContext& decode_context);
    void         localGenerate(DecodeGenerateContext& decode_context);

    ErrorInfo              loadCache(const LoadKVCacheContext& load_context);
    ErrorInfo              loadCacheForAllRank(DecodeGenerateContext& decode_context);
    ErrorInfo              loadCacheAsyncForTp(DecodeGenerateContext& decode_context, LoadKVCacheContext& load_context);
    ErrorInfo              loadCacheSyncForTp(DecodeGenerateContext& decode_context, LoadKVCacheContext& load_context);
    BroadcastLoadRequestPB constructRemoteLoadRequest(const LoadKVCacheContext&       load_context,
                                                      int                             index,
                                                      const std::vector<std::string>& peer_ips) const;
    BroadcastLoadRequestPB constructRemoteLoadRequestForMla(const LoadKVCacheContext&       load_context,
                                                            int                             index,
                                                            const std::vector<std::string>& peer_ips) const;

private:
    autil::ThreadPoolBasePtr thread_pool_;
    std::atomic<size_t>      onflight_load_cache_requests_{0};
    size_t                   model_id;
};

}  // namespace rtp_llm
