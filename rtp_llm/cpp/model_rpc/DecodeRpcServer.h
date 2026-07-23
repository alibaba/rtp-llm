#pragma once

#include <algorithm>

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
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                      py::object                                             mm_process_engine);

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
                           const std::vector<std::string>&  group_tags,
                           int64_t                          reuse_block_size,
                           int64_t                          timeout_ms,
                           int                              partition_count,
                           int                              partition_id,
                           grpc::ServerContext*             server_context,
                           int32_t                          prefill_cp_size = 1):
            request_id(request_id),
            request_key(request_key),
            peer_addrs(peer_addrs),
            cache_keys(cache_keys),
            block_ids_by_group(block_ids_by_group),
            group_tags(group_tags),
            reuse_block_size(reuse_block_size),
            timeout_ms(timeout_ms),
            partition_count(partition_count),
            partition_id(partition_id),
            server_context(server_context),
            prefill_cp_size(prefill_cp_size) {}

        const std::shared_ptr<BlockIds>& groupBlock(std::string_view tag) const {
            const auto it = std::find(group_tags.begin(), group_tags.end(), tag);
            RTP_LLM_CHECK_WITH_INFO(
                it != group_tags.end(), "cache tag missing from block_ids_by_group: tag=%s", std::string(tag).c_str());
            const size_t group_index = static_cast<size_t>(std::distance(group_tags.begin(), it));
            RTP_LLM_CHECK_WITH_INFO(group_index < block_ids_by_group.size(),
                                    "cache group index=%zu out of range=%zu for tag=%s",
                                    group_index,
                                    block_ids_by_group.size(),
                                    std::string(tag).c_str());
            RTP_LLM_CHECK_WITH_INFO(
                block_ids_by_group[group_index] != nullptr, "null group_block: tag=%s", std::string(tag).c_str());
            return block_ids_by_group[group_index];
        }
        int64_t                          request_id;
        const std::string&               request_key;
        const std::vector<std::string>&  peer_addrs;
        const std::vector<CacheKeyType>& cache_keys;
        const GroupBlockIds&             block_ids_by_group;
        const std::vector<std::string>&  group_tags;
        int64_t                          reuse_block_size;
        int64_t                          timeout_ms;
        int                              partition_count;
        int                              partition_id;

        grpc::ServerContext* server_context;
        int32_t              prefill_cp_size;
    };

private:
    struct MTPModuleLoadPlan {
        size_t                  module_index;
        const EngineInitParams* engine_init_params;
        size_t                  cache_model_id;
    };

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
    static GroupBlockIds   decodeGroupBlockIds(const BroadcastLoadRequestPB& request, const CacheTopology& topology);
    static std::string     makeTaggedRequestKey(int64_t request_id, size_t layer_id, const std::string& tag);
    static std::string
    makeMTPModuleCacheKey(size_t mtp_base_model_id, const std::string& token_id_str, size_t layer_id);
    static std::vector<MTPModuleLoadPlan> makeMTPModuleLoadPlan(const ProposeModelEngineInitParams* propose_params);
    static void                           logReadFailures(int64_t                         request_id,
                                                          const std::string&              peer_addr,
                                                          ErrorCode                       error_code,
                                                          const std::string&              error_message,
                                                          const std::vector<std::string>& buffer_debug_infos);

private:
    autil::ThreadPoolBasePtr thread_pool_;
    std::atomic<size_t>      onflight_load_cache_requests_{0};
    size_t                   model_id;
};

}  // namespace rtp_llm
