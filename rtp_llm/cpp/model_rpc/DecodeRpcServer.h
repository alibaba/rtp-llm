#pragma once

#include <atomic>
#include <map>
#include <utility>

#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServer.h"
#include "rtp_llm/cpp/model_rpc/DecodeGenerateContext.h"
#include "rtp_llm/cpp/cache/BufferTypes.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"

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
        LoadKVCacheContext(int64_t                                request_id,
                           std::string                            request_key,
                           const std::vector<std::string>&        peer_addrs,
                           const std::vector<CacheKeyType>&       cache_keys,
                           const std::map<std::string, BlockIds>& block_ids_by_group,
                           int64_t                                reuse_block_size,
                           int64_t                                timeout_ms,
                           int                                    partition_count,
                           int                                    partition_id,
                           grpc::ServerContext*                   server_context,
                           int32_t                                prefill_cp_size = 1):
            request_id(request_id),
            request_key(std::move(request_key)),
            peer_addrs(peer_addrs),
            cache_keys(cache_keys),
            block_ids_by_group(block_ids_by_group),
            reuse_block_size(reuse_block_size),
            timeout_ms(timeout_ms),
            partition_count(partition_count),
            partition_id(partition_id),
            server_context(server_context),
            prefill_cp_size(prefill_cp_size) {}
        int64_t                          request_id;
        std::string                      request_key;
        const std::vector<std::string>&  peer_addrs;
        const std::vector<CacheKeyType>& cache_keys;
        // Tag-bearing cache group records; the record order is not identity.
        const std::map<std::string, BlockIds>& block_ids_by_group;
        int64_t                                reuse_block_size;
        int64_t                                timeout_ms;
        int                                    partition_count;
        int                                    partition_id;

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
    // Report a terminal early failure to FlexLB via meta_->finishTask(); guaranteed at most once per
    // request. MUST NOT be called inside functions driven by EXECUTE_WITH_RETRY (would report while
    // retries could still succeed); only call at final failure points.
    void
    reportEarlyFinishTask(DecodeGenerateContext& decode_context, int64_t error_code, const std::string& error_message);

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
    static ErrorInfo       decodeGroupBlockIds(const BroadcastLoadRequestPB&    request,
                                               const CacheConfig&               cache_config,
                                               std::map<std::string, BlockIds>& block_ids_by_group);
    static ErrorInfo       validateGroupBlockIdsGeometry(const CacheConfig&                     cache_config,
                                                         const std::map<std::string, BlockIds>& block_ids_by_group,
                                                         size_t                                 cache_key_count,
                                                         int64_t                                reuse_block_size,
                                                         int32_t                                prefill_cp_size);
    static std::string     makeRequestKeyForGroup(int64_t request_id, size_t layer_id, const std::string& tag);
    static std::string
    makeMTPModuleCacheKey(size_t mtp_base_model_id, const std::string& token_id_str, size_t layer_id);
    static std::vector<MTPModuleLoadPlan> makeMTPModuleLoadPlan(const ProposeModelEngineInitParams* propose_params);
    // Projects the producer's buildCacheStorePlan onto one group's decode block
    // table: key_index indexes the global cache keys, offset_index the group-local
    // BlockIds::blocks(). Derived from group policy plus block geometry only, never
    // from how many prefill peers answered this load. Static so it is testable.
    static std::vector<CacheStoreBlockPair> buildGroupLoadPlan(const CacheGroupPolicy& policy,
                                                               size_t                  local_block_num,
                                                               size_t                  cache_key_count,
                                                               size_t                  reuse_block_size,
                                                               bool                    use_hybrid,
                                                               size_t                  group_seq_size_per_block,
                                                               size_t                  base_seq_size_per_block);
    static grpc::Status                     generateRequestReadFailureStatus(bool cancelled);
    // Classifies error.type for the synthesized Decode phase spans. Static and
    // side-effect free so the classification itself is unit testable.
    static const char* phaseErrorType(bool                         request_ok,
                                      DecodeStatInfo::ExecuteStage stage,
                                      const ErrorInfo&             error_info,
                                      const grpc::Status&          error_status);
    static void        logReadFailures(int64_t                         request_id,
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
