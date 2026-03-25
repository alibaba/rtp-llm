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
                      py::object                                             mm_process_engine,
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
                           grpc::ServerContext*             server_context,
                           int32_t                          prefill_cp_size = 1):
            request_id(request_id),
            request_key(request_key),
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
        const std::string&               request_key;
        const std::vector<std::string>&  peer_addrs;
        const std::vector<CacheKeyType>& cache_keys;
        const GroupBlockIds&             block_ids_by_group;
        int64_t                          reuse_block_size;
        int64_t                          timeout_ms;
        int                              partition_count;
        int                              partition_id;

        grpc::ServerContext* server_context;
        int32_t              prefill_cp_size;
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

    // CP sharded KV cache helpers
    std::vector<CacheKeyType> recomputeVirtualCacheKeys(GenerateStream* stream, int32_t cp_size) const;
    /// Build RDMA receive descriptors for one layer from one peer.
    /// Data is received into temp_buffer at the peer's slice, not into decode blocks.
    void buildCPShardedLayerCache(std::shared_ptr<RequestBlockBuffer>& load_layer_cache,
                                  const LoadKVCacheContext&            load_context,
                                  size_t                               layer_id,
                                  int                                  peer_index,
                                  size_t                               model_id,
                                  void*                                temp_kv,
                                  size_t                               block_stride_bytes,
                                  void*                                temp_scale,
                                  size_t                               scale_stride_bytes) const;

    /// After RDMA completes, scatter interleaved tokens from per-layer temp buffers
    /// into contiguous decode KV cache blocks.
    void scatterCPTempToDecodeBlocks(const LoadKVCacheContext&     load_context,
                                     const std::vector<BufferPtr>& temp_kv_bufs,
                                     const std::vector<BufferPtr>& temp_scale_bufs,
                                     const std::vector<size_t>&    kv_strides,
                                     const std::vector<size_t>&    scale_strides) const;

private:
    autil::ThreadPoolBasePtr thread_pool_;
    std::atomic<size_t>      onflight_load_cache_requests_{0};
    size_t                   model_id;

    /// Lazy-initialized staging buffer for CP sharded PD transfer.
    /// Allocated on first use, registered for RDMA, reused across requests.
    struct CPStagingBuffer {
        BufferPtr kv_buf;     // [layer_num * max_temp_slots * kv_stride]
        BufferPtr scale_buf;  // [layer_num * max_temp_slots * scale_stride] (may be null)
        size_t    kv_stride      = 0;
        size_t    scale_stride   = 0;
        int       max_temp_slots = 0;  // max vblock_count * cp_size
        size_t    layer_num      = 0;

        /// Get KV address for (layer_id, slot_index) within the staging buffer.
        void* kvAddr(size_t layer_id, int slot_index) const {
            size_t offset = (layer_id * max_temp_slots + slot_index) * kv_stride;
            return static_cast<char*>(kv_buf->data()) + offset;
        }
        /// Get scale address for (layer_id, slot_index), or nullptr if no scale.
        void* scaleAddr(size_t layer_id, int slot_index) const {
            if (!scale_buf || scale_stride == 0)
                return nullptr;
            size_t offset = (layer_id * max_temp_slots + slot_index) * scale_stride;
            return static_cast<char*>(scale_buf->data()) + offset;
        }
    };
    std::mutex                       staging_mutex_;
    std::unique_ptr<CPStagingBuffer> cp_staging_;

    /// Ensure staging buffer is allocated for the given cp_size.
    /// Thread-safe; only allocates once (or re-allocates if cp_size grows).
    void ensureCPStagingBuffer(int cp_size);
};

}  // namespace rtp_llm
