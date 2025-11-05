#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store_new/LayerCacheBuffer.h"
#include "rtp_llm/cpp/disaggregate/cache_store_new/proto/service.pb.h"
#include "rtp_llm/cpp/disaggregate/cache_store_new/LoadContext.h"
#include "rtp_llm/cpp/disaggregate/cache_store_new/TcpClient.h"

namespace rtp_llm {

class CacheStoreClientClosure: public google::protobuf::Closure {
public:
    CacheStoreClientClosure(const std::shared_ptr<CacheLoadRequest>&  cache_load_request,
                            const std::shared_ptr<CacheLoadResponse>& cache_load_response,
                            arpc::ANetRPCController*                  controller,
                            const std::shared_ptr<LoadContext>&       load_context);
    ~CacheStoreClientClosure() {
        if (controller_) {
            delete controller_;
        }
    };

public:
    void Run() override;

private:
    std::shared_ptr<CacheLoadRequest>  cache_load_request_;
    std::shared_ptr<CacheLoadResponse> cache_load_response_;
    arpc::ANetRPCController*           controller_ = nullptr;
    std::shared_ptr<LoadContext>       load_context_;
};

class CacheStoreClient {
public:
    CacheStoreClient(const std::shared_ptr<TcpClient>& tcp_client, int tp_size, int tp_rank);
    ~CacheStoreClient();

public:
    std::shared_ptr<LoadContext> asyncLoad(const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                                           int64_t                                               timeout_ms,
                                           const std::string&                                    ip,
                                           uint32_t                                              port,
                                           int                                                   tp_size,
                                           int                                                   tp_rank);

private:
    int64_t generateContextId();
    bool    generateCacheLoadRequest(const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                                     int64_t                                               deadline_ms,
                                     int64_t                                               context_id,
                                     const std::shared_ptr<CacheLoadRequest>&              cache_load_request);

private:
    std::shared_ptr<TcpClient> tcp_client_;
    int                        tp_size_;
    int                        tp_rank_;
    std::string                ip_;
    uint32_t                   port_;

    std::mutex                                                load_context_map_mutex_;
    std::unordered_map<int64_t, std::shared_ptr<LoadContext>> load_context_map_;
};

}  // namespace rtp_llm