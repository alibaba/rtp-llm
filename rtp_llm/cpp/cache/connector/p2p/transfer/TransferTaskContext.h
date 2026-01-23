#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/transfer/LayerCacheBufferTask.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/LayerBlockConvertor.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferMetric.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/proto/service.pb.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/RdmaInterface.h"
#include <memory>

namespace rtp_llm {

class TransferTaskContext {

public:
    TransferTaskContext(::google::protobuf::RpcController*           controller,
                        const ::transfer::LayerBlockTransferRequest* request,
                        ::transfer::LayerBlockTransferResponse*      response,
                        ::google::protobuf::Closure*                 done,
                        const std::shared_ptr<LayerBlockConvertor>&  layer_block_convector,
                        const kmonitor::MetricsReporterPtr&          metrics_reporter);
    ~TransferTaskContext();

public:
    void                                         addTask(const std::shared_ptr<LayerCacheBufferTask>& task);
    const std::shared_ptr<LayerCacheBufferTask>& getTask() const;

    std::string                                                                     getUniqueKey() const;
    std::vector<std::pair<BufferPtr, std::shared_ptr<::transfer::BlockBufferInfo>>> getTcpBlockPair();
    std::vector<std::pair<BufferPtr, std::shared_ptr<RemoteBuffer>>>                getRdmaBlockPair();
    std::pair<std::string, uint32_t>                                                getServerRdmaInfo() const;
    bool                                                                            isTimeout() const;
    uint64_t                                                                        getDeadlineMs() const;

    void run(bool success, const std::string& info);

private:
    ::google::protobuf::RpcController*           controller_;
    const ::transfer::LayerBlockTransferRequest* request_;
    ::transfer::LayerBlockTransferResponse*      response_;
    ::google::protobuf::Closure*                 done_;
    std::shared_ptr<LayerBlockConvertor>         layer_block_convector_;
    kmonitor::MetricsReporterPtr                 metrics_reporter_;
    std::string                                  unique_key_;
    int64_t                                      partition_count_ = 1;
    int64_t                                      partition_id_    = 0;
    std::string                                  server_rdma_ip_;
    uint32_t                                     server_rdma_port_ = 0;

    std::shared_ptr<TransferServerTransferMetricsCollector> collector_;
    int64_t                                                 start_time_us_ = 0;

    std::shared_ptr<LayerCacheBufferTask> task_;
    std::shared_ptr<LayerCacheBuffer>     layer_cache_buffer_;
};

}  // namespace rtp_llm