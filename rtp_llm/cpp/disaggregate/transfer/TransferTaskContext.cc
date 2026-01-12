#include "rtp_llm/cpp/disaggregate/transfer/TransferTaskContext.h"
#
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

TransferTaskContext::TransferTaskContext(::google::protobuf::RpcController*           controller,
                                         const ::transfer::LayerBlockTransferRequest* request,
                                         ::transfer::LayerBlockTransferResponse*      response,
                                         ::google::protobuf::Closure*                 done,
                                         const std::shared_ptr<LayerBlockConvertor>&  layer_block_convector,
                                         const kmonitor::MetricsReporterPtr&          metrics_reporter):
    controller_(controller),
    request_(request),
    response_(response),
    done_(done),
    layer_block_convector_(layer_block_convector),
    metrics_reporter_(metrics_reporter),
    unique_key_(request->unique_key()),
    partition_count_(request->partition_count()),
    partition_id_(request->partition_id()),
    server_rdma_ip_(request->server_rdma_ip()),
    server_rdma_port_(request->server_rdma_port()),
    collector_(std::make_shared<TransferServerTransferMetricsCollector>()),
    start_time_us_(currentTimeUs()) {}

TransferTaskContext::~TransferTaskContext() {
    if (done_) {
        run(false, "transfer task context destroyed");
    }
}

void TransferTaskContext::addTask(const std::shared_ptr<LayerCacheBufferTask>& task) {
    task_                                = task;
    collector_->wait_task_run_latency_us = currentTimeUs() - start_time_us_;
}

const std::shared_ptr<LayerCacheBufferTask>& TransferTaskContext::getTask() const {
    return task_;
}

std::string TransferTaskContext::getUniqueKey() const {
    return unique_key_;
}

bool TransferTaskContext::isTimeout() const {
    return currentTimeMs() > request_->deadline_ms();
}

std::vector<std::pair<BufferPtr, std::shared_ptr<::transfer::BlockBufferInfo>>> TransferTaskContext::getTcpBlockPair() {
    if (task_ == nullptr || partition_count_ == 0) {
        RTP_LLM_LOG_WARNING("get tcp block pair failed, unique_key: %s, task is nullptr or partition count is 0",
                            unique_key_.c_str());
        return {};
    }
    auto& layer_block_info = request_->layer_block();
    auto  layer_id         = layer_block_info.layer_id();

    layer_cache_buffer_ = task_->loadingLayerCacheBuffer(layer_id, partition_count_, partition_id_);
    if (layer_cache_buffer_ == nullptr) {
        RTP_LLM_LOG_WARNING(
            "get layer cache buffer failed, unique_key: %s, layer_id: %d", unique_key_.c_str(), layer_id);
        return {};
    }

    int                                                                             transfer_count = 0;
    std::vector<std::pair<BufferPtr, std::shared_ptr<::transfer::BlockBufferInfo>>> block_pair;
    for (const auto& block_info : layer_block_info.blocks()) {
        int64_t key      = block_info.key();
        auto    block_id = layer_cache_buffer_->getBlockId(key);
        if (block_id == -1) {
            // no need to transfer
            continue;
        }
        auto buffers =
            layer_block_convector_->convertIndexToBuffer(layer_id, block_id, partition_count_, partition_id_);
        if (buffers.size() != block_info.blocks_size()) {
            RTP_LLM_LOG_WARNING(
                "buffers mismatch, unique_key: %s, layer_id: %d, block_id: %d, cache_key: %lld expected: %d, actual: %d",
                unique_key_.c_str(),
                layer_id,
                block_id,
                key,
                block_info.blocks_size(),
                buffers.size());
            return {};
        }
        for (int i = 0; i < buffers.size(); i++) {
            auto buffer            = buffers[i];
            auto block_buffer_info = std::make_shared<::transfer::BlockBufferInfo>();
            block_buffer_info->CopyFrom(block_info.blocks(i));
            block_pair.push_back({buffer, block_buffer_info});
            collector_->total_block_size += block_buffer_info->len();
        }
        transfer_count++;
    }
    collector_->block_count = transfer_count;
    if (transfer_count != layer_cache_buffer_->blockIdMap().size()) {
        RTP_LLM_LOG_WARNING("transfer count mismatch, unique_key: %s, layer_id: %d, expected: %d, actual: %d",
                            unique_key_.c_str(),
                            layer_id,
                            layer_cache_buffer_->blockIdMap().size(),
                            transfer_count);
        return {};
    }
    return block_pair;
}

std::vector<std::pair<BufferPtr, std::shared_ptr<RemoteBuffer>>> TransferTaskContext::getRdmaBlockPair() {
    if (task_ == nullptr || partition_count_ == 0) {
        RTP_LLM_LOG_WARNING("get rdma block pair failed, task is nullptr or partition count is 0");
        return {};
    }

    auto& layer_block_info = request_->layer_block();
    auto  layer_id         = layer_block_info.layer_id();

    layer_cache_buffer_ = task_->loadingLayerCacheBuffer(layer_id, partition_count_, partition_id_);
    if (layer_cache_buffer_ == nullptr) {
        RTP_LLM_LOG_WARNING(
            "get layer cache buffer failed, unique_key: %s, layer_id: %d", unique_key_.c_str(), layer_id);
        return {};
    }

    auto convert_to_remote_buffer_func =
        [](const ::transfer::BlockBufferInfo& block_buffer_info) -> std::shared_ptr<RemoteBuffer> {
        const auto& rdma_info = block_buffer_info.rdma_info();
        auto        nic_rkeys = std::make_shared<std::map<uint32_t, uint32_t>>();

        for (const auto& nic_rkey : rdma_info.nic_rkeys()) {
            nic_rkeys->insert({nic_rkey.nicid(), nic_rkey.rkey()});
        }
        return std::make_shared<RemoteBuffer>(
            static_cast<int64_t>(rdma_info.addr()), block_buffer_info.len(), nic_rkeys);
    };

    int                                                              transfer_count = 0;
    std::vector<std::pair<BufferPtr, std::shared_ptr<RemoteBuffer>>> block_pair;
    for (const auto& block_info : layer_block_info.blocks()) {
        int64_t key      = block_info.key();
        auto    block_id = layer_cache_buffer_->getBlockId(key);
        if (block_id == -1) {
            continue;  // no need to transfer
        }
        auto buffers =
            layer_block_convector_->convertIndexToBuffer(layer_id, block_id, partition_count_, partition_id_);
        if (buffers.size() != block_info.blocks_size()) {
            RTP_LLM_LOG_WARNING(
                "buffers mismatch, unique_key: %s, layer_id: %d, block_id: %d, cache_key: %lld expected: %d, actual: %d",
                unique_key_.c_str(),
                layer_id,
                block_id,
                key,
                block_info.blocks_size(),
                buffers.size());
            return {};
        }
        for (int i = 0; i < buffers.size(); i++) {
            auto buffer            = buffers[i];
            auto block_buffer_info = block_info.blocks(i);
            auto remote_buffer     = convert_to_remote_buffer_func(block_buffer_info);
            block_pair.push_back({buffer, remote_buffer});
            collector_->total_block_size += block_buffer_info.len();
        }
        transfer_count++;
    }

    if (transfer_count != layer_cache_buffer_->blockIdMap().size()) {
        RTP_LLM_LOG_WARNING("transfer count mismatch, unique_key: %s, layer_id: %d, expected: %d, actual: %d",
                            unique_key_.c_str(),
                            layer_id,
                            layer_cache_buffer_->blockIdMap().size(),
                            transfer_count);
        return {};
    }
    collector_->block_count = transfer_count;
    return block_pair;
}

std::pair<std::string, uint32_t> TransferTaskContext::getServerRdmaInfo() const {
    return std::make_pair(server_rdma_ip_, server_rdma_port_);
}

void TransferTaskContext::run(bool success, const std::string& info) {
    if (!done_) {
        return;
    }

    collector_->success               = success;
    collector_->total_cost_latency_us = currentTimeUs() - start_time_us_;
    if (metrics_reporter_) {
        metrics_reporter_->report<TransferMetric, TransferServerTransferMetricsCollector>(nullptr, collector_.get());
    }

    if (task_ && layer_cache_buffer_) {
        auto layer_id = layer_cache_buffer_->getLayerId();
        task_->notifyDone(layer_id, success, partition_count_, partition_id_);
    }
    response_->set_success(success);
    response_->set_info(info);
    done_->Run();
    done_ = nullptr;
}

}  // namespace rtp_llm