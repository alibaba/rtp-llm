#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/TcpTaskContext.h"

#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferErrorCode.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace transfer {
namespace tcp {

TcpTaskContext::TcpTaskContext(::google::protobuf::RpcController*                  controller,
                               const ::tcp_transfer::TcpLayerBlockTransferRequest* request,
                               ::tcp_transfer::TcpLayerBlockTransferResponse*      response,
                               ::google::protobuf::Closure*                        done,
                               const kmonitor::MetricsReporterPtr&                 metrics_reporter):
    controller_(controller),
    request_(request),
    response_(response),
    done_(done),
    metrics_reporter_(metrics_reporter),
    unique_key_(request->unique_key()),
    collector_(std::make_shared<TransferServerMetricsCollector>()),
    start_time_us_(currentTimeUs()) {}

TcpTaskContext::~TcpTaskContext() {
    if (done_) {
        run(false, TransferErrorCode::UNKNOWN, "TcpTaskContext destroyed without completion");
    }
}

void TcpTaskContext::setTask(const std::shared_ptr<TransferTask>& task) {
    task_                                = task;
    collector_->wait_task_run_latency_us = currentTimeUs() - start_time_us_;
}

bool TcpTaskContext::startTransfer() {
    if (!task_) {
        RTP_LLM_LOG_WARNING("startTransfer: task is null, unique_key: %s", unique_key_.c_str());
        return false;
    }
    return task_->startTransfer();
}

const std::string& TcpTaskContext::getUniqueKey() const {
    return unique_key_;
}

bool TcpTaskContext::isTimeout() const {
    return currentTimeMs() > request_->deadline_ms();
}

uint64_t TcpTaskContext::getDeadlineMs() const {
    return static_cast<uint64_t>(request_->deadline_ms());
}

bool TcpTaskContext::executeCopy(CudaCopyUtil& cuda_copy_util) {
    if (!task_) {
        RTP_LLM_LOG_WARNING("executeCopy: task is null, unique_key: %s", unique_key_.c_str());
        return false;
    }

    // Build a lookup index from the request for O(1) access by cache_key.
    std::unordered_map<int64_t, const ::tcp_transfer::TcpCacheKeyBlockBufferInfo*> request_index;
    request_index.reserve(request_->blocks_size());
    for (const auto& cache_key_block : request_->blocks()) {
        request_index[cache_key_block.key()] = &cache_key_block;
    }

    // Iterate over task's expected blocks as the authoritative source of truth.
    // Every non-empty BlockInfo in the task must be present in the request with a matching size.
    std::vector<CopyTask> copy_tasks;
    for (const auto& [cache_key, kbi_ptr] : task_->getBlockInfos()) {
        auto req_it = request_index.find(cache_key);
        if (req_it == request_index.end()) {
            RTP_LLM_LOG_WARNING(
                "executeCopy: cache_key %lld missing in request, unique_key: %s", cache_key, unique_key_.c_str());
            return false;
        }
        const auto* req_block = req_it->second;

        for (int i = 0; i < static_cast<int>(kbi_ptr->blocks.size()); ++i) {
            const BlockInfo& bi = kbi_ptr->blocks[i];
            if (bi.addr == nullptr || bi.size_bytes == 0) {
                continue;
            }
            if (i >= req_block->blocks_size()) {
                RTP_LLM_LOG_WARNING("executeCopy: cache_key %lld sub_block %d missing in request, unique_key: %s",
                                    cache_key,
                                    i,
                                    unique_key_.c_str());
                return false;
            }
            const auto& proto_block = req_block->blocks(i);
            if (proto_block.len() != static_cast<uint32_t>(bi.size_bytes)) {
                RTP_LLM_LOG_WARNING(
                    "executeCopy: size mismatch cache_key %lld sub %d: expected %zu got %u, unique_key: %s",
                    cache_key,
                    i,
                    bi.size_bytes,
                    proto_block.len(),
                    unique_key_.c_str());
                return false;
            }
            CopyTask task;
            task.src_ptr = const_cast<char*>(proto_block.content().data());
            task.size    = proto_block.len();
            task.dst_ptr = static_cast<char*>(bi.addr);
            copy_tasks.push_back(task);
            collector_->total_block_size += proto_block.len();
        }
        ++collector_->block_count;
    }

    if (copy_tasks.empty()) {
        RTP_LLM_LOG_WARNING("executeCopy: no blocks to transfer, unique_key: %s", unique_key_.c_str());
        return false;
    }
    return cuda_copy_util.batchCopyToDevice(copy_tasks);
}

namespace {

inline ::tcp_transfer::TcpTransferErrorCodePB toTcpProtoErrorCode(TransferErrorCode ec) {
    switch (ec) {
        case TransferErrorCode::OK:
            return ::tcp_transfer::TCP_TRANSFER_NONE_ERROR;
        case TransferErrorCode::BUFFER_MISMATCH:
            return ::tcp_transfer::TCP_TRANSFER_BUFFER_MISMATCH;
        case TransferErrorCode::TIMEOUT:
            return ::tcp_transfer::TCP_TRANSFER_CONTEXT_TIMEOUT;
        case TransferErrorCode::CANCELLED:
            return ::tcp_transfer::TCP_TRANSFER_TASK_CANCELLED;
        default:
            return ::tcp_transfer::TCP_TRANSFER_UNKNOWN_ERROR;
    }
}

}  // anonymous namespace

void TcpTaskContext::run(bool success, TransferErrorCode error_code, const std::string& error_message) {
    if (!done_) {
        return;
    }

    collector_->total_cost_latency_us = currentTimeUs() - start_time_us_;

    if (task_) {
        task_->notifyDone(success, success ? TransferErrorCode::OK : error_code, error_message);
        // notifyDone() may override the result (e.g. cancel_requested_ was set during transfer).
        // Read back the authoritative error code so sender and receiver see the same outcome.
        error_code = task_->errorCode();
        success    = (error_code == TransferErrorCode::OK);
    }

    auto proto_error_code = toTcpProtoErrorCode(error_code);

    collector_->success = success;
    if (metrics_reporter_) {
        metrics_reporter_->report<TransferMetric, TransferServerMetricsCollector>(nullptr, collector_.get());
    }

    response_->set_error_code(proto_error_code);
    response_->set_error_message(error_message);
    done_->Run();
    done_ = nullptr;
}

}  // namespace tcp
}  // namespace transfer
}  // namespace rtp_llm
