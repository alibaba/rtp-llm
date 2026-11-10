#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaTaskContext.h"

#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/proto/rdma_service.pb.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferErrorCode.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace transfer {
namespace barex_rdma {

RdmaTaskContext::RdmaTaskContext(::google::protobuf::RpcController*                     controller,
                                 const ::rdma_transfer::RdmaLayerBlockTransferRequest*  request,
                                 ::rdma_transfer::RdmaLayerBlockTransferResponse*       response,
                                 ::google::protobuf::Closure*                           done,
                                 const kmonitor::MetricsReporterPtr&                    metrics_reporter):
    controller_(controller),
    request_(request),
    response_(response),
    done_(done),
    metrics_reporter_(metrics_reporter),
    unique_key_(request->unique_key()),
    collector_(std::make_shared<TransferServerMetricsCollector>()),
    start_time_us_(currentTimeUs()) {}

RdmaTaskContext::~RdmaTaskContext() {
    if (done_) {
        run(false, TransferErrorCode::UNKNOWN, "RdmaTaskContext destroyed without completion");
    }
}

void RdmaTaskContext::setTask(const std::shared_ptr<TransferTask>& task) {
    task_                                = task;
    collector_->wait_task_run_latency_us = currentTimeUs() - start_time_us_;
}

bool RdmaTaskContext::startTransfer() {
    if (!task_) {
        RTP_LLM_LOG_WARNING("startTransfer: task is null, unique_key: %s", unique_key_.c_str());
        return false;
    }
    return task_->startTransfer();
}

std::string RdmaTaskContext::getUniqueKey() const {
    return unique_key_;
}

bool RdmaTaskContext::isTimeout() const {
    return currentTimeMs() > request_->deadline_ms();
}

uint64_t RdmaTaskContext::getDeadlineMs() const {
    return static_cast<uint64_t>(request_->deadline_ms());
}

std::string RdmaTaskContext::getRdmaServerIp() const {
    return request_->server_rdma_ip();
}

uint32_t RdmaTaskContext::getRdmaServerPort() const {
    return request_->server_rdma_port();
}

std::optional<torch::Tensor> RdmaTaskContext::findLocalBuffer(int64_t cache_key, int sub_block_idx) const {
    if (!task_) {
        return std::nullopt;
    }
    const auto& block_infos = task_->getBlockInfos();
    auto        it          = block_infos.find(cache_key);
    if (it == block_infos.end() || !it->second) {
        return std::nullopt;
    }
    const auto& kbi = *it->second;
    if (sub_block_idx < 0 || static_cast<size_t>(sub_block_idx) >= kbi.blocks.size()) {
        return std::nullopt;
    }
    const BlockInfo& bi = kbi.blocks[sub_block_idx];
    if (bi.addr == nullptr || bi.size_bytes == 0) {
        return std::nullopt;
    }
    auto options = torch::TensorOptions().dtype(torch::kUInt8);
    if (bi.is_cuda) {
        options = options.device(torch::kCUDA, bi.device_index);
    }
    return torch::from_blob(bi.addr, {static_cast<int64_t>(bi.size_bytes)}, options);
}

std::vector<std::pair<torch::Tensor, std::shared_ptr<RemoteBuffer>>> RdmaTaskContext::getRdmaBlockPair() {
    if (!task_) {
        RTP_LLM_LOG_WARNING("getRdmaBlockPair: task is null, unique_key: %s", unique_key_.c_str());
        return {};
    }

    std::vector<std::pair<torch::Tensor, std::shared_ptr<RemoteBuffer>>> result;

    auto build_remote_buffer = [](const ::rdma_transfer::RdmaBlockBufferInfo& info) {
        const auto& rdma_info = info.rdma_info();
        auto        nic_rkeys = std::make_shared<std::map<uint32_t, uint32_t>>();
        for (const auto& nic_rkey : rdma_info.nic_rkeys()) {
            nic_rkeys->insert({nic_rkey.nicid(), nic_rkey.rkey()});
        }
        return std::make_shared<RemoteBuffer>(rdma_info.addr(), info.len(), nic_rkeys);
    };

    for (const auto& cache_key_block : request_->blocks()) {
        int64_t cache_key = cache_key_block.key();
        for (int i = 0; i < cache_key_block.blocks_size(); ++i) {
            const auto& block_info = cache_key_block.blocks(i);
            if (!block_info.has_rdma_info() || !block_info.rdma_info().has_addr()) {
                continue;
            }
            auto local_buf = findLocalBuffer(cache_key, i);
            if (!local_buf) {
                RTP_LLM_LOG_WARNING(
                    "getRdmaBlockPair: local buffer not found, cache_key: %lld, sub: %d", cache_key, i);
                continue;
            }
            auto remote_buf = build_remote_buffer(block_info);
            result.push_back({*local_buf, remote_buf});
            collector_->total_block_size += block_info.len();
        }
    }
    collector_->block_count = static_cast<int>(result.size());
    return result;
}

namespace {

inline ::rdma_transfer::RdmaTransferErrorCodePB toRdmaProtoErrorCode(TransferErrorCode ec) {
    switch (ec) {
        case TransferErrorCode::OK:              return ::rdma_transfer::RDMA_TRANSFER_NONE_ERROR;
        case TransferErrorCode::BUFFER_MISMATCH: return ::rdma_transfer::RDMA_TRANSFER_BUFFER_MISMATCH;
        case TransferErrorCode::RDMA_FAILED:     return ::rdma_transfer::RDMA_TRANSFER_RDMA_FAILED;
        case TransferErrorCode::TIMEOUT:         return ::rdma_transfer::RDMA_TRANSFER_CONTEXT_TIMEOUT;
        case TransferErrorCode::CANCELLED:       return ::rdma_transfer::RDMA_TRANSFER_TASK_CANCELLED;
        default:                                 return ::rdma_transfer::RDMA_TRANSFER_UNKNOWN_ERROR;
    }
}

}  // anonymous namespace

void RdmaTaskContext::run(bool               success,
                           TransferErrorCode  error_code,
                           const std::string& error_message) {
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

    auto proto_error_code = toRdmaProtoErrorCode(error_code);

    collector_->success = success;
    if (metrics_reporter_) {
        metrics_reporter_->report<TransferMetric, TransferServerMetricsCollector>(nullptr,
                                                                                         collector_.get());
    }

    response_->set_error_code(proto_error_code);
    response_->set_error_message(error_message);
    done_->Run();
    done_ = nullptr;
}

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm
