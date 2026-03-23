#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/TcpKVCacheSender.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "aios/network/arpc/arpc/ANetRPCController.h"

namespace rtp_llm {
namespace transfer {
namespace tcp {

namespace {

inline TransferErrorCode transTcpErrorCode(::tcp_transfer::TcpTransferErrorCodePB error_code) {
    switch (error_code) {
        case ::tcp_transfer::TCP_TRANSFER_NONE_ERROR:
            return TransferErrorCode::OK;
        case ::tcp_transfer::TCP_TRANSFER_BUFFER_MISMATCH:
            return TransferErrorCode::BUFFER_MISMATCH;
        case ::tcp_transfer::TCP_TRANSFER_CONTEXT_TIMEOUT:
            return TransferErrorCode::TIMEOUT;
        case ::tcp_transfer::TCP_TRANSFER_TASK_CANCELLED:
            return TransferErrorCode::CANCELLED;
        default:
            return TransferErrorCode::UNKNOWN;
    }
}

class TcpTransferClosure: public ::google::protobuf::Closure {
public:
    TcpTransferClosure(const std::string&                                                    peer_ip,
                       uint32_t                                                              peer_port,
                       const std::shared_ptr<::tcp_transfer::TcpLayerBlockTransferRequest>&  request,
                       const std::shared_ptr<::tcp_transfer::TcpLayerBlockTransferResponse>& response,
                       arpc::ANetRPCController*                                              controller,
                       std::function<void(TransferErrorCode, const std::string&)>            callback):
        peer_ip_(peer_ip),
        peer_port_(peer_port),
        request_(request),
        response_(response),
        controller_(controller),
        callback_(callback) {}

    ~TcpTransferClosure() {
        delete controller_;
    }

    void Run() override {
        TransferErrorCode error_code = TransferErrorCode::OK;
        std::string       error_msg;

        if (controller_->Failed()) {
            error_code = TransferErrorCode::RPC_FAILED;
            error_msg  = "tcp transfer failed: " + controller_->ErrorText() + " peer [" + peer_ip_ + ":"
                        + std::to_string(peer_port_) + "]";
        } else if (response_->has_error_code() && response_->error_code() != ::tcp_transfer::TCP_TRANSFER_NONE_ERROR) {
            error_code = transTcpErrorCode(response_->error_code());
            error_msg  = response_->has_error_message() ? response_->error_message() : "";
        }

        if (callback_) {
            callback_(error_code, error_msg);
        }
    }

private:
    std::string                                                    peer_ip_;
    uint32_t                                                       peer_port_;
    std::shared_ptr<::tcp_transfer::TcpLayerBlockTransferRequest>  request_;
    std::shared_ptr<::tcp_transfer::TcpLayerBlockTransferResponse> response_;
    arpc::ANetRPCController*                                       controller_;
    std::function<void(TransferErrorCode, const std::string&)>     callback_;
};

}  // anonymous namespace

TcpKVCacheSender::TcpKVCacheSender(const kmonitor::MetricsReporterPtr& metrics_reporter):
    metrics_reporter_(metrics_reporter), cuda_copy_util_(std::make_unique<CudaCopyUtil>()) {}

bool TcpKVCacheSender::init(int                       io_thread_count,
                            std::chrono::milliseconds channel_idle_ttl,
                            std::uint64_t             sweep_interval_calls) {
    tcp_client_ = std::make_shared<transfer::TcpClient>();
    if (!tcp_client_->init(io_thread_count, channel_idle_ttl, sweep_interval_calls)) {
        RTP_LLM_LOG_WARNING("TcpKVCacheSender: create tcp client failed");
        return false;
    }
    return true;
}

bool TcpKVCacheSender::regMem(const BlockInfo& /*block_info*/, uint64_t /*aligned_size*/) {

    return true;
}

bool TcpKVCacheSender::setBlockBufferInfo(::tcp_transfer::TcpBlockBufferInfo* block_buffer_info,
                                          int64_t                             cache_key,
                                          const BlockInfo&                    block_info,
                                          std::vector<CopyTask>&              copy_tasks) {
    block_buffer_info->set_len(static_cast<uint32_t>(block_info.size_bytes));

    auto* content = block_buffer_info->mutable_content();
    content->resize(block_info.size_bytes);

    CopyTask task;
    task.src_ptr = block_info.addr;
    task.size    = block_info.size_bytes;
    task.dst_ptr = content->data();
    copy_tasks.push_back(task);
    return true;
}

std::shared_ptr<::tcp_transfer::TcpLayerBlockTransferRequest>
TcpKVCacheSender::makeTransferRequest(const transfer::SendRequest&                           request,
                                      const std::shared_ptr<TransferClientMetricsCollector>& collector) {
    auto transfer_request = std::make_shared<::tcp_transfer::TcpLayerBlockTransferRequest>();
    transfer_request->set_unique_key(request.unique_key);
    transfer_request->set_deadline_ms(
        std::max(request.deadline_ms - 10, currentTimeMs()));  // 10ms for network latency and processing time

    if (request.block_info.empty()) {
        return nullptr;
    }

    std::vector<CopyTask> copy_tasks;
    for (const auto& [cache_key, kbi_ptr] : request.block_info) {
        auto cache_key_block_info = transfer_request->add_blocks();
        cache_key_block_info->set_key(cache_key);

        for (const auto& bi : kbi_ptr->blocks) {
            if (bi.addr != nullptr && bi.size_bytes > 0) {
                auto block_buffer_info = cache_key_block_info->add_blocks();
                if (!setBlockBufferInfo(block_buffer_info, cache_key, bi, copy_tasks)) {
                    return nullptr;
                }
                collector->block_count++;
                collector->total_block_size += bi.size_bytes;
            }
        }
    }

    if (!copy_tasks.empty()) {
        if (!cuda_copy_util_->batchCopyToHost(copy_tasks)) {
            RTP_LLM_LOG_WARNING("TcpKVCacheSender: batchCopyToHost failed, unique_key: %s", request.unique_key.c_str());
            return nullptr;
        }
    }
    return transfer_request;
}

void TcpKVCacheSender::loadToRemote(
    const std::string&                                                   ip,
    uint32_t                                                             port,
    const std::shared_ptr<::tcp_transfer::TcpLayerBlockTransferRequest>& transfer_request,
    std::function<void(TransferErrorCode, const std::string&)>           callback,
    int64_t                                                              deadline_ms) {
    auto channel = tcp_client_->getChannel(ip, port);
    if (!channel) {
        if (callback) {
            callback(TransferErrorCode::CONNECTION_FAILED, "get channel failed");
        }
        return;
    }

    auto transfer_response = std::make_shared<::tcp_transfer::TcpLayerBlockTransferResponse>();
    auto controller        = new arpc::ANetRPCController();
    auto timeout_ms        = deadline_ms - currentTimeMs();
    controller->SetExpireTime(timeout_ms > 0 ? timeout_ms : 1);

    auto closure = new TcpTransferClosure(ip, port, transfer_request, transfer_response, controller, callback);
    ::tcp_transfer::TcpTransferService_Stub stub((::google::protobuf::RpcChannel*)(channel.get()),
                                                 ::google::protobuf::Service::STUB_DOESNT_OWN_CHANNEL);
    stub.transfer(controller, transfer_request.get(), transfer_response.get(), closure);
}

void TcpKVCacheSender::send(const transfer::SendRequest&                               request,
                            std::function<void(TransferErrorCode, const std::string&)> callback) {
    auto collector     = std::make_shared<TransferClientMetricsCollector>();
    auto start_time_us = currentTimeUs();
    auto callback2     = [callback, collector, start_time_us, metrics_reporter = metrics_reporter_](
                         TransferErrorCode error_code, const std::string& error_msg) {
        collector->success    = (error_code == TransferErrorCode::OK);
        collector->latency_us = currentTimeUs() - start_time_us;
        if (metrics_reporter) {
            metrics_reporter->report<TransferMetric, TransferClientMetricsCollector>(nullptr, collector.get());
        }
        callback(error_code, error_msg);
    };

    auto transfer_request = makeTransferRequest(request, collector);
    if (!transfer_request) {
        callback2(TransferErrorCode::BUILD_REQUEST_FAILED, "make transfer request failed");
        return;
    }

    loadToRemote(request.ip, request.port, transfer_request, callback2, request.deadline_ms);
}

}  // namespace tcp
}  // namespace transfer
}  // namespace rtp_llm
