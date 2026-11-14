#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaKVCacheSender.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "aios/network/arpc/arpc/ANetRPCController.h"
#include "autil/NetUtil.h"
#include <cstdlib>
#include <memory>

namespace rtp_llm {
namespace transfer {
namespace barex_rdma {

namespace {

inline TransferErrorCode transRdmaErrorCode(::rdma_transfer::RdmaTransferErrorCodePB error_code) {
    switch (error_code) {
        case ::rdma_transfer::RDMA_TRANSFER_NONE_ERROR:
            return TransferErrorCode::OK;
        case ::rdma_transfer::RDMA_TRANSFER_BUFFER_MISMATCH:
            return TransferErrorCode::BUFFER_MISMATCH;
        case ::rdma_transfer::RDMA_TRANSFER_RDMA_FAILED:
            return TransferErrorCode::RDMA_FAILED;
        case ::rdma_transfer::RDMA_TRANSFER_CONTEXT_TIMEOUT:
            return TransferErrorCode::TIMEOUT;
        case ::rdma_transfer::RDMA_TRANSFER_TASK_CANCELLED:
            return TransferErrorCode::CANCELLED;
        case ::rdma_transfer::RDMA_TRANSFER_CONNECTION_FAILED:
            return TransferErrorCode::CONNECTION_FAILED;
        case ::rdma_transfer::RDMA_TRANSFER_NOT_INITIALIZED:
            return TransferErrorCode::NOT_INITIALIZED;
        case ::rdma_transfer::RDMA_TRANSFER_QUEUE_REJECTED:
            return TransferErrorCode::QUEUE_REJECTED;
        case ::rdma_transfer::RDMA_TRANSFER_COPY_FAILED:
            return TransferErrorCode::COPY_FAILED;
        case ::rdma_transfer::RDMA_TRANSFER_CONTEXT_DESTROYED:
            return TransferErrorCode::CONTEXT_DESTROYED;
        default:
            return TransferErrorCode::UNKNOWN;
    }
}

class RdmaTransferClosure: public ::google::protobuf::Closure {
public:
    RdmaTransferClosure(const std::string&                                                      peer_ip,
                        uint32_t                                                                peer_port,
                        const std::shared_ptr<::rdma_transfer::RdmaLayerBlockTransferRequest>&  request,
                        const std::shared_ptr<::rdma_transfer::RdmaLayerBlockTransferResponse>& response,
                        arpc::ANetRPCController*                                                controller,
                        std::function<void(TransferErrorCode, const std::string&)>              callback):
        peer_ip_(peer_ip),
        peer_port_(peer_port),
        request_(request),
        response_(response),
        controller_(controller),
        callback_(std::move(callback)) {}

    ~RdmaTransferClosure() {
        delete controller_;
    }

    void Run() override {
        std::unique_ptr<RdmaTransferClosure> self_guard(this);
        TransferErrorCode error_code = TransferErrorCode::OK;
        std::string       error_msg;

        if (controller_->Failed()) {
            error_code = TransferErrorCode::RPC_FAILED;
            error_msg  = "rdma transfer failed: rpc_code=" + std::to_string(controller_->GetErrorCode())
                        + " " + controller_->ErrorText() + " peer [" + peer_ip_ + ":"
                        + std::to_string(peer_port_) + "]";
        } else if (response_->has_error_code()
                   && response_->error_code() != ::rdma_transfer::RDMA_TRANSFER_NONE_ERROR) {
            error_code = transRdmaErrorCode(response_->error_code());
            error_msg  = "wire_code=" + std::to_string(static_cast<int>(response_->error_code())) + ": "
                         + (response_->has_error_message() ? response_->error_message() : "");
        }

        if (error_code != TransferErrorCode::OK) {
            error_msg += " [received_at location=RDMATransferClosure::Run peer=" + peer_ip_ + ":"
                         + std::to_string(peer_port_) + " partition_layer_key=" + request_->unique_key() + "]";
            RTP_LLM_LOG_WARNING("P2P RDMA send RPC failed, partition_layer_key=%s peer=%s:%u "
                                "transfer_code=%d transfer_name=%s rpc_code=%d wire_code=%d error=%s",
                                request_->unique_key().c_str(),
                                peer_ip_.c_str(),
                                peer_port_,
                                static_cast<int>(error_code),
                                transferErrorCodeToString(error_code),
                                controller_->GetErrorCode(),
                                response_->has_error_code() ? static_cast<int>(response_->error_code()) : -1,
                                error_msg.c_str());
        }
        callback_(error_code, error_msg);
    }

private:
    std::string                                                      peer_ip_;
    uint32_t                                                         peer_port_;
    std::shared_ptr<::rdma_transfer::RdmaLayerBlockTransferRequest>  request_;
    std::shared_ptr<::rdma_transfer::RdmaLayerBlockTransferResponse> response_;
    arpc::ANetRPCController*                                         controller_;
    std::function<void(TransferErrorCode, const std::string&)>       callback_;
};

}  // anonymous namespace

RdmaKVCacheSender::RdmaKVCacheSender(const std::shared_ptr<IRdmaMemoryManager>& rdma_memory_manager,
                                     const kmonitor::MetricsReporterPtr&        metrics_reporter):
    rdma_memory_manager_(rdma_memory_manager), metrics_reporter_(metrics_reporter) {}

bool RdmaKVCacheSender::init(int tcp_io_thread_count, int rdma_io_thread_count, int rdma_worker_thread_count) {
    if (!rdma_memory_manager_) {
        RTP_LLM_LOG_WARNING("rdma memory manager is null");
        return false;
    }

    tcp_client_ = std::make_shared<transfer::TcpClient>();
    if (!tcp_client_->init(tcp_io_thread_count)) {
        RTP_LLM_LOG_WARNING("create tcp client failed");
        return false;
    }

    // Barex XServerTransport keeps requested port in RdmaServer::listen_port_; port 0 stays 0 after bind
    // (kernel picks ephemeral port but we never read it back). Use an explicit free port like TCP tests.
    const uint32_t rdma_listen_port = static_cast<uint32_t>(autil::NetUtil::randomPort());
    rdma_server_ = createRdmaServer(rdma_memory_manager_, rdma_listen_port, rdma_io_thread_count, rdma_worker_thread_count);
    if (!rdma_server_) {
        RTP_LLM_LOG_WARNING("create rdma server failed");
        return false;
    }
    return true;
}

bool RdmaKVCacheSender::regMem(const BlockInfo& block_info, uint64_t aligned_size) {
    if (!rdma_memory_manager_) {
        RTP_LLM_LOG_WARNING("rdma memory manager is null, regMem failed");
        return false;
    }
    return rdma_memory_manager_->regUserMr(block_info, aligned_size);
}

void RdmaKVCacheSender::setBlockBufferInfo(::rdma_transfer::RdmaBlockBufferInfo* block_buffer_info,
                                           int64_t cache_key,
                                           const BlockInfo& block_info) {
    auto remote_buffer = rdma_memory_manager_->findMemoryMr(block_info);
    // RDMA MR 通常在进程初始化时统一注册管理。发送范围找不到 MR，可能是越界访问，
    // 很可能伴随脏读、脏写，此时服务已不可用；直接 abort 暴露问题，禁止拷贝注册后继续发送。
    // 入表时保证 rkey 集合非空；发送时发现该约束被破坏也必须立即终止。
    if (!remote_buffer || !remote_buffer->nic_rkeys || remote_buffer->nic_rkeys->empty()) {
        RTP_LLM_LOG_ERROR("RDMA send MR/rkey lookup failed, cache_key=%ld addr=%p size=%lu cuda=%d",
                          cache_key,
                          block_info.addr,
                          block_info.size_bytes,
                          block_info.is_cuda);
        std::abort();
    }

    block_buffer_info->set_len(static_cast<uint32_t>(block_info.size_bytes));
    auto* rdma_info = block_buffer_info->mutable_rdma_info();
    rdma_info->set_addr(remote_buffer->addr);
    for (const auto& [nicid, rkey] : *remote_buffer->nic_rkeys) {
        auto nic_rkey = rdma_info->add_nic_rkeys();
        nic_rkey->set_nicid(nicid);
        nic_rkey->set_rkey(rkey);
    }
}

std::shared_ptr<::rdma_transfer::RdmaLayerBlockTransferRequest>
RdmaKVCacheSender::makeTransferRequest(const transfer::SendRequest&                           request,
                                       const std::shared_ptr<TransferClientMetricsCollector>& collector) {
    auto transfer_request = std::make_shared<::rdma_transfer::RdmaLayerBlockTransferRequest>();
    transfer_request->set_unique_key(request.unique_key);
    transfer_request->set_timeout_ms(request.timeout_ms);

    transfer_request->set_server_rdma_ip(autil::NetUtil::getBindIp());
    transfer_request->set_server_rdma_port(rdma_server_->getListenPort());

    if (request.block_info.empty()) {
        return nullptr;
    }

    for (const auto& [cache_key, kbi_ptr] : request.block_info) {
        if (!kbi_ptr) {
            continue;
        }
        const auto& kbi                  = *kbi_ptr;
        auto        cache_key_block_info = transfer_request->add_blocks();
        cache_key_block_info->set_key(kbi.cache_key);

        for (const auto& block : kbi.blocks) {
            if (block.addr == nullptr || block.size_bytes == 0) {
                continue;
            }
            auto block_buffer_info = cache_key_block_info->add_blocks();
            setBlockBufferInfo(block_buffer_info, kbi.cache_key, block);
            collector->block_count++;
            collector->total_block_size += block.size_bytes;
        }
    }
    return transfer_request;
}

void RdmaKVCacheSender::loadToRemote(
    const std::string&                                                     ip,
    uint32_t                                                               port,
    const std::shared_ptr<::rdma_transfer::RdmaLayerBlockTransferRequest>& transfer_request,
    std::function<void(TransferErrorCode, const std::string&)>             callback,
    int64_t                                                                deadline_ms) {
    if (!callback) {
        RTP_LLM_LOG_ERROR("loadToRemote callback is null");
        return;
    }

    auto channel = tcp_client_->getChannel(ip, port);
    if (!channel) {
        callback(TransferErrorCode::CONNECTION_FAILED, "get channel failed");
        return;
    }

    auto transfer_response = std::make_shared<::rdma_transfer::RdmaLayerBlockTransferResponse>();
    auto controller        = new arpc::ANetRPCController();
    auto timeout_ms        = deadline_ms - currentTimeMs();
    controller->SetExpireTime(timeout_ms > 0 ? timeout_ms : 1);

    auto closure = new RdmaTransferClosure(
        ip, port, transfer_request, transfer_response, controller, callback);
    ::rdma_transfer::RdmaTransferService_Stub stub((::google::protobuf::RpcChannel*)(channel.get()),
                                                   ::google::protobuf::Service::STUB_DOESNT_OWN_CHANNEL);
    stub.transfer(controller, transfer_request.get(), transfer_response.get(), closure);
}

void RdmaKVCacheSender::send(const transfer::SendRequest&                               request,
                             std::function<void(TransferErrorCode, const std::string&)> callback) {
    if (!callback) {
        RTP_LLM_LOG_ERROR("send callback is null");
        return;
    }

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
        callback2(TransferErrorCode::BUILD_REQUEST_FAILED, "make rdma transfer request failed");
        return;
    }
    loadToRemote(request.ip, request.port, transfer_request, callback2, request.deadline_ms);
}

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm
