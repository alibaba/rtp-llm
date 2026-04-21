#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeKVCacheSender.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <thread>
#include <unordered_map>

#include "aios/network/arpc/arpc/ANetRPCController.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TcpClient.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/proto/mooncake_service.pb.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
namespace transfer {
namespace mooncake {

namespace {

std::string blockKey(int64_t cache_key, uint32_t block_index) {
    return std::to_string(cache_key) + "#" + std::to_string(block_index);
}

::mooncake_transfer::MooncakeTransferErrorCodePB toProtoErrorCode(TransferErrorCode error_code) {
    switch (error_code) {
        case TransferErrorCode::OK:
            return ::mooncake_transfer::MOONCAKE_TRANSFER_NONE_ERROR;
        case TransferErrorCode::TIMEOUT:
            return ::mooncake_transfer::MOONCAKE_TRANSFER_CONTEXT_TIMEOUT;
        case TransferErrorCode::CANCELLED:
            return ::mooncake_transfer::MOONCAKE_TRANSFER_TASK_CANCELLED;
        case TransferErrorCode::BUFFER_MISMATCH:
            return ::mooncake_transfer::MOONCAKE_TRANSFER_BUFFER_MISMATCH;
        default:
            return ::mooncake_transfer::MOONCAKE_TRANSFER_UNKNOWN_ERROR;
    }
}

TransferErrorCode fromProtoErrorCode(::mooncake_transfer::MooncakeTransferErrorCodePB error_code) {
    switch (error_code) {
        case ::mooncake_transfer::MOONCAKE_TRANSFER_NONE_ERROR:
            return TransferErrorCode::OK;
        case ::mooncake_transfer::MOONCAKE_TRANSFER_CONTEXT_TIMEOUT:
            return TransferErrorCode::TIMEOUT;
        case ::mooncake_transfer::MOONCAKE_TRANSFER_TASK_CANCELLED:
            return TransferErrorCode::CANCELLED;
        case ::mooncake_transfer::MOONCAKE_TRANSFER_BUFFER_MISMATCH:
            return TransferErrorCode::BUFFER_MISMATCH;
        default:
            return TransferErrorCode::UNKNOWN;
    }
}

class BlockingClosure : public ::google::protobuf::Closure {
public:
    void Run() override {
        promise_.set_value();
    }

    bool waitFor(std::chrono::milliseconds timeout) {
        return future_.wait_for(timeout) == std::future_status::ready;
    }

private:
    std::promise<void> promise_;
    std::future<void>  future_ = promise_.get_future();
};

class ArpcMooncakeControlPlaneClient : public IMooncakeControlPlaneClient {
public:
    bool init(int io_thread_count) override {
        tcp_client_ = std::make_shared<transfer::TcpClient>();
        return tcp_client_->init(io_thread_count);
    }

    bool prepare(const std::string& ip,
                 uint32_t port,
                 const std::string& unique_key,
                 int64_t deadline_ms,
                 MooncakeRemoteDescriptor* descriptor,
                 TransferErrorCode* error_code,
                 std::string* error_message) override {
        if (!descriptor || !error_code || !error_message) {
            return false;
        }
        if (!tcp_client_) {
            *error_code = TransferErrorCode::UNKNOWN;
            *error_message = "Mooncake control plane client is not initialized";
            return false;
        }

        auto channel = tcp_client_->getChannel(ip, port);
        if (!channel) {
            *error_code = TransferErrorCode::CONNECTION_FAILED;
            *error_message = "Mooncake control plane prepare get channel failed";
            return false;
        }

        ::mooncake_transfer::MooncakePrepareRequest request;
        request.set_unique_key(unique_key);
        request.set_deadline_ms(deadline_ms);

        ::mooncake_transfer::MooncakePrepareResponse response;
        arpc::ANetRPCController controller;
        const auto timeout_ms = deadline_ms > 0 ? std::max<int64_t>(deadline_ms - currentTimeMs(), 1) : 1000;
        controller.SetExpireTime(timeout_ms);

        BlockingClosure done;
        ::mooncake_transfer::MooncakeTransferService_Stub stub(
            (::google::protobuf::RpcChannel*)(channel.get()),
            ::google::protobuf::Service::STUB_DOESNT_OWN_CHANNEL);
        stub.prepare(&controller, &request, &response, &done);
        if (!done.waitFor(std::chrono::milliseconds(timeout_ms + 50))) {
            *error_code = TransferErrorCode::TIMEOUT;
            *error_message = "Mooncake control plane prepare timed out";
            return false;
        }
        if (controller.Failed()) {
            *error_code = TransferErrorCode::RPC_FAILED;
            *error_message = controller.ErrorText();
            return false;
        }
        if (response.has_error_code() && response.error_code() != ::mooncake_transfer::MOONCAKE_TRANSFER_NONE_ERROR) {
            *error_code = fromProtoErrorCode(response.error_code());
            *error_message = response.has_error_message() ? response.error_message() : std::string();
            return false;
        }

        descriptor->segment_name = response.has_segment_name() ? response.segment_name() : std::string();
        descriptor->blocks.clear();
        descriptor->blocks.reserve(response.descriptors_size());
        for (const auto& block : response.descriptors()) {
            MooncakeRemoteBlockDescriptor descriptor_block;
            descriptor_block.cache_key = block.has_cache_key() ? block.cache_key() : 0;
            descriptor_block.block_index = block.has_block_index() ? block.block_index() : 0;
            descriptor_block.target_addr = block.has_target_addr() ? block.target_addr() : 0;
            descriptor_block.len = block.has_len() ? block.len() : 0;
            descriptor->blocks.push_back(descriptor_block);
        }
        *error_code = TransferErrorCode::OK;
        error_message->clear();
        return true;
    }

    bool finish(const std::string& ip,
                uint32_t port,
                const std::string& unique_key,
                bool success,
                TransferErrorCode error_code,
                const std::string& error_message,
                TransferErrorCode* response_error_code,
                std::string* response_error_message) override {
        if (response_error_code) {
            *response_error_code = TransferErrorCode::OK;
        }
        if (response_error_message) {
            response_error_message->clear();
        }
        if (!tcp_client_) {
            if (response_error_code) {
                *response_error_code = TransferErrorCode::UNKNOWN;
            }
            if (response_error_message) {
                *response_error_message = "Mooncake control plane client is not initialized";
            }
            return false;
        }
        auto channel = tcp_client_->getChannel(ip, port);
        if (!channel) {
            if (response_error_code) {
                *response_error_code = TransferErrorCode::CONNECTION_FAILED;
            }
            if (response_error_message) {
                *response_error_message = "Mooncake control plane finish get channel failed";
            }
            return false;
        }

        ::mooncake_transfer::MooncakeFinishRequest request;
        request.set_unique_key(unique_key);
        request.set_success(success);
        request.set_error_code(toProtoErrorCode(error_code));
        if (!error_message.empty()) {
            request.set_error_message(error_message);
        }

        ::mooncake_transfer::MooncakeFinishResponse response;
        arpc::ANetRPCController controller;
        controller.SetExpireTime(1000);
        BlockingClosure done;
        ::mooncake_transfer::MooncakeTransferService_Stub stub(
            (::google::protobuf::RpcChannel*)(channel.get()),
            ::google::protobuf::Service::STUB_DOESNT_OWN_CHANNEL);
        stub.finish(&controller, &request, &response, &done);
        if (!done.waitFor(std::chrono::milliseconds(1050))) {
            if (response_error_code) {
                *response_error_code = TransferErrorCode::TIMEOUT;
            }
            if (response_error_message) {
                *response_error_message = "Mooncake control plane finish timed out";
            }
            return false;
        }
        if (controller.Failed()) {
            if (response_error_code) {
                *response_error_code = TransferErrorCode::RPC_FAILED;
            }
            if (response_error_message) {
                *response_error_message = controller.ErrorText();
            }
            return false;
        }
        if (response.has_error_code() && response.error_code() != ::mooncake_transfer::MOONCAKE_TRANSFER_NONE_ERROR) {
            if (response_error_code) {
                *response_error_code = fromProtoErrorCode(response.error_code());
            }
            if (response_error_message) {
                *response_error_message = response.has_error_message() ? response.error_message() : std::string();
            }
            return false;
        }
        return true;
    }

private:
    std::shared_ptr<transfer::TcpClient> tcp_client_;
};

}  // namespace

IMooncakeControlPlaneClientPtr createMooncakeControlPlaneClient() {
    return std::make_shared<ArpcMooncakeControlPlaneClient>();
}

MooncakeKVCacheSender::MooncakeKVCacheSender(const IMooncakeTransferEngineAdapterPtr& adapter,
                                             const IMooncakeControlPlaneClientPtr& control_plane_client,
                                             const kmonitor::MetricsReporterPtr& metrics_reporter):
    adapter_(adapter), control_plane_client_(control_plane_client), metrics_reporter_(metrics_reporter) {}

bool MooncakeKVCacheSender::init(const TransferBackendConfig& config) {
    config_ = config.mooncake;
    if (!adapter_) {
        RTP_LLM_LOG_ERROR("MooncakeKVCacheSender init failed: adapter is null");
        return false;
    }
    if (!adapter_->init(config_)) {
        return false;
    }
    if (control_plane_client_) {
        return control_plane_client_->init(config.messager_io_thread_count);
    }
    return true;
}

bool MooncakeKVCacheSender::regMem(const BlockInfo& block_info, uint64_t aligned_size) {
    if (!adapter_) {
        RTP_LLM_LOG_ERROR("MooncakeKVCacheSender regMem failed: adapter is null");
        return false;
    }
    return adapter_->registerLocalMemory(block_info, aligned_size);
}

bool MooncakeKVCacheSender::buildWriteRequests(const SendRequest& request,
                                               const MooncakeRemoteDescriptor& descriptor,
                                               std::vector<MooncakeWriteRequest>* write_requests,
                                               TransferErrorCode* error_code,
                                               std::string* error_message) const {
    if (!write_requests || !error_code || !error_message) {
        return false;
    }
    if (request.block_info.empty()) {
        *error_code = TransferErrorCode::BUILD_REQUEST_FAILED;
        *error_message = "Mooncake send failed: block_info is empty";
        return false;
    }
    if (descriptor.segment_name.empty()) {
        *error_code = TransferErrorCode::BUILD_REQUEST_FAILED;
        *error_message = "Mooncake send failed: descriptor segment_name is empty";
        return false;
    }

    std::unordered_map<std::string, MooncakeRemoteBlockDescriptor> descriptor_index;
    descriptor_index.reserve(descriptor.blocks.size());
    for (const auto& block : descriptor.blocks) {
        descriptor_index.emplace(blockKey(block.cache_key, block.block_index), block);
    }

    std::vector<int64_t> cache_keys;
    cache_keys.reserve(request.block_info.size());
    for (const auto& [cache_key, _] : request.block_info) {
        cache_keys.push_back(cache_key);
    }
    std::sort(cache_keys.begin(), cache_keys.end());

    write_requests->clear();
    for (auto cache_key : cache_keys) {
        const auto info_it = request.block_info.find(cache_key);
        if (info_it == request.block_info.end() || !info_it->second) {
            continue;
        }
        const auto& blocks = info_it->second->blocks;
        for (size_t i = 0; i < blocks.size(); ++i) {
            const auto& block = blocks[i];
            if (!block.addr || block.size_bytes == 0) {
                continue;
            }
            const auto descriptor_it = descriptor_index.find(blockKey(cache_key, static_cast<uint32_t>(i)));
            if (descriptor_it == descriptor_index.end()) {
                *error_code = TransferErrorCode::BUFFER_MISMATCH;
                *error_message = "Mooncake send failed: descriptor block missing";
                write_requests->clear();
                return false;
            }
            if (descriptor_it->second.len != block.size_bytes) {
                *error_code = TransferErrorCode::BUFFER_MISMATCH;
                *error_message = "Mooncake send failed: descriptor length mismatch";
                write_requests->clear();
                return false;
            }
            if (descriptor_it->second.target_addr == 0) {
                *error_code = TransferErrorCode::BUFFER_MISMATCH;
                *error_message = "Mooncake send failed: descriptor target address is empty";
                write_requests->clear();
                return false;
            }

            MooncakeWriteRequest write_request;
            write_request.source_addr = block.addr;
            write_request.segment_name = descriptor.segment_name;
            write_request.target_addr = descriptor_it->second.target_addr;
            write_request.length = descriptor_it->second.len;
            write_request.cache_key = cache_key;
            write_request.block_index = static_cast<uint32_t>(i);
            write_requests->push_back(write_request);
        }
    }

    if (write_requests->empty()) {
        *error_code = TransferErrorCode::BUILD_REQUEST_FAILED;
        *error_message = "Mooncake send failed: no non-empty block to transfer";
        return false;
    }
    if (write_requests->size() != descriptor.blocks.size()) {
        *error_code = TransferErrorCode::BUFFER_MISMATCH;
        *error_message = "Mooncake send failed: descriptor count mismatch";
        write_requests->clear();
        return false;
    }

    *error_code = TransferErrorCode::OK;
    error_message->clear();
    return true;
}

bool MooncakeKVCacheSender::waitTransferDone(uint64_t batch_id,
                                             int64_t deadline_ms,
                                             TransferErrorCode* error_code,
                                             std::string* error_message) const {
    if (!error_code || !error_message) {
        return false;
    }

    while (true) {
        bool finished = false;
        *error_code = adapter_->getTransferStatus(batch_id, &finished, error_message);
        if (finished) {
            return *error_code == TransferErrorCode::OK;
        }
        if (deadline_ms > 0 && currentTimeMs() >= deadline_ms) {
            *error_code = TransferErrorCode::TIMEOUT;
            *error_message = "Mooncake send failed: wait transfer status timed out";
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

uint32_t MooncakeKVCacheSender::resolveControlPlanePort(const SendRequest& request) const {
    if (request.port > 0) {
        return request.port;
    }
    return config_.control_plane_port > 0 ? static_cast<uint32_t>(config_.control_plane_port) : 0;
}

void MooncakeKVCacheSender::send(const SendRequest& request,
                                 std::function<void(TransferErrorCode, const std::string&)> callback) {
    if (!callback) {
        return;
    }
    if (!adapter_ || !control_plane_client_) {
        callback(TransferErrorCode::UNKNOWN, "Mooncake sender is not fully initialized");
        return;
    }

    const auto control_plane_port = resolveControlPlanePort(request);
    if (control_plane_port == 0) {
        callback(TransferErrorCode::CONNECTION_FAILED, "Mooncake send failed: control plane port is empty");
        return;
    }

    MooncakeRemoteDescriptor descriptor;
    TransferErrorCode        error_code = TransferErrorCode::OK;
    std::string              error_message;
    if (!control_plane_client_->prepare(
            request.ip, control_plane_port, request.unique_key, request.deadline_ms, &descriptor, &error_code, &error_message)) {
        callback(error_code, error_message);
        return;
    }

    if (!adapter_->openSegment(descriptor.segment_name)) {
        error_code = TransferErrorCode::UNKNOWN;
        error_message = "Mooncake send failed: openSegment failed";
        TransferErrorCode finish_error_code = TransferErrorCode::OK;
        std::string finish_error_message;
        control_plane_client_->finish(request.ip,
                                      control_plane_port,
                                      request.unique_key,
                                      false,
                                      error_code,
                                      error_message,
                                      &finish_error_code,
                                      &finish_error_message);
        callback(error_code, error_message);
        return;
    }

    std::vector<MooncakeWriteRequest> write_requests;
    if (!buildWriteRequests(request, descriptor, &write_requests, &error_code, &error_message)) {
        TransferErrorCode finish_error_code = TransferErrorCode::OK;
        std::string finish_error_message;
        control_plane_client_->finish(request.ip,
                                      control_plane_port,
                                      request.unique_key,
                                      false,
                                      error_code,
                                      error_message,
                                      &finish_error_code,
                                      &finish_error_message);
        callback(error_code, error_message);
        return;
    }

    const auto batch_id = adapter_->allocateBatchID(write_requests.size());
    if (batch_id == kInvalidMooncakeBatchId) {
        error_code = TransferErrorCode::UNKNOWN;
        error_message = "Mooncake send failed: allocateBatchID failed";
        TransferErrorCode finish_error_code = TransferErrorCode::OK;
        std::string finish_error_message;
        control_plane_client_->finish(request.ip,
                                      control_plane_port,
                                      request.unique_key,
                                      false,
                                      error_code,
                                      error_message,
                                      &finish_error_code,
                                      &finish_error_message);
        callback(error_code, error_message);
        return;
    }

    if (!adapter_->submitTransfer(batch_id, write_requests)) {
        error_code = TransferErrorCode::UNKNOWN;
        error_message = "Mooncake send failed: submitTransfer failed";
        adapter_->freeBatchID(batch_id);
        TransferErrorCode finish_error_code = TransferErrorCode::OK;
        std::string finish_error_message;
        control_plane_client_->finish(request.ip,
                                      control_plane_port,
                                      request.unique_key,
                                      false,
                                      error_code,
                                      error_message,
                                      &finish_error_code,
                                      &finish_error_message);
        callback(error_code, error_message);
        return;
    }

    (void)waitTransferDone(batch_id, request.deadline_ms, &error_code, &error_message);
    adapter_->freeBatchID(batch_id);

    TransferErrorCode finish_error_code = TransferErrorCode::OK;
    std::string       finish_error_message;
    const bool finish_ok = control_plane_client_->finish(request.ip,
                                                         control_plane_port,
                                                         request.unique_key,
                                                         error_code == TransferErrorCode::OK,
                                                         error_code,
                                                         error_message,
                                                         &finish_error_code,
                                                         &finish_error_message);
    if (error_code == TransferErrorCode::OK && (!finish_ok || finish_error_code != TransferErrorCode::OK)) {
        error_code = finish_error_code;
        error_message = finish_error_message;
    }
    callback(error_code, error_code == TransferErrorCode::OK ? std::string() : error_message);
}

}  // namespace mooncake
}  // namespace transfer
}  // namespace rtp_llm
