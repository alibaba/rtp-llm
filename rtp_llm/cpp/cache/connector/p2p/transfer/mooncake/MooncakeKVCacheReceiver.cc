#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeKVCacheReceiver.h"

#include <algorithm>
#include <vector>

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
namespace transfer {
namespace mooncake {
namespace {

std::string resolveLocalServerName(const MooncakeBackendConfig& config) {
    if (!config.classic.local_server_name.empty()) {
        return config.classic.local_server_name;
    }
    const auto host = config.classic.ip_or_host_name.empty() ? std::string("127.0.0.1") : config.classic.ip_or_host_name;
    return host + ":" + std::to_string(config.classic.rpc_port);
}

}  // namespace

MooncakeKVCacheReceiver::MooncakeKVCacheReceiver(const IMooncakeTransferEngineAdapterPtr& adapter,
                                                 const kmonitor::MetricsReporterPtr& metrics_reporter):
    adapter_(adapter), task_store_(std::make_shared<TransferTaskStore>()), metrics_reporter_(metrics_reporter) {}

MooncakeKVCacheReceiver::~MooncakeKVCacheReceiver() {
    tcp_server_.reset();
    transfer_service_.reset();
}

bool MooncakeKVCacheReceiver::init(const TransferBackendConfig& config) {
    config_ = config.mooncake;
    if (!adapter_) {
        RTP_LLM_LOG_ERROR("MooncakeKVCacheReceiver init failed: adapter is null");
        return false;
    }
    if (!adapter_->init(config_)) {
        RTP_LLM_LOG_ERROR("MooncakeKVCacheReceiver init failed: adapter init failed");
        return false;
    }

    transfer_service_ = std::make_shared<MooncakeTransferService>(this, metrics_reporter_);
    return startControlPlaneServer(config);
}

bool MooncakeKVCacheReceiver::regMem(const BlockInfo& block_info, uint64_t aligned_size) {
    if (!adapter_) {
        RTP_LLM_LOG_ERROR("MooncakeKVCacheReceiver regMem failed: adapter is null");
        return false;
    }
    return adapter_->registerLocalMemory(block_info, aligned_size);
}

IKVCacheRecvTaskPtr MooncakeKVCacheReceiver::recv(const RecvRequest& request) {
    auto task = task_store_->addTask(request.unique_key, request.block_info, request.deadline_ms);
    if (!task) {
        return nullptr;
    }

    auto descriptor = buildDescriptor(request.unique_key, request.block_info);
    {
        std::unique_lock<std::shared_mutex> lock(descriptor_mutex_);
        descriptor_index_[request.unique_key] = std::move(descriptor);
    }
    return task;
}

void MooncakeKVCacheReceiver::stealTask(const std::string& unique_key) {
    task_store_->stealTask(unique_key);
    eraseDescriptor(unique_key);
}

IKVCacheRecvTaskPtr MooncakeKVCacheReceiver::getTask(const std::string& unique_key) {
    return task_store_->getTask(unique_key);
}

bool MooncakeKVCacheReceiver::prepareDescriptor(const std::string& unique_key,
                                                int64_t deadline_ms,
                                                MooncakeRemoteDescriptor* descriptor,
                                                TransferErrorCode* error_code,
                                                std::string* error_message) {
    if (!descriptor || !error_code || !error_message) {
        return false;
    }
    if (deadline_ms > 0 && currentTimeMs() >= deadline_ms) {
        *error_code = TransferErrorCode::TIMEOUT;
        *error_message = "Mooncake prepare timed out before descriptor lookup";
        return false;
    }

    auto task = task_store_->getTask(unique_key);
    if (!task) {
        *error_code = TransferErrorCode::CANCELLED;
        *error_message = "Mooncake prepare failed: task not found";
        eraseDescriptor(unique_key);
        return false;
    }
    if (task->done()) {
        *error_code = task->errorCode();
        *error_message = task->errorMessage();
        eraseDescriptor(unique_key);
        return false;
    }

    {
        std::shared_lock<std::shared_mutex> lock(descriptor_mutex_);
        auto it = descriptor_index_.find(unique_key);
        if (it == descriptor_index_.end()) {
            *error_code = TransferErrorCode::CANCELLED;
            *error_message = "Mooncake prepare failed: descriptor not found";
            return false;
        }
        *descriptor = it->second;
    }

    if (!task->startTransfer()) {
        *error_code = task->errorCode();
        *error_message = task->errorMessage().empty()
                             ? std::string("Mooncake prepare failed: task cancelled before transfer")
                             : task->errorMessage();
        eraseDescriptor(unique_key);
        return false;
    }

    *error_code = TransferErrorCode::OK;
    error_message->clear();
    return true;
}

bool MooncakeKVCacheReceiver::finishTransfer(const std::string& unique_key,
                                             bool success,
                                             TransferErrorCode error_code,
                                             const std::string& error_message,
                                             TransferErrorCode* response_error_code,
                                             std::string* response_error_message) {
    if (response_error_code) {
        *response_error_code = TransferErrorCode::OK;
    }
    if (response_error_message) {
        response_error_message->clear();
    }
    if (unique_key.empty()) {
        if (response_error_code) {
            *response_error_code = TransferErrorCode::UNKNOWN;
        }
        if (response_error_message) {
            *response_error_message = "Mooncake finish failed: unique_key is empty";
        }
        return false;
    }

    auto task = task_store_->getTask(unique_key);
    if (!task) {
        if (response_error_code) {
            *response_error_code = TransferErrorCode::CANCELLED;
        }
        if (response_error_message) {
            *response_error_message = "Mooncake finish failed: task not found";
        }
        eraseDescriptor(unique_key);
        return false;
    }

    task->notifyDone(success, success ? TransferErrorCode::OK : error_code, error_message);
    if (response_error_code) {
        *response_error_code = task->errorCode();
    }
    if (response_error_message) {
        *response_error_message = task->errorMessage();
    }
    eraseDescriptor(unique_key);
    return true;
}

bool MooncakeKVCacheReceiver::startControlPlaneServer(const TransferBackendConfig& config) {
    if (!transfer_service_) {
        RTP_LLM_LOG_ERROR("MooncakeKVCacheReceiver init failed: transfer_service is null");
        return false;
    }
    const int64_t listen_port = config_.control_plane_port > 0 ? config_.control_plane_port : config.cache_store_listen_port;
    if (listen_port <= 0) {
        RTP_LLM_LOG_INFO("MooncakeKVCacheReceiver init skip control plane server because listen port is %ld",
                         listen_port);
        return true;
    }

    tcp_server_ = std::make_shared<transfer::TcpServer>();
    if (!tcp_server_->init(config.messager_io_thread_count,
                           config.messager_worker_thread_count,
                           static_cast<uint32_t>(listen_port),
                           true,
                           static_cast<uint32_t>(config.cache_store_tcp_anet_rpc_thread_num),
                           static_cast<uint32_t>(config.cache_store_tcp_anet_rpc_queue_num))) {
        RTP_LLM_LOG_ERROR("MooncakeKVCacheReceiver init failed: control plane server init failed");
        return false;
    }
    if (!tcp_server_->registerService(transfer_service_.get())) {
        RTP_LLM_LOG_ERROR("MooncakeKVCacheReceiver init failed: register control plane service failed");
        return false;
    }
    if (!tcp_server_->start()) {
        RTP_LLM_LOG_ERROR("MooncakeKVCacheReceiver init failed: start control plane server failed");
        return false;
    }
    return true;
}

MooncakeRemoteDescriptor MooncakeKVCacheReceiver::buildDescriptor(const std::string& unique_key,
                                                                  const KeyBlockInfoMap& block_infos) const {
    (void)unique_key;
    MooncakeRemoteDescriptor descriptor;
    descriptor.segment_name = adapter_ ? adapter_->getLocalServerName() : std::string();
    if (descriptor.segment_name.empty()) {
        descriptor.segment_name = resolveLocalServerName(config_);
    }

    std::vector<int64_t> cache_keys;
    cache_keys.reserve(block_infos.size());
    for (const auto& [cache_key, _] : block_infos) {
        cache_keys.push_back(cache_key);
    }
    std::sort(cache_keys.begin(), cache_keys.end());

    for (auto cache_key : cache_keys) {
        const auto it = block_infos.find(cache_key);
        if (it == block_infos.end() || !it->second) {
            continue;
        }
        const auto& blocks = it->second->blocks;
        for (size_t i = 0; i < blocks.size(); ++i) {
            const auto& block_info = blocks[i];
            if (!block_info.addr || block_info.size_bytes == 0) {
                continue;
            }
            MooncakeRemoteBlockDescriptor descriptor_block;
            descriptor_block.cache_key = cache_key;
            descriptor_block.block_index = static_cast<uint32_t>(i);
            descriptor_block.target_addr = reinterpret_cast<uint64_t>(block_info.addr);
            descriptor_block.len = static_cast<uint64_t>(block_info.size_bytes);
            descriptor.blocks.push_back(descriptor_block);
        }
    }
    return descriptor;
}

void MooncakeKVCacheReceiver::eraseDescriptor(const std::string& unique_key) {
    std::unique_lock<std::shared_mutex> lock(descriptor_mutex_);
    descriptor_index_.erase(unique_key);
}

}  // namespace mooncake
}  // namespace transfer
}  // namespace rtp_llm
