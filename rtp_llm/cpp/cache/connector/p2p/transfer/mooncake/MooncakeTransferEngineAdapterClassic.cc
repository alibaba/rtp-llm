#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeTransferEngineAdapterProvider.h"

#include <mutex>
#include <unordered_map>

#include "transfer_engine.h"
#include "transfer_metadata.h"

#ifdef LOG_EVERY_N_VARNAME_CONCAT
#undef LOG_EVERY_N_VARNAME_CONCAT
#endif

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace transfer {
namespace mooncake {
namespace {

std::string resolveLocalServerName(const MooncakeTransferEngineInitConfig& config) {
    if (!config.local_server_name.empty()) {
        return config.local_server_name;
    }
    const auto host = config.ip_or_host_name.empty() ? std::string("127.0.0.1") : config.ip_or_host_name;
    return host + ":" + std::to_string(config.rpc_port);
}

std::string resolveTransportProto(const MooncakeTransferEngineInitConfig& config) {
    return config.transport.empty() ? std::string("tcp") : config.transport;
}

constexpr ::mooncake::SegmentHandle kInvalidSegmentHandle = static_cast<::mooncake::SegmentHandle>(-1);

bool isValidSegmentHandle(::mooncake::SegmentHandle handle) {
    return static_cast<int64_t>(handle) >= 0;
}

class MooncakeClassicTransferEngineAdapter : public IMooncakeTransferEngineAdapter {
public:
    ~MooncakeClassicTransferEngineAdapter() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (engine_) {
            engine_->freeEngine();
        }
    }

    bool init(const MooncakeBackendConfig& config) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (engine_) {
            return true;
        }

        config_ = config;
        engine_ = std::make_unique<::mooncake::TransferEngine>(false);

        const auto metadata_conn_string = config.classic.metadata_conn_string.empty()
                                              ? std::string(P2PHANDSHAKE)
                                              : config.classic.metadata_conn_string;
        const auto local_server_name = resolveLocalServerName(config.classic);
        const auto transport_proto = resolveTransportProto(config.classic);
        const auto ip_or_host_name = config.classic.ip_or_host_name.empty()
                                         ? std::string("127.0.0.1")
                                         : config.classic.ip_or_host_name;
        const int init_rc = engine_->init(
            metadata_conn_string, local_server_name, ip_or_host_name, config.classic.rpc_port);
        if (init_rc != 0) {
            RTP_LLM_LOG_ERROR("MooncakeClassicTransferEngineAdapter init failed, rc=%d", init_rc);
            engine_.reset();
            return false;
        }

        if (!engine_->installTransport(transport_proto, nullptr)) {
            RTP_LLM_LOG_ERROR("MooncakeClassicTransferEngineAdapter install transport failed, transport=%s",
                              transport_proto.c_str());
            engine_->freeEngine();
            engine_.reset();
            return false;
        }
        return true;
    }

    bool registerLocalMemory(const BlockInfo& block_info, uint64_t aligned_size) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!engine_ || !block_info.addr) {
            return false;
        }

        const uint64_t length = aligned_size > 0 ? aligned_size : static_cast<uint64_t>(block_info.size_bytes);
        auto           it = registered_memory_.find(block_info.addr);
        if (it != registered_memory_.end()) {
            return it->second == length;
        }

        const int rc = engine_->registerLocalMemory(
            block_info.addr, length, config_.location, config_.remote_accessible, config_.update_metadata);
        if (rc != 0) {
            RTP_LLM_LOG_ERROR("MooncakeClassicTransferEngineAdapter registerLocalMemory failed, rc=%d, addr=%p, length=%lu",
                              rc,
                              block_info.addr,
                              length);
            return false;
        }
        registered_memory_[block_info.addr] = length;
        return true;
    }

    bool openSegment(const std::string& segment_name) override {
        std::lock_guard<std::mutex> lock(mutex_);
        return isValidSegmentHandle(getOrOpenSegmentLocked(segment_name));
    }

    std::string getLocalServerName() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!engine_) {
            return resolveLocalServerName(config_.classic);
        }
        const auto local_ip_and_port = engine_->getLocalIpAndPort();
        return local_ip_and_port.empty() ? resolveLocalServerName(config_.classic) : local_ip_and_port;
    }

    uint64_t allocateBatchID(size_t request_count) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!engine_) {
            return kInvalidMooncakeBatchId;
        }
        const auto batch_id = engine_->allocateBatchID(request_count);
        return batch_id == ::mooncake::INVALID_BATCH_ID ? kInvalidMooncakeBatchId : batch_id;
    }

    void freeBatchID(uint64_t batch_id) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!engine_ || batch_id == kInvalidMooncakeBatchId) {
            return;
        }
        engine_->freeBatchID(batch_id);
    }

    bool submitTransfer(uint64_t batch_id, const std::vector<MooncakeWriteRequest>& requests) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!engine_ || batch_id == kInvalidMooncakeBatchId || requests.empty()) {
            return false;
        }

        std::vector<::mooncake::TransferRequest> te_requests;
        te_requests.reserve(requests.size());
        for (const auto& request : requests) {
            const auto handle = getOrOpenSegmentLocked(request.segment_name);
            if (!isValidSegmentHandle(handle) || !request.source_addr || request.target_addr == 0 || request.length == 0) {
                RTP_LLM_LOG_ERROR("MooncakeClassicTransferEngineAdapter submitTransfer invalid request, segment=%s, source=%p, target=%lu, length=%lu",
                                  request.segment_name.c_str(),
                                  request.source_addr,
                                  request.target_addr,
                                  request.length);
                return false;
            }

            ::mooncake::TransferRequest te_request;
            te_request.opcode = ::mooncake::TransferRequest::WRITE;
            te_request.source = const_cast<void*>(request.source_addr);
            te_request.target_id = handle;
            te_request.target_offset = request.target_addr;
            te_request.length = request.length;
            te_requests.push_back(te_request);
        }

        auto status = engine_->submitTransfer(batch_id, te_requests);
        if (!status.ok()) {
            RTP_LLM_LOG_ERROR("MooncakeClassicTransferEngineAdapter submitTransfer failed, status=%s",
                              status.ToString().c_str());
            return false;
        }
        return true;
    }

    TransferErrorCode getTransferStatus(uint64_t batch_id, bool* finished, std::string* error_message) override {
        if (finished) {
            *finished = false;
        }
        if (error_message) {
            error_message->clear();
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if (!engine_) {
            if (finished) {
                *finished = true;
            }
            if (error_message) {
                *error_message = "MooncakeClassicTransferEngineAdapter engine is not initialized";
            }
            return TransferErrorCode::UNKNOWN;
        }

        ::mooncake::TransferStatus transfer_status;
        auto status = engine_->getBatchTransferStatus(batch_id, transfer_status);
        if (!status.ok()) {
            if (finished) {
                *finished = true;
            }
            if (error_message) {
                *error_message = status.ToString();
            }
            return TransferErrorCode::UNKNOWN;
        }

        switch (transfer_status.s) {
            case ::mooncake::TransferStatusEnum::WAITING:
            case ::mooncake::TransferStatusEnum::PENDING:
                return TransferErrorCode::OK;
            case ::mooncake::TransferStatusEnum::COMPLETED:
                if (finished) {
                    *finished = true;
                }
                return TransferErrorCode::OK;
            case ::mooncake::TransferStatusEnum::CANCELED:
                if (finished) {
                    *finished = true;
                }
                if (error_message) {
                    *error_message = "Mooncake transfer cancelled";
                }
                return TransferErrorCode::CANCELLED;
            case ::mooncake::TransferStatusEnum::TIMEOUT:
                if (finished) {
                    *finished = true;
                }
                if (error_message) {
                    *error_message = "Mooncake transfer timed out";
                }
                return TransferErrorCode::TIMEOUT;
            case ::mooncake::TransferStatusEnum::FAILED:
            case ::mooncake::TransferStatusEnum::INVALID:
            default:
                if (finished) {
                    *finished = true;
                }
                if (error_message) {
                    *error_message = "Mooncake transfer failed";
                }
                return TransferErrorCode::UNKNOWN;
        }
    }

private:
    ::mooncake::SegmentHandle getOrOpenSegmentLocked(const std::string& segment_name) {
        if (!engine_ || segment_name.empty()) {
            return kInvalidSegmentHandle;
        }
        auto it = segment_handles_.find(segment_name);
        if (it != segment_handles_.end()) {
            return it->second;
        }

        const auto handle = engine_->openSegment(segment_name);
        if (!isValidSegmentHandle(handle)) {
            RTP_LLM_LOG_ERROR("MooncakeClassicTransferEngineAdapter openSegment failed, segment=%s, handle=%ld",
                              segment_name.c_str(),
                              static_cast<int64_t>(handle));
            return kInvalidSegmentHandle;
        }
        segment_handles_[segment_name] = handle;
        return handle;
    }

private:
    std::mutex                                          mutex_;
    MooncakeBackendConfig                               config_;
    std::unique_ptr<::mooncake::TransferEngine>         engine_;
    std::unordered_map<void*, uint64_t>                 registered_memory_;
    std::unordered_map<std::string, ::mooncake::SegmentHandle> segment_handles_;
};

}  // namespace

IMooncakeTransferEngineAdapterPtr createMooncakeTransferEngineAdapter() {
    return std::make_shared<MooncakeClassicTransferEngineAdapter>();
}

}  // namespace mooncake
}  // namespace transfer
}  // namespace rtp_llm
