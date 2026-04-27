#pragma once

#include <string>

#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferErrorCode.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeRemoteDescriptor.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/proto/mooncake_service.pb.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

namespace rtp_llm {
namespace transfer {
namespace mooncake {

class IMooncakeControlPlaneHandler {
public:
    virtual ~IMooncakeControlPlaneHandler() = default;

    virtual bool prepareDescriptor(const std::string& unique_key,
                                   int64_t deadline_ms,
                                   MooncakeRemoteDescriptor* descriptor,
                                   TransferErrorCode* error_code,
                                   std::string* error_message) = 0;

    virtual bool finishTransfer(const std::string& unique_key,
                                bool success,
                                TransferErrorCode error_code,
                                const std::string& error_message,
                                TransferErrorCode* response_error_code,
                                std::string* response_error_message) = 0;
};

class MooncakeTransferService : public ::mooncake_transfer::MooncakeTransferService {
public:
    explicit MooncakeTransferService(IMooncakeControlPlaneHandler* handler,
                                     const kmonitor::MetricsReporterPtr& metrics_reporter = nullptr);

    void prepare(::google::protobuf::RpcController* controller,
                 const ::mooncake_transfer::MooncakePrepareRequest* request,
                 ::mooncake_transfer::MooncakePrepareResponse* response,
                 ::google::protobuf::Closure* done) override;

    void finish(::google::protobuf::RpcController* controller,
                const ::mooncake_transfer::MooncakeFinishRequest* request,
                ::mooncake_transfer::MooncakeFinishResponse* response,
                ::google::protobuf::Closure* done) override;

private:
    IMooncakeControlPlaneHandler* handler_ = nullptr;
    kmonitor::MetricsReporterPtr  metrics_reporter_;
};

}  // namespace mooncake
}  // namespace transfer
}  // namespace rtp_llm
