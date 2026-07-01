#pragma once

#include <chrono>
#include <functional>
#include <algorithm>
#include <chrono>
#include <string>
#include <vector>
#include <torch/python.h>
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalTypes.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace py = pybind11;

namespace rtp_llm {

class RemoteMultimodalProcessor: public MultimodalProcessor {
public:
    RemoteMultimodalProcessor(const MMModelConfig&         mm_model_config,
                              int64_t                      max_seq_len,
                              kmonitor::MetricsReporterPtr metrics_reporter = nullptr):
        MultimodalProcessor(py::none(), mm_model_config, max_seq_len, metrics_reporter) {}

private:
    MultimodalRpcPool pool_;
    std::string       vit_cluster_name_;

    void reportRpcClientError(const std::string& ip_port,
                              const std::string& reason,
                              const std::string& grpc_code = "") const {
        if (!metrics_reporter_) {
            return;
        }
        kmonitor::MetricsTags error_tags;
        error_tags.AddTag("source", "inference_client");
        error_tags.AddTag("target", ip_port);
        error_tags.AddTag("reason", reason);
        if (!grpc_code.empty()) {
            error_tags.AddTag("grpc_code", grpc_code);
        }
        metrics_reporter_->report(1, "rtp_llm_vit_rpc_client_error_qps", kmonitor::MetricType::QPS, &error_tags, true);
    }

    void reportRpcMetrics(const std::string&  ip_port,
                          int64_t             cost_us,
                          int64_t             request_bytes,
                          int64_t             response_bytes,
                          const grpc::Status* status = nullptr) const {
        if (!metrics_reporter_) {
            return;
        }

        kmonitor::MetricsTags tags;
        tags.AddTag("source", "inference_client");
        tags.AddTag("target", ip_port);
        metrics_reporter_->report(cost_us, "rtp_llm_vit_rpc_client_rt_us", kmonitor::MetricType::GAUGE, &tags, true);
        metrics_reporter_->report(
            request_bytes, "rtp_llm_vit_rpc_request_bytes", kmonitor::MetricType::GAUGE, &tags, true);
        metrics_reporter_->report(
            response_bytes, "rtp_llm_vit_rpc_response_bytes", kmonitor::MetricType::GAUGE, &tags, true);

        if (status != nullptr && !status->ok()) {
            reportRpcClientError(ip_port, "grpc_error", std::to_string(static_cast<int>(status->error_code())));
        }
    }

    ErrorResult<MultimodalOutput> MultimodalEmbedding(const std::vector<rtp_llm::MultimodalInput> mm_inputs,
                                                      std::string                                 ip_port = "") {
        if (ip_port == "") {
            return ErrorInfo(ErrorCode::MM_NOT_SUPPORTED_ERROR, "ip:port is empty in remote multimodal processing");
        }
        auto connection_status = pool_.getConnection(ip_port);
        if (!connection_status.ok()) {
            reportRpcClientError(ip_port, "connection_error");
            return ErrorInfo(ErrorCode::MM_EMPTY_ENGINE_ERROR, connection_status.status().ToString());
        }
        auto&               connection = connection_status.value();
        auto                stub       = connection.stub;
        MultimodalOutputPB  output_pb;
        grpc::ClientContext context;

        // Set gRPC deadline from the max mm_timeout_ms across all inputs.
        // Start from 0 so explicitly small timeouts (e.g. 1000ms) are honored;
        // fall back to the 30s default only when no input configures a positive timeout.
        int32_t max_timeout_ms = 0;
        for (const auto& mm_input : mm_inputs) {
            max_timeout_ms = std::max(max_timeout_ms, mm_input.mm_preprocess_config.mm_timeout_ms);
        }
        if (max_timeout_ms <= 0) {
            max_timeout_ms = 30000;  // default 30 s
        }
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(max_timeout_ms));

        auto                request_pb    = QueryConverter::transMMInputsPB(mm_inputs);
        const int64_t       request_bytes = request_pb.ByteSizeLong();
        const auto          start         = std::chrono::steady_clock::now();
        auto                status        = stub->RemoteMultimodalEmbedding(&context, request_pb, &output_pb);
        const int64_t       cost_us =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
        reportRpcMetrics(ip_port, cost_us, request_bytes, output_pb.ByteSizeLong(), &status);

        if (!status.ok()) {
            // Remove the bad connection on failure (timeout, unavailable, etc.)
            // so subsequent requests don't reuse a dead/stuck VIT worker.
            pool_.removeConnection(ip_port);
            return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, status.error_message());
        }
        return QueryConverter::transMMOutput(&output_pb);
    }
};

}  // namespace rtp_llm
