#pragma once

#include <functional>
#include <chrono>
#include <algorithm>
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
#include "rtp_llm/cpp/multimodal_processor/MMRdmaTransport.h"

namespace py = pybind11;

namespace rtp_llm {

class RemoteMultimodalProcessor: public MultimodalProcessor {
public:
    RemoteMultimodalProcessor(py::object           mm_process_engine,
                              const MMModelConfig& mm_model_config,
                              int64_t              max_seq_len,
                              const VitConfig&     vit_config = {}):
        MultimodalProcessor(mm_process_engine, mm_model_config, max_seq_len),
        vit_config_(vit_config),
        rdma_transport_(createMMRdmaTransport(vit_config, MMRdmaRole::LLM_CLIENT)) {}

private:
    MultimodalRpcPool                pool_;
    VitConfig                        vit_config_;
    std::shared_ptr<MMRdmaTransport> rdma_transport_;

    ErrorResult<MultimodalOutput> MultimodalEmbedding(const std::vector<rtp_llm::MultimodalInput> mm_inputs,
                                                      std::string                                 ip_port = "",
                                                      grpc::ClientContext* rpc_context                    = nullptr) {
        return remoteEmbedding(QueryConverter::transMMInputsPB(mm_inputs), mm_inputs.size(), ip_port, rpc_context);
    }

    ErrorResult<MultimodalOutput> V41MultimodalEmbedding(const V41RequestInputs& inputs,
                                                         const std::string&      ip_port,
                                                         grpc::ClientContext*    rpc_context) override {
        MultimodalInputsPB request;
        auto*              typed = request.mutable_v41_inputs();
        typed->set_schema_version(1);
        auto token_types = inputs.token_types.to(torch::kCPU).to(torch::kInt32).contiguous();
        auto image_mask  = inputs.image_mask.to(torch::kCPU).to(torch::kBool).contiguous();
        for (int64_t index = 0; index < token_types.numel(); ++index) {
            typed->add_token_types(token_types.data_ptr<int32_t>()[index]);
        }
        for (int64_t index = 0; index < image_mask.numel(); ++index) {
            typed->add_image_mask(image_mask.data_ptr<bool>()[index]);
        }
        for (const auto& image : inputs.images) {
            auto* output = typed->add_images();
            output->set_start(image.start);
            output->set_n_vit_h(image.n_vit_h);
            output->set_n_vit_w(image.n_vit_w);
            QueryConverter::transTensorPB(output->mutable_patches(), image.patches);
            auto types = image.types.to(torch::kCPU).to(torch::kInt32).contiguous();
            for (int64_t index = 0; index < types.numel(); ++index) {
                output->add_types(types.data_ptr<int32_t>()[index]);
            }
            output->set_content_sha256(image.content_sha256);
            output->set_processor_identity(image.processor_identity);
        }
        return remoteEmbedding(std::move(request), inputs.images.size(), ip_port, rpc_context);
    }

    ErrorResult<MultimodalOutput> remoteEmbedding(MultimodalInputsPB   request,
                                                  size_t               expected_images,
                                                  const std::string&   ip_port,
                                                  grpc::ClientContext* rpc_context) {
        if (ip_port == "") {
            return ErrorInfo(ErrorCode::MM_NOT_SUPPORTED_ERROR, "ip:port is empty in remote multimodal processing");
        }
        auto connection_status = pool_.getConnection(ip_port);
        if (!connection_status.ok()) {
            return ErrorInfo(ErrorCode::MM_EMPTY_ENGINE_ERROR, connection_status.status().ToString());
        }
        auto& connection = connection_status.value();

        auto                stub = connection.stub;
        MultimodalOutputsPB output_pb;
        grpc::ClientContext local_context;
        auto*               context = rpc_context ? rpc_context : &local_context;
        request.set_support_rdma(rdma_transport_ != nullptr);
        auto status = stub->RemoteMultimodalEmbedding(context, request, &output_pb);
        if (!status.ok()) {
            auto code = status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED ? ErrorCode::GENERATE_TIMEOUT :
                        status.error_code() == grpc::StatusCode::CANCELLED         ? ErrorCode::CANCELLED :
                                                                                     ErrorCode::MM_PROCESS_ERROR;
            return ErrorInfo(code, status.error_message());
        }
        std::vector<std::string> releasable;
        for (const auto& item : output_pb.multimodal_outputs()) {
            if (item.has_output_rdma()) {
                releasable.push_back(item.output_rdma().handle());
            }
        }
        auto release_handles = [stub, timeout_ms = vit_config_.mm_rdma_release_timeout_ms](
                                   const std::vector<std::string>& handles) noexcept {
            try {
                if (handles.empty()) {
                    return;
                }
                // Cleanup has its own short bound; the embedding context is already consumed.
                grpc::ClientContext release_context;
                release_context.set_deadline(std::chrono::system_clock::now()
                                             + std::chrono::milliseconds(timeout_ms));
                ReleaseEmbeddingPB release_request;
                for (const auto& handle : handles) {
                    release_request.add_handle(handle);
                }
                EmptyPB ignored;
                auto    released = stub->ReleaseEmbedding(&release_context, release_request, &ignored);
                if (!released.ok()) {
                    RTP_LLM_LOG_WARNING("ViT slot release failed; slots remain bounded until restart: %s",
                                        released.error_message().c_str());
                }
            } catch (const std::exception& error) {
                try {
                    RTP_LLM_LOG_WARNING("ViT slot cleanup failed: %s", error.what());
                } catch (...) {}
            } catch (...) {
                // Best-effort cleanup must not terminate a reclaimer or unwind
                // past release_slots, including when logging cannot allocate.
            }
        };
        auto release_slots = [&]() noexcept { release_handles(releasable); };
        if (vit_config_.mm_transport_mode == "rdma"
            && (output_pb.multimodal_outputs_size() != expected_images || releasable.size() != expected_images)) {
            // No READ has started. Reclaim descriptors from a mixed response
            // before rejecting it; strict mode must never accept inline features.
            release_slots();
            return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, "Strict ViT RDMA requires a descriptor for every image");
        }
        auto convert = [&]() -> ErrorResult<MultimodalOutput> {
            if ((!releasable.empty() && !rdma_transport_) || output_pb.multimodal_outputs_size() != expected_images) {
                return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, "Unexpected ViT RDMA output count or transport");
            }
            MultimodalOutput output;
            for (const auto& item : output_pb.multimodal_outputs()) {
                if (item.has_output_rdma()) {
                    const auto& desc = item.output_rdma();
                    if (item.has_multimodal_embedding() || !validateMMRdmaDescriptor(desc)) {
                        return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, "Invalid ViT RDMA descriptor");
                    }
                    auto remaining_ms = vit_config_.mm_rdma_read_timeout_ms;
                    if (context->deadline() != std::chrono::system_clock::time_point::max()) {
                        remaining_ms = std::min<int64_t>(remaining_ms,
                                                         std::chrono::duration_cast<std::chrono::milliseconds>(
                                                             context->deadline() - std::chrono::system_clock::now())
                                                             .count());
                    }
                    if (remaining_ms <= 0) {
                        return ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "ViT RDMA request deadline exceeded");
                    }
                    // Until READ completion is confirmed, neither an exception nor a
                    // deadline may release this source slot to another request.
                    std::function<void()> release_after_completion = [release_handles, handle = desc.handle()]() {
                        release_handles({handle});
                    };
                    auto defer_release = [&]() {
                        releasable.erase(std::remove(releasable.begin(), releasable.end(), desc.handle()),
                                         releasable.end());
                    };
                    std::vector<torch::Tensor> tensors;
                    bool release_deferred = false;
                    MMRdmaReadStatus read_status;
                    try {
                        read_status = rdma_transport_->readEmbedding(
                            desc, &tensors, remaining_ms, std::move(release_after_completion), &release_deferred);
                    } catch (...) {
                        if (release_deferred) {
                            defer_release();
                        }
                        throw;
                    }
                    if (release_deferred || read_status == MMRdmaReadStatus::IN_FLIGHT) {
                        defer_release();
                    }
                    if (read_status != MMRdmaReadStatus::SUCCESS || tensors.size() != 1) {
                        auto code = context->deadline() <= std::chrono::system_clock::now() ?
                                        ErrorCode::GENERATE_TIMEOUT :
                                        ErrorCode::MM_PROCESS_ERROR;
                        return ErrorInfo(code,
                                         "ViT RDMA READ failed: " + std::to_string(static_cast<int>(read_status)));
                    }
                    output.mm_features.emplace_back(std::move(tensors[0]));
                } else {
                    if (!item.has_multimodal_embedding()) {
                        return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, "ViT response is missing an embedding");
                    }
                    output.mm_features.emplace_back(QueryConverter::transTensor(item.multimodal_embedding()));
                }
                if (item.has_multimodal_pos_id()) {
                    if (!output.mm_position_ids) {
                        output.mm_position_ids.emplace();
                    }
                    output.mm_position_ids->emplace_back(QueryConverter::transTensor(item.multimodal_pos_id()));
                }
            }
            if (output.mm_position_ids && output.mm_position_ids->size() != output.mm_features.size()) {
                return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, "ViT embedding/position count mismatch");
            }
            return output;
        };
        try {
            auto output = convert();
            release_slots();
            return output;
        } catch (const std::exception& error) {
            release_slots();
            return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, error.what());
        }
    }
};

}  // namespace rtp_llm
