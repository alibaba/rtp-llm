#pragma once

#include <functional>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>
#include <torch/python.h>
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalTypes.h"
#include "rtp_llm/cpp/multimodal_processor/MMRdmaTransport.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace py = pybind11;

namespace rtp_llm {

class RemoteMultimodalProcessor: public MultimodalProcessor {
public:
    RemoteMultimodalProcessor(const MMModelConfig& mm_model_config,
                              int64_t              max_seq_len,
                              const VitConfig&     vit_config = VitConfig()):
        MultimodalProcessor(py::none(), mm_model_config, max_seq_len) {
        // LLM consumer side of the encoder<->LLM RDMA fast path. nullptr when disabled /
        // unavailable, in which case every request transparently uses the inline-bytes path.
        rdma_transport_         = createMMRdmaTransport(vit_config, MMRdmaRole::LLM_CLIENT);
        rdma_release_timeout_ms_ = vit_config.mm_rdma_release_timeout_ms;
    }

private:
    MultimodalRpcPool                pool_;
    std::string                      vit_cluster_name_;
    std::shared_ptr<MMRdmaTransport> rdma_transport_;
    // Deadline (ms) for the best-effort slot-release RPC; keeps it off the critical path.
    int64_t                          rdma_release_timeout_ms_ = 1000;

    // Best-effort: tell the encoder it can return the slot to its free list.
    // This runs on the inference path, so the RPC is bounded by a short deadline: a slow
    // or hung encoder must never stall the request. Failure is logged only — the encoder's
    // slot-GC timeout (mm_rdma_slot_gc_timeout_ms) reclaims the slot as a backstop.
    template<typename Stub>
    void releaseRemoteSlot(Stub& stub, const std::string& handle) {
        grpc::ClientContext rel_ctx;
        rel_ctx.set_deadline(std::chrono::system_clock::now()
                             + std::chrono::milliseconds(rdma_release_timeout_ms_));
        ReleaseEmbeddingPB rel;
        rel.add_handle(handle);
        EmptyPB empty;
        auto    rel_status = stub->ReleaseMultimodalEmbedding(&rel_ctx, rel, &empty);
        if (!rel_status.ok()) {
            // Not fatal: the encoder's timeout GC will reclaim the slot.
            RTP_LLM_LOG_WARNING("ReleaseMultimodalEmbedding(handle=%s) failed: %s",
                                handle.c_str(),
                                rel_status.error_message().c_str());
        }
    }

    // Assemble a MultimodalOutput from the tensors pulled out of the single RDMA slot.
    // `mm_tensors` is parallel to output_pb->output_rdma().tensors() (same order the encoder
    // packed: embedding, optional pos_id, then per-image extra_input); we dispatch by role.
    MultimodalOutput assembleRdmaOutput(const std::vector<torch::Tensor>& mm_tensors,
                                        const MultimodalOutputPB*         output_pb) {
        const auto& desc = output_pb->output_rdma();
        RTP_LLM_CHECK_WITH_INFO(static_cast<int>(mm_tensors.size()) == desc.tensors_size(),
                                "rdma read tensor count=%zu does not match manifest size=%d",
                                mm_tensors.size(),
                                desc.tensors_size());

        torch::Tensor              mm_embedding;
        bool                       has_embedding = false;
        torch::Tensor              mm_position_id;
        bool                       has_pos_id = false;
        std::vector<torch::Tensor> extra_inputs;
        for (int i = 0; i < desc.tensors_size(); ++i) {
            switch (desc.tensors(i).role()) {
                case MMRdmaTensorPB::EMBEDDING:
                    RTP_LLM_CHECK_WITH_INFO(!has_embedding, "rdma manifest carries more than one embedding");
                    mm_embedding  = mm_tensors[i];
                    has_embedding = true;
                    break;
                case MMRdmaTensorPB::POS_ID:
                    RTP_LLM_CHECK_WITH_INFO(!has_pos_id, "rdma manifest carries more than one pos_id");
                    // PositionIdsGenerator reads pos_id on the host (data_ptr<int32_t>()), but the
                    // RDMA read lands it in GPU memory. Bring it back to CPU so the host deref is
                    // valid — matching the inline-bytes path (and the embedding/extra_input tensors
                    // stay on GPU, where their consumers move them with .to(kCUDA) anyway).
                    mm_position_id = mm_tensors[i].to(torch::kCPU);
                    has_pos_id     = true;
                    break;
                case MMRdmaTensorPB::EXTRA_INPUT:
                    extra_inputs.emplace_back(mm_tensors[i]);
                    break;
                default:
                    RTP_LLM_CHECK_WITH_INFO(false, "rdma manifest has unknown tensor role %d", desc.tensors(i).role());
            }
        }
        RTP_LLM_CHECK_WITH_INFO(has_embedding, "rdma manifest has no embedding tensor");

        std::vector<int64_t> split_sizes;
        for (auto split_size : output_pb->split_size()) {
            split_sizes.push_back(split_size);
        }
        const int64_t split_total = std::accumulate(split_sizes.begin(), split_sizes.end(), int64_t{0});
        RTP_LLM_CHECK_WITH_INFO(!split_sizes.empty() && split_total == mm_embedding.size(0),
                                "split_sizes sum=%ld does not match rdma mm_embedding.size(0)=%ld",
                                split_total,
                                mm_embedding.size(0));

        MultimodalOutput mm_output;
        mm_output.mm_features = mm_embedding.split(split_sizes, 0);

        if (has_pos_id) {
            RTP_LLM_CHECK_WITH_INFO(split_total == mm_position_id.size(0),
                                    "split_sizes sum=%ld does not match mm_position_id.size(0)=%ld",
                                    split_total,
                                    mm_position_id.size(0));
            mm_output.mm_position_ids = mm_position_id.split(split_sizes, 0);
        }
        if (!extra_inputs.empty()) {
            // extra_input is one tensor per image, so its count must match the image count
            // (== number of split_sizes); the order is the per-image order preserved above.
            RTP_LLM_CHECK_WITH_INFO(extra_inputs.size() == split_sizes.size(),
                                    "rdma extra_input count=%zu does not match image count=%zu",
                                    extra_inputs.size(),
                                    split_sizes.size());
            mm_output.mm_extra_input = std::move(extra_inputs);
        }
        return mm_output;
    }

    ErrorResult<MultimodalOutput> MultimodalEmbedding(const std::vector<rtp_llm::MultimodalInput> mm_inputs,
                                                      std::string                                 ip_port = "") {
        if (ip_port == "") {
            return ErrorInfo(ErrorCode::MM_NOT_SUPPORTED_ERROR, "ip:port is empty in remote multimodal processing");
        }
        auto connection_status = pool_.getConnection(ip_port);
        if (!connection_status.ok()) {
            return ErrorInfo(ErrorCode::MM_EMPTY_ENGINE_ERROR, connection_status.status().ToString());
        }
        auto&               connection = connection_status.value();
        auto                stub       = connection.stub;
        MultimodalOutputPB  output_pb;
        grpc::ClientContext context;

        auto request = QueryConverter::transMMInputsPB(mm_inputs);
        if (rdma_transport_ != nullptr) {
            request.set_support_rdma(true);
        }
        auto status = stub->RemoteMultimodalEmbedding(&context, request, &output_pb);
        if (!status.ok()) {
            return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, status.error_message());
        }

        // RDMA fast path: encoder returned a descriptor instead of inline output bytes.
        if (rdma_transport_ != nullptr && output_pb.has_output_rdma()) {
            const auto&                desc = output_pb.output_rdma();
            std::vector<torch::Tensor> mm_tensors;
            const bool                 read_ok = rdma_transport_->readEmbedding(desc, &mm_tensors);
            releaseRemoteSlot(stub, desc.handle());  // either way, free the encoder slot
            if (read_ok) {
                // Benchmark hook: MM_RDMA_READ_ONLY=1 aborts the request right after a
                // successful RDMA READ (already timed + logged as [MM-RDMA-BW]), so we can
                // measure pure read bandwidth without running prefill/decode. The request
                // fails cleanly at the multimodal stage; the engine stays healthy for the
                // next request.
                static const bool read_only = [] {
                    const char* e = std::getenv("MM_RDMA_READ_ONLY");
                    return e != nullptr && std::string(e) == "1";
                }();
                if (read_only) {
                    return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, "MM_RDMA_READ_ONLY: aborted after rdma read");
                }
                return assembleRdmaOutput(mm_tensors, &output_pb);
            }
            // RDMA read failed. As agreed, fall back to the inline-bytes path: re-issue the
            // request with support_rdma=false so the encoder returns the embedding as bytes
            // instead of a descriptor. The slot from the failed attempt was already released
            // above (and is GC-backed regardless).
            RTP_LLM_LOG_WARNING("rdma read of multimodal embedding failed (handle=%s), "
                                "falling back to inline bytes",
                                desc.handle().c_str());
            request.set_support_rdma(false);
            MultimodalOutputPB  fallback_pb;
            grpc::ClientContext fallback_context;
            auto fallback_status = stub->RemoteMultimodalEmbedding(&fallback_context, request, &fallback_pb);
            if (!fallback_status.ok()) {
                return ErrorInfo(ErrorCode::MM_PROCESS_ERROR,
                                 "rdma read failed and inline-bytes fallback also failed: "
                                     + fallback_status.error_message());
            }
            // With support_rdma=false the encoder must answer with inline bytes; a descriptor
            // here would mean a protocol violation we cannot consume.
            if (fallback_pb.has_output_rdma()) {
                return ErrorInfo(ErrorCode::MM_PROCESS_ERROR,
                                 "rdma read failed and fallback response unexpectedly carried "
                                 "an rdma descriptor");
            }
            return QueryConverter::transMMOutput(&fallback_pb);
        }

        return QueryConverter::transMMOutput(&output_pb);
    }
};

}  // namespace rtp_llm
