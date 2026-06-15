#pragma once

#include <functional>
#include <algorithm>
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
        rdma_transport_ = createMMRdmaTransport(vit_config, MMRdmaRole::LLM_CLIENT);
    }

private:
    MultimodalRpcPool                pool_;
    std::string                      vit_cluster_name_;
    std::shared_ptr<MMRdmaTransport> rdma_transport_;

    // Best-effort: tell the encoder it can return the slot to its free list.
    template<typename Stub>
    void releaseRemoteSlot(Stub& stub, const std::string& handle) {
        grpc::ClientContext rel_ctx;
        ReleaseEmbeddingPB  rel;
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

    // Assemble a MultimodalOutput from an RDMA-read embedding tensor plus the inline
    // small tensors (pos_id / extra_input) carried in the same response.
    MultimodalOutput assembleRdmaOutput(const torch::Tensor& mm_embedding, const MultimodalOutputPB* output_pb) {
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

        if (output_pb->has_multimodal_pos_id()) {
            torch::Tensor mm_position_id = QueryConverter::transTensor(output_pb->multimodal_pos_id());
            RTP_LLM_CHECK_WITH_INFO(split_total == mm_position_id.size(0),
                                    "split_sizes sum=%ld does not match mm_position_id.size(0)=%ld",
                                    split_total,
                                    mm_position_id.size(0));
            mm_output.mm_position_ids = mm_position_id.split(split_sizes, 0);
        }
        if (output_pb->multimodal_extra_input_size() > 0) {
            std::vector<torch::Tensor> extra_inputs;
            extra_inputs.reserve(output_pb->multimodal_extra_input_size());
            for (const auto& extra_input_pb : output_pb->multimodal_extra_input()) {
                extra_inputs.emplace_back(QueryConverter::transTensor(extra_input_pb));
            }
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

        // RDMA fast path: encoder returned a descriptor instead of inline embedding bytes.
        if (rdma_transport_ != nullptr && output_pb.has_embedding_rdma()) {
            const auto&   desc = output_pb.embedding_rdma();
            torch::Tensor mm_embedding;
            const bool    read_ok = rdma_transport_->readEmbedding(desc, &mm_embedding);
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
                return assembleRdmaOutput(mm_embedding, &output_pb);
            }
            // No inline fallback exists for a descriptor-only response.
            return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, "rdma read of multimodal embedding failed");
        }

        return QueryConverter::transMMOutput(&output_pb);
    }
};

}  // namespace rtp_llm
