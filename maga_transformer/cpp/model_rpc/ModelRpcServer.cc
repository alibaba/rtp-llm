#include "maga_transformer/cpp/model_rpc/ModelRpcServer.h"
#include "maga_transformer/cpp/dataclass/Query.h"
#include "maga_transformer/cpp/normal_engine/NormalEngine.h"
#include "maga_transformer/cpp/model_rpc/QueryConverter.h"
#include "maga_transformer/cpp/proto/model_rpc_service.pb.h"
#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <thread>

using namespace std;

namespace rtp_llm {

ModelRpcServiceImpl::ModelRpcServiceImpl(
    const MagaInitParams&                                                   maga_init_params,
    const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layer_weights,
    const std::unordered_map<std::string, ft::ConstBufferPtr>&              weights) {
    engine_.reset(new NormalEngine(maga_init_params, layer_weights, weights));
    auto kmon_tags = kmonitor::MetricsTags();
    metrics_reporter_.reset(new kmonitor::MetricsReporter("", "", kmon_tags));
}

grpc::Status ModelRpcServiceImpl::generate_stream(grpc::ServerContext*                  context,
                                                  const GenerateInputPB*                request,
                                                  grpc::ServerWriter<GenerateOutputPB>* writer) {
    FT_LOG_DEBUG("rec request %ld", request->request_id());
    auto stream = QueryConverter::transQuery(engine_->resourceContext(), request);
    FT_LOG_DEBUG("req:%ld trans to stream success", request->request_id());
    auto status = engine_->enqueue(stream);
    FT_LOG_DEBUG("req:%ld enqueue success", request->request_id());
    if (!status.ok()) {
        return grpc::Status(grpc::StatusCode::INTERNAL, status.ToString());
    }
    while (!stream->finished()) {
        if (context->IsCancelled()) {
            // std::cout << "cancel:" << count << std::endl;
            stream->cancel();
            FT_LOG_DEBUG("req:%ld cancel", request->request_id());
            break;
        }
        const auto output_status = stream->nextOutput();
        if (!output_status.ok()) {
            FT_LOG_DEBUG("req:%ld generate error %s", request->request_id(), output_status.status().ToString().c_str());
            return grpc::Status(grpc::StatusCode::INTERNAL, output_status.status().ToString());
        }
        FT_LOG_DEBUG("req:%ld generate next output suc", request->request_id());
        GenerateOutputPB output_pb;
        QueryConverter::transResponse(&output_pb, &(output_status.value()));
        writer->Write(output_pb);
    }
    FT_LOG_DEBUG("req:%ld generate over", request->request_id());
    return grpc::Status::OK;
}

void ModelRpcServiceImpl::addLoRA(const int64_t                                                   lora_id,
                       const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_a_weights,
                       const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_b_weights) {
    engine_->addLoRA(lora_id, lora_a_weights, lora_b_weights);
}

void ModelRpcServiceImpl::removeLoRA(const int64_t lora_id) {
    engine_->removeLoRA(lora_id);
}


}  // namespace rtp_llm
