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
    const EngineInitParams& maga_init_params) {
    engine_.reset(new NormalEngine(maga_init_params));
}

grpc::Status ModelRpcServiceImpl::generate_stream(grpc::ServerContext*                  context,
                                                  const GenerateInputPB*                request,
                                                  grpc::ServerWriter<GenerateOutputsPB>* writer) {
    FT_LOG_DEBUG("receive request %ld", request->request_id());
    auto input = QueryConverter::transQuery(request);
    FT_LOG_DEBUG("request:[%ld] trans to stream success", request->request_id());
    auto stream = engine_->enqueue(input);
    FT_LOG_DEBUG("request:[%ld] enqueue success", request->request_id());
    while (!stream->finished()) {
        if (context->IsCancelled()) {
            stream->cancel();
            FT_LOG_DEBUG("request:[%ld] cancel", request->request_id());
            break;
        }
        const auto output_status = stream->nextOutput();
        if (!output_status.ok()) {
            FT_LOG_DEBUG("request:[%ld] generate error %s", request->request_id(), output_status.status().ToString().c_str());
            return grpc::Status(grpc::StatusCode::INTERNAL, output_status.status().ToString());
        }
        FT_LOG_DEBUG("request:[%ld] generate next output success", request->request_id());
        GenerateOutputsPB outputs_pb;
        QueryConverter::transResponse(&outputs_pb, &(output_status.value()));
        if (!writer->Write(outputs_pb)) {
            FT_LOG_DEBUG("request:[%ld] write outputs pb failed", request->request_id());
            break;
        }
    }
    FT_LOG_DEBUG("request:[%ld] generate done", request->request_id());
    return grpc::Status::OK;
}

void ModelRpcServiceImpl::addLoRA(const int64_t                                                   lora_id,
                       const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_a_weights,
                       const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_b_weights) {
    (void)engine_->addLoRA(lora_id, lora_a_weights, lora_b_weights);
}

void ModelRpcServiceImpl::removeLoRA(const int64_t lora_id) {
    (void)engine_->removeLoRA(lora_id);
}


}  // namespace rtp_llm
