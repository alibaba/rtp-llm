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
#include <unordered_map>

using namespace std;

namespace rtp_llm {

// TODO: not use absl::status
int transErrorCode(absl::StatusCode code) {
    const static std::unordered_map<int, int> error_code_map = {
        {8, 602}, // kResourceExhausted, MALLOC_ERROR
        {4, 603}, // kDeadlineExceeded, TIMEOUT_ERROR
        {13, 514} // kInternal, UNKNOWN_ERROR
    };
    auto it = error_code_map.find((int)code);
    if (it != error_code_map.end()) {
        return it->second;
    } else {
        return 514;
    }
}

ModelRpcServiceImpl::ModelRpcServiceImpl(
    const EngineInitParams& maga_init_params) {
    engine_.reset(new NormalEngine(maga_init_params));
}

grpc::Status ModelRpcServiceImpl::generate_stream(grpc::ServerContext*                  context,
                                                  const GenerateInputPB*                request,
                                                  grpc::ServerWriter<GenerateOutputsPB>* writer) {
    FT_LOG_DEBUG("receive request %ld", request->request_id());
    auto input = QueryConverter::transQuery(request);
    std::shared_mutex* inner_mutex = nullptr;
    if (input->lora_id != -1) {
        std::lock_guard<std::mutex> g_lk(global_mutex_);

        auto it = lora_map_mutex_.find(input->lora_id);
        if (it != lora_map_mutex_.end() && it->second->alive_) {
            inner_mutex = it->second->mutex_.get();
        } else {
            FT_LOG_INFO("request:[%ld] error lora id[%ld] is not alive", request->request_id(), input->lora_id);
            return grpc::Status::CANCELLED;

        }
    }
    auto lock_scope = (inner_mutex == nullptr) ?
                      std::shared_lock<std::shared_mutex>() : std::shared_lock<std::shared_mutex>(*inner_mutex);
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
        if (context->IsCancelled()) {
            stream->cancel();
            FT_LOG_DEBUG("request:[%ld] cancel", request->request_id());
            break;
        }
        if (!output_status.ok()) {
            FT_LOG_INFO("request:[%ld] generate error %s", request->request_id(), output_status.status().ToString().c_str());
            auto status = output_status.status();
            ErrorDetailsPB error_details;
            error_details.set_error_code(transErrorCode(status.code()));
            error_details.set_error_message(status.ToString());
            std::string error_details_serialized;
            if (error_details.SerializeToString(&error_details_serialized)) {
                return grpc::Status(grpc::StatusCode::INTERNAL, status.ToString(), error_details_serialized);
            } else {
                FT_LOG_INFO("request:[%ld] SerializeToString error", request->request_id());
                return grpc::Status(grpc::StatusCode::INTERNAL, status.ToString());
            }
        }
        FT_LOG_DEBUG("request:[%ld] generate next output success", request->request_id());
        GenerateOutputsPB outputs_pb;
        QueryConverter::transResponse(&outputs_pb, &(output_status.value()));
        if (context->IsCancelled()) {
            stream->cancel();
            FT_LOG_DEBUG("request:[%ld] cancel", request->request_id());
            break;
        }
        if (!writer->Write(outputs_pb)) {
            FT_LOG_INFO("request:[%ld] write outputs pb failed", request->request_id());
            stream->cancel();
            break;
        }
    }
    FT_LOG_DEBUG("request:[%ld] generate done", request->request_id());
    return grpc::Status::OK;
}

void ModelRpcServiceImpl::addLoRA(const int64_t                                                   lora_id,
                       const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_a_weights,
                       const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_b_weights) {
    std::shared_mutex* inner_mutex = nullptr;
    {
        std::lock_guard<std::mutex> g_lk(global_mutex_);

        auto it = lora_map_mutex_.find(lora_id);
        if (it == lora_map_mutex_.end()) {
            auto lora_mutex_ptr = std::make_unique<LoraMutex>(
                LoraMutex({false, std::make_unique<std::shared_mutex>()}));
            it = lora_map_mutex_.emplace(lora_id, std::move(lora_mutex_ptr)).first;
        }
        inner_mutex = it->second->mutex_.get();
    }
    {
        std::unique_lock<std::shared_mutex> c_lk(*inner_mutex);
        (void)engine_->addLoRA(lora_id, lora_a_weights, lora_b_weights);
    }
    {
        std::lock_guard<std::mutex> g_lk(global_mutex_);
        auto it = lora_map_mutex_.find(lora_id);
        it->second->alive_ = true;
    }
}

void ModelRpcServiceImpl::removeLoRA(const int64_t lora_id) {
    std::shared_mutex* inner_mutex = nullptr;
    {
        std::lock_guard<std::mutex> g_lk(global_mutex_);

        auto it = lora_map_mutex_.find(lora_id);
        if (it == lora_map_mutex_.end() || it->second->alive_ == false) {
            return;
        }
        inner_mutex = it->second->mutex_.get();
        it->second->alive_ = false;
    }
    {
        std::unique_lock<std::shared_mutex> c_lk(*inner_mutex);
        (void)engine_->removeLoRA(lora_id);
    }
}

KVCacheInfo ModelRpcServiceImpl::getKVCacheInfo() const {
    return engine_->getKVCacheInfo();
}

}  // namespace rtp_llm
