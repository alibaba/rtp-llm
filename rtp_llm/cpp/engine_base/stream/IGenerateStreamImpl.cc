#include "rtp_llm/cpp/engine_base/stream/IGenerateStreamImpl.h"

#include <torch/extension.h>
#include <torch/all.h>
#include <cstring>
#include <stdexcept>
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

// Helper function: Convert TensorPB to torch::Tensor
torch::Tensor transTensor(const TensorPB& tensor_pb) {
    std::vector<int64_t> shape(tensor_pb.shape().begin(), tensor_pb.shape().end());
    void*                data_ptr = nullptr;
    switch (tensor_pb.data_type()) {
        case TensorPB::FP32: {
            data_ptr     = const_cast<char*>(tensor_pb.fp32_data().data());
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            return torch::from_blob(data_ptr, shape, options).clone();
        }
        case TensorPB::INT32: {
            data_ptr     = const_cast<char*>(tensor_pb.int32_data().data());
            auto options = torch::TensorOptions().dtype(torch::kInt32);
            return torch::from_blob(data_ptr, shape, options).clone();
        }
        case TensorPB::FP16: {
            data_ptr     = const_cast<char*>(tensor_pb.fp16_data().data());
            auto options = torch::TensorOptions().dtype(torch::kFloat16);
            return torch::from_blob(data_ptr, shape, options).clone();
        }
        case TensorPB::BF16: {
            data_ptr     = const_cast<char*>(tensor_pb.bf16_data().data());
            auto options = torch::TensorOptions().dtype(torch::kBFloat16);
            return torch::from_blob(data_ptr, shape, options).clone();
        }
        default:
            throw std::runtime_error("Unsupported data type.");
    }
}

// Helper function: Convert Buffer to TensorPB
void transTensorPB(TensorPB* t, const rtp_llm::Buffer* buffer) {
    RTP_LLM_CHECK(t != nullptr);
    RTP_LLM_CHECK_WITH_INFO(buffer->where() != rtp_llm::MemoryType::MEMORY_GPU,
                            "buffer is on gpu, not supported transfer to tensorpb");
    auto shape       = t->mutable_shape();
    auto shape_array = buffer->shape();
    shape->Resize(shape_array.size(), 0);
    memcpy(shape->mutable_data(), shape_array.data(), shape_array.size() * sizeof(int64_t));

    TensorPB_DataType data_type;
    switch (buffer->type()) {
        case rtp_llm::DataType::TYPE_FP32:
            data_type = TensorPB_DataType::TensorPB_DataType_FP32;
            t->set_fp32_data(reinterpret_cast<const char*>(buffer->data()), buffer->sizeBytes());
            break;
        case rtp_llm::DataType::TYPE_INT32:
            data_type = TensorPB_DataType::TensorPB_DataType_INT32;
            t->set_int32_data(reinterpret_cast<const char*>(buffer->data()), buffer->sizeBytes());
            break;
        case rtp_llm::DataType::TYPE_FP16:
            data_type = TensorPB_DataType::TensorPB_DataType_FP16;
            t->set_fp16_data(reinterpret_cast<const char*>(buffer->data()), buffer->sizeBytes());
            break;
        case rtp_llm::DataType::TYPE_BF16:
            data_type = TensorPB_DataType::TensorPB_DataType_BF16;
            t->set_bf16_data(reinterpret_cast<const char*>(buffer->data()), buffer->sizeBytes());
            break;
        default:
            throw std::invalid_argument("unsupport buffer data type: " + std::to_string(buffer->type()));
            break;
    }
    t->set_data_type(data_type);
}

}  // namespace

IGenerateStreamImpl::IGenerateStreamImpl(const std::shared_ptr<GenerateStream>& stream, rtp_llm::DeviceBase* device):
    stream_(stream), device_(device) {}

// appendTokenId - 参考 DecodeRpcServer.cc:160-183
void IGenerateStreamImpl::appendTokenId(int batch_id, int token_id) {
    stream_->setIsContextStream(false);
    stream_->step();

    auto new_tokens = device_->allocateBuffer(
        {rtp_llm::DataType::TYPE_INT32, {(size_t)stream_->nextBatchSize(), (size_t)1}, rtp_llm::AllocationType::HOST},
        {});

    auto data = new_tokens->data<int32_t>();
    *data     = token_id;

    // stream_->incLastOutputPos();
    stream_->update({new_tokens, 1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr});
    RTP_LLM_LOG_DEBUG("append token id: %d, stream id: %ld", token_id, stream_->streamId());
}

// currentExecuteTokens - 参考 GenerateStream.h:224
std::vector<int> IGenerateStreamImpl::currentExecuteTokens(int batch_id) {
    return stream_->currentExecuteTokens(batch_id);
}

// appendSPInfo - 参考 DecodeRpcServer.cc:184-208
void IGenerateStreamImpl::appendSPInfo(const std::vector<int>& propose_tokens,
                                       const TensorPB&         propose_probs,
                                       const TensorPB&         propose_hidden) {
    stream_->setReuseLength(stream_->seqLength() - 1);
    stream_->setSpEditRun(false);
    stream_->setMtpTokenIndex(stream_->seqLength() - 1);
    stream_->setContainProposeToken(true);
    stream_->setProposeToken(propose_tokens);

    auto sp_output_buffer = std::make_shared<SpeculativeExecutorStreamOutput>();
    auto propose_token =
        device_->allocateBuffer({DataType::TYPE_INT32, {1, propose_tokens.size()}, AllocationType::HOST});
    memcpy(propose_token->data<int>(), propose_tokens.data(), propose_tokens.size() * sizeof(int));
    sp_output_buffer->tokens = propose_token;

    auto propose_probs_t  = transTensor(propose_probs);
    auto propose_hidden_t = transTensor(propose_hidden);

    auto& tensors_holder = sp_output_buffer->tensors_holder;
    tensors_holder.emplace_back(std::move(propose_probs_t));
    tensors_holder.emplace_back(std::move(propose_hidden_t));

    stream_->setSPOutputBuffer(sp_output_buffer);
    RTP_LLM_LOG_DEBUG("append sp info, stream id: %ld", stream_->streamId());
}

// getSPInfoPB - 返回 TensorPB 格式的 propose 信息
std::optional<std::tuple<std::vector<int>, TensorPB, TensorPB>> IGenerateStreamImpl::getSPInfoPB() {
    auto& propose_tokens = stream_->getProposeToken();
    if (propose_tokens.empty()) {
        return std::nullopt;
    }

    auto sp_output_buffer = stream_->getSPOutputBuffer();
    if (!sp_output_buffer) {
        return std::nullopt;
    }

    // Clone to HOST if on GPU
    if (sp_output_buffer->all_probs && sp_output_buffer->all_probs->where() == rtp_llm::MemoryType::MEMORY_GPU) {
        sp_output_buffer->all_probs = device_->clone({*sp_output_buffer->all_probs, rtp_llm::AllocationType::HOST});
    }

    if (!sp_output_buffer->hidden_states) {
        // dummy hidden states, so datatype is not important
        sp_output_buffer->hidden_states =
            device_->allocateBuffer({rtp_llm::DataType::TYPE_FP16, {0}, rtp_llm::AllocationType::HOST});
    }

    if (sp_output_buffer->hidden_states->where() == rtp_llm::MemoryType::MEMORY_GPU) {
        sp_output_buffer->hidden_states =
            device_->clone({*sp_output_buffer->hidden_states, rtp_llm::AllocationType::HOST});
    }

    // Convert Buffer to TensorPB
    TensorPB probs_pb;
    TensorPB hidden_pb;
    transTensorPB(&probs_pb, sp_output_buffer->all_probs.get());
    transTensorPB(&hidden_pb, sp_output_buffer->hidden_states.get());

    return std::make_tuple(propose_tokens, probs_pb, hidden_pb);
}

// reuseBlockNum - 计算 reuse blocks 数量
int IGenerateStreamImpl::reuseBlockNum() {
    return static_cast<int>(stream_->reuseBlockSize());
}

// getReuseLength - 参考 GenerateStream 方法
std::tuple<int, int, int> IGenerateStreamImpl::getReuseLength() {
    return std::make_tuple(stream_->initialReuseLength(), stream_->localReuseLength(), stream_->remoteReuseLength());
}

// setPrefillReuseLength - 设置 prefill 返回的 reuse 长度信息
void IGenerateStreamImpl::setPrefillReuseLength(int reuse_length, int local_reuse_length, int remote_reuse_length) {
    stream_->setPrefillReuseLength(reuse_length, local_reuse_length, remote_reuse_length);
}

// getPrefillAddr - 获取 prefill 地址
std::pair<std::string, uint32_t> IGenerateStreamImpl::getPrefillAddr() {
    return stream_->prefillAddr();
}

// getContextPositionIdsPB - 获取 context position ids
std::vector<int32_t> IGenerateStreamImpl::getContextPositionIdsPB() {
    auto context_position_ids = stream_->getContextPositionIds();
    if (!context_position_ids) {
        return {};
    }
    return std::vector<int32_t>(context_position_ids->data<int32_t>(),
                                context_position_ids->data<int32_t>() + context_position_ids->size());
}

// setContextPositionIds - 设置 context position ids
void IGenerateStreamImpl::setContextPositionIds(const std::vector<int32_t>& ids) {
    if (ids.empty()) {
        return;
    }
    auto context_position_ids =
        device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {ids.size()}, rtp_llm::AllocationType::HOST}, {});
    memcpy(context_position_ids->data<int32_t>(), ids.data(), ids.size() * sizeof(int32_t));
    stream_->setContextPositionIds(context_position_ids);
}

// waitForRemoteGenerate - 等待 first token 生成并通过 updateOutput 设置到 stream
bool IGenerateStreamImpl::waitForRemoteGenerate() {
    return stream_->waitForRemoteGenerate();
}

// getOriginalRequest - 获取原始请求
const GenerateInputPB* IGenerateStreamImpl::getOriginalRequest() const {
    return stream_->getOriginalRequest();
}

// needCallPrefill - 检查是否需要调用 prefill server
bool IGenerateStreamImpl::needCallPrefill() const {
    return stream_->needCallPrefill();
}

// setStop - 设置 stream 停止并设置错误码
void IGenerateStreamImpl::setStop(ErrorCode error_code, const std::string& error_msg) {
    stream_->setStop(error_code, error_msg);
}

}  // namespace rtp_llm
