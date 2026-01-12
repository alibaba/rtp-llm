#include "rtp_llm/cpp/engine_base/stream/IGenerateStreamImpl.h"

#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

namespace rtp_llm {

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
    RTP_LLM_LOG_INFO("append token id: %d, stream id: %ld", token_id, stream_->streamId());
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

    auto propose_probs_t  = QueryConverter::transTensor(propose_probs);
    auto propose_hidden_t = QueryConverter::transTensor(propose_hidden);

    auto& tensors_holder = sp_output_buffer->tensors_holder;
    tensors_holder.emplace_back(std::move(propose_probs_t));
    tensors_holder.emplace_back(std::move(propose_hidden_t));

    stream_->setSPOutputBuffer(sp_output_buffer);
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
    QueryConverter::transTensorPB(&probs_pb, sp_output_buffer->all_probs.get());
    QueryConverter::transTensorPB(&hidden_pb, sp_output_buffer->hidden_states.get());

    return std::make_tuple(propose_tokens, probs_pb, hidden_pb);
}

// reuseBlockNum - 参考 ReuseInfo
int IGenerateStreamImpl::reuseBlockNum() {
    return static_cast<int>(stream_->reuseInfo().reuseBlocksNum());
}

// getReuseLength - 参考 GenerateStream 方法
std::tuple<int, int, int> IGenerateStreamImpl::getReuseLength() {
    return std::make_tuple(stream_->initialReuseLength(), stream_->localReuseLength(), stream_->remoteReuseLength());
}

// setPrefillReuseLength - 参考 P2PConnectorServerCaller.cc:185-188
void IGenerateStreamImpl::setPrefillReuseLength(int reuse_length, int local_reuse_length, int remote_reuse_length) {
    stream_->reuseInfo().prefill_total_reuse_len  = reuse_length;
    stream_->reuseInfo().prefill_local_reuse_len  = local_reuse_length;
    stream_->reuseInfo().prefill_remote_reuse_len = remote_reuse_length;
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

}  // namespace rtp_llm
