#include "rtp_llm/cpp/engine_base/stream/IGenerateStreamImpl.h"

#include <torch/extension.h>
#include <torch/all.h>
#include <cstring>

#include "rtp_llm/cpp/model_rpc/TensorPbConvert.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

IGenerateStreamImpl::IGenerateStreamImpl(const std::shared_ptr<GenerateStream>& stream): stream_(stream) {}

// 与 DecodeRpcServer::localGenerate 中首 token 注入方式一致：torch 张量 + update(StreamUpdateInfo)
void IGenerateStreamImpl::appendTokenId(int batch_id, int token_id) {
    (void)batch_id;
    stream_->setIsContextStream(false);
    stream_->step();

    auto new_tokens                   = torch::zeros({(int64_t)stream_->nextBatchSize(), 1}, torch::kInt32);
    new_tokens.data_ptr<int32_t>()[0] = token_id;
    stream_->incLastOutputPos();
    stream_->update({new_tokens,
                     1,
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor()});
    RTP_LLM_LOG_DEBUG("append token id: %d, stream id: %ld", token_id, stream_->streamId());
}

int64_t IGenerateStreamImpl::deadlineMs() const {
    return stream_->deadlineMs();
}

std::string IGenerateStreamImpl::uniqueKey() const {
    return stream_->uniqueKey();
}

int64_t IGenerateStreamImpl::requestId() const {
    return stream_->streamId();
}

std::vector<int> IGenerateStreamImpl::currentExecuteTokens(int batch_id) {
    return stream_->currentExecuteTokens(batch_id);
}

void IGenerateStreamImpl::appendSPInfo(const std::vector<int>& propose_tokens,
                                       const TensorPB&         propose_probs,
                                       const TensorPB&         propose_hidden) {
    stream_->setReuseLength(stream_->seqLength() - 1);
    stream_->setSpEditRun(false);
    stream_->setMtpTokenIndex(stream_->seqLength() - 1);
    stream_->setContainProposeToken(true);
    stream_->setProposeToken(propose_tokens);

    auto sp_output_buffer    = std::make_shared<SpeculativeExecutorStreamOutput>();
    sp_output_buffer->tokens = torch::zeros({1, (int64_t)propose_tokens.size()}, torch::kInt32);
    memcpy(sp_output_buffer->tokens.data_ptr<int>(), propose_tokens.data(), propose_tokens.size() * sizeof(int));

    auto propose_probs_t  = TensorPbConvert::pbToTorch(propose_probs);
    auto propose_hidden_t = TensorPbConvert::pbToTorch(propose_hidden);

    auto& tensors_holder = sp_output_buffer->tensors_holder;
    tensors_holder.emplace_back(std::move(propose_probs_t));
    tensors_holder.emplace_back(std::move(propose_hidden_t));

    stream_->setSPOutputBuffer(sp_output_buffer);
    RTP_LLM_LOG_DEBUG("append sp info, stream id: %ld", stream_->streamId());
}

std::optional<std::tuple<std::vector<int>, TensorPB, TensorPB>> IGenerateStreamImpl::getSPInfoPB() {
    auto& propose_tokens = stream_->getProposeToken();
    if (propose_tokens.empty()) {
        return std::nullopt;
    }

    auto sp_output_buffer = stream_->getSPOutputBuffer();
    if (!sp_output_buffer) {
        return std::nullopt;
    }

    torch::Tensor probs = sp_output_buffer->all_probs;
    if (probs.defined() && probs.is_cuda()) {
        probs = probs.cpu();
    }
    if (!probs.defined()) {
        probs = torch::empty({0}, torch::dtype(torch::kFloat32));
    }

    torch::Tensor hidden = sp_output_buffer->hidden_states;
    if (!hidden.defined() || hidden.numel() == 0) {
        hidden = torch::empty({0}, torch::dtype(torch::kFloat16));
    } else if (hidden.is_cuda()) {
        hidden = hidden.cpu();
    }

    TensorPB probs_pb;
    TensorPB hidden_pb;
    TensorPbConvert::torchToPb(&probs_pb, probs.contiguous());
    TensorPbConvert::torchToPb(&hidden_pb, hidden.contiguous());

    return std::make_tuple(propose_tokens, probs_pb, hidden_pb);
}

int IGenerateStreamImpl::reuseBlockNum() {
    return static_cast<int>(stream_->reuseBlockSize());
}

std::tuple<int, int, int, int> IGenerateStreamImpl::getReuseLength() {
    return std::make_tuple(stream_->initialReuseLength(),
                           stream_->localReuseLength(),
                           stream_->remoteReuseLength(),
                           stream_->memoryReuseLength());
}

void IGenerateStreamImpl::setPrefillReuseLength(int reuse_length,
                                                int local_reuse_length,
                                                int remote_reuse_length,
                                                int memory_reuse_length) {
    stream_->setPrefillReuseLength(
        reuse_length, local_reuse_length, remote_reuse_length, static_cast<int64_t>(memory_reuse_length));
}

std::pair<std::string, uint32_t> IGenerateStreamImpl::getPrefillAddr() {
    return stream_->prefillAddr();
}

std::vector<int32_t> IGenerateStreamImpl::getContextPositionIdsPB() {
    auto t = stream_->getContextPositionIds();
    if (!t.defined() || t.numel() == 0) {
        return {};
    }
    auto cpu = t.is_cuda() ? t.cpu().contiguous() : t.contiguous();
    RTP_LLM_CHECK(cpu.dtype() == torch::kInt32);
    const int32_t* p = cpu.data_ptr<int32_t>();
    return std::vector<int32_t>(p, p + cpu.numel());
}

void IGenerateStreamImpl::setContextPositionIds(const std::vector<int32_t>& ids) {
    if (ids.empty()) {
        return;
    }
    auto context_position_ids = torch::tensor(ids, torch::dtype(torch::kInt32).device(torch::kCPU));
    stream_->setContextPositionIds(std::move(context_position_ids));
}

bool IGenerateStreamImpl::waitForRemoteGenerate() {
    return stream_->waitForRemoteGenerate();
}

int IGenerateStreamImpl::getPrefillTpSize() const {
    return stream_->getPrefillTpSize();
}

void IGenerateStreamImpl::setStop(ErrorCode error_code, const std::string& error_msg) {
    stream_->setStop(error_code, error_msg);
}

}  // namespace rtp_llm
