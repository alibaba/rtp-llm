#include "ATen/ops/ones.h"
#include "c10/core/ScalarType.h"
#include "maga_transformer/cpp/utils/StatusUtil.h"
#include "maga_transformer/cpp/embedding_engine/EmbeddingExecutor.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/core/Types.h"
#include "maga_transformer/cpp/models/GptModel.h"
#include "maga_transformer/cpp/models/Sampler.h"
#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "maga_transformer/cpp/engine_base/Executor.h"
#include "maga_transformer/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include <algorithm>

using namespace std;
using namespace fastertransformer;

namespace rtp_llm {

EmbeddingExecutor::EmbeddingExecutor(const EngineInitParams& params, ft::DeviceBase* device, py::object handler):
    handler_(handler),
    device_(device),
    metrics_reporter_(params.metrics_reporter),
    params_(params.gpt_init_parameter)
{
    model_.reset(new GptModel({device_, params.gpt_weights, Executor::genModelDescription(params_)}));
    init_position_ids(params_.max_seq_len_);
}

void EmbeddingExecutor::init_position_ids(int max_seq_len) {
    max_position_ids_buf_ = device_->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)max_seq_len}, ft::AllocationType::HOST}, {});
    int*  position_ids    = (int*)max_position_ids_buf_->data();
    for (int i = 0; i < max_seq_len; i++) {
        position_ids[i] = i;
    }
}

absl::StatusOr<GptModelInputs> EmbeddingExecutor::gatherModelInput(const std::list<EmbeddingStreamPtr>& streams) const {
    int64_t token_num = 0;
    int64_t batch_size = 0;
    calcTokenNum(streams, token_num, batch_size);
    GptModelInputs model_input;
    model_input.combo_tokens =
        device_->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)token_num}, ft::AllocationType::HOST}, {});
    model_input.combo_tokens_type_ids =
        device_->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)token_num}, ft::AllocationType::HOST}, {});
    model_input.combo_position_ids =
        device_->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)token_num}, ft::AllocationType::HOST}, {});
    model_input.input_lengths =
        device_->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)batch_size}, ft::AllocationType::HOST}, {});
    model_input.sequence_lengths =
        device_->allocateBuffer({ft::DataType::TYPE_INT32, {0}, ft::AllocationType::HOST}, {});
    model_input.prefix_lengths =
        device_->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)batch_size}, ft::AllocationType::HOST}, {});
    memset(model_input.prefix_lengths->data(), 0, model_input.prefix_lengths->sizeBytes());
    int*      merged_tokens    = model_input.combo_tokens->data<int>();
    int*      input_lengths    = model_input.input_lengths->data<int>();
    int*      merged_positon_ids = model_input.combo_position_ids->data<int>();
    int*      merged_token_type_ids = model_input.combo_tokens_type_ids->data<int>();
    int token_idx = 0;
    int batch_idx = 0;
    int position_bias = 0;
    if (params_.position_ids_style_ == 1) {
        position_bias = params_.special_tokens_.pad_token_id_ + 1;
    }
    std::vector<ft::BufferPtr> gathered_mm_features;
    std::vector<int> new_locs;
    std::vector<int> merged_text_mask;
    merged_text_mask.resize(token_num, 1);
    for (auto& stream: streams) {
        int length = stream->inputLength();
        int batchSize = stream->batchSize();
        const auto& mm_feature = stream->multimodalFeature();
        if (mm_feature.has_value()) {
            for (const auto& feature: mm_feature.value().features) {
                gathered_mm_features.emplace_back(torchTensor2Buffer(feature));
            }
            const auto mm_locs = mm_feature.value().locs;
            for (int i = 0; i < mm_locs->size(); ++i) {
                new_locs.push_back(*mm_locs->dataWithOffset<int>(i) + token_idx);
            }
            const auto text_token_mask = mm_feature.value().text_tokens_mask;
            memcpy(merged_text_mask.data() + token_idx, text_token_mask->data(), text_token_mask->size() * sizeof(int));
        }
        memcpy(merged_tokens + (int)token_idx, stream->embeddingInput()->token_ids->data(), length * sizeof(int32_t));
        memcpy(merged_token_type_ids + (int)token_idx, stream->embeddingInput()->token_type_ids->data(), length * sizeof(int32_t));
        memcpy(input_lengths + (int)batch_idx, stream->embeddingInput()->input_lengths->data(), stream->batchSize() * sizeof(int32_t));
        int length_idx = 0;
        for (int i = 0; i < batchSize; i++) {
            int seqLen = stream->embeddingInput()->input_lengths->data<int32_t>()[i];
            FT_CHECK_WITH_INFO(seqLen + position_bias <= int(max_position_ids_buf_->shape()[0]), "position index exceed max_position_length");
            memcpy(merged_positon_ids + token_idx + length_idx, max_position_ids_buf_->data<int32_t>() + position_bias, seqLen * sizeof(int32_t));
            length_idx += seqLen;
        }
        if (length_idx != length) {
            return absl::InternalError("stream total_length not equal to sum of lengths");
        }
        batch_idx += stream->batchSize();
        token_idx += length;
    }
    if (!gathered_mm_features.empty()) {
        model_input.multimodal_features = std::move(gathered_mm_features);
        model_input.mm_features_locs = device_->clone({*vector2Buffer(new_locs), ft::AllocationType::HOST});
        model_input.text_tokens_mask = device_->clone({*vector2Buffer(merged_text_mask), ft::AllocationType::HOST});
    }
    size_t max_seq_len = *std::max_element(input_lengths, input_lengths + batch_size);
    reportMetrics(batch_size, token_num, max_seq_len);
    return model_input;
}

ModelRequest EmbeddingExecutor::generateOldModelRequest(GptModelInputs& model_input) {
    ModelRequest model_request;
    model_request.generate_batch_size  = 0;
    model_request.context_batch_size   = model_input.input_lengths->shape()[0];
    model_request.combo_tokens         = model_input.combo_tokens;
    model_request.combo_position_ids   = model_input.combo_position_ids;
    model_request.combo_token_type_ids = model_input.combo_tokens_type_ids;
    model_request.input_lengths        = model_input.input_lengths;
    model_request.sequence_lengths     = model_input.sequence_lengths;
    model_request.prefix_lengths       = model_input.prefix_lengths;
    model_request.attention_mask       = model_input.attention_mask;
    return model_request;
}

void EmbeddingExecutor::calcTokenNum(const list<EmbeddingStreamPtr>& streams, int64_t& token_num, int64_t& batch_size) const {
    token_num = 0;
    batch_size = 0;
    for (auto& stream: streams) {
        token_num += stream->inputLength();
        batch_size += stream->batchSize();
    }
}

unique_ptr<GptModelOutputs> EmbeddingExecutor::copyResultToCPU(th::Tensor gpu_outputs) const {
    auto output = std::make_unique<GptModelOutputs>();
    auto buffer_ptr = torchTensor2Buffer(gpu_outputs);
    output->hidden_states = device_->allocateBuffer({buffer_ptr->type(), buffer_ptr->shape(), ft::AllocationType::HOST}, {});
    device_->copy({*(output->hidden_states), *(buffer_ptr)});
    return output;
}

absl::Status EmbeddingExecutor::updateStreams(th::Tensor gpu_outputs, const std::list<EmbeddingStreamPtr>& streams) const {
    auto cpu_output = copyResultToCPU(gpu_outputs);
    int index = 0;
    for (auto& stream: streams) {
        auto hidden_states_buf = (*cpu_output->hidden_states).view(index, stream->batchSize());
        auto new_buffer_ptr = device_->allocateBuffer({hidden_states_buf.type(), hidden_states_buf.shape(), AllocationType::HOST});
        device_->copy({*new_buffer_ptr, hidden_states_buf});
        stream->updateOutput(new_buffer_ptr);
        index += stream->batchSize();
    }
    return absl::OkStatus();
}

absl::StatusOr<th::Tensor> EmbeddingExecutor::postProcess(const ModelRequest& model_request, const GptModelOutputs& gpu_outputs) {
    py::gil_scoped_acquire acquire;
    try {
        torch::Tensor hidden_states = Buffer2torchTensor(gpu_outputs.all_hidden_states, false);
        torch::Tensor input_lengths = Buffer2torchTensor(model_request.input_lengths, false);
        torch::Tensor input_ids = Buffer2torchTensor(model_request.combo_tokens, false);
        torch::Tensor output = handler_.attr("forward")(input_ids, hidden_states, input_lengths).cast<th::Tensor>();
        return output;
    } catch (const exception& e) {
        return absl::InternalError("meet error when run handler " + std::string(e.what()));
    }
}

absl::Status EmbeddingExecutor::process(const std::list<EmbeddingStreamPtr>& streams) {
    CHECK_AND_RETURN_REF(model_input, gatherModelInput(streams));
    FT_LOG_DEBUG("model_input: %s", model_input.debugString().c_str());
    auto         merged_output        = std::make_unique<MergedOutput>();
    GptModelOutputs model_output;
    ModelRequest model_request = generateOldModelRequest(model_input);
    model_output = std::move(model_->forward(model_input));
    CHECK_AND_RETURN_REF(post, postProcess(model_request, model_output));
    return updateStreams(post, streams);
}

void EmbeddingExecutor::reportMetrics(size_t context_batch_size, size_t combo_token_num, size_t max_seq_len) const {
    if (metrics_reporter_) {
        RtpLLMExecutorMetricsCollector collector;
        collector.context_batch_size = context_batch_size;
        collector.generate_batch_size = 0;
        collector.execute_token_size = combo_token_num;
        collector.max_seq_len = max_seq_len;
        metrics_reporter_->report<RtpLLMExecutorMetrics, RtpLLMExecutorMetricsCollector>(nullptr, &collector);
    }
}

}  // namespace rtp_llma
