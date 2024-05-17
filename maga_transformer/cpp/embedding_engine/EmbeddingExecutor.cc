#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/embedding_engine/EmbeddingExecutor.h"
#include "maga_transformer/cpp/deprecated/ParallelModelWrapper.h"
#include "src/fastertransformer/core/Types.h"
#include "maga_transformer/cpp/models/GptModel.h"
#include "maga_transformer/cpp/models/Sampler.h"

using namespace std;
using namespace fastertransformer;

namespace rtp_llm {

EmbeddingExecutor::EmbeddingExecutor(
        const MagaInitParams&                                params,
        ft::NcclParam                                        tensor_para,
        ft::NcclParam                                        pipeline_para,
        const vector<unordered_map<string, ConstBufferPtr>>& layer_weights,
        const unordered_map<string, ConstBufferPtr>&         weights,
        const HandlerBase&                                   handler):
    handler_(handler),
    tensor_para_(tensor_para),
    pipeline_para_(pipeline_para)
{
    // need init model and sampler
    unique_ptr<GptModelInitParams> model_params;
    // model_.reset(new GptModel(*model_params));
    SamplerInitParams sampler_params;
    device_               = ft::DeviceFactory::getDevice(DeviceType::Cuda);
    data_type_ = ft::getDataType(params.gpt_init_parameter->data_type_);
    sampler_params.device = device_;
    model_wrapper_.reset(
        new ParallelModelWrapper(*params.gpt_init_parameter, weights, layer_weights)
    );
    init_position_ids(params.gpt_init_parameter->max_seq_len_);
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
        device_->allocateBuffer({ft::DataType::TYPE_INT32, {0}, ft::AllocationType::HOST}, {});
    int*      merged_tokens    = (int*)model_input.combo_tokens->data();
    int*      input_lengths    = (int*)model_input.input_lengths->data();
    int*      merged_positon_ids = (int*)model_input.combo_position_ids->data();
    int*      merged_token_type_ids = (int*)model_input.combo_tokens_type_ids->data();
    int token_idx = 0;
    int batch_idx = 0;

    for (auto& stream: streams) {
        int length = stream->inputLength();
        int batchSize = stream->batchSize();
        memcpy(merged_tokens + (int)token_idx, stream->embeddingInput()->token_ids->data(), length * sizeof(int32_t));
        memcpy(merged_token_type_ids + (int)token_idx, stream->embeddingInput()->token_type_ids->data(), length * sizeof(int32_t));
        memcpy(input_lengths + (int)batch_idx, stream->embeddingInput()->input_lengths->data(), stream->batchSize() * sizeof(int32_t));
        int length_idx = 0;
        for (int i = 0; i < batchSize; i++) {
            int seqLen = stream->embeddingInput()->input_lengths->data<int32_t>()[i];
            memcpy(merged_positon_ids + token_idx + length_idx, max_position_ids_buf_->data(), seqLen * sizeof(int32_t));
            length_idx += seqLen;
        }
        if (length_idx != length) {
            return absl::InternalError("stream total_length not equal to sum of lengths");
        }
        batch_idx += stream->batchSize();
        token_idx += length;
    }

    return model_input;
}

ModelRequest EmbeddingExecutor::generateOldModelRequest(GptModelInputs& model_input) {
    ModelRequest model_request;
    model_request.generate_batch_size  = 0;
    model_request.context_batch_size   = model_input.input_lengths->shape()[0];
    model_request.combo_tokens         = std::move(model_input.combo_tokens);
    model_request.combo_position_ids   = std::move(model_input.combo_position_ids);
    model_request.combo_token_type_ids = std::move(model_input.combo_tokens_type_ids);
    model_request.input_lengths        = std::move(model_input.input_lengths);
    model_request.sequence_lengths     = std::move(model_input.sequence_lengths);
    model_request.prefix_lengths       = std::move(model_input.prefix_lengths);
    model_request.attention_mask       = std::move(model_input.attention_mask);
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

unique_ptr<GptModelOutputs> EmbeddingExecutor::copyResultToCPU(const GptModelOutputs& gpu_outputs) const {
    auto output = std::make_unique<GptModelOutputs>();
    output->hidden_states = device_->allocateBuffer({gpu_outputs.hidden_states->type(), gpu_outputs.hidden_states->shape(), ft::AllocationType::HOST}, {});
    device_->copy({*(output->hidden_states), *(gpu_outputs.hidden_states)});
    return output;
}

absl::Status EmbeddingExecutor::updateStreams(std::unique_ptr<GptModelOutputs>& gpu_outputs, const std::list<EmbeddingStreamPtr>& streams) const {
    auto cpu_output = copyResultToCPU(*gpu_outputs);
    int index = 0;
    for (auto& stream: streams) {
        auto hidden_states_buf = (*cpu_output->hidden_states).slice(index, index + stream->batchSize());
        auto new_buffer_ptr = device_->allocateBuffer({hidden_states_buf.type(), hidden_states_buf.shape(), AllocationType::HOST});
        device_->copy({*new_buffer_ptr, hidden_states_buf});
        stream->updateOutput(new_buffer_ptr);
        index += stream->batchSize();
    }
    return absl::OkStatus();
}

absl::Status EmbeddingExecutor::process(const std::list<EmbeddingStreamPtr>& streams) {
    auto model_input_status = gatherModelInput(streams);
    RETURN_IF_STATUS_OR_ERROR(model_input_status);
    auto& model_input = model_input_status.value();
    FT_LOG_DEBUG("model_input: %s", model_input.debugString().c_str());
    auto         merged_output        = std::make_unique<MergedOutput>();
    ModelRequest model_request        = std::move(generateOldModelRequest(model_input));

    auto output = model_wrapper_->forward(model_request);
    auto handler_output = handler_.forward(model_request, *output);
    RETURN_IF_STATUS_OR_ERROR(handler_output);
    return updateStreams(handler_output.value(), streams);
    // return updateStreams(output, streams);
}
}  // namespace rtp_llma
