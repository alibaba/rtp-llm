#include "ATen/ops/ones.h"
#include "c10/core/ScalarType.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingExecutor.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include <ATen/TensorIndexing.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include "rtp_llm/cpp/utils/DebugUtils.h"
using namespace std;
using namespace at::indexing;

namespace rtp_llm {

namespace HandlerArgs {

static const char* names[] = {
    "input_lengths",
    "hidden_states",
    "input_ids",
    "attention_mask",
    "moe_gating",
};
static_assert(sizeof(names) / sizeof(names[0]) <= NUM_INPUT_TYPES, "redundant handler arg name");
static_assert(sizeof(names) / sizeof(names[0]) >= NUM_INPUT_TYPES, "missing handler arg name");

static bool set_by_str(Flag& flag, const char* name) {
    for (size_t i = 0; i < NUM_INPUT_TYPES; ++i) {
        if (std::strcmp(names[i], name) == 0) {
            flag.set(i);
            return true;
        }
    }
    return false;
}

static const char* get_name(Arg idx) {
    return names[static_cast<size_t>(idx)];
}

static bool has_arg(const Flag& flag, Arg idx) {
    return flag.test(static_cast<size_t>(idx));
}

}  // namespace HandlerArgs

EmbeddingExecutor::EmbeddingExecutor(const EngineInitParams& params, py::object handler):
    handler_(handler),
    handler_args_(),
    metrics_reporter_(params.metrics_reporter),
    model_config_(params.model_config_),
    parallelism_config(params.parallelism_config),
    eplb_config(params.eplb_config) {
    GptModelInitParams model_init_params({
        params.gpt_weights,
        Executor::genModelDescription(model_config_, parallelism_config, eplb_config, params.moe_config),
        nullopt,  // no kv cache buffer for embedding executor
        0,
        parallelism_config,
        ExecInitParams{buildDeviceType()},
    });

    RTP_LLM_CHECK_WITH_INFO(!params.py_model.is_none(), "py_model must be provided, legacy C++ GptModel path removed");
    RTP_LLM_LOG_INFO("init executor with python model");
    model_.reset(new PyWrappedModel(model_init_params, params.py_model, true));

    init_position_ids(model_config_.max_seq_len);
    std::vector<std::string> handler_args;
    {
        py::gil_scoped_acquire acquire;
        torch_type_  = py::module::import("torch").attr("Tensor");
        handler_args = py::cast<std::vector<std::string>>(handler_.attr("extend_forward_args")());
    }

    for (const auto& name : handler_args) {
        if (!HandlerArgs::set_by_str(handler_args_, name.c_str())) {
            RTP_LLM_LOG_WARNING("unknown handler arg: \"%s\", ignored", name.c_str());
        }
    }
}

void EmbeddingExecutor::init_position_ids(int max_seq_len) {
    max_position_ids_tensor_ = torch::arange(max_seq_len, torch::kInt32);
}

absl::StatusOr<GptModelInputs> EmbeddingExecutor::gatherModelInput(const std::list<EmbeddingStreamPtr>& streams) const {
    int64_t token_num  = 0;
    int64_t batch_size = 0;
    calcTokenNum(streams, token_num, batch_size);
    GptModelInputs model_input;
    auto           i32_options = torch::TensorOptions(torch::kInt32).pinned_memory(true);

    model_input.combo_tokens          = torch::empty({token_num}, i32_options);
    model_input.combo_tokens_type_ids = torch::empty({token_num}, i32_options);
    model_input.combo_position_ids    = torch::empty({token_num}, i32_options);
    model_input.input_lengths         = torch::empty({batch_size}, i32_options);
    model_input.sequence_lengths      = torch::empty({0}, i32_options);
    model_input.prefix_lengths        = torch::zeros({batch_size}, i32_options);
    int* merged_tokens                = model_input.combo_tokens.data_ptr<int>();
    int* input_lengths                = model_input.input_lengths.data_ptr<int>();
    int* merged_positon_ids           = model_input.combo_position_ids.data_ptr<int>();
    int* merged_token_type_ids        = model_input.combo_tokens_type_ids.data_ptr<int>();
    int  token_idx                    = 0;
    int  batch_idx                    = 0;
    int  position_bias                = 0;
    if (model_config_.position_ids_style == 1) {
        position_bias = model_config_.special_tokens.pad_token_id + 1;
    }

    std::vector<torch::Tensor> gathered_mm_features;
    std::vector<int>           new_locs;
    std::vector<int>           merged_text_mask;
    std::vector<torch::Tensor> gathered_input_embeddings;
    std::vector<int>           gathered_input_embeddings_locs;
    merged_text_mask.resize(token_num, 1);
    for (auto& stream : streams) {
        int         length     = stream->inputLength();
        int         batchSize  = stream->batchSize();
        const auto& mm_feature = stream->multimodalFeature();
        if (mm_feature.has_value()) {
            for (const auto& feature : mm_feature.value().features) {
                gathered_mm_features.emplace_back(feature);
            }
            const auto mm_locs      = mm_feature.value().locs;
            auto       mm_locs_data = mm_locs.data_ptr<int>();
            for (int i = 0; i < mm_locs.numel(); ++i) {
                new_locs.push_back(mm_locs_data[i] + token_idx);
            }
            const auto text_token_mask = mm_feature.value().text_tokens_mask;
            memcpy(merged_text_mask.data() + token_idx,
                   text_token_mask.data_ptr<int>(),
                   text_token_mask.numel() * sizeof(int));
        }

        if (stream->embeddingInput()->input_embeddings.has_value()) {
            gathered_input_embeddings.emplace_back(stream->embeddingInput()->input_embeddings.value().cpu());
            gathered_input_embeddings_locs.push_back(token_idx);
        }
        memcpy(
            merged_tokens + (int)token_idx, stream->embeddingInput()->token_ids.data_ptr(), length * sizeof(int32_t));
        memcpy(merged_token_type_ids + (int)token_idx,
               stream->embeddingInput()->token_type_ids.data_ptr(),
               length * sizeof(int32_t));
        memcpy(input_lengths + (int)batch_idx,
               stream->embeddingInput()->input_lengths.data_ptr(),
               stream->batchSize() * sizeof(int32_t));
        int length_idx = 0;
        for (int i = 0; i < batchSize; i++) {
            int seqLen = stream->embeddingInput()->input_lengths.data_ptr<int32_t>()[i];
            RTP_LLM_CHECK_WITH_INFO(seqLen + position_bias <= (int)max_position_ids_tensor_.size(0),
                                    "seqlen(%d) + position_bias(%d) exceed max_position_length(%d)",
                                    int(seqLen),
                                    int(position_bias),
                                    (int)max_position_ids_tensor_.size(0));
            memcpy(merged_positon_ids + token_idx + length_idx,
                   max_position_ids_tensor_.data_ptr<int32_t>() + position_bias,
                   seqLen * sizeof(int32_t));
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
        model_input.mm_features_locs =
            torch::from_blob(new_locs.data(), {(int64_t)new_locs.size()}, torch::kInt32).clone();
        model_input.text_tokens_mask =
            torch::from_blob(merged_text_mask.data(), {(int64_t)merged_text_mask.size()}, torch::kInt32).clone();
    }

    if (!gathered_input_embeddings.empty()) {
        model_input.input_embeddings      = std::move(gathered_input_embeddings);
        model_input.input_embeddings_locs = torch::from_blob(gathered_input_embeddings_locs.data(),
                                                             {(int64_t)gathered_input_embeddings_locs.size()},
                                                             torch::kInt32)
                                                .clone();
    }

    size_t max_seq_len = *std::max_element(input_lengths, input_lengths + batch_size);
    if (HandlerArgs::has_arg(handler_args_, HandlerArgs::Arg::MOE_GATING)) {
        model_input.need_moe_gating = true;
    }
    reportMetrics(batch_size, token_num, max_seq_len);
    return model_input;
}

ModelRequest EmbeddingExecutor::generateOldModelRequest(GptModelInputs& model_input) {
    ModelRequest model_request;
    model_request.generate_batch_size  = 0;
    model_request.context_batch_size   = model_input.input_lengths.size(0);
    model_request.combo_tokens         = model_input.combo_tokens;
    model_request.combo_position_ids   = model_input.combo_position_ids;
    model_request.combo_token_type_ids = model_input.combo_tokens_type_ids;
    model_request.input_lengths        = model_input.input_lengths;
    model_request.sequence_lengths     = model_input.sequence_lengths;
    model_request.prefix_lengths       = model_input.prefix_lengths;
    model_request.attention_mask       = model_input.attention_mask;
    return model_request;
}

void EmbeddingExecutor::calcTokenNum(const list<EmbeddingStreamPtr>& streams,
                                     int64_t&                        token_num,
                                     int64_t&                        batch_size) const {
    token_num  = 0;
    batch_size = 0;
    for (auto& stream : streams) {
        token_num += stream->inputLength();
        batch_size += stream->batchSize();
    }
}

unique_ptr<GptModelOutputs> EmbeddingExecutor::copyResultToCPU(th::Tensor gpu_outputs) const {
    auto output           = std::make_unique<GptModelOutputs>();
    output->hidden_states = gpu_outputs.cpu();
    return output;
}

absl::Status EmbeddingExecutor::sliceTensor(py::object                           tensor,
                                            const std::list<EmbeddingStreamPtr>& streams,
                                            int                                  total_batch_size) const {
    auto gpu_tensors = py::cast<torch::Tensor>(tensor);
    auto cpu_tensors = gpu_tensors.cpu();
    if (total_batch_size != cpu_tensors.size(0)) {
        std::ostringstream error_msg;
        error_msg << "total batch size not equal to output tensor at dim 0: " << total_batch_size << " vs "
                  << cpu_tensors.size(0);
        return absl::InternalError(error_msg.str());
    }
    int index = 0;
    for (auto& stream : streams) {
        if (index + stream->batchSize() > cpu_tensors.size(0)) {
            std::ostringstream error_msg;
            error_msg << "current index exceed output tensor at dim 0: " << index << ":" << index + stream->batchSize()
                      << "tensor size: " << cpu_tensors.size(0);
            return absl::InternalError(error_msg.str());
        }
        torch::Tensor sliced_tensor = cpu_tensors.slice(0, index, index + stream->batchSize(), 1);
        stream->updateTensorOutput(sliced_tensor);
        index += stream->batchSize();
    }
    return absl::OkStatus();
}

absl::Status EmbeddingExecutor::slicePyList(py::object                           gpu_outputs,
                                            const std::list<EmbeddingStreamPtr>& streams,
                                            int                                  total_batch_size) const {
    auto output_list = py::cast<py::list>(gpu_outputs);
    if (total_batch_size != output_list.size()) {
        std::ostringstream error_msg;
        error_msg << "total batch size not equal to output list: " << total_batch_size << " vs " << output_list.size();
        return absl::InternalError(error_msg.str());
    }
    int index = 0;
    for (auto& stream : streams) {
        if (index + stream->batchSize() > output_list.size()) {
            std::ostringstream error_msg;
            error_msg << "current index exceed output list max index: " << index << ":" << index + stream->batchSize()
                      << "list size: " << output_list.size();
            return absl::InternalError(error_msg.str());
        }
        py::slice slice(index, index + stream->batchSize(), 1);
        auto      res = pyListToTensorMapVec(output_list[slice]);
        stream->updateMapOutput(res);
        index += stream->batchSize();
    }
    return absl::OkStatus();
}

// embedding postprocess output has two vaild values:
// 1. tensor, which shape is [total_batch_size, ...]
// 2. list<map<str, tensor>, which shape is [total_batch_size]
absl::Status EmbeddingExecutor::updateStreams(py::object                           post_process_output,
                                              const std::list<EmbeddingStreamPtr>& streams,
                                              int                                  total_batch_size) const {
    if (pybind11::isinstance<py::list>(post_process_output)) {
        return slicePyList(post_process_output, streams, total_batch_size);
        // need use python class type to check
    } else if (py::isinstance(post_process_output, torch_type_)) {
        return sliceTensor(post_process_output, streams, total_batch_size);
    } else {
        return absl::InternalError("unknown output type");
    }
}

absl::StatusOr<py::object> EmbeddingExecutor::postProcess(const ModelRequest&    model_request,
                                                          const GptModelOutputs& gpu_outputs) {
    using namespace HandlerArgs;

    try {
        py::dict kwargs;
        if (has_arg(handler_args_, Arg::INPUT_LENGTHS)) {
            kwargs[get_name(Arg::INPUT_LENGTHS)] = model_request.input_lengths;
        }
        if (has_arg(handler_args_, Arg::HIDDEN_STATES)) {
            kwargs[get_name(Arg::HIDDEN_STATES)] = gpu_outputs.all_hidden_states;
        }
        if (has_arg(handler_args_, Arg::INPUT_IDS)) {
            kwargs[get_name(Arg::INPUT_IDS)] = model_request.combo_tokens;
        }
        if (has_arg(handler_args_, Arg::ATTENTION_MASK)) {
            kwargs[get_name(Arg::ATTENTION_MASK)] = py::none();  // mark to be generated by python
        }
        if (has_arg(handler_args_, Arg::MOE_GATING)) {
            py::list moe_gating;
            for (const auto& gating : gpu_outputs.moe_gating) {
                if (gating.defined()) {
                    moe_gating.append(gating);
                } else {
                    moe_gating.append(py::none());
                }
            }
            kwargs[get_name(Arg::MOE_GATING)] = moe_gating;
        }

        py::object output = handler_.attr("extend_forward")(**kwargs);
        return output;
    } catch (const exception& e) {
        return absl::InternalError("meet error when run handler " + std::string(e.what()));
    }
}

absl::Status EmbeddingExecutor::process(const std::list<EmbeddingStreamPtr>& streams) {
    CHECK_AND_RETURN_REF(model_input, gatherModelInput(streams));
    auto            merged_output = std::make_unique<MergedOutput>();
    GptModelOutputs model_output;
    ModelRequest    model_request    = generateOldModelRequest(model_input);
    auto            total_batch_size = model_request.context_batch_size;
    model_->releaseBuffers();
    model_output = std::move(model_->forward(model_input));
    py::gil_scoped_acquire acquire;
    // for py::list, handler should ensure object to cpu in the python impl,
    // for torch::Tensor, we manually move it to cpu during updateStreams()
    CHECK_AND_RETURN_REF(post, postProcess(model_request, model_output));
    auto res = updateStreams(post, streams, total_batch_size);
    model_->releaseBuffers();
    return res;
}

void EmbeddingExecutor::reportMetrics(size_t context_batch_size, size_t combo_token_num, size_t max_seq_len) const {
    if (metrics_reporter_) {
        RtpLLMExecutorMetricsCollector collector;
        collector.context_batch_size  = context_batch_size;
        collector.generate_batch_size = 0;
        collector.execute_token_size  = combo_token_num;
        collector.max_seq_len         = max_seq_len;
        metrics_reporter_->report<RtpLLMExecutorMetrics, RtpLLMExecutorMetricsCollector>(nullptr, &collector);
    }
}

}  // namespace rtp_llm
