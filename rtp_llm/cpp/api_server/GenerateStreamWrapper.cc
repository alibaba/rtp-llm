#include "rtp_llm/cpp/api_server/GenerateStreamWrapper.h"

#include "rtp_llm/cpp/api_server/Exception.h"

namespace rtp_llm {

GenerateStreamWrapper::GenerateStreamWrapper(const std::shared_ptr<ApiServerMetricReporter>& metric_reporter,
                                             const std::shared_ptr<TokenProcessor>&          token_processor):
    metric_reporter_(metric_reporter), token_processor_(token_processor) {}

void GenerateStreamWrapper::init(const std::shared_ptr<GenerateInput>& input,
                                 const std::shared_ptr<EngineBase>&    engine) {
    input_ids_       = input->input_ids;
    generate_config_ = input->generate_config;
    stream_          = engine->enqueue(input);
}

void GenerateStreamWrapper::init(GenerateStreamPtr stream, const std::shared_ptr<EngineBase>& engine) {
    auto input       = stream->generateInput();
    input_ids_       = input->input_ids;
    generate_config_ = input->generate_config;
    stream_          = stream;
}

std::pair<MultiSeqsResponse, bool> GenerateStreamWrapper::generateResponse() {
    if (stream_->finished() && stream_->hasOutput() == false) {
        RTP_LLM_LOG_INFO("stream finished.");
        return std::make_pair(MultiSeqsResponse(), true);
    }

    const auto result = stream_->nextOutput();
    if (!result.ok()) {
        RTP_LLM_LOG_INFO("stream nextOutput failed.");
        return std::make_pair(MultiSeqsResponse(), true);
    }
    auto outputs = result.value();

    if (outputs_cache_.generate_outputs.size() == 0) {
        outputs_cache_.generate_outputs = outputs.generate_outputs;
    } else {
        for (int i = 0; i < outputs_cache_.generate_outputs.size(); i++) {
            if (outputs_cache_.generate_outputs[i].finished == false) {
                outputs_cache_.generate_outputs[i] = outputs.generate_outputs[i];
            }
        }
    }
    autil::ScopedTime2 timer;
    if (token_processor_ctx_ == nullptr) {
        token_processor_ctx_ = token_processor_->getTokenProcessorCtx(
            generate_config_->hasNumBeams(), outputs_cache_.generate_outputs.size(), token_processor_);
    }
    std::vector<std::string> texts =
        token_processor_->decodeTokens(token_processor_ctx_, outputs_cache_, output_lens_, generate_config_);
    if (metric_reporter_) {
        metric_reporter_->reportFTPostTokenProcessorRtMetric(timer.done_ms());
    }
    auto response = formatResponse(texts, outputs_cache_, generate_config_, input_ids_);

    bool all_finished = std::all_of(outputs_cache_.generate_outputs.begin(),
                                    outputs_cache_.generate_outputs.end(),
                                    [](const auto& out) { return out.finished; });
    if (all_finished && metric_reporter_ && outputs_cache_.generate_outputs.size() > 0) {
        metric_reporter_->reportFTIterateCountMetric(outputs_cache_.generate_outputs[0].aux_info.iter_count);
        for (const auto& len : output_lens_) {
            metric_reporter_->reportFTOutputTokenLengthMetric(len);
        }
    }
    return std::make_pair(response, false);
}

MultiSeqsResponse GenerateStreamWrapper::formatResponse(const std::vector<std::string>&        generate_texts,
                                                        const GenerateOutputs&                 generate_outputs,
                                                        const std::shared_ptr<GenerateConfig>& generate_config,
                                                        const torch::Tensor&                   input_ids) {
    if (generate_texts.size() == 0) {
        RTP_LLM_LOG_WARNING("generate_texts is empty!");
        return MultiSeqsResponse();
    }
    if (generate_outputs.generate_outputs.size() == 0) {
        RTP_LLM_LOG_WARNING("generate_outputs is empty!");
        return MultiSeqsResponse();
    }

    MultiSeqsResponse res;
    res.response = generate_texts;
    res.finished = std::all_of(generate_outputs.generate_outputs.begin(),
                               generate_outputs.generate_outputs.end(),
                               [](const auto& out) { return out.finished; });

    if (generate_config->aux_info) {
        std::transform(generate_outputs.generate_outputs.begin(),
                       generate_outputs.generate_outputs.end(),
                       std::back_inserter(res.aux_info),
                       [generate_config, generate_texts](const auto& out) {
                           auto aux_info = AuxInfoAdapter(out.aux_info);
                           return aux_info;
                       });
    }

    if (generate_config->return_logits)
        res.logits.emplace();
    if (generate_config->calculate_loss)
        res.loss.emplace();
    if (generate_config->return_hidden_states)
        res.hidden_states.emplace();
    if (generate_config->return_output_ids)
        res.output_ids.emplace();
    if (generate_config->return_input_ids)
        res.input_ids.emplace();

    for (const auto& generate_output : generate_outputs.generate_outputs) {
        auto logits        = generate_output.logits;
        auto loss          = generate_output.loss;
        auto hidden_states = generate_output.hidden_states;
        auto output_ids    = generate_output.output_ids;

        if (generate_config->return_logits && logits.has_value()) {
            auto tensor = logits.value().to(torch::kFloat).contiguous();
            res.logits.value().push_back(
                std::vector<float>(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel()));
        }
        if (generate_config->calculate_loss && loss.has_value()) {
            auto tensor = loss.value().to(torch::kFloat).contiguous();
            res.loss.value().push_back(
                std::vector<float>(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel()));
        }
        if (generate_config->return_hidden_states && hidden_states.has_value()) {
            auto hidden_states_tensor = hidden_states.value().to(torch::kFloat).to(torch::kCPU).contiguous();
            std::vector<float> hidden_states_vec(hidden_states_tensor.data_ptr<float>(),
                                                 hidden_states_tensor.data_ptr<float>() + hidden_states_tensor.numel());
            res.hidden_states.value().push_back(hidden_states_vec);
        }
        if (generate_config->return_output_ids) {
            auto ids_contig = output_ids.contiguous();
            res.output_ids.value().push_back(std::vector<int32_t>(ids_contig.data_ptr<int32_t>(),
                                                                  ids_contig.data_ptr<int32_t>() + ids_contig.numel()));
        }
        if (generate_config->return_input_ids) {
            auto ids_cpu = input_ids.contiguous();
            res.input_ids.value().push_back(
                std::vector<int32_t>(ids_cpu.data_ptr<int32_t>(), ids_cpu.data_ptr<int32_t>() + ids_cpu.numel()));
        }
    }

    return res;
}

}  // namespace rtp_llm
