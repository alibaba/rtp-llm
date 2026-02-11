#include "rtp_llm/cpp/api_server/GenerateStreamWrapper.h"

#include "rtp_llm/cpp/api_server/Exception.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

namespace rtp_llm {

GenerateStreamWrapper::GenerateStreamWrapper(const std::shared_ptr<ApiServerMetricReporter>& metric_reporter,
                                             const std::shared_ptr<TokenProcessor>&          token_processor):
    metric_reporter_(metric_reporter), token_processor_(token_processor) {}

void GenerateStreamWrapper::init(const std::shared_ptr<GenerateInput>& input,
                                 const std::shared_ptr<EngineBase>&    engine) {
    input_ids_       = input->input_ids;
    generate_config_ = input->generate_config;
    // align life cycle with stream
    lora_guard_ =
        std::make_shared<lora::LoraResourceGuard>(engine->getLoraManager(), input->generate_config->adapter_name);
    stream_ = engine->enqueue(input);
}

void GenerateStreamWrapper::init(GenerateStreamPtr stream, const std::shared_ptr<EngineBase>& engine) {
    auto input       = stream->generateInput();
    input_ids_       = input->input_ids;
    generate_config_ = input->generate_config;
    // align life cycle with stream
    lora_guard_ =
        std::make_shared<lora::LoraResourceGuard>(engine->getLoraManager(), input->generate_config->adapter_name);
    stream_ = stream;
}

std::pair<MultiSeqsResponse, bool> GenerateStreamWrapper::generateResponse() {
    if (stream_->finished() && stream_->hasOutput() == false) {
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
                                                        rtp_llm::BufferPtr                     input_ids) {
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
            auto buffer = logits.value();
            res.logits.value().push_back(rtp_llm::buffer2vector<float>(*buffer));
        }
        if (generate_config->calculate_loss && loss.has_value()) {
            auto buffer = loss.value();
            res.loss.value().push_back(rtp_llm::buffer2vector<float>(*buffer));
        }
        if (generate_config->return_hidden_states && hidden_states.has_value()) {
            auto buffer                          = hidden_states.value();
            auto hidden_states_tensor            = Buffer2torchTensor(buffer);
            hidden_states_tensor                 = hidden_states_tensor.to(torch::kFloat).to(torch::kCPU);
            std::vector<float> hidden_states_vec = std::vector<float>(hidden_states_tensor.numel());
            memcpy(hidden_states_vec.data(),
                   hidden_states_tensor.data_ptr<float>(),
                   hidden_states_tensor.numel() * sizeof(float));
            res.hidden_states.value().push_back(hidden_states_vec);
        }
        if (generate_config->return_output_ids) {
            res.output_ids.value().push_back(rtp_llm::buffer2vector<int32_t>(*output_ids));
        }
        if (generate_config->return_input_ids) {
            res.input_ids.value().push_back(rtp_llm::buffer2vector<int32_t>(*input_ids));
        }
    }

    return res;
}

}  // namespace rtp_llm
