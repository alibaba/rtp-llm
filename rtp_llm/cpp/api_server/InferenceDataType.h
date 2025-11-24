#pragma once

#include "autil/legacy/jsonizable.h"

#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
#include "rtp_llm/cpp/normal_engine/NormalEngine.h"

using namespace autil::legacy;
using namespace autil::legacy::json;

namespace rtp_llm {

// used for de-serialization
#define JSONIZE_OPTIONAL(field)                                                                                        \
    try {                                                                                                              \
        using Type = decltype(field)::value_type;                                                                      \
        Type field##Tmp;                                                                                               \
        json.Jsonize(#field, field##Tmp);                                                                              \
        field = field##Tmp;                                                                                            \
    } catch (autil::legacy::ExceptionBase & e) {                                                                       \
        if (field.has_value() == false) {                                                                              \
            field = std::nullopt;                                                                                      \
        }                                                                                                              \
    }

class EmbeddingRequest: public autil::legacy::Jsonizable {
public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("source", source, "unknown");
        json.Jsonize("private_request", private_request, false);
    }
    std::string source;
    bool        private_request;
};

class RawRequest: public autil::legacy::Jsonizable {
public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("source", source, "unknown");
        json.Jsonize("yield_generator", yield_generator, false);
        json.Jsonize("private_request", private_request, false);
        JSONIZE_OPTIONAL(prompt);
        try {
            std::string text;
            json.Jsonize("text", text);
            prompt = text;
        } catch (autil::legacy::ExceptionBase& e) {
            if (prompt.has_value() == false) {
                prompt = std::nullopt;
            }
        }
        JSONIZE_OPTIONAL(prompt_batch);

        try {
            std::vector<std::string> images_;
            json.Jsonize("images", images_);
            images = images_;
        } catch (autil::legacy::ExceptionBase& e) {
            try {
                std::vector<std::vector<std::string>> images_batch_;
                json.Jsonize("images", images_batch_);
                images_batch = images_batch_;
            } catch (autil::legacy::ExceptionBase& e) {
                images_batch = std::nullopt;
                images       = std::nullopt;
            }
        }

        JSONIZE_OPTIONAL(generate_config);
    }
    std::string                                          source;
    bool                                                 private_request;
    bool                                                 yield_generator;
    std::optional<std::string>                           prompt;
    std::optional<std::vector<std::string>>              prompt_batch;
    std::optional<std::vector<std::vector<std::string>>> images_batch;
    std::optional<std::vector<std::string>>              images;
    std::optional<GenerateConfig>                        generate_config;
};

class AuxInfoAdapter: public Jsonizable, public AuxInfo {
public:
    void Jsonize(Jsonizable::JsonWrapper& json) override {
        json.Jsonize("cost_time", cost_time_ms, cost_time_ms);
        json.Jsonize("iter_count", iter_count, iter_count);
        json.Jsonize("input_len", input_len, input_len);
        json.Jsonize("prefix_len", prefix_len, prefix_len);
        json.Jsonize("reuse_len", reuse_len, reuse_len);
        json.Jsonize("output_len", output_len, output_len);
        json.Jsonize("pd_sep", pd_sep, pd_sep);
        json.Jsonize("step_output_len", step_output_len, step_output_len);
        json.Jsonize("beam_responses", beam_responses, beam_responses);
        if (json.GetMode() == FastJsonizableBase::Mode::TO_JSON && cum_log_probs.has_value()) {
            auto buffer = cum_log_probs.value();
            json.Jsonize("cum_log_probs", rtp_llm::buffer2vector<float>(*buffer));
        }
        json.Jsonize("local_reuse_len", local_reuse_len, local_reuse_len);
        json.Jsonize("remote_reuse_len", remote_reuse_len, remote_reuse_len);
    }
    AuxInfoAdapter() {
        AuxInfo();
    }
    AuxInfoAdapter(const AuxInfo& base) {
        cost_time_us     = base.cost_time_us;
        iter_count       = base.iter_count;
        input_len        = base.input_len;
        prefix_len       = base.prefix_len;
        reuse_len        = base.reuse_len;
        output_len       = base.output_len;
        step_output_len  = base.step_output_len;
        pd_sep           = base.pd_sep;
        cum_log_probs    = base.cum_log_probs;
        local_reuse_len  = base.local_reuse_len;
        remote_reuse_len = base.remote_reuse_len;

        cost_time_ms = cost_time_us / 1000.0;
    }
    float                    cost_time_ms;
    std::vector<std::string> beam_responses;
};

class MultiSeqsResponse: public Jsonizable {
public:
    void Jsonize(Jsonizable::JsonWrapper& json) override {
        if (response.size() == 1) {
            json.Jsonize("response", response[0], response[0]);
        } else {
            json.Jsonize("response", response, response);
        }
        json.Jsonize("finished", finished, finished);
        json.Jsonize("aux_info", aux_info, aux_info);
        if (logits.has_value())
            json.Jsonize("logits", logits.value(), logits.value());
        if (loss.has_value())
            json.Jsonize("loss", loss.value(), loss.value());
        if (hidden_states.has_value())
            json.Jsonize("hidden_states", hidden_states.value(), hidden_states.value());
        if (output_ids.has_value())
            json.Jsonize("output_ids", output_ids.value(), output_ids.value());
        if (input_ids.has_value())
            json.Jsonize("input_ids", input_ids.value(), input_ids.value());
    }
    bool                                           finished;
    std::vector<std::string>                       response;
    std::vector<AuxInfoAdapter>                    aux_info;
    std::optional<std::vector<std::vector<float>>> logits;
    std::optional<std::vector<std::vector<float>>> loss;
    std::optional<std::vector<std::vector<float>>> hidden_states;
    std::optional<std::vector<std::vector<int>>>   output_ids;
    std::optional<std::vector<std::vector<int>>>   input_ids;
};

class BatchResponse: public Jsonizable {
public:
    BatchResponse(std::vector<MultiSeqsResponse> batch_state): response_batch(batch_state) {}
    void Jsonize(Jsonizable::JsonWrapper& json) override {
        json.Jsonize("response_batch", response_batch, response_batch);
    }
    std::vector<MultiSeqsResponse> response_batch;
};

#undef JSONIZE_OPTIONAL

}  // namespace rtp_llm
