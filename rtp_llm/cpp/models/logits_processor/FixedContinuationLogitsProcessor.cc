#include "rtp_llm/cpp/models/logits_processor/FixedContinuationLogitsProcessor.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>
#include <string_view>
#include <fcntl.h>
#include <unistd.h>

#include "autil/legacy/json.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

constexpr std::string_view kTaskPrefix = "__rtp_v41_acceptance__/";

ErrorInfo invalid(const std::string& message) {
    return {ErrorCode::INVALID_PARAMS, "fixed continuation: " + message};
}

struct FixtureCase: autil::legacy::Jsonizable {
    std::vector<int32_t> input_ids;
    std::vector<int32_t> continuation_tokens;
    void Jsonize(JsonWrapper& json) override {
        json.Jsonize("input_ids", input_ids);
        json.Jsonize("continuation_tokens", continuation_tokens);
    }
};

struct Fixture: autil::legacy::Jsonizable {
    int32_t                            schema_version{0};
    int32_t                            vocab_size{0};
    std::map<std::string, FixtureCase> cases;
    void Jsonize(JsonWrapper& json) override {
        json.Jsonize("schema_version", schema_version);
        json.Jsonize("vocab_size", vocab_size);
        json.Jsonize("cases", cases);
    }
};

void writeExclusive(const std::filesystem::path& path, const void* data, size_t bytes) {
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) {
        throw std::runtime_error("cannot exclusively create " + path.string());
    }
    const char* cursor = static_cast<const char*>(data);
    while (bytes) {
        const auto written = ::write(fd, cursor, bytes);
        if (written < 0 && errno == EINTR) {
            continue;
        }
        if (written <= 0) {
            ::close(fd);
            throw std::runtime_error("capture write failed: " + path.string());
        }
        cursor += written;
        bytes -= written;
    }
    if (::close(fd) != 0) {
        throw std::runtime_error("capture close failed: " + path.string());
    }
}

}  // namespace

ErrorResult<BaseLogitsProcessorPtr>
FixedContinuationLogitsProcessor::fromGenerateInput(const std::shared_ptr<GenerateInput>& input) {
    auto& config = *input->generate_config;
    const char* fixture_path = std::getenv("RTP_LLM_FIXED_CONTINUATION_FIXTURE");
    if (!fixture_path || !*fixture_path || !config.task_id.has_value()
        || config.task_id->compare(0, kTaskPrefix.size(), kTaskPrefix) != 0) {
        return BaseLogitsProcessorPtr{};
    }
    const char* output_path = std::getenv("RTP_LLM_FIXED_CONTINUATION_OUTPUT");
    if (!output_path || !*output_path) {
        return invalid("operator capture directory is required");
    }
    if (!config.return_logits || !config.is_streaming || config.hasNumBeams()
        || config.num_return_sequences > 1 || !config.force_disable_sp_run || config.reuse_cache
        || !config.ignore_eos || !config.stop_words_list.empty() || !config.select_tokens_id.empty()
        || config.repetition_penalty != 1.0f || config.presence_penalty != 0.0f
        || config.frequency_penalty != 0.0f || config.no_repeat_ngram_size.value_or(0) != 0) {
        return invalid("requires streaming raw logits, one sequence, no speculation, reuse or sampling constraints");
    }
    try {
        const auto case_id = config.task_id->substr(kTaskPrefix.size());
        if (case_id.empty() || !std::all_of(case_id.begin(), case_id.end(), [](unsigned char c) {
                return std::isalnum(c) || c == '_' || c == '-';
            })) {
            return invalid("invalid fixture case id");
        }
        std::ifstream reader(fixture_path);
        if (!reader) {
            return invalid("cannot read operator fixture");
        }
        std::string json{std::istreambuf_iterator<char>(reader), std::istreambuf_iterator<char>()};
        Fixture fixture;
        autil::legacy::FromJsonString(fixture, json);
        if (fixture.schema_version != 1 || fixture.vocab_size <= 0 || fixture.vocab_size > 1048576
            || fixture.cases.count(case_id) != 1) {
            return invalid("fixture schema, vocabulary or case mismatch");
        }
        auto& selected = fixture.cases.at(case_id);
        const auto valid_id = [&](int32_t id) { return id >= 0 && id < fixture.vocab_size; };
        if (selected.input_ids.empty() || selected.continuation_tokens.empty()
            || !std::all_of(selected.input_ids.begin(), selected.input_ids.end(), valid_id)
            || !std::all_of(selected.continuation_tokens.begin(), selected.continuation_tokens.end(), valid_id)
            || config.max_new_tokens != static_cast<int>(selected.continuation_tokens.size())
            || config.min_new_tokens != config.max_new_tokens) {
            return invalid("fixture IDs or exact output length mismatch");
        }
        if (!input->input_ids.device().is_cpu() || input->input_ids.scalar_type() != torch::kInt32
            || input->input_ids.dim() != 1 || input->input_ids.numel() != selected.input_ids.size()
            || !std::equal(selected.input_ids.begin(), selected.input_ids.end(), input->input_ids.data_ptr<int32_t>())) {
            return invalid("request input IDs differ from frozen fixture");
        }
        BaseLogitsProcessorPtr result = std::make_shared<FixedContinuationLogitsProcessor>(
            std::move(selected.input_ids), std::move(selected.continuation_tokens), fixture.vocab_size,
            output_path, case_id, input->request_id);
        RTP_LLM_LOG_INFO("fixed continuation installed: request=%ld case=%s", input->request_id, case_id.c_str());
        return std::move(result);
    } catch (const std::exception& error) {
        return invalid(error.what());
    }
}

FixedContinuationLogitsProcessor::FixedContinuationLogitsProcessor(std::vector<int32_t> input_ids,
                                                                   std::vector<int32_t> continuation,
                                                                   int32_t              vocab_size,
                                                                   std::string          output_directory,
                                                                   std::string          case_id,
                                                                   int64_t              request_id):
    input_ids_(std::move(input_ids)), continuation_(std::move(continuation)), vocab_size_(vocab_size),
    output_directory_(std::move(output_directory)), case_id_(std::move(case_id)), request_id_(request_id) {}

std::optional<ErrorInfo>
FixedContinuationLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    if (finish_idx != start_idx + 1 || pending_raw_logits_.defined() || committed_ >= continuation_.size()
        || !inputs.token_ids.defined() || !inputs.sequence_lengths.defined() || !inputs.logits.defined()
        || !inputs.token_ids.device().is_cpu() || inputs.token_ids.scalar_type() != torch::kInt32
        || !inputs.sequence_lengths.device().is_cpu() || inputs.sequence_lengths.scalar_type() != torch::kInt32
        || inputs.token_ids.dim() != 2 || inputs.token_ids.size(0) <= start_idx
        || inputs.sequence_lengths.numel() <= start_idx
        || inputs.logits.dim() != 2 || inputs.logits.size(0) <= start_idx || inputs.logits.size(1) < vocab_size_) {
        return invalid("unsupported sampler shape, device, dtype or pending state");
    }
    const auto seq_len = inputs.sequence_lengths.data_ptr<int32_t>()[start_idx];
    if (seq_len != input_ids_.size() + committed_ || inputs.token_ids.size(1) < seq_len) {
        return invalid("sampler context length differs from frozen history");
    }
    const auto* history = inputs.token_ids.data_ptr<int32_t>() + start_idx * inputs.token_ids.size(1);
    if (!std::equal(input_ids_.begin(), input_ids_.end(), history)
        || !std::equal(continuation_.begin(), continuation_.begin() + committed_, history + input_ids_.size())) {
        return invalid("sampler token history differs from frozen history");
    }
    // This is the gatherer's private clone of the real LM-head row. This is
    // the only processor for this request, before EOS/temperature/top-k. Keep
    // the snapshot on device until the committed-token host boundary.
    auto row = inputs.logits.select(0, start_idx);
    source_scalar_type_ = static_cast<int>(row.scalar_type());
    pending_raw_logits_ = row.narrow(0, 0, vocab_size_).clone().to(torch::kFloat32);
    row.fill_(-std::numeric_limits<float>::infinity());
    row.select(0, continuation_[committed_]).fill_(0);
    return std::nullopt;
}

std::optional<ErrorInfo>
FixedContinuationLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    if (num_new_tokens != 1 || !new_tokens.defined() || !new_tokens.device().is_cpu()
        || new_tokens.scalar_type() != torch::kInt32
        || new_tokens.dim() != 2 || new_tokens.size(0) != 1 || new_tokens.size(1) != 1
        || committed_ >= continuation_.size() || new_tokens.data_ptr<int32_t>()[0] != continuation_[committed_]) {
        return invalid("committed token differs from frozen continuation");
    }
    // Decode receives the already-sampled first Prefill token before its own
    // first forward. Verify that token without inventing a local logits row.
    if (pending_raw_logits_.defined()) {
        try {
            const auto directory = std::filesystem::path(output_directory_)
                                   / (case_id_ + "-request-" + std::to_string(request_id_));
            std::filesystem::create_directories(directory);
            auto host = pending_raw_logits_.to(torch::kCPU).contiguous();
            const auto stem = directory / ("row-" + std::to_string(committed_));
            writeExclusive(stem.string() + ".f32", host.data_ptr<float>(), vocab_size_ * sizeof(float));
            const char* role_env = std::getenv("ROLE_TYPE");
            std::string role = role_env ? role_env : "UNKNOWN";
            if (role != "PREFILL" && role != "DECODE" && role != "PDFUSION") {
                role = "UNKNOWN";
            }
            std::string metadata = "{\"schema_version\":1,\"dtype\":\"float32\",\"vocab_size\":"
                                   + std::to_string(vocab_size_) + ",\"row\":" + std::to_string(committed_)
                                   + ",\"context_length\":" + std::to_string(input_ids_.size() + committed_)
                                   + ",\"target_token\":" + std::to_string(continuation_[committed_])
                                   + ",\"request_id\":" + std::to_string(request_id_)
                                   + ",\"pid\":" + std::to_string(getpid())
                                   + ",\"source_scalar_type\":" + std::to_string(source_scalar_type_)
                                   + ",\"case_id\":\"" + case_id_ + "\",\"role\":\"" + role
                                   + "\",\"source\":\"native_lm_head_before_sampling\"}\n";
            writeExclusive(stem.string() + ".json", metadata.data(), metadata.size());
            RTP_LLM_LOG_INFO("fixed continuation captured: request=%ld case=%s row=%ld context=%zu vocab=%d",
                             request_id_, case_id_.c_str(), committed_, input_ids_.size() + committed_, vocab_size_);
            pending_raw_logits_ = torch::Tensor();
        } catch (const std::exception& error) {
            return invalid(error.what());
        }
    }
    ++committed_;
    return std::nullopt;
}

}  // namespace rtp_llm
