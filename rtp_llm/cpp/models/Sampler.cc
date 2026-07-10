#include "rtp_llm/cpp/models/Sampler.h"
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include <unordered_map>
#include <unordered_set>
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "autil/TimeUtility.h"

using namespace std;

namespace rtp_llm {

Sampler::Sampler(const SamplerInitParams& params) {}

namespace {

std::string samplerTraceEnv(const char* name, const std::string& default_value = "") {
    const char* value = std::getenv(name);
    return value == nullptr ? default_value : std::string(value);
}

bool samplerTraceEnabled() {
    static const bool enabled = [] {
        auto value = samplerTraceEnv("RTPLLM_SAMPLER_TRACE");
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
        return value == "1" || value == "true" || value == "yes" || value == "on";
    }();
    return enabled;
}

std::vector<std::string> samplerTraceFilters() {
    static const std::vector<std::string> filters = [] {
        std::vector<std::string> result;
        std::stringstream        ss(samplerTraceEnv("RTPLLM_SAMPLER_TRACE_FILTER"));
        std::string              item;
        while (std::getline(ss, item, ',')) {
            auto begin = item.find_first_not_of(" \t\r\n");
            if (begin == std::string::npos) {
                continue;
            }
            auto end = item.find_last_not_of(" \t\r\n");
            result.emplace_back(item.substr(begin, end - begin + 1));
        }
        return result;
    }();
    return filters;
}

bool samplerTraceMatches(const std::string& trace_id) {
    if (!samplerTraceEnabled()) {
        return false;
    }
    const auto filters = samplerTraceFilters();
    if (filters.empty()) {
        return true;
    }
    return std::any_of(filters.begin(), filters.end(), [&](const std::string& filter) {
        return trace_id.find(filter) != std::string::npos;
    });
}

int samplerTraceMaxOutputLen() {
    static const int max_output_len = [] {
        const auto value = samplerTraceEnv("RTPLLM_SAMPLER_TRACE_MAX_OUTPUT_LEN");
        if (value.empty()) {
            return -1;
        }
        return std::atoi(value.c_str());
    }();
    return max_output_len;
}

std::string samplerTraceRankString() {
    auto rank = samplerTraceEnv("WORLD_RANK");
    if (rank.empty()) {
        rank = samplerTraceEnv("RANK");
    }
    if (rank.empty()) {
        rank = samplerTraceEnv("LOCAL_RANK", "0");
    }
    return rank;
}

bool samplerTraceEnsureDir(const std::string& path) {
    if (path.empty()) {
        return true;
    }
    std::string current;
    current.reserve(path.size());
    for (size_t i = 0; i < path.size(); ++i) {
        current.push_back(path[i]);
        if (path[i] != '/' || current.size() == 1) {
            continue;
        }
        if (::mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
            return false;
        }
    }
    return ::mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

std::string samplerTraceDirname(const std::string& path) {
    auto pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return "";
    }
    if (pos == 0) {
        return "/";
    }
    return path.substr(0, pos);
}

std::string samplerTraceOutputPath() {
    auto explicit_file = samplerTraceEnv("RTPLLM_SAMPLER_TRACE_FILE");
    if (!explicit_file.empty()) {
        return explicit_file;
    }
    auto dir = samplerTraceEnv("RTPLLM_SAMPLER_TRACE_DIR", "/tmp/rtpllm_sampler_trace");
    std::ostringstream oss;
    oss << dir << "/sampler_trace_rank" << samplerTraceRankString() << "_pid" << ::getpid() << ".jsonl";
    return oss.str();
}

std::ofstream& samplerTraceOutputStream() {
    static std::ofstream stream;
    static bool          initialized = false;
    if (!initialized) {
        initialized = true;
        const auto path = samplerTraceOutputPath();
        if (samplerTraceEnsureDir(samplerTraceDirname(path))) {
            stream.open(path, std::ios::out | std::ios::app);
        }
    }
    return stream;
}

std::mutex& samplerTraceMutex() {
    static std::mutex mutex;
    return mutex;
}

std::string samplerTraceJsonEscape(const std::string& value) {
    std::ostringstream os;
    for (char c : value) {
        switch (c) {
            case '\\':
                os << "\\\\";
                break;
            case '"':
                os << "\\\"";
                break;
            case '\n':
                os << "\\n";
                break;
            case '\r':
                os << "\\r";
                break;
            case '\t':
                os << "\\t";
                break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    const char* hex = "0123456789abcdef";
                    os << "\\u00" << hex[(c >> 4) & 0xf] << hex[c & 0xf];
                } else {
                    os << c;
                }
        }
    }
    return os.str();
}

bool samplerTraceChunkMatches(const SamplerInputs& inputs, size_t from_batch_idx_in, size_t batch_size_in) {
    if (!samplerTraceEnabled()) {
        return false;
    }
    for (size_t i = 0; i < batch_size_in; ++i) {
        const auto idx = from_batch_idx_in + i;
        const auto trace_id = idx < inputs.trace_ids.size() ? inputs.trace_ids[idx] : "";
        if (samplerTraceMatches(trace_id)) {
            return true;
        }
    }
    return false;
}

void samplerTraceAppendTensorInts(std::ostream& os, const torch::Tensor& values) {
    auto cpu = values.to(torch::kCPU).contiguous();
    os << "[";
    const auto* ptr = cpu.data_ptr<int32_t>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        if (i) {
            os << ",";
        }
        os << ptr[i];
    }
    os << "]";
}

void samplerTraceAppendTopK(std::ostream& os, const torch::Tensor& row, int64_t topn) {
    const auto limit = std::min<int64_t>(std::max<int64_t>(1, topn), row.size(0));
    auto       vals_inds = row.to(torch::kFloat32).topk(limit);
    auto       vals_cpu  = std::get<0>(vals_inds).to(torch::kCPU).contiguous();
    auto       inds_cpu  = std::get<1>(vals_inds).to(torch::kCPU).contiguous();
    const auto* vals     = vals_cpu.data_ptr<float>();
    const auto* inds     = inds_cpu.data_ptr<int64_t>();
    os << "[";
    for (int64_t i = 0; i < limit; ++i) {
        if (i) {
            os << ",";
        }
        os << "{\"id\":" << inds[i] << ",\"value\":" << vals[i] << "}";
    }
    os << "]";
}

std::string samplerTopKJson(const torch::Tensor& row, int64_t topn) {
    std::ostringstream os;
    samplerTraceAppendTopK(os, row, topn);
    return os.str();
}

void samplerTraceLogRows(const char*          stage,
                         const SamplerInputs& inputs,
                         size_t               from_batch_idx_in,
                         size_t               batch_size_in,
                         const torch::Tensor& logits,
                         const torch::Tensor& token_ids) {
    if (!samplerTraceChunkMatches(inputs, from_batch_idx_in, batch_size_in)) {
        return;
    }
    const int topn = std::max(1, std::atoi(samplerTraceEnv("RTPLLM_SAMPLER_TRACE_TOPK", "8").c_str()));
    std::ostringstream row;
    row << "{\"ts_us\":" << autil::TimeUtility::currentTimeInMicroSeconds() << ",\"pid\":" << ::getpid()
        << ",\"rank\":\"" << samplerTraceJsonEscape(samplerTraceRankString()) << "\",\"event\":\"sampler_trace\""
        << ",\"stage\":\"" << stage << "\",\"from_batch_idx_in\":" << from_batch_idx_in
        << ",\"batch_size_in\":" << batch_size_in << ",\"step\":" << inputs.step << ",\"rows\":[";

    const auto* input_lengths       = inputs.input_lengths.data_ptr<int32_t>();
    const auto* sequence_lengths    = inputs.sequence_lengths.data_ptr<int32_t>();
    const auto* top_k               = inputs.top_k.data_ptr<int32_t>();
    const auto* top_p               = inputs.top_p.data_ptr<float>();
    const auto* temperature         = inputs.temperature.data_ptr<float>();
    const auto* repetition_penalty  = inputs.repetition_penalty.data_ptr<float>();
    const auto* presence_penalty    = inputs.presence_penalty.data_ptr<float>();
    const auto* frequency_penalty   = inputs.frequency_penalty.data_ptr<float>();
    const auto* do_sample           = inputs.do_sample.data_ptr<bool>();
    const int   max_output_len      = samplerTraceMaxOutputLen();
    bool        wrote               = false;
    for (size_t i = 0; i < batch_size_in; ++i) {
        const auto global_idx = from_batch_idx_in + i;
        const auto trace_id   = global_idx < inputs.trace_ids.size() ? inputs.trace_ids[global_idx] : "";
        if (!samplerTraceMatches(trace_id)) {
            continue;
        }
        const int32_t output_len = sequence_lengths[global_idx] - input_lengths[global_idx];
        if (max_output_len >= 0 && output_len > max_output_len) {
            continue;
        }
        if (wrote) {
            row << ",";
        }
        wrote = true;
        const int32_t seq_len    = sequence_lengths[global_idx];
        const int64_t tail_begin = std::max<int64_t>(0, std::min<int64_t>(seq_len, inputs.step + 1) - 16);
        const int64_t tail_end   = std::max<int64_t>(tail_begin, std::min<int64_t>(seq_len, inputs.step + 1));
        row << "{\"trace_id\":\"" << samplerTraceJsonEscape(trace_id) << "\",\"global_idx\":" << global_idx
            << ",\"local_idx\":" << i << ",\"input_len\":" << input_lengths[global_idx]
            << ",\"seq_len\":" << seq_len << ",\"output_len\":" << output_len << ",\"top_k\":" << top_k[global_idx]
            << ",\"top_p\":" << top_p[global_idx] << ",\"temperature\":" << temperature[global_idx]
            << ",\"repetition_penalty\":" << repetition_penalty[global_idx]
            << ",\"presence_penalty\":" << presence_penalty[global_idx]
            << ",\"frequency_penalty\":" << frequency_penalty[global_idx]
            << ",\"do_sample\":" << (do_sample[global_idx] ? "true" : "false") << ",\"token_tail\":";
        samplerTraceAppendTensorInts(row, token_ids[i].slice(0, tail_begin, tail_end));
        row << ",\"selected_token\":" << token_ids[i][inputs.step].item<int32_t>() << ",\"top_logits\":";
        samplerTraceAppendTopK(row, logits[i], topn);
        row << "}";
    }
    row << "]}";

    std::lock_guard<std::mutex> lock(samplerTraceMutex());
    auto&                       os = samplerTraceOutputStream();
    if (os.good()) {
        os << row.str() << "\n";
        os.flush();
    }
}

bool samplerBadWatchEnabled() {
    static const bool enabled = [] {
        auto value = samplerTraceEnv("RTPLLM_SAMPLER_BAD_WATCH");
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
        return value == "1" || value == "true" || value == "yes" || value == "on";
    }();
    return enabled;
}

int samplerBadWatchEnvInt(const char* name, int default_value) {
    const auto value = samplerTraceEnv(name);
    if (value.empty()) {
        return default_value;
    }
    try {
        return std::stoi(value);
    } catch (...) {
        return default_value;
    }
}

std::vector<std::string> samplerBadWatchFilters() {
    static const std::vector<std::string> filters = [] {
        std::vector<std::string> result;
        std::stringstream        ss(samplerTraceEnv("RTPLLM_SAMPLER_BAD_WATCH_FILTER"));
        std::string              item;
        while (std::getline(ss, item, ',')) {
            auto begin = item.find_first_not_of(" \t\r\n");
            if (begin == std::string::npos) {
                continue;
            }
            auto end = item.find_last_not_of(" \t\r\n");
            result.emplace_back(item.substr(begin, end - begin + 1));
        }
        return result;
    }();
    return filters;
}

bool samplerBadWatchMatches(const std::string& trace_id) {
    if (!samplerBadWatchEnabled()) {
        return false;
    }
    const auto filters = samplerBadWatchFilters();
    if (filters.empty()) {
        return true;
    }
    return std::any_of(filters.begin(), filters.end(), [&](const std::string& filter) {
        return trace_id.find(filter) != std::string::npos;
    });
}

bool samplerBadWatchChunkMatches(const SamplerInputs& inputs, size_t from_batch_idx_in, size_t batch_size_in) {
    if (!samplerBadWatchEnabled()) {
        return false;
    }
    for (size_t i = 0; i < batch_size_in; ++i) {
        const auto idx      = from_batch_idx_in + i;
        const auto trace_id = idx < inputs.trace_ids.size() ? inputs.trace_ids[idx] : "";
        if (samplerBadWatchMatches(trace_id)) {
            return true;
        }
    }
    return false;
}

std::string samplerBadWatchOutputPath() {
    auto explicit_file = samplerTraceEnv("RTPLLM_SAMPLER_BAD_WATCH_FILE");
    if (!explicit_file.empty()) {
        return explicit_file;
    }
    auto dir = samplerTraceEnv("RTPLLM_SAMPLER_BAD_WATCH_DIR", "/tmp/rtpllm_sampler_bad_watch");
    std::ostringstream oss;
    oss << dir << "/sampler_bad_watch_rank" << samplerTraceRankString() << "_pid" << ::getpid() << ".jsonl";
    return oss.str();
}

std::ofstream& samplerBadWatchOutputStream() {
    static std::ofstream stream;
    static bool          initialized = false;
    if (!initialized) {
        initialized = true;
        const auto path = samplerBadWatchOutputPath();
        if (samplerTraceEnsureDir(samplerTraceDirname(path))) {
            stream.open(path, std::ios::out | std::ios::app);
        }
    }
    return stream;
}

std::mutex& samplerBadWatchMutex() {
    static std::mutex mutex;
    return mutex;
}

void samplerBadWatchAppendIntVector(std::ostream& os, const std::vector<int>& values) {
    os << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) {
            os << ",";
        }
        os << values[i];
    }
    os << "]";
}

struct SamplerBadWatchRepeatedSuffixInfo {
    bool             matched      = false;
    int              pattern_size = 0;
    int              repeat_count = 0;
    std::vector<int> pattern;
};

SamplerBadWatchRepeatedSuffixInfo samplerBadWatchFindRepeatedSuffix(const std::vector<int>& values,
                                                                     int                    max_pattern_size,
                                                                     int                    min_repeats) {
    SamplerBadWatchRepeatedSuffixInfo info;
    if (min_repeats <= 1) {
        return info;
    }
    for (int pattern_size = 1; pattern_size <= max_pattern_size; ++pattern_size) {
        const int need = pattern_size * min_repeats;
        if ((int)values.size() < need) {
            continue;
        }
        int repeat_count = 1;
        for (int start = static_cast<int>(values.size()) - 2 * pattern_size; start >= 0; start -= pattern_size) {
            bool same = true;
            for (int offset = 0; offset < pattern_size; ++offset) {
                if (values[start + offset] != values[values.size() - pattern_size + offset]) {
                    same = false;
                    break;
                }
            }
            if (!same) {
                break;
            }
            ++repeat_count;
        }
        if (repeat_count >= min_repeats) {
            info.matched      = true;
            info.pattern_size = pattern_size;
            info.repeat_count = repeat_count;
            info.pattern.assign(values.end() - pattern_size, values.end());
            return info;
        }
    }
    return info;
}

struct SamplerBadWatchState {
    std::vector<int> token_tail;
    bool             triggered      = false;
    int              cf_total       = 0;
    size_t           cf_match_index = 0;
};

struct SamplerPreprocessWatchSnapshot {
    int64_t     step       = -1;
    int32_t     input_len  = -1;
    int32_t     seq_len    = -1;
    int32_t     output_len = -1;
    std::string raw_top_logits;
};

std::unordered_map<std::string, SamplerBadWatchState>& samplerBadWatchStates() {
    static std::unordered_map<std::string, SamplerBadWatchState> states;
    return states;
}

bool samplerPreprocessWatchEnabled() {
    static const bool enabled = [] {
        auto value = samplerTraceEnv("RTPLLM_SAMPLER_PREPROCESS_WATCH");
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
        return value == "1" || value == "true" || value == "yes" || value == "on";
    }();
    return enabled;
}

std::vector<std::string> samplerPreprocessWatchFilters() {
    static const std::vector<std::string> filters = [] {
        std::vector<std::string> result;
        std::stringstream        ss(samplerTraceEnv("RTPLLM_SAMPLER_PREPROCESS_WATCH_FILTER"));
        std::string              item;
        while (std::getline(ss, item, ',')) {
            auto begin = item.find_first_not_of(" \t\r\n");
            if (begin == std::string::npos) {
                continue;
            }
            auto end = item.find_last_not_of(" \t\r\n");
            result.emplace_back(item.substr(begin, end - begin + 1));
        }
        if (result.empty()) {
            return samplerBadWatchFilters();
        }
        return result;
    }();
    return filters;
}

bool samplerPreprocessWatchMatches(const std::string& trace_id) {
    if (!samplerPreprocessWatchEnabled()) {
        return false;
    }
    const auto filters = samplerPreprocessWatchFilters();
    if (filters.empty()) {
        return true;
    }
    return std::any_of(filters.begin(), filters.end(), [&](const std::string& filter) {
        return trace_id.find(filter) != std::string::npos;
    });
}

std::mutex& samplerPreprocessWatchMutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<std::string, SamplerPreprocessWatchSnapshot>& samplerPreprocessWatchSnapshots() {
    static std::unordered_map<std::string, SamplerPreprocessWatchSnapshot> snapshots;
    return snapshots;
}

int samplerPreprocessWatchTopK() {
    return std::max(1, samplerBadWatchEnvInt("RTPLLM_SAMPLER_PREPROCESS_WATCH_TOPK", 8));
}

void samplerPreprocessWatchCaptureRaw(const SamplerInputs& inputs) {
    if (!samplerPreprocessWatchEnabled() || !inputs.logits.defined() || inputs.logits.dim() != 2) {
        return;
    }

    const auto* input_lengths    = inputs.input_lengths.data_ptr<int32_t>();
    const auto* sequence_lengths = inputs.sequence_lengths.data_ptr<int32_t>();
    const int   topn             = samplerPreprocessWatchTopK();

    std::lock_guard<std::mutex> lock(samplerPreprocessWatchMutex());
    auto&                       snapshots = samplerPreprocessWatchSnapshots();
    for (size_t i = 0; i < inputs.batch_size; ++i) {
        const auto trace_id = i < inputs.trace_ids.size() ? inputs.trace_ids[i] : "";
        if (!samplerPreprocessWatchMatches(trace_id)) {
            continue;
        }
        const int32_t seq_len    = sequence_lengths[i];
        const int32_t input_len  = input_lengths[i];
        const int32_t output_len = seq_len >= input_len ? seq_len - input_len : -1;
        snapshots[trace_id] = SamplerPreprocessWatchSnapshot{
            (int64_t)inputs.step, input_len, seq_len, output_len, samplerTopKJson(inputs.logits[i], topn)};
    }
}

std::optional<SamplerPreprocessWatchSnapshot> samplerPreprocessWatchLookup(const std::string& trace_id, int64_t step) {
    if (!samplerPreprocessWatchEnabled()) {
        return std::nullopt;
    }
    std::lock_guard<std::mutex> lock(samplerPreprocessWatchMutex());
    auto&                       snapshots = samplerPreprocessWatchSnapshots();
    auto                        iter      = snapshots.find(trace_id);
    if (iter == snapshots.end() || iter->second.step != step) {
        return std::nullopt;
    }
    return iter->second;
}

void samplerBadWatchUpdateCfCount(SamplerBadWatchState&    state,
                                  int                      token,
                                  const std::vector<int>&  pattern) {
    if (pattern.empty()) {
        return;
    }
    if (state.cf_match_index >= pattern.size()) {
        state.cf_match_index = 0;
    }
    while (state.cf_match_index > 0 && token != pattern[state.cf_match_index]) {
        state.cf_match_index = 0;
    }
    if (token != pattern[state.cf_match_index]) {
        return;
    }
    ++state.cf_match_index;
    if (state.cf_match_index == pattern.size()) {
        ++state.cf_total;
        state.cf_match_index = 0;
    }
}

void samplerBadWatchMaybeLog(const SamplerInputs& inputs,
                             size_t               from_batch_idx_in,
                             size_t               batch_size_in,
                             const torch::Tensor& logits,
                             const torch::Tensor& token_ids) {
    if (!samplerBadWatchChunkMatches(inputs, from_batch_idx_in, batch_size_in)) {
        return;
    }
    const int tail_size        = std::max(16, samplerBadWatchEnvInt("RTPLLM_SAMPLER_BAD_WATCH_TAIL_SIZE", 160));
    const int min_cf           = std::max(2, samplerBadWatchEnvInt("RTPLLM_SAMPLER_BAD_WATCH_MIN_CF", 4));
    const int max_pattern_size = std::max(1, samplerBadWatchEnvInt("RTPLLM_SAMPLER_BAD_WATCH_MAX_PATTERN_SIZE", 8));
    const int min_repeats      = std::max(2, samplerBadWatchEnvInt("RTPLLM_SAMPLER_BAD_WATCH_MIN_REPEATS", 8));
    const int topn             = std::max(1, samplerBadWatchEnvInt("RTPLLM_SAMPLER_BAD_WATCH_TOPK", 8));

    auto selected_cpu =
        token_ids.slice(1, inputs.step, inputs.step + 1).squeeze(1).to(torch::kCPU).contiguous();
    const auto* selected_ptr = selected_cpu.data_ptr<int32_t>();
    const auto* input_lengths       = inputs.input_lengths.data_ptr<int32_t>();
    const auto* sequence_lengths    = inputs.sequence_lengths.data_ptr<int32_t>();
    const auto* top_k               = inputs.top_k.data_ptr<int32_t>();
    const auto* top_p               = inputs.top_p.data_ptr<float>();
    const auto* temperature         = inputs.temperature.data_ptr<float>();
    const auto* repetition_penalty  = inputs.repetition_penalty.data_ptr<float>();
    const auto* presence_penalty    = inputs.presence_penalty.data_ptr<float>();
    const auto* frequency_penalty   = inputs.frequency_penalty.data_ptr<float>();
    const auto* do_sample           = inputs.do_sample.data_ptr<bool>();

    std::vector<int> chunk_selected;
    chunk_selected.reserve(batch_size_in);
    for (size_t i = 0; i < batch_size_in; ++i) {
        chunk_selected.push_back(selected_ptr[i]);
    }

    static const std::vector<int> cf_pattern = {27, 9500, 1419, 9500, 29};
    std::lock_guard<std::mutex> lock(samplerBadWatchMutex());
    for (size_t i = 0; i < batch_size_in; ++i) {
        const auto global_idx = from_batch_idx_in + i;
        const auto trace_id   = global_idx < inputs.trace_ids.size() ? inputs.trace_ids[global_idx] : "";
        if (!samplerBadWatchMatches(trace_id)) {
            continue;
        }
        auto& state = samplerBadWatchStates()[trace_id];
        const int token = selected_ptr[i];
        samplerBadWatchUpdateCfCount(state, token, cf_pattern);
        state.token_tail.push_back(token);
        if ((int)state.token_tail.size() > tail_size) {
            state.token_tail.erase(state.token_tail.begin(),
                                   state.token_tail.begin() + (state.token_tail.size() - tail_size));
        }
        const auto repeated_suffix =
            samplerBadWatchFindRepeatedSuffix(state.token_tail, max_pattern_size, min_repeats);
        const bool cf_tail_repeat = state.cf_total >= min_cf;
        if (state.triggered || (!cf_tail_repeat && !repeated_suffix.matched)) {
            continue;
        }
        state.triggered = true;

        const int32_t seq_len    = sequence_lengths[global_idx];
        const int32_t input_len  = input_lengths[global_idx];
        const int32_t output_len = seq_len >= input_len ? seq_len - input_len : -1;
        std::ostringstream row;
        row << "{\"ts_us\":" << autil::TimeUtility::currentTimeInMicroSeconds()
            << ",\"pid\":" << ::getpid()
            << ",\"rank\":\"" << samplerTraceJsonEscape(samplerTraceRankString()) << "\""
            << ",\"event\":\"sampler_bad_watch_trigger\""
            << ",\"reason\":\"" << (cf_tail_repeat ? "cf_tail_repeat" : "repeated_suffix") << "\""
            << ",\"trace_id\":\"" << samplerTraceJsonEscape(trace_id) << "\""
            << ",\"from_batch_idx_in\":" << from_batch_idx_in
            << ",\"batch_size_in\":" << batch_size_in
            << ",\"global_idx\":" << global_idx
            << ",\"local_idx\":" << i
            << ",\"step\":" << inputs.step
            << ",\"input_len\":" << input_len
            << ",\"seq_len\":" << seq_len
            << ",\"output_len\":" << output_len
            << ",\"top_k\":" << top_k[global_idx]
            << ",\"top_p\":" << top_p[global_idx]
            << ",\"temperature\":" << temperature[global_idx]
            << ",\"repetition_penalty\":" << repetition_penalty[global_idx]
            << ",\"presence_penalty\":" << presence_penalty[global_idx]
            << ",\"frequency_penalty\":" << frequency_penalty[global_idx]
            << ",\"do_sample\":" << (do_sample[global_idx] ? "true" : "false")
            << ",\"selected_token\":" << token
            << ",\"cf_total\":" << state.cf_total
            << ",\"repeat_pattern_size\":" << repeated_suffix.pattern_size
            << ",\"repeat_count\":" << repeated_suffix.repeat_count
            << ",\"repeat_pattern\":";
        samplerBadWatchAppendIntVector(row, repeated_suffix.pattern);
        row << ",\"token_tail\":";
        samplerBadWatchAppendIntVector(row, state.token_tail);
        row << ",\"chunk_selected_tokens\":";
        samplerBadWatchAppendIntVector(row, chunk_selected);
        row << ",\"preprocess_watch_enabled\":" << (samplerPreprocessWatchEnabled() ? "true" : "false")
            << ",\"logits_processor_state_present\":"
            << (inputs.logits_processor_states_ptr != nullptr ? "true" : "false")
            << ",\"logits_processor_count\":"
            << (inputs.logits_processor_states_ptr != nullptr ? inputs.logits_processor_states_ptr->size() : 0);
        const auto raw_snapshot = samplerPreprocessWatchLookup(trace_id, inputs.step);
        if (raw_snapshot.has_value()) {
            row << ",\"raw_preprocess_step\":" << raw_snapshot->step
                << ",\"raw_preprocess_input_len\":" << raw_snapshot->input_len
                << ",\"raw_preprocess_seq_len\":" << raw_snapshot->seq_len
                << ",\"raw_preprocess_output_len\":" << raw_snapshot->output_len
                << ",\"raw_preprocess_top_logits\":" << raw_snapshot->raw_top_logits;
        } else {
            row << ",\"raw_preprocess_top_logits\":null";
        }
        const auto post_top_logits = samplerTopKJson(logits[i], topn);
        row << ",\"post_preprocess_top_logits\":" << post_top_logits;
        row << ",\"top_logits\":" << post_top_logits;
        row << "}";

        auto& os = samplerBadWatchOutputStream();
        if (os.good()) {
            os << row.str() << "\n";
            os.flush();
        }
    }
}

}  // namespace

SamplerOutput Sampler::forward(const SamplerInputs& inputs) {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    RTP_LLM_PROFILE_SCOPE("sampler.forward");
    // Helper: narrow a tensor if defined, else return undefined tensor
    auto mayNarrow = [](const torch::Tensor& t, int64_t offset, int64_t size) -> torch::Tensor {
        return t.defined() ? t.narrow(0, offset, size) : torch::Tensor();
    };

    // Helper: convert optional tensor slice to std::optional<torch::Tensor>
    auto mayOptNarrow = [](const torch::Tensor& t, int64_t offset, int64_t size) -> std::optional<torch::Tensor> {
        return t.defined() ? std::optional<torch::Tensor>(t.narrow(0, offset, size)) : std::nullopt;
    };

    samplerPreprocessWatchCaptureRaw(inputs);
    preprocessLogits(inputs);

    uint64_t max_seq_len   = inputs.token_ids.size(1);
    auto     num_beams_in  = inputs.num_beams_in.data_ptr<int64_t>();
    auto     num_beams_out = inputs.num_beams_out.data_ptr<int64_t>();

    bool has_num_beams = std::any_of(num_beams_in, num_beams_in + inputs.batch_size, [](auto n) { return n > 1; })
                         || std::any_of(num_beams_out, num_beams_out + inputs.batch_size, [](auto n) { return n > 1; });
    bool variable_num_beams = inputs.batch_size != inputs.batch_size_out;

    // allocate output tensors
    // Keep success on CUDA to avoid a blocking D2H copy: the GPU sampling kernel writes success
    // directly, and callers that need CPU access should call .cpu() explicitly.
    auto all_success =
        torch::empty({(int64_t)inputs.batch_size}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    auto all_beam_indices =
        has_num_beams ? torch::empty({(int64_t)inputs.batch_size_out}, torch::kInt32) : torch::Tensor();
    // Move token_ids to CUDA once so sampleGreedy writes GPU→GPU (no blocking D2H sync).
    // Callers that need CPU access should call .cpu() explicitly.
    // Use blocking transfer: on ROCm, hipMemcpyAsync from pageable memory is truly async
    // and can cause memory access faults if a kernel reads the buffer before transfer completes.
    auto inputs_token_ids_cuda = inputs.token_ids.to(torch::kCUDA);
    auto all_token_ids_out     = variable_num_beams ?
                                     torch::empty({(int64_t)inputs.batch_size_out, (int64_t)max_seq_len},
                                              torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA)) :
                                     inputs_token_ids_cuda;
    auto all_cum_log_probs_out = variable_num_beams && inputs.cum_log_probs.defined() ?
                                     torch::empty({(int64_t)inputs.batch_size_out}, torch::kFloat32) :
                                     inputs.cum_log_probs;

    size_t from_batch_idx_in = 0, to_batch_idx_in = 0;
    size_t from_batch_idx_out = 0;

    while (from_batch_idx_in < inputs.batch_size) {
        auto cur_num_beams_in  = num_beams_in[from_batch_idx_in];
        auto cur_num_beams_out = num_beams_out[from_batch_idx_in];
        ++to_batch_idx_in;
        while (to_batch_idx_in < inputs.batch_size && num_beams_in[to_batch_idx_in] == cur_num_beams_in
               && num_beams_out[to_batch_idx_in] == cur_num_beams_out) {
            ++to_batch_idx_in;
        }

        // now from_batch_idx to to_batch_idx have the same beam size, sample once.
        const auto batch_size_in    = to_batch_idx_in - from_batch_idx_in;
        const auto beam_batch_size  = batch_size_in / cur_num_beams_in;
        const auto batch_size_out   = beam_batch_size * cur_num_beams_out;
        const auto to_batch_idx_out = from_batch_idx_out + batch_size_out;

        auto success           = all_success.narrow(0, from_batch_idx_in, batch_size_in);
        auto logits            = inputs.logits.narrow(0, from_batch_idx_in, batch_size_in);
        auto token_ids_in      = inputs_token_ids_cuda.narrow(0, from_batch_idx_in, batch_size_in);
        auto token_ids_out     = all_token_ids_out.narrow(0, from_batch_idx_out, batch_size_out);
        auto input_lengths     = inputs.input_lengths.narrow(0, from_batch_idx_in, batch_size_in);
        auto sequence_lengths  = inputs.sequence_lengths.narrow(0, from_batch_idx_in, batch_size_in);
        auto cum_log_probs_in  = mayNarrow(inputs.cum_log_probs, from_batch_idx_in, batch_size_in);
        auto cum_log_probs_out = mayNarrow(all_cum_log_probs_out, from_batch_idx_out, batch_size_out);

        if (cur_num_beams_in == 1 && cur_num_beams_out == 1) {
            const auto decoder_batch_size = (int64_t)inputs.sequence_lengths.size(0);
            auto       sequence_lengths_in =
                (int64_t)from_batch_idx_in < decoder_batch_size ?
                          inputs.sequence_lengths.narrow(
                        0,
                        from_batch_idx_in,
                        min((int64_t)batch_size_in, decoder_batch_size - (int64_t)from_batch_idx_in)) :
                          torch::empty({0}, torch::kInt32);

            // TODO(zhangjianning.zjn): would be better to eliminate the copy
            if (cum_log_probs_out.defined() && cum_log_probs_in.defined()) {
                cum_log_probs_out.copy_(cum_log_probs_in);
            }

            auto top_k                = inputs.top_k.narrow(0, from_batch_idx_in, batch_size_in);
            auto top_p                = inputs.top_p.narrow(0, from_batch_idx_in, batch_size_in);
            auto temperature          = inputs.temperature.narrow(0, from_batch_idx_in, batch_size_in);
            auto repetition_penalty   = mayOptNarrow(inputs.repetition_penalty, from_batch_idx_in, batch_size_in);
            auto presence_penalty     = mayOptNarrow(inputs.presence_penalty, from_batch_idx_in, batch_size_in);
            auto frequency_penalty    = mayOptNarrow(inputs.frequency_penalty, from_batch_idx_in, batch_size_in);
            auto no_repeat_ngram_size = mayOptNarrow(inputs.no_repeat_ngram_size, from_batch_idx_in, batch_size_in);
            auto all_probs            = mayOptNarrow(inputs.all_probs, from_batch_idx_in, batch_size_in);
            auto do_sample            = mayOptNarrow(inputs.do_sample, from_batch_idx_in, batch_size_in);
            auto generator            = std::vector<at::Generator>{inputs.generator.begin() + from_batch_idx_in,
                                                                   inputs.generator.begin() + from_batch_idx_in + batch_size_in};
            const bool trace_chunk     = samplerTraceChunkMatches(inputs, from_batch_idx_in, batch_size_in);

            RTP_LLM_PROFILE_SCOPE("sampler.forward.execSampleGreedy");
            if (trace_chunk) {
                samplerTraceLogRows("before", inputs, from_batch_idx_in, batch_size_in, logits, token_ids_in);
            }
            auto greedy_output = execSampleGreedy(
                {logits,
                 input_lengths,
                 sequence_lengths_in,
                 token_ids_in,
                 inputs.step,
                 top_k,
                 top_p,
                 temperature,
                 repetition_penalty,
                 no_repeat_ngram_size,
                 cum_log_probs_out.defined() ? std::optional<torch::Tensor>(cum_log_probs_out) : std::nullopt,
                 std::nullopt,  // output_log_probs
                 all_probs,
                 presence_penalty,
                 frequency_penalty,
                 do_sample,
                 generator});
            if (trace_chunk) {
                samplerTraceLogRows("after", inputs, from_batch_idx_in, batch_size_in, logits, token_ids_in);
            }
            samplerBadWatchMaybeLog(inputs, from_batch_idx_in, batch_size_in, logits, token_ids_in);
            if (greedy_output.success.defined()) {
                success.copy_(greedy_output.success);
                // TODO(zhangjianning.zjn): would be better to eliminate the copy
                if (variable_num_beams) {
                    token_ids_out.copy_(token_ids_in);
                }
            } else {
                success.fill_(true);
            }
        } else {
            RTP_LLM_LOG_DEBUG("current_num_beams_in is %d", cur_num_beams_in);
            RTP_LLM_LOG_DEBUG("current_num_beams_out is %d", cur_num_beams_out);
            RTP_LLM_LOG_DEBUG("current_beam_batch is %d", beam_batch_size);
            RTP_LLM_CHECK_WITH_INFO((batch_size_in % cur_num_beams_in == 0),
                                    "sample_batch_size[%d] must devide by current_num_beams_in[%d]");

            const size_t vocab_size      = inputs.logits.size(1);
            const size_t max_seq_len_val = inputs.token_ids.size(1);

            auto beam_indices = all_beam_indices.narrow(0, from_batch_idx_out, batch_size_out);

            // Reshape for beam search: [batch, beams, ...]
            auto logits_reshaped =
                logits.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_in, (int64_t)vocab_size});
            auto token_ids_in_reshaped =
                token_ids_in.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_in, (int64_t)max_seq_len_val});
            auto input_lengths_reshaped = input_lengths.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_in});
            auto sequence_lengths_reshaped =
                sequence_lengths.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_in});
            auto cum_log_probs_in_reshaped =
                cum_log_probs_in.defined() ?
                    cum_log_probs_in.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_in}) :
                    torch::zeros({(int64_t)beam_batch_size, (int64_t)cur_num_beams_in});

            auto logits_t           = logits_reshaped.to(torch::kCUDA);
            auto token_ids_in_t     = token_ids_in_reshaped.to(torch::kCUDA);
            auto input_lengths_t    = input_lengths_reshaped.to(torch::kCUDA);
            auto sequence_lengths_t = sequence_lengths_reshaped.to(torch::kCUDA);
            auto cum_log_probs_in_t = cum_log_probs_in_reshaped.to(torch::kCUDA);

            auto output = execSampleBeamSearch({logits_t,
                                                token_ids_in_t,
                                                input_lengths_t,
                                                sequence_lengths_t,
                                                cum_log_probs_in_t,
                                                (size_t)cur_num_beams_out});

            auto token_ids_out_reshaped =
                token_ids_out.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_out, (int64_t)max_seq_len_val});
            auto cum_log_probs_out_reshaped =
                cum_log_probs_out.defined() ?
                    cum_log_probs_out.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_out}) :
                    torch::Tensor();

            token_ids_out_reshaped.copy_(output.token_ids);
            if (cum_log_probs_out_reshaped.defined()) {
                cum_log_probs_out_reshaped.copy_(output.cum_log_probs);
            }
            beam_indices.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_out}).copy_(output.beam_indices);

            success.fill_(true);
        }

        // prepare for next sampling
        from_batch_idx_in  = to_batch_idx_in;
        from_batch_idx_out = to_batch_idx_out;
    }

    return SamplerOutput({std::move(all_token_ids_out),
                          std::move(all_cum_log_probs_out),
                          std::move(inputs.all_probs),
                          std::move(all_beam_indices),
                          std::move(all_success)});
}

void Sampler::preprocessLogits(const SamplerInputs& inputs) {
    if (inputs.logits_processor_states_ptr != nullptr) {
        inputs.logits_processor_states_ptr->batchProcess(inputs);
    }
}

}  // namespace rtp_llm
