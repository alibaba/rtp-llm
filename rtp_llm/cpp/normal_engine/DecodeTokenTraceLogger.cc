#include "rtp_llm/cpp/normal_engine/DecodeTokenTraceLogger.h"

#include <algorithm>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <utility>

#include "autil/TimeUtility.h"

namespace rtp_llm {
namespace {

std::string getenvString(const char* name, const std::string& default_value = "") {
    const char* value = std::getenv(name);
    return value == nullptr ? default_value : std::string(value);
}

bool getenvBool(const char* name, bool default_value = false) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return default_value;
    }
    std::string lower(value);
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
    return lower == "1" || lower == "true" || lower == "yes" || lower == "on";
}

int getenvInt(const char* name, int default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return default_value;
    }
    try {
        return std::stoi(value);
    } catch (...) {
        return default_value;
    }
}

std::vector<std::string> splitCsv(const std::string& value) {
    std::vector<std::string> result;
    std::stringstream        ss(value);
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
}

std::string rankString() {
    auto rank = getenvString("WORLD_RANK");
    if (rank.empty()) {
        rank = getenvString("RANK");
    }
    if (rank.empty()) {
        rank = getenvString("LOCAL_RANK", "0");
    }
    return rank;
}

bool ensureDir(const std::string& path) {
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

std::string dirnameOf(const std::string& path) {
    auto pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return "";
    }
    if (pos == 0) {
        return "/";
    }
    return path.substr(0, pos);
}

std::string defaultOutputPath() {
    auto explicit_file = getenvString("RTPLLM_DECODE_TOKEN_TRACE_FILE");
    if (!explicit_file.empty()) {
        return explicit_file;
    }
    auto dir = getenvString("RTPLLM_DECODE_TOKEN_TRACE_DIR", "/tmp/rtpllm_decode_token_trace");
    std::ostringstream oss;
    oss << dir << "/decode_token_trace_rank" << rankString() << "_pid" << ::getpid() << ".jsonl";
    return oss.str();
}

std::string defaultBadWatchOutputPath() {
    auto explicit_file = getenvString("RTPLLM_DECODE_BAD_WATCH_FILE");
    if (!explicit_file.empty()) {
        return explicit_file;
    }
    auto dir = getenvString("RTPLLM_DECODE_BAD_WATCH_DIR", "/tmp/rtpllm_decode_bad_watch");
    std::ostringstream oss;
    oss << dir << "/decode_bad_watch_rank" << rankString() << "_pid" << ::getpid() << ".jsonl";
    return oss.str();
}

DecodeTokenTraceConfig& config() {
    static DecodeTokenTraceConfig cfg = DecodeTokenTraceConfig::fromEnv();
    return cfg;
}

std::mutex& outputMutex() {
    static std::mutex mutex;
    return mutex;
}

std::ofstream& outputStream() {
    static std::ofstream stream;
    static bool          initialized = false;
    if (!initialized) {
        initialized = true;
        const auto& path = config().output_path;
        if (!path.empty() && ensureDir(dirnameOf(path))) {
            stream.open(path, std::ios::out | std::ios::app);
        }
    }
    return stream;
}

std::ofstream& badWatchOutputStream() {
    static std::ofstream stream;
    static bool          initialized = false;
    if (!initialized) {
        initialized = true;
        const auto& path = config().bad_watch_output_path;
        if (!path.empty() && ensureDir(dirnameOf(path))) {
            stream.open(path, std::ios::out | std::ios::app);
        }
    }
    return stream;
}

void appendIntVector(std::ostream& os, const std::vector<int>& values) {
    os << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) {
            os << ",";
        }
        os << values[i];
    }
    os << "]";
}

struct WatchStep {
    int              total_decode_batch_size  = 0;
    int              total_context_batch_size = 0;
    int              seq_len                  = 0;
    int              output_len               = 0;
    int              iter_count               = 0;
    int              current_batch_size       = 0;
    int              next_batch_size          = 0;
    int              read_logical_block       = -1;
    int              write_logical_block      = -1;
    int              cf_total                 = 0;
    std::vector<int> new_tokens;
    std::vector<int> read_blocks;
    std::vector<int> write_blocks;
};

struct WatchState {
    std::vector<int>   token_tail;
    std::deque<WatchStep> history;
    bool               triggered      = false;
    int                cf_total       = 0;
    size_t             cf_match_index = 0;
};

std::unordered_map<std::string, WatchState>& watchStates() {
    static std::unordered_map<std::string, WatchState> states;
    return states;
}

int countSubsequence(const std::vector<int>& values, const std::vector<int>& pattern) {
    if (pattern.empty() || values.size() < pattern.size()) {
        return 0;
    }
    int count = 0;
    for (size_t i = 0; i + pattern.size() <= values.size(); ++i) {
        bool matched = true;
        for (size_t j = 0; j < pattern.size(); ++j) {
            if (values[i + j] != pattern[j]) {
                matched = false;
                break;
            }
        }
        if (matched) {
            ++count;
            i += pattern.size() - 1;
        }
    }
    return count;
}

void updateStreamingSubsequenceCount(WatchState& state,
                                     const std::vector<int>& tokens,
                                     const std::vector<int>& pattern) {
    if (pattern.empty()) {
        return;
    }
    for (const auto token : tokens) {
        if (state.cf_match_index >= pattern.size()) {
            state.cf_match_index = 0;
        }
        while (state.cf_match_index > 0 && token != pattern[state.cf_match_index]) {
            state.cf_match_index = 0;
        }
        if (token != pattern[state.cf_match_index]) {
            continue;
        }
        ++state.cf_match_index;
        if (state.cf_match_index == pattern.size()) {
            ++state.cf_total;
            state.cf_match_index = 0;
        }
    }
}

bool hasRepeatedSuffix(const std::vector<int>& values, int max_pattern_size, int min_repeats) {
    if (min_repeats <= 1) {
        return false;
    }
    for (int pattern_size = 1; pattern_size <= max_pattern_size; ++pattern_size) {
        const int need = pattern_size * min_repeats;
        if ((int)values.size() < need) {
            continue;
        }
        const int start = values.size() - need;
        bool      same  = true;
        for (int i = start + pattern_size; i < (int)values.size(); ++i) {
            if (values[i] != values[start + (i - start) % pattern_size]) {
                same = false;
                break;
            }
        }
        if (same) {
            return true;
        }
    }
    return false;
}

void appendBlockSlice(std::ostream& os, const BlockIndicesType& blocks, int begin, int end) {
    os << "[";
    begin = std::max(0, std::min<int>(begin, blocks.size()));
    end   = std::max(begin, std::min<int>(end, blocks.size()));
    for (int i = begin; i < end; ++i) {
        if (i != begin) {
            os << ",";
        }
        os << "{\"pos\":" << i << ",\"block\":" << blocks[i] << "}";
    }
    os << "]";
}

void appendBlockGroups(std::ostream& os,
                       const GenerateStreamPtr& stream,
                       int                      max_blocks_per_group,
                       int                      read_logical_block  = -1,
                       int                      write_logical_block = -1) {
    os << "[";
    if (stream->curBlocksNum() > 0) {
        const auto& kv_cache   = stream->kvCache();
        const int   group_nums = kv_cache.groupNums();
        for (int group_id = 0; group_id < group_nums; ++group_id) {
            if (group_id) {
                os << ",";
            }
            const auto& blocks = kv_cache.blocks(0, group_id);
            os << "{\"group\":" << group_id << ",\"blocks\":[";
            const int limit = std::min<int>(blocks.size(), std::max(0, max_blocks_per_group));
            for (int i = 0; i < limit; ++i) {
                if (i) {
                    os << ",";
                }
                os << blocks[i];
            }
            os << "],\"total\":" << blocks.size();

            if (!blocks.empty()) {
                const int tail_limit = std::min<int>(blocks.size(), std::max(0, max_blocks_per_group));
                os << ",\"tail\":";
                appendBlockSlice(os, blocks, static_cast<int>(blocks.size()) - tail_limit, static_cast<int>(blocks.size()));
            }
            if (read_logical_block >= 0 || write_logical_block >= 0) {
                const int center_begin = std::max(0, std::min(read_logical_block, write_logical_block) - 4);
                const int center_end   = std::max(read_logical_block, write_logical_block) + 5;
                os << ",\"current_window\":";
                appendBlockSlice(os, blocks, center_begin, center_end);
            }
            os << "}";
        }
    }
    os << "]";
}

std::vector<int> collectGroupBlocksAt(const GenerateStreamPtr& stream, int logical_block) {
    std::vector<int> result;
    if (logical_block < 0 || stream->curBlocksNum() <= 0) {
        return result;
    }
    const auto& kv_cache   = stream->kvCache();
    const int   group_nums = kv_cache.groupNums();
    result.reserve(group_nums);
    for (int group_id = 0; group_id < group_nums; ++group_id) {
        const auto& blocks = kv_cache.blocks(0, group_id);
        result.push_back(logical_block < static_cast<int>(blocks.size()) ? blocks[logical_block] : -999999);
    }
    return result;
}

void appendWatchHistory(std::ostream& os, const std::deque<WatchStep>& history) {
    os << "[";
    for (size_t i = 0; i < history.size(); ++i) {
        if (i) {
            os << ",";
        }
        const auto& step = history[i];
        os << "{\"total_decode_batch_size\":" << step.total_decode_batch_size
           << ",\"total_context_batch_size\":" << step.total_context_batch_size
           << ",\"seq_len\":" << step.seq_len
           << ",\"output_len\":" << step.output_len
           << ",\"iter_count\":" << step.iter_count
           << ",\"current_batch_size\":" << step.current_batch_size
           << ",\"next_batch_size\":" << step.next_batch_size
           << ",\"read_logical_block\":" << step.read_logical_block
           << ",\"write_logical_block\":" << step.write_logical_block
           << ",\"cf_total\":" << step.cf_total
           << ",\"new_tokens\":";
        appendIntVector(os, step.new_tokens);
        os << ",\"read_blocks\":";
        appendIntVector(os, step.read_blocks);
        os << ",\"write_blocks\":";
        appendIntVector(os, step.write_blocks);
        os << "}";
    }
    os << "]";
}

}  // namespace

DecodeTokenTraceConfig DecodeTokenTraceConfig::fromValues(bool               enabled,
                                                          const std::string& filter_csv,
                                                          const std::string& output_path,
                                                          bool               capture_peers,
                                                          int                max_blocks_per_group,
                                                          bool               bad_watch_enabled,
                                                          const std::string& bad_watch_output_path,
                                                          int                bad_watch_tail_size,
                                                          int                bad_watch_min_cf,
                                                          int                bad_watch_history_size) {
    DecodeTokenTraceConfig cfg;
    cfg.enabled              = enabled;
    cfg.filters              = splitCsv(filter_csv);
    cfg.output_path          = output_path;
    cfg.capture_peers        = capture_peers;
    cfg.max_blocks_per_group = std::max(0, max_blocks_per_group);
    cfg.bad_watch_enabled    = bad_watch_enabled;
    cfg.bad_watch_output_path = bad_watch_output_path;
    cfg.bad_watch_tail_size  = std::max(16, bad_watch_tail_size);
    cfg.bad_watch_min_cf     = std::max(2, bad_watch_min_cf);
    cfg.bad_watch_history_size = std::max(0, bad_watch_history_size);
    return cfg;
}

DecodeTokenTraceConfig DecodeTokenTraceConfig::fromEnv() {
    return fromValues(getenvBool("RTPLLM_DECODE_TOKEN_TRACE", false),
                      getenvString("RTPLLM_DECODE_TOKEN_TRACE_FILTER"),
                      defaultOutputPath(),
                      getenvBool("RTPLLM_DECODE_TOKEN_TRACE_CAPTURE_PEERS", true),
                      getenvInt("RTPLLM_DECODE_TOKEN_TRACE_MAX_BLOCKS_PER_GROUP", 16),
                      getenvBool("RTPLLM_DECODE_BAD_WATCH", false),
                      defaultBadWatchOutputPath(),
                      getenvInt("RTPLLM_DECODE_BAD_WATCH_TAIL_SIZE", 128),
                      getenvInt("RTPLLM_DECODE_BAD_WATCH_MIN_CF", 4),
                      getenvInt("RTPLLM_DECODE_BAD_WATCH_HISTORY_SIZE", 128));
}

bool DecodeTokenTraceConfig::matches(const std::string& trace_id) const {
    if (!enabled) {
        return false;
    }
    if (filters.empty()) {
        return true;
    }
    return std::any_of(filters.begin(), filters.end(), [&](const std::string& filter) {
        return trace_id.find(filter) != std::string::npos;
    });
}

bool DecodeTokenTraceLogger::enabled() {
    return config().enabled;
}

std::string DecodeTokenTraceLogger::jsonEscape(const std::string& value) {
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
                    os << "\\u00";
                    const char* hex = "0123456789abcdef";
                    os << hex[(c >> 4) & 0xf] << hex[c & 0xf];
                } else {
                    os << c;
                }
        }
    }
    return os.str();
}

void DecodeTokenTraceLogger::logDispatchBatch(const StreamGroups&   stream_groups,
                                              const torch::Tensor& token_ids_cpu,
                                              const torch::Tensor& success_cpu) {
    const auto& cfg = config();
    if ((!cfg.enabled && !cfg.bad_watch_enabled) || !token_ids_cpu.defined() || token_ids_cpu.numel() == 0) {
        return;
    }

    const auto all_streams = stream_groups.allStreams();
    bool       has_match   = false;
    for (const auto& stream : all_streams) {
        if (cfg.matches(stream->traceId())) {
            has_match = true;
            break;
        }
    }
    if (cfg.enabled && !has_match) {
        return;
    }

    const auto token_stride = token_ids_cpu.size(1);
    const auto* token_ptr   = token_ids_cpu.data_ptr<int32_t>();
    const auto* success_ptr = success_cpu.defined() ? success_cpu.data_ptr<bool>() : nullptr;

    if (cfg.bad_watch_enabled) {
        int batch_idx_in  = 0;
        int batch_idx_out = 0;
        for (const auto& stream : all_streams) {
            const auto cur_bs  = stream->currentBatchSize();
            const auto next_bs = stream->nextBatchSize();
            std::vector<int> new_tokens;
            new_tokens.reserve(next_bs);
            for (int i = 0; i < next_bs; ++i) {
                new_tokens.push_back(token_ptr[(batch_idx_out + i) * token_stride + token_stride - 1]);
            }
            auto& state = watchStates()[stream->traceId()];
            static const std::vector<int> cf_pattern = {27, 9500, 1419, 9500, 29};
            updateStreamingSubsequenceCount(state, new_tokens, cf_pattern);
            state.token_tail.insert(state.token_tail.end(), new_tokens.begin(), new_tokens.end());
            if ((int)state.token_tail.size() > cfg.bad_watch_tail_size) {
                state.token_tail.erase(state.token_tail.begin(),
                                       state.token_tail.begin() + (state.token_tail.size() - cfg.bad_watch_tail_size));
            }
            const int cf_count = countSubsequence(state.token_tail, cf_pattern);
            const bool repeated_suffix = hasRepeatedSuffix(state.token_tail, 8, 8);
            const bool cf_total_repeat = state.cf_total >= cfg.bad_watch_min_cf;
            const int seq_size_per_block =
                stream->seqSizePerBlock() > 0 ? stream->seqSizePerBlock() : std::max(1, stream->seqLength());
            const int read_logical_block =
                stream->seqLength() > 0 ? (stream->seqLength() - 1) / seq_size_per_block : -1;
            const int write_logical_block = stream->seqLength() / seq_size_per_block;
            if (cfg.bad_watch_history_size > 0) {
                WatchStep step;
                step.total_decode_batch_size  = stream_groups.totalDecodeBatchSize();
                step.total_context_batch_size = stream_groups.totalContextBatchSize();
                step.seq_len                  = stream->seqLength();
                step.output_len               = stream->outputTokenLen();
                step.iter_count               = stream->iterCount();
                step.current_batch_size       = cur_bs;
                step.next_batch_size          = next_bs;
                step.read_logical_block       = read_logical_block;
                step.write_logical_block      = write_logical_block;
                step.cf_total                 = state.cf_total;
                step.new_tokens               = new_tokens;
                step.read_blocks              = collectGroupBlocksAt(stream, read_logical_block);
                step.write_blocks             = collectGroupBlocksAt(stream, write_logical_block);
                state.history.push_back(std::move(step));
                while (static_cast<int>(state.history.size()) > cfg.bad_watch_history_size) {
                    state.history.pop_front();
                }
            }
            if (!state.triggered && (cf_total_repeat || cf_count >= cfg.bad_watch_min_cf || repeated_suffix)) {
                state.triggered = true;
                std::ostringstream watch_row;
                watch_row << "{\"ts_us\":" << autil::TimeUtility::currentTimeInMicroSeconds()
                          << ",\"pid\":" << ::getpid() << ",\"rank\":\"" << jsonEscape(rankString())
                          << "\",\"event\":\"decode_bad_watch_trigger\""
                          << ",\"reason\":\""
                          << (cf_total_repeat ? "cf_total_repeat" :
                              (cf_count >= cfg.bad_watch_min_cf ? "cf_tail_repeat" : "repeated_suffix"))
                          << "\",\"cf_count\":" << cf_count
                          << ",\"cf_total\":" << state.cf_total
                          << ",\"total_decode_batch_size\":" << stream_groups.totalDecodeBatchSize()
                          << ",\"total_context_batch_size\":" << stream_groups.totalContextBatchSize()
                          << ",\"total_sampler_batch_size_in\":" << stream_groups.totalSamplerBatchSizeIn()
                          << ",\"total_sampler_batch_size_out\":" << stream_groups.totalSamplerBatchSizeOut()
                          << ",\"max_seq_len\":" << stream_groups.maxSeqLen()
                          << ",\"trace_id\":\"" << jsonEscape(stream->traceId()) << "\""
                          << ",\"stream_id\":" << stream->streamId()
                          << ",\"seq_len\":" << stream->seqLength()
                          << ",\"input_len\":" << stream->inputLength()
                          << ",\"output_len\":" << stream->outputTokenLen()
                          << ",\"iter_count\":" << stream->iterCount()
                          << ",\"seq_size_per_block\":" << seq_size_per_block
                          << ",\"read_logical_block\":" << read_logical_block
                          << ",\"write_logical_block\":" << write_logical_block
                          << ",\"current_batch_size\":" << cur_bs
                          << ",\"next_batch_size\":" << next_bs
                          << ",\"reuse_len\":" << stream->reuseLength()
                          << ",\"initial_reuse_len\":" << stream->initialReuseLength()
                          << ",\"local_reuse_len\":" << stream->localReuseLength()
                          << ",\"memory_reuse_len\":" << stream->memoryReuseLength()
                          << ",\"cur_blocks_num\":" << stream->curBlocksNum()
                          << ",\"new_tokens\":";
                appendIntVector(watch_row, new_tokens);
                if (success_ptr != nullptr && batch_idx_in < success_cpu.numel()) {
                    watch_row << ",\"success\":" << (success_ptr[batch_idx_in] ? "true" : "false");
                }
                watch_row << ",\"token_tail\":";
                appendIntVector(watch_row, state.token_tail);
                watch_row << ",\"history\":";
                appendWatchHistory(watch_row, state.history);
                watch_row << ",\"kv_groups\":";
                appendBlockGroups(watch_row, stream, cfg.max_blocks_per_group, read_logical_block, write_logical_block);
                watch_row << "}";
                std::lock_guard<std::mutex> lock(outputMutex());
                auto&                       os = badWatchOutputStream();
                if (os.good()) {
                    os << watch_row.str() << "\n";
                }
            }
            batch_idx_in += cur_bs;
            batch_idx_out += next_bs;
        }
    }

    if (!cfg.enabled || !has_match) {
        return;
    }

    std::ostringstream row;
    row << "{\"ts_us\":" << autil::TimeUtility::currentTimeInMicroSeconds() << ",\"pid\":" << ::getpid()
        << ",\"rank\":\"" << jsonEscape(rankString()) << "\",\"event\":\"decode_dispatch_batch\""
        << ",\"total_decode_batch_size\":" << stream_groups.totalDecodeBatchSize()
        << ",\"total_context_batch_size\":" << stream_groups.totalContextBatchSize()
        << ",\"total_sampler_batch_size_in\":" << stream_groups.totalSamplerBatchSizeIn()
        << ",\"total_sampler_batch_size_out\":" << stream_groups.totalSamplerBatchSizeOut()
        << ",\"max_seq_len\":" << stream_groups.maxSeqLen()
        << ",\"cur_blocks_num\":" << stream_groups.curBlocksNum() << ",\"streams\":[";

    int         batch_idx_in  = 0;
    int         batch_idx_out = 0;
    bool        wrote_stream  = false;
    for (const auto& stream : all_streams) {
        const bool matched = config().matches(stream->traceId());
        const auto cur_bs  = stream->currentBatchSize();
        const auto next_bs = stream->nextBatchSize();
        if (!matched && !config().capture_peers) {
            batch_idx_in += cur_bs;
            batch_idx_out += next_bs;
            continue;
        }
        if (wrote_stream) {
            row << ",";
        }
        wrote_stream = true;
        std::vector<int> new_tokens;
        new_tokens.reserve(next_bs);
        for (int i = 0; i < next_bs; ++i) {
            new_tokens.push_back(token_ptr[(batch_idx_out + i) * token_stride + token_stride - 1]);
        }
        const int seq_size_per_block =
            stream->seqSizePerBlock() > 0 ? stream->seqSizePerBlock() : std::max(1, stream->seqLength());
        const int read_logical_block =
            stream->seqLength() > 0 ? (stream->seqLength() - 1) / seq_size_per_block : -1;
        const int write_logical_block = stream->seqLength() / seq_size_per_block;
        row << "{\"matched\":" << (matched ? "true" : "false") << ",\"trace_id\":\""
            << jsonEscape(stream->traceId()) << "\",\"stream_id\":" << stream->streamId()
            << ",\"is_context\":" << (stream->isContextStream() ? "true" : "false")
            << ",\"seq_len\":" << stream->seqLength() << ",\"input_len\":" << stream->inputLength()
            << ",\"output_len\":" << stream->outputTokenLen() << ",\"iter_count\":" << stream->iterCount()
            << ",\"seq_size_per_block\":" << seq_size_per_block
            << ",\"read_logical_block\":" << read_logical_block
            << ",\"write_logical_block\":" << write_logical_block
            << ",\"current_batch_size\":" << cur_bs << ",\"next_batch_size\":" << next_bs
            << ",\"reuse_len\":" << stream->reuseLength()
            << ",\"initial_reuse_len\":" << stream->initialReuseLength()
            << ",\"local_reuse_len\":" << stream->localReuseLength()
            << ",\"memory_reuse_len\":" << stream->memoryReuseLength()
            << ",\"cur_blocks_num\":" << stream->curBlocksNum() << ",\"new_tokens\":";
        appendIntVector(row, new_tokens);
        if (success_ptr != nullptr && batch_idx_in < success_cpu.numel()) {
            row << ",\"success\":" << (success_ptr[batch_idx_in] ? "true" : "false");
        }
        row << ",\"kv_groups\":";
        appendBlockGroups(row, stream, config().max_blocks_per_group, read_logical_block, write_logical_block);
        row << "}";
        batch_idx_in += cur_bs;
        batch_idx_out += next_bs;
    }
    row << "]}";

    std::lock_guard<std::mutex> lock(outputMutex());
    auto&                       os = outputStream();
    if (os.good()) {
        os << row.str() << "\n";
    }
}

}  // namespace rtp_llm
