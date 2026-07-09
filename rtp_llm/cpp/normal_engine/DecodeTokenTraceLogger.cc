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

DecodeBlockTraceInfo computeBlockTrace(int seq_len, int seq_size_per_block) {
    DecodeBlockTraceInfo info;
    if (seq_size_per_block <= 0) {
        return info;
    }
    // logDispatchBatch runs after sampling and before stream->update(). For the
    // token being dispatched now, Qwen3Next decode kernels saw sequence_lengths_plus_1=seq_len.
    info.model_read_logical_block  = seq_len >= 2 ? (seq_len - 2) / seq_size_per_block : -1;
    info.model_write_logical_block = seq_len >= 1 ? (seq_len - 1) / seq_size_per_block : -1;
    // These are the blocks that the next decode forward will read/write after this token is appended.
    info.next_read_logical_block  = seq_len >= 1 ? (seq_len - 1) / seq_size_per_block : -1;
    info.next_write_logical_block = seq_len >= 1 ? seq_len / seq_size_per_block : -1;
    return info;
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
    int              model_read_logical_block = -1;
    int              model_write_logical_block = -1;
    int              next_read_logical_block  = -1;
    int              next_write_logical_block = -1;
    int              cf_total                 = 0;
    std::vector<int> new_tokens;
    std::vector<int> model_read_blocks;
    std::vector<int> model_write_blocks;
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

DecodeRepeatedSuffixInfo findRepeatedSuffix(const std::vector<int>& values, int max_pattern_size, int min_repeats) {
    DecodeRepeatedSuffixInfo info;
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
           << ",\"model_read_logical_block\":" << step.model_read_logical_block
           << ",\"model_write_logical_block\":" << step.model_write_logical_block
           << ",\"next_read_logical_block\":" << step.next_read_logical_block
           << ",\"next_write_logical_block\":" << step.next_write_logical_block
           << ",\"cf_total\":" << step.cf_total
           << ",\"new_tokens\":";
        appendIntVector(os, step.new_tokens);
        os << ",\"model_read_blocks\":";
        appendIntVector(os, step.model_read_blocks);
        os << ",\"model_write_blocks\":";
        appendIntVector(os, step.model_write_blocks);
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

DecodeRepeatedSuffixInfo DecodeTokenTraceLogger::debugFindRepeatedSuffixForTest(const std::vector<int>& values,
                                                                                 int                    max_pattern_size,
                                                                                 int                    min_repeats) {
    return findRepeatedSuffix(values, max_pattern_size, min_repeats);
}

DecodeBlockTraceInfo DecodeTokenTraceLogger::debugComputeBlockTraceForTest(int seq_len, int seq_size_per_block) {
    return computeBlockTrace(seq_len, seq_size_per_block);
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
            const auto repeated_suffix = findRepeatedSuffix(state.token_tail, 8, 8);
            const int seq_size_per_block =
                stream->seqSizePerBlock() > 0 ? stream->seqSizePerBlock() : std::max(1, stream->seqLength());
            const auto block_trace = computeBlockTrace(stream->seqLength(), seq_size_per_block);
            const int  read_logical_block  = block_trace.next_read_logical_block;
            const int  write_logical_block = block_trace.next_write_logical_block;
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
                step.model_read_logical_block = block_trace.model_read_logical_block;
                step.model_write_logical_block = block_trace.model_write_logical_block;
                step.next_read_logical_block  = block_trace.next_read_logical_block;
                step.next_write_logical_block = block_trace.next_write_logical_block;
                step.cf_total                 = state.cf_total;
                step.new_tokens               = new_tokens;
                step.model_read_blocks        = collectGroupBlocksAt(stream, block_trace.model_read_logical_block);
                step.model_write_blocks       = collectGroupBlocksAt(stream, block_trace.model_write_logical_block);
                step.read_blocks              = collectGroupBlocksAt(stream, read_logical_block);
                step.write_blocks             = collectGroupBlocksAt(stream, write_logical_block);
                state.history.push_back(std::move(step));
                while (static_cast<int>(state.history.size()) > cfg.bad_watch_history_size) {
                    state.history.pop_front();
                }
            }
            const bool cf_tail_repeat = cf_count >= cfg.bad_watch_min_cf;
            if (!state.triggered && (cf_tail_repeat || repeated_suffix.matched)) {
                state.triggered = true;
                std::ostringstream watch_row;
                watch_row << "{\"ts_us\":" << autil::TimeUtility::currentTimeInMicroSeconds()
                          << ",\"pid\":" << ::getpid() << ",\"rank\":\"" << jsonEscape(rankString())
                          << "\",\"event\":\"decode_bad_watch_trigger\""
                          << ",\"reason\":\""
                          << (cf_tail_repeat ? "cf_tail_repeat" : "repeated_suffix")
                          << "\",\"cf_count\":" << cf_count
                          << ",\"cf_total\":" << state.cf_total
                          << ",\"repeat_pattern_size\":" << repeated_suffix.pattern_size
                          << ",\"repeat_count\":" << repeated_suffix.repeat_count
                          << ",\"repeat_pattern\":";
                appendIntVector(watch_row, repeated_suffix.pattern);
                watch_row << ",\"total_decode_batch_size\":" << stream_groups.totalDecodeBatchSize()
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
                          << ",\"model_read_logical_block\":" << block_trace.model_read_logical_block
                          << ",\"model_write_logical_block\":" << block_trace.model_write_logical_block
                          << ",\"next_read_logical_block\":" << block_trace.next_read_logical_block
                          << ",\"next_write_logical_block\":" << block_trace.next_write_logical_block
                          << ",\"current_batch_size\":" << cur_bs
                          << ",\"next_batch_size\":" << next_bs
                          << ",\"batch_idx_in\":" << batch_idx_in
                          << ",\"batch_idx_out\":" << batch_idx_out
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
                watch_row << ",\"peers\":[";
                int  peer_batch_idx_in  = 0;
                int  peer_batch_idx_out = 0;
                bool wrote_peer         = false;
                for (const auto& peer : all_streams) {
                    const auto peer_cur_bs  = peer->currentBatchSize();
                    const auto peer_next_bs = peer->nextBatchSize();
                    if (wrote_peer) {
                        watch_row << ",";
                    }
                    wrote_peer = true;
                    std::vector<int> peer_new_tokens;
                    peer_new_tokens.reserve(peer_next_bs);
                    for (int i = 0; i < peer_next_bs; ++i) {
                        const int token_row = peer_batch_idx_out + i;
                        if (token_row >= 0 && token_row < token_ids_cpu.size(0)) {
                            peer_new_tokens.push_back(token_ptr[token_row * token_stride + token_stride - 1]);
                        }
                    }
                    const int peer_seq_size_per_block =
                        peer->seqSizePerBlock() > 0 ? peer->seqSizePerBlock() : std::max(1, peer->seqLength());
                    const auto peer_block_trace = computeBlockTrace(peer->seqLength(), peer_seq_size_per_block);
                    const int  peer_read_logical_block  = peer_block_trace.next_read_logical_block;
                    const int  peer_write_logical_block = peer_block_trace.next_write_logical_block;
                    watch_row << "{\"target\":" << (peer.get() == stream.get() ? "true" : "false")
                              << ",\"trace_id\":\"" << jsonEscape(peer->traceId()) << "\""
                              << ",\"stream_id\":" << peer->streamId()
                              << ",\"batch_idx_in\":" << peer_batch_idx_in
                              << ",\"batch_idx_out\":" << peer_batch_idx_out
                              << ",\"current_batch_size\":" << peer_cur_bs
                              << ",\"next_batch_size\":" << peer_next_bs
                              << ",\"seq_len\":" << peer->seqLength()
                              << ",\"input_len\":" << peer->inputLength()
                              << ",\"output_len\":" << peer->outputTokenLen()
                              << ",\"iter_count\":" << peer->iterCount()
                              << ",\"seq_size_per_block\":" << peer_seq_size_per_block
                              << ",\"read_logical_block\":" << peer_read_logical_block
                              << ",\"write_logical_block\":" << peer_write_logical_block
                              << ",\"model_read_logical_block\":" << peer_block_trace.model_read_logical_block
                              << ",\"model_write_logical_block\":" << peer_block_trace.model_write_logical_block
                              << ",\"next_read_logical_block\":" << peer_block_trace.next_read_logical_block
                              << ",\"next_write_logical_block\":" << peer_block_trace.next_write_logical_block
                              << ",\"reuse_len\":" << peer->reuseLength()
                              << ",\"initial_reuse_len\":" << peer->initialReuseLength()
                              << ",\"local_reuse_len\":" << peer->localReuseLength()
                              << ",\"memory_reuse_len\":" << peer->memoryReuseLength()
                              << ",\"cur_blocks_num\":" << peer->curBlocksNum()
                              << ",\"new_tokens\":";
                    appendIntVector(watch_row, peer_new_tokens);
                    if (success_ptr != nullptr && peer_batch_idx_in < success_cpu.numel()) {
                        watch_row << ",\"success\":" << (success_ptr[peer_batch_idx_in] ? "true" : "false");
                    }
                    watch_row << ",\"kv_groups\":";
                    appendBlockGroups(
                        watch_row, peer, cfg.max_blocks_per_group, peer_read_logical_block, peer_write_logical_block);
                    watch_row << "}";
                    peer_batch_idx_in += peer_cur_bs;
                    peer_batch_idx_out += peer_next_bs;
                }
                watch_row << "]";
                watch_row << "}";
                std::lock_guard<std::mutex> lock(outputMutex());
                auto&                       os = badWatchOutputStream();
                if (os.good()) {
                    os << watch_row.str() << "\n";
                    os.flush();
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
        const auto block_trace = computeBlockTrace(stream->seqLength(), seq_size_per_block);
        const int  read_logical_block  = block_trace.next_read_logical_block;
        const int  write_logical_block = block_trace.next_write_logical_block;
        row << "{\"matched\":" << (matched ? "true" : "false") << ",\"trace_id\":\""
            << jsonEscape(stream->traceId()) << "\",\"stream_id\":" << stream->streamId()
            << ",\"is_context\":" << (stream->isContextStream() ? "true" : "false")
            << ",\"seq_len\":" << stream->seqLength() << ",\"input_len\":" << stream->inputLength()
            << ",\"output_len\":" << stream->outputTokenLen() << ",\"iter_count\":" << stream->iterCount()
            << ",\"seq_size_per_block\":" << seq_size_per_block
            << ",\"read_logical_block\":" << read_logical_block
            << ",\"write_logical_block\":" << write_logical_block
            << ",\"model_read_logical_block\":" << block_trace.model_read_logical_block
            << ",\"model_write_logical_block\":" << block_trace.model_write_logical_block
            << ",\"next_read_logical_block\":" << block_trace.next_read_logical_block
            << ",\"next_write_logical_block\":" << block_trace.next_write_logical_block
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
        os.flush();
    }
}

}  // namespace rtp_llm
