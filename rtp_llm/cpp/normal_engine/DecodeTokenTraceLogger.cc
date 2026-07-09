#include "rtp_llm/cpp/normal_engine/DecodeTokenTraceLogger.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

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

void appendBlockGroups(std::ostream& os, const GenerateStreamPtr& stream, int max_blocks_per_group) {
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
            os << "],\"total\":" << blocks.size() << "}";
        }
    }
    os << "]";
}

}  // namespace

DecodeTokenTraceConfig DecodeTokenTraceConfig::fromValues(bool               enabled,
                                                          const std::string& filter_csv,
                                                          const std::string& output_path,
                                                          bool               capture_peers,
                                                          int                max_blocks_per_group) {
    DecodeTokenTraceConfig cfg;
    cfg.enabled              = enabled;
    cfg.filters              = splitCsv(filter_csv);
    cfg.output_path          = output_path;
    cfg.capture_peers        = capture_peers;
    cfg.max_blocks_per_group = std::max(0, max_blocks_per_group);
    return cfg;
}

DecodeTokenTraceConfig DecodeTokenTraceConfig::fromEnv() {
    return fromValues(getenvBool("RTPLLM_DECODE_TOKEN_TRACE", false),
                      getenvString("RTPLLM_DECODE_TOKEN_TRACE_FILTER"),
                      defaultOutputPath(),
                      getenvBool("RTPLLM_DECODE_TOKEN_TRACE_CAPTURE_PEERS", true),
                      getenvInt("RTPLLM_DECODE_TOKEN_TRACE_MAX_BLOCKS_PER_GROUP", 16));
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
    if (!enabled() || !token_ids_cpu.defined() || token_ids_cpu.numel() == 0) {
        return;
    }

    const auto all_streams = stream_groups.allStreams();
    bool       has_match   = false;
    for (const auto& stream : all_streams) {
        if (config().matches(stream->traceId())) {
            has_match = true;
            break;
        }
    }
    if (!has_match) {
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

    const auto token_stride = token_ids_cpu.size(1);
    const auto* token_ptr   = token_ids_cpu.data_ptr<int32_t>();
    const auto* success_ptr = success_cpu.defined() ? success_cpu.data_ptr<bool>() : nullptr;
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
        row << "{\"matched\":" << (matched ? "true" : "false") << ",\"trace_id\":\""
            << jsonEscape(stream->traceId()) << "\",\"stream_id\":" << stream->streamId()
            << ",\"is_context\":" << (stream->isContextStream() ? "true" : "false")
            << ",\"seq_len\":" << stream->seqLength() << ",\"input_len\":" << stream->inputLength()
            << ",\"output_len\":" << stream->outputTokenLen() << ",\"iter_count\":" << stream->iterCount()
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
        appendBlockGroups(row, stream, config().max_blocks_per_group);
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
