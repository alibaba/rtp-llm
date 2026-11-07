#pragma once

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/models/ModelTypes.h"

namespace rtp_llm::smoke_logits_capture {

// Explicit accuracy diagnostics only. Ordinary requests never copy tensors or
// wait here. KVCache transfer, publication and reference ownership are untouched.
inline bool enabled() {
    static const bool value = [] {
        const auto flag = std::getenv("KIMI_K3_SMOKE_EVIDENCE");
        const auto path = std::getenv("KIMI_K3_SMOKE_LOGITS_DIR");
        return flag && std::string(flag) == "1" && path && path[0];
    }();
    return value;
}

inline std::vector<int64_t> hostIntegers(const torch::Tensor& value) {
    auto cpu = value.to(torch::kCPU).to(torch::kInt64).contiguous().view({-1});
    return {cpu.data_ptr<int64_t>(), cpu.data_ptr<int64_t>() + cpu.numel()};
}

inline void writeExclusive(const std::filesystem::path& path, const void* data, size_t bytes) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
    if (fd < 0) {
        throw std::runtime_error("cannot create smoke logits artifact: " + path.string());
    }
    const auto* cursor = static_cast<const char*>(data);
    while (bytes) {
        const auto count = ::write(fd, cursor, bytes);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            ::close(fd);
            throw std::runtime_error("cannot write smoke logits artifact: " + path.string());
        }
        cursor += count;
        bytes -= count;
    }
    if (::close(fd) != 0) {
        throw std::runtime_error("cannot close smoke logits artifact: " + path.string());
    }
}

template<typename T>
inline std::string jsonIntegers(const std::vector<T>& values) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        out << (i ? "," : "") << values[i];
    }
    out << ']';
    return out.str();
}

inline std::filesystem::path captureRoot() {
    const auto env = std::getenv("KIMI_K3_SMOKE_LOGITS_DIR");
    if (!env || !env[0]) {
        throw std::runtime_error("smoke logits directory is missing");
    }
    const auto path = std::filesystem::canonical(env);
    if (!path.is_absolute() || !std::filesystem::is_directory(path)) {
        throw std::runtime_error("smoke logits directory must already exist");
    }
    return path;
}

inline std::string sourceSha() {
    const auto env = std::getenv("KIMI_K3_SMOKE_SOURCE_SHA");
    const std::string sha = env ? env : "";
    if (sha.size() != 40 || sha.find_first_not_of("0123456789abcdef") != std::string::npos) {
        throw std::runtime_error("smoke logits capture requires a pinned source SHA");
    }
    return sha;
}

// One frame records only rows whose preceding draft tokens were accepted.
// Its positions are absolute generation offsets, independent of MTP grouping.
inline void writeFrame(const std::filesystem::path& root,
                       const std::string& sha,
                       int64_t request_id,
                       const char* phase,
                       int64_t begin,
                       const std::vector<int>& prompt,
                       const std::vector<int64_t>& row_inputs,
                       const std::vector<int64_t>& accepted,
                       const torch::Tensor& logits,
                       int64_t vocab_size) {
    if (prompt.empty() || begin < 0 || accepted.empty() || accepted.size() != row_inputs.size()
        || logits.dim() != 2 || logits.size(0) < static_cast<int64_t>(accepted.size())
        || vocab_size <= 0 || logits.size(1) < vocab_size) {
        throw std::runtime_error("invalid smoke logits frame geometry");
    }
    if (begin > 3) {
        return;
    }
    const std::string input_file = std::to_string(request_id) + ".input.i32";
    const auto input_path = root / input_file;
    if (!std::filesystem::exists(input_path)) {
        static_assert(sizeof(int) == 4, "smoke inputs use int32");
        writeExclusive(input_path, prompt.data(), prompt.size() * sizeof(int));
    } else {
        std::ifstream input(input_path, std::ios::binary);
        std::vector<int> previous(prompt.size());
        input.read(reinterpret_cast<char*>(previous.data()), prompt.size() * sizeof(int));
        if (!input || previous != prompt || input.peek() != std::char_traits<char>::eof()) {
            throw std::runtime_error("request identity reused with different smoke input tokens");
        }
    }
    std::vector<int64_t> indices, positions;
    for (size_t i = 0; i < accepted.size(); ++i) {
        const auto position = begin + i;
        if (position == 0 || position == 1 || position == 3) {
            indices.push_back(i);
            positions.push_back(position);
        }
        if (i && row_inputs[i] != accepted[i-1]) {
            throw std::runtime_error("smoke logits row follows an unaccepted draft token");
        }
    }
    auto selected = logits.narrow(1, 0, vocab_size).narrow(0, 0, 0).to(torch::kCPU).to(torch::kFloat32);
    if (!indices.empty()) {
        const auto index = torch::tensor(indices, torch::TensorOptions().dtype(torch::kLong)).to(logits.device());
        selected = logits.index_select(0, index).narrow(1, 0, vocab_size)
                       .to(torch::kCPU).to(torch::kFloat32).contiguous();
    }
    const std::string name = std::to_string(request_id) + "." + phase + "." + std::to_string(begin);
    const std::string data_file = name + ".f32";
    writeExclusive(root / data_file, selected.data_ptr<float>(), selected.numel() * sizeof(float));
    std::ostringstream metadata;
    metadata << "{\"schema\":1,\"source_commit\":\"" << sha << "\",\"request_id\":" << request_id
             << ",\"phase\":\"" << phase << "\",\"begin_output_position\":" << begin
             << ",\"prompt_tokens\":" << prompt.size() << ",\"input_file\":\"" << input_file
             << "\",\"row_input_tokens\":" << jsonIntegers(row_inputs)
             << ",\"accepted_tokens\":" << jsonIntegers(accepted)
             << ",\"positions\":" << jsonIntegers(positions)
             << ",\"vocab_size\":" << vocab_size << ",\"dtype\":\"float32_le\",\"data_file\":\""
             << data_file << "\",\"time_ns\":"
             << std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count() << "}\n";
    const auto text = metadata.str();
    writeExclusive(root / (name + ".json"), text.data(), text.size());
}

inline bool isPrecisionRequest(const GenerateStreamPtr& stream) {
    return stream->generateConfig()->trace_id.rfind("k31-last-n-precision:", 0) == 0;
}

template<typename Streams>
inline bool requested(const Streams& streams) {
    return enabled() && std::any_of(streams.begin(), streams.end(), [](const auto& stream) {
        return isPrecisionRequest(stream);
    });
}

inline std::vector<int> promptTokens(const GenerateStreamPtr& stream) {
    auto tokens = stream->completeTokenIdsVec();
    if (tokens.size() < static_cast<size_t>(stream->inputLength())) {
        throw std::runtime_error("smoke stream does not contain its original prompt");
    }
    tokens.resize(stream->inputLength());
    return tokens;
}

inline bool prefillRequested(const StreamGroups& groups) {
    const auto& streams = groups.contextStreams();
    return enabled() && std::any_of(streams.begin(), streams.end(), [](const auto& stream) {
        return isPrecisionRequest(stream) && !stream->isMiddleChunk();
    });
}

struct PrefillBatch {
    // The sampler may overwrite the target tensor with its probabilities.
    explicit PrefillBatch(const torch::Tensor& target_logits): logits(target_logits.clone()) {}

    static std::vector<int64_t> sampledTokens(const torch::Tensor& token_history) {
        if (token_history.dim() != 2 || token_history.size(1) == 0) {
            throw std::runtime_error("invalid smoke Prefill sampler history");
        }
        return hostIntegers(token_history.select(1, token_history.size(1) - 1));
    }

    void write(const std::filesystem::path& root, const std::string& sha, int64_t request_id,
               const std::vector<int>& prompt, int64_t logit_row, int64_t sampled, int64_t vocab_size) const {
        if (prompt.empty() || logit_row < 0 || logit_row >= logits.size(0)) {
            throw std::runtime_error("invalid smoke Prefill logits row");
        }
        writeFrame(root, sha, request_id, "prefill", 0, prompt,
                   {prompt.back()}, {sampled}, logits.narrow(0, logit_row, 1), vocab_size);
    }

    const torch::Tensor logits;
};

inline void prefill(const StreamGroups& groups, const PrefillBatch& batch,
                    const torch::Tensor& token_history, int64_t vocab_size) {
    if (!groups.decodeStreams().empty()) {
        throw std::runtime_error("smoke Prefill logits require a context-only batch");
    }
    const auto root = captureRoot();
    const auto sha = sourceSha();
    const auto sampled = PrefillBatch::sampledTokens(token_history);
    int64_t logit_row = 0, output_row = 0;
    for (const auto& stream : groups.contextStreams()) {
        if (isPrecisionRequest(stream) && !stream->isMiddleChunk()) {
            if (stream->maxBatchSize() != 1 || output_row >= static_cast<int64_t>(sampled.size())) {
                throw std::runtime_error("smoke Prefill logits require single-row requests");
            }
            batch.write(root, sha, stream->streamId(), promptTokens(stream), logit_row,
                        sampled[output_row], vocab_size);
        }
        logit_row += stream->currentBatchSize();
        output_row += stream->nextBatchSize();
    }
}

struct DecodeBatch {
    DecodeBatch(const GptModelInputs& inputs, const torch::Tensor& accept_len, const torch::Tensor& accept_tokens):
        prefixes(hostIntegers(inputs.prefix_lengths)),
        tokens(hostIntegers(inputs.combo_tokens)),
        lengths(hostIntegers(accept_len)),
        accepted(hostIntegers(accept_tokens)) {}

    void write(const std::filesystem::path& root, const std::string& sha, int64_t request_id,
               const std::vector<int>& prompt, const torch::Tensor& logits,
               int64_t row, int64_t width, int64_t vocab_size) const {
        const auto start = row * width;
        if (row < 0 || width <= 0 || row >= static_cast<int64_t>(lengths.size())
            || row >= static_cast<int64_t>(prefixes.size())
            || start + width > static_cast<int64_t>(tokens.size())
            || start + width > static_cast<int64_t>(accepted.size())
            || start + width > logits.size(0) || lengths[row] <= 0 || lengths[row] > width) {
            throw std::runtime_error("invalid single-row MTP smoke logits geometry");
        }
        // Target verify appends contiguous tokens to the committed prefix.
        // NoPE models do not provide combo_position_ids.
        const int64_t begin = prefixes[row] + 1 - static_cast<int64_t>(prompt.size());
        const auto count = lengths[row];
        writeFrame(root, sha, request_id, "decode", begin, prompt,
                   {tokens.begin()+start, tokens.begin()+start+count},
                   {accepted.begin()+start, accepted.begin()+start+count},
                   logits.narrow(0, start, count), vocab_size);
    }

    const std::vector<int64_t> prefixes, tokens, lengths, accepted;
};

inline void decode(const StreamGroups& groups, const GptModelInputs& inputs,
                   const torch::Tensor& logits, const torch::Tensor& accept_len,
                   const torch::Tensor& accept_tokens, int64_t width, int64_t vocab_size) {
    const auto& streams = groups.decodeStreams();
    if (!requested(streams)) {
        return;
    }
    const DecodeBatch batch(inputs, accept_len, accept_tokens);
    const auto root = captureRoot();
    const auto sha = sourceSha();
    int64_t row = 0;
    for (const auto& stream : streams) {
        if (isPrecisionRequest(stream)) {
            if (stream->maxBatchSize() != 1) {
                throw std::runtime_error("smoke Decode logits require single-row requests");
            }
            batch.write(root, sha, stream->streamId(), promptTokens(stream), logits, row, width, vocab_size);
        }
        row += stream->currentBatchSize();
    }
}

}  // namespace rtp_llm::smoke_logits_capture
