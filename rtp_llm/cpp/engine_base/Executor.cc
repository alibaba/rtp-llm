#include "rtp_llm/cpp/engine_base/Executor.h"

namespace rtp_llm {

std::shared_ptr<RecordedBatch> Executor::recordBatchInputs(const GptModelInputs& inputs,
                                                           const StreamGroups&   groups) const {
    auto& recorder = ExecutionRecorder::instance();
    if (!recorder.enabled() || !inputs.record_execution_id)
        return nullptr;
    if (!inputs.record_lengths)
        throw std::runtime_error("batch CPU length snapshot missing");
    const auto& lengths = *inputs.record_lengths;
    using Json          = ExecutionRecorder::Json;
    std::vector<Json> rows;
    auto              append = [&](const auto& streams, const char* phase) {
        for (const auto& stream : streams) {
            const auto request = stream->recordedRequest();
            for (int i = 0; i < stream->currentBatchSize(); ++i) {
                Json row{{"phase", std::string(phase)},
                                      {"input_len", stream->inputLength()},
                                      {"reuse_len", stream->initialReuseLength()}};
                if (request)
                    row["request_id"] = "r" + std::to_string(request->id());
                if (i != 0)
                    row["sequence_id"] = i;
                if (stream->isFakeStream())
                    row["is_fake"] = true;
                rows.push_back(std::move(row));
            }
        }
    };
    append(groups.decodeStreams(), "decode");
    append(groups.contextStreams(), "prefill");
    if (rows.size() != lengths.q_tokens.size() || rows.size() != lengths.kv_tokens_before.size()
        || rows.size() != static_cast<size_t>(inputs.input_lengths.numel())
        || groups.totalDecodeBatchSize() != static_cast<size_t>(inputs.sequence_lengths.numel())
        || groups.totalContextBatchSize() != static_cast<size_t>(inputs.prefix_lengths.numel()))
        throw std::runtime_error("batch snapshot slot/length count mismatch");
    // JSON numbers lose int64 precision in JavaScript and other double-based consumers.
    Json record{
        {"schema_version", 1}, {"owner_id", recorder.owner()}, {"exec_id", std::to_string(inputs.record_execution_id)}};
    for (size_t i = 0; i < rows.size(); ++i) {
        // Preserve every batch slot; omitted lengths mean unknown, not zero.
        if (lengths.q_tokens[i] > 0) {
            rows[i]["q_len"] = lengths.q_tokens[i];
        }
        if (lengths.kv_tokens_before[i] >= 0) {
            rows[i]["kv_len"] = lengths.kv_tokens_before[i];
        }
    }
    return std::make_shared<RecordedBatch>(recorder, std::move(record), std::move(rows));
}

bool Executor::shouldRecordBatch(bool owner, bool has_request_stream, size_t total_model_batch_size) {
    return owner && has_request_stream && total_model_batch_size > 0;
}

bool Executor::sampleDecodeStep() {
    // Deterministic ratio sampling: accumulate the rate per decode step and
    // record whenever the credit crosses 1.0 (Bresenham-style). rate=1.0
    // records every step (legacy behavior), 0.0 records none, and any rate
    // keeps an exact recorded fraction over the long run with even spacing.
    record_decode_sample_credit_ += record_decode_sample_rate_;
    if (record_decode_sample_credit_ < 1.0)
        return false;
    record_decode_sample_credit_ -= 1.0;
    return true;
}

}  // namespace rtp_llm
