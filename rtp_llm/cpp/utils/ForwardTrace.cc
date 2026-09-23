#include "rtp_llm/cpp/utils/ForwardTrace.h"

#include <exception>
#include <algorithm>
#include <stdexcept>
#if USING_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#endif

namespace rtp_llm {
namespace {
thread_local ForwardTraceSession* current_session = nullptr;
}

ForwardTraceSession* activeForwardTrace() { return current_session; }
void setActiveForwardTrace(ForwardTraceSession* session) { current_session = session; }

ForwardTraceSession::ForwardTraceSession(int64_t value_capacity, int64_t record_capacity):
    record_capacity_(record_capacity) {
    records.reserve(record_capacity);
    sources_.reserve(record_capacity);
#if USING_CUDA
    if (value_capacity > 0) {
        arena_ = torch::empty({value_capacity}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    }
    producer_streams_.reserve(16);
    ready_events_.reserve(16);
#endif
}

ForwardTraceRecord* ForwardTraceSession::append() {
    if (sealed || static_cast<int64_t>(records.size()) >= record_capacity_) {
        ++dropped;
        return nullptr;
    }
    auto record = std::make_unique<ForwardTraceRecord>();
    record->id = records.size() + 1;
    record->parent_id = parent_id;
    auto* ptr = record.get();
    records.push_back(std::move(record));
    return ptr;
}

void ForwardTraceSession::snapshot(ForwardTraceRecord& record, const std::string& key,
                                  const torch::Tensor& values) {
    auto& out = record.arrays[key];
    out = ForwardTraceValues{};
    if (!values.defined()) {
        return;
    }
    out.count = values.numel();
    if (values.scalar_type() != torch::kInt32 || !values.is_contiguous()) {
        out.valid = false;
        return;
    }
    if (out.count == 0) {
        return;
    }
    if (values.device().is_cpu()) {
        const auto* data = values.data_ptr<int32_t>();
        out.host.assign(data, data + out.count);
        return;
    }
#if USING_CUDA
    if (values.is_cuda() && arena_.defined() && values.device() == arena_.device()
        && used_ + out.count <= arena_.numel()) {
        RECORD_FUNCTION("RTP::forward_metadata.snapshot_d2d", {});
        // This is D2D on the producer's current stream, before its next write.
        // Both buffers already exist; no host read, stream wait or allocator call.
        const auto stream = c10::cuda::getCurrentCUDAStream(values.get_device());
        auto error = cudaMemcpyAsync(arena_.data_ptr<int32_t>() + used_, values.data_ptr<int32_t>(),
                                     out.count * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream.stream());
        if (error == cudaSuccess) {
            out.offset = used_;
            used_ += out.count;
            sources_.push_back(values);  // prevent source storage reuse before the copy completes
            const c10::Stream producer = stream;
            if (std::find(producer_streams_.begin(), producer_streams_.end(), producer) == producer_streams_.end()) {
                producer_streams_.push_back(producer);
            }
            return;
        }
    }
#endif
    out.valid = false;
    ++dropped;
}

void ForwardTraceSession::seal() {
    sealed = true;
#if USING_CUDA
    for (const auto& stream : producer_streams_) {
        auto event = std::make_shared<torch::Event>(stream.device_type());
        event->record(stream);
        ready_events_.push_back(std::move(event));
    }
#endif
}

void ForwardTraceSession::materialize() {
    if (!sealed) {
        throw std::logic_error("cannot export an active forward trace");
    }
#if USING_CUDA
    if (used_) {
        c10::cuda::CUDAGuard guard(arena_.device());
        for (const auto& event : ready_events_) event->synchronize();  // SAVE WORKER ONLY
        auto host = arena_.narrow(0, 0, used_).cpu();
        const auto* data = host.data_ptr<int32_t>();
        for (auto& record : records) {
            for (auto& item : record->arrays) {
                auto& values = item.second;
                if (values.valid && values.offset >= 0) {
                    values.host.assign(data + values.offset, data + values.offset + values.count);
                }
            }
        }
    }
#endif
    sources_.clear();
}

ForwardTraceScope::ForwardTraceScope(): session_(activeForwardTrace()), exceptions_(std::uncaught_exceptions()) {
    if (!session_) {
        return;
    }
    previous_parent_ = session_->parent_id;
    record_ = session_->append();
    if (!record_) {
        return;
    }
    session_->parent_id = record_->id;
    record_->strings["status"] = "incomplete";
    name_ = "RTP::model_forward(id=" + std::to_string(record_->id) + ")";
    scope_ = std::make_unique<at::RecordFunction>(at::RecordScope::FUNCTION);
    if (scope_->isActive()) {
        scope_->before(name_.c_str());
    }
}

ForwardTraceScope::~ForwardTraceScope() {
    if (record_) {
        record_->strings["status"] = std::uncaught_exceptions() > exceptions_ ? "error" : "ok";
        session_->parent_id = previous_parent_;
    }
}

void ForwardTraceScope::snapshot(const std::string& key, const torch::Tensor& values) {
    if (record_) {
        session_->snapshot(*record_, key, values);
    }
}

int64_t recordForwardTraceChunk(int64_t round, const std::vector<int64_t>& rows,
                                const std::vector<int64_t>& q, const std::vector<int64_t>& prefix,
                                int64_t physical_requests, int64_t physical_tokens) {
    auto* session = activeForwardTrace();
    if (!session) return 0;
    auto* record = session->append();
    if (!record) return 0;
    record->integers = {{"chunk_index", round}, {"physical_requests", physical_requests},
                        {"physical_tokens", physical_tokens}, {"logical_sequences", static_cast<int64_t>(q.size())}};
    record->strings = {{"phase", "prefill_target"}, {"kind", "chunk"}, {"status", "incomplete"}};
    record->arrays["original_batch_indices"].host = rows;
    record->arrays["q_lens"].host = q;
    record->arrays["prefix_lens"].host = prefix;
    return record->id;
}

void finishForwardTraceChunk(int64_t id) {
    auto* session = activeForwardTrace();
    if (session && id > 0 && id <= static_cast<int64_t>(session->records.size())) {
        session->records[id - 1]->strings["status"] = "ok";
    }
}
bool normalizeForwardTraceLengths(ForwardTraceRecord& record, std::vector<int64_t>& q, std::vector<int64_t>& prefix) {
    auto get = [&](const char* key) -> const ForwardTraceValues* {
        auto it = record.arrays.find(key);
        return it != record.arrays.end() && it->second.valid ? &it->second : nullptr;
    };
    if (record.strings["kind"] == "chunk") {
        const auto* queries = get("q_lens");
        const auto* prefixes = get("prefix_lens");
        if (!queries || !prefixes) return false;
        q = queries->host;
        prefix = prefixes->host;
    } else {
        const auto* inputs = get("input_lengths");
        const auto* sequences = get("sequence_lengths");
        const auto* prefixes = get("prefix_lengths");
        if (!inputs || !sequences || !prefixes) return false;
        const auto batch = inputs->host.size();
        const auto decode = sequences->host.size();
        if (decode > batch || (decode < batch && prefixes->host.size() != batch - decode)) return false;
        q.assign(batch, 1);
        prefix = sequences->host;
        if (decode < batch) prefix.insert(prefix.end(), prefixes->host.begin(), prefixes->host.end());
        for (size_t i = decode; i < batch; ++i) q[i] = inputs->host[i];
        const auto logical_it = record.integers.find("logical_sequences");
        if (logical_it == record.integers.end()) return false;
        const auto logical = logical_it->second;
        if (logical < 0 || static_cast<size_t>(logical) > batch) return false;
        q.resize(logical);
        prefix.resize(logical);
    }
    if (q.size() != prefix.size()) return false;
    for (size_t i = 0; i < q.size(); ++i) {
        if (q[i] <= 0 || prefix[i] < 0) return false;
    }
    return true;
}

}  // namespace rtp_llm
