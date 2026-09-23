#pragma once

#include <ATen/record_function.h>
#include <torch/torch.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

// A session is owned by the profiler thread, then moved to the save worker.
// No locks, D2H, CUDA allocation, events or waits are permitted in snapshot().
struct ForwardTraceValues {
    std::vector<int64_t> host;
    int64_t offset = -1;
    int64_t count = 0;
    bool valid = true;
};

struct ForwardTraceRecord {
    int64_t id = 0;
    int64_t parent_id = 0;
    std::map<std::string, int64_t> integers;
    std::map<std::string, std::string> strings;
    std::map<std::string, ForwardTraceValues> arrays;
};

class ForwardTraceSession {
public:
    // Called before enableProfiler(), outside model forward. Fixed capacities
    // deliberately fail closed instead of growing CUDA storage during inference.
    explicit ForwardTraceSession(int64_t value_capacity = 4 * 1024 * 1024,
                                 int64_t record_capacity = 65536);
    ForwardTraceRecord* append();
    void snapshot(ForwardTraceRecord& record, const std::string& key, const torch::Tensor& values);
    void seal();  // profiler thread, after the last forward
    void materialize();  // save worker ONLY; may wait and copy to CPU

    std::vector<std::unique_ptr<ForwardTraceRecord>> records;
    int64_t dropped = 0;
    int64_t parent_id = 0;
    bool sealed = false;

private:
    int64_t record_capacity_;
    int64_t used_ = 0;
    torch::Tensor arena_;
    std::vector<torch::Tensor> sources_;
    std::vector<c10::Stream> producer_streams_;
    std::vector<std::shared_ptr<torch::Event>> ready_events_;
};

ForwardTraceSession* activeForwardTrace();
void setActiveForwardTrace(ForwardTraceSession* session);

class ForwardTraceScope {
public:
    ForwardTraceScope();
    ~ForwardTraceScope();
    ForwardTraceScope(const ForwardTraceScope&) = delete;
    ForwardTraceScope& operator=(const ForwardTraceScope&) = delete;
    explicit operator bool() const { return record_ != nullptr; }
    ForwardTraceRecord& record() { return *record_; }
    void snapshot(const std::string& key, const torch::Tensor& values);

private:
    ForwardTraceSession* session_ = nullptr;
    ForwardTraceRecord* record_ = nullptr;
    int64_t previous_parent_ = 0;
    int exceptions_ = 0;
    std::string name_;
    std::unique_ptr<at::RecordFunction> scope_;
};

// CPU-only entry point used by the K3 chunk planner. The vectors are already
// planner outputs, not Tensor.tolist() calls on CUDA values.
int64_t recordForwardTraceChunk(int64_t round, const std::vector<int64_t>& rows,
                                const std::vector<int64_t>& q, const std::vector<int64_t>& prefix,
                                int64_t physical_requests, int64_t physical_tokens);
void finishForwardTraceChunk(int64_t id);

// CPU schema normalization, called only after deferred values are materialized.
bool normalizeForwardTraceLengths(ForwardTraceRecord& record, std::vector<int64_t>& q,
                                   std::vector<int64_t>& prefix);

// Runs after ProfilerResult::save, exclusively on the export path.
void enrichForwardTrace(const std::string& path, ForwardTraceSession& session);

}  // namespace rtp_llm
