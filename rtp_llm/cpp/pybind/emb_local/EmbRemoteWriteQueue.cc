#if USING_CUDA
#include "rtp_llm/cpp/pybind/emb_local/EmbRemoteWriteQueue.h"

#include <rapidjson/writer.h>

#include <atomic>
#include <cmath>
#include <map>
#include <unistd.h>

namespace rtp_llm::emb_local {
namespace {
constexpr int64_t kAlignment = 64;
constexpr size_t  kMaxHeader = 64 * 1024;

struct HeaderBuffer {
    using Ch = char;
    std::string value;

    void Put(char byte) {
        if (value.size() == kMaxHeader) {
            throw std::invalid_argument("EMB object metadata exceeds limit");
        }
        value.push_back(byte);
    }
    void Flush() {}
};
using Json = rapidjson::Writer<HeaderBuffer>;

int64_t align(int64_t bytes) {
    return (bytes + kAlignment - 1) / kAlignment * kAlignment;
}

const char* dtypeName(at::ScalarType type) {
    switch (type) {
        case at::kBool:
            return "torch.bool";
        case at::kByte:
            return "torch.uint8";
        case at::kChar:
            return "torch.int8";
        case at::kShort:
            return "torch.int16";
        case at::kInt:
            return "torch.int32";
        case at::kLong:
            return "torch.int64";
        case at::kHalf:
            return "torch.float16";
        case at::kBFloat16:
            return "torch.bfloat16";
        case at::kFloat:
            return "torch.float32";
        case at::kDouble:
            return "torch.float64";
        default:
            throw std::invalid_argument("unsupported EMB dtype");
    }
}

void string(Json& json, const std::string& value) {
    if (value.size() > kMaxHeader) {
        throw std::invalid_argument("oversized EMB object string");
    }
    json.String(value.data(), static_cast<rapidjson::SizeType>(value.size()));
}

void tree(Json& json, const Tree& node, size_t depth, size_t& nodes) {
    TORCH_CHECK(depth <= 32 && ++nodes <= 4096, "EMB object structure exceeds limits");
    using Kind = Tree::Kind;
    json.StartArray();
    switch (node.kind) {
        case Kind::Tensor:
            json.String("tensor");
            json.Int64(node.integer);
            break;
        case Kind::List:
        case Kind::Tuple:
            json.String(node.kind == Kind::List ? "list" : "tuple");
            json.StartArray();
            for (const auto& child : node.items) {
                tree(json, child, depth + 1, nodes);
            }
            json.EndArray();
            break;
        case Kind::Dict:
            json.String("dict");
            json.StartArray();
            for (size_t i = 0; i < node.items.size(); i += 2) {
                TORCH_CHECK(node.items[i].kind == Kind::String, "unsupported EMB object dictionary key");
                json.StartArray();
                string(json, node.items[i].text);
                tree(json, node.items[i + 1], depth + 1, nodes);
                json.EndArray();
            }
            json.EndArray();
            break;
        default:
            json.String("scalar");
            switch (node.kind) {
                case Kind::None:
                    json.Null();
                    break;
                case Kind::Bool:
                    json.Bool(node.integer != 0);
                    break;
                case Kind::Integer:
                    json.Int64(node.integer);
                    break;
                case Kind::Float:
                    TORCH_CHECK(std::isfinite(node.real), "nonfinite EMB object scalar");
                    json.Double(node.real);
                    break;
                case Kind::String:
                    string(json, node.text);
                    break;
                default:
                    throw std::invalid_argument("unsupported EMB object scalar");
            }
    }
    json.EndArray();
}

void beginField(Json& json, const char* field) {
    json.StartArray();
    json.String(field);
    json.StartArray();
}

void endField(Json& json) {
    json.EndArray();
    json.EndArray();
}

template<typename Emit>
void scalarField(Json& json, const char* field, Emit emit) {
    beginField(json, field);
    json.String("scalar");
    emit();
    endField(json);
}

std::string objectPrefix(const std::string&           key,
                         const Audit&                 audit,
                         bool                         request_scoped,
                         const Tree&                  result,
                         const std::vector<Slice>&    slices,
                         int64_t                      max_bytes,
                         std::vector<EmbTensorInput>& tensors,
                         int64_t&                     payload_bytes) {
    HeaderBuffer buffer;
    Json         json(buffer);
    json.StartObject();
    json.Key("tree");
    json.StartArray();
    json.String("dict");
    json.StartArray();
    scalarField(json, "version", [&] { json.Int(1); });
    json.StartArray();
    json.String("result");
    size_t nodes = audit.verdict_present ? 9 : 4;
    tree(json, result, 1, nodes);
    json.EndArray();
    if (audit.verdict_present) {
        beginField(json, "greennet");
        json.String("dict");
        json.StartArray();
        scalarField(json, "checked", [&] { json.Bool(audit.checked); });
        scalarField(json, "passed", [&] { json.Bool(audit.passed); });
        scalarField(json, "code", [&] { json.Int64(audit.code); });
        scalarField(json, "message", [&] { string(json, audit.message); });
        scalarField(json, "request_scoped", [&] { json.Bool(request_scoped); });
        json.EndArray();
        endField(json);
    } else {
        scalarField(json, "greennet", [&] { json.Null(); });
    }
    scalarField(json, "cache_key", [&] { string(json, key); });
    json.EndArray();
    json.EndArray();

    json.Key("tensors");
    json.StartArray();
    for (const auto& slice : slices) {
        const auto offset = align(payload_bytes);
        TORCH_CHECK(offset <= max_bytes && slice.nbytes <= max_bytes - offset, "EMB object exceeds byte limit");
        payload_bytes = offset + slice.nbytes;
        tensors.emplace_back(offset, slice.nbytes, at::Tensor{});
        json.StartObject();
        json.Key("shape");
        json.StartArray();
        for (int64_t size : slice.sizes) {
            json.Int64(size);
        }
        json.EndArray();
        json.Key("dtype");
        json.String(dtypeName(slice.type));
        json.Key("device");
        json.String(slice.original.is_cuda() ? "cuda" : "cpu");
        json.Key("offset");
        json.Int64(offset);
        json.Key("nbytes");
        json.Int64(slice.nbytes);
        json.EndObject();
    }
    json.EndArray();
    json.EndObject();

    const auto  size = buffer.value.size();
    std::string prefix("RTPMM002");
    for (unsigned i = 0; i < 8; ++i) {
        prefix.push_back(static_cast<char>((static_cast<uint64_t>(size) >> (i * 8)) & 0xff));
    }
    prefix.append(buffer.value);
    return prefix;
}
}  // namespace

RemoteWriteQueue::RemoteWriteQueue(std::shared_ptr<EmbRegisteredArena> arena,
                                   at::Tensor                          root,
                                   KvcmObjectNativeApiV2               api,
                                   int64_t                             max_object_bytes,
                                   int64_t                             max_pending,
                                   int64_t                             max_bytes,
                                   int64_t                             timeout_ms,
                                   int                                 workers,
                                   bool                                request_scoped):
    arena_(std::move(arena)),
    root_(std::move(root)),
    api_(api),
    max_object_bytes_(max_object_bytes),
    max_pending_(max_pending),
    max_bytes_(max_bytes),
    timeout_(timeout_ms),
    request_scoped_(request_scoped) {
    TORCH_CHECK((!arena_
                 || (root_.is_cuda() && root_.dim() == 1 && root_.scalar_type() == at::kByte && root_.is_contiguous()
                     && max_object_bytes <= root_.numel()))
                    && max_object_bytes > 0 && max_object_bytes <= 1024 * 1024 * 1024 && max_pending > 0
                    && max_bytes > 0 && timeout_ms > 0 && timeout_ms <= 600000 && workers > 0,
                "invalid native EMB eviction writer configuration");
    workers_.reserve(workers);
    try {
        for (int i = 0; i < workers; ++i) {
            workers_.emplace_back(&RemoteWriteQueue::run, this);
        }
    } catch (...) {
        close();
        throw;
    }
}

RemoteWriteQueue::~RemoteWriteQueue() {
    close();
}

std::shared_timed_mutex& RemoteWriteQueue::graphGate() {
    static auto* gate = new std::shared_timed_mutex;
    return *gate;
}

void RemoteWriteQueue::submit(const std::string&                     key,
                              const Audit&                           audit,
                              const std::shared_ptr<const Resident>& record,
                              std::optional<c10::cuda::CUDAStream>   stream) {
    enqueue(key, {}, audit, *record->tree, record->slices, [&](Job&) {
        // Keep the current placement for the detached snapshot. The header
        // already carries the original devices to use on restore.
        return copyOut(record, stream, false);
    });
}

void RemoteWriteQueue::submitEncoded(const std::string&                                             key,
                                     const std::string&                                             trace,
                                     const Audit&                                                   audit,
                                     const Tree&                                                    result,
                                     const std::vector<at::Tensor>&                                 inputs,
                                     const std::vector<std::pair<c10::DeviceIndex, c10::StreamId>>& producers) {
    std::vector<Slice> slices;
    slices.reserve(inputs.size());
    for (const auto& input : inputs) {
        TORCH_CHECK(input.layout() == at::kStrided && (input.is_cuda() || input.device().is_cpu()),
                    "unsupported EMB object tensor");
        slices.push_back({0,
                          static_cast<int64_t>(input.nbytes()),
                          input.scalar_type(),
                          input.sizes().vec(),
                          input.device(),
                          input.is_cuda() ? Tier::Gpu : Tier::Cpu});
    }
    enqueue(key, trace, audit, result, slices, [&](Job& job) {
        std::vector<at::Tensor>                         detached(inputs.size());
        std::map<c10::DeviceIndex, std::vector<size_t>> by_device;
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (inputs[i].is_cuda()) {
                by_device[inputs[i].get_device()].push_back(i);
            } else {
                detached[i] = inputs[i].detach().clone(at::MemoryFormat::Contiguous);
            }
        }
        for (const auto& [device, indices] : by_device) {
            const auto producer = std::find_if(
                producers.begin(), producers.end(), [device](const auto& stream) { return stream.first == device; });
            std::vector<at::Tensor> tensors;
            tensors.reserve(indices.size());
            for (size_t index : indices) {
                tensors.push_back(inputs[index]);
            }
            auto [snapshots, ready] = snapshotEmbCudaTensors(
                tensors, producer == producers.end() ? std::nullopt : std::optional(producer->second), true);
            job.events.push_back(std::move(ready));
            for (size_t i = 0; i < indices.size(); ++i) {
                detached[indices[i]] = std::move(snapshots[i]);
            }
        }
        return detached;
    });
}

template<typename Snapshot>
void RemoteWriteQueue::enqueue(const std::string&        key,
                               const std::string&        trace,
                               const Audit&              audit,
                               const Tree&               result,
                               const std::vector<Slice>& slices,
                               Snapshot                  snapshot) {
    if (key.empty() || (audit.verdict_present && !audit.passed)) {
        return;
    }
    if (arena_ && arena_->deviceFailed()) {
        std::lock_guard lock(mutex_);
        ++stats_.skipped;
        return;
    }
    Job job;
    job.deadline  = WriteClock::now() + timeout_;
    bool reserved = false;
    try {
        int64_t payload_bytes = 0;
        job.prefix =
            objectPrefix(key, audit, request_scoped_, result, slices, max_object_bytes_, job.tensors, payload_bytes);
        job.payload_offset = align(job.prefix.size());
        job.bytes          = job.payload_offset + payload_bytes;
        job.charge         = job.bytes * (arena_ ? 1 : 3);
        if (job.bytes > max_object_bytes_) {
            std::lock_guard lock(mutex_);
            ++stats_.skipped;
            return;
        }
        {
            std::lock_guard lock(mutex_);
            if (closed_ || (!arena_ && stats_.inflight_failure) || stats_.pending >= max_pending_ - external_tasks_
                || job.charge > max_bytes_ - stats_.bytes - external_bytes_ || !keys_.insert(key).second) {
                ++stats_.skipped;
                return;
            }
            reserved = true;
            ++stats_.pending;
            stats_.bytes += job.charge;
        }
        job.key       = key;
        job.trace     = trace;
        auto detached = snapshot(job);
        for (size_t i = 0; i < detached.size(); ++i) {
            std::get<0>(job.tensors[i]) += job.payload_offset;
            std::get<2>(job.tensors[i]) = std::move(detached[i]);
        }
        const auto charge = job.charge;
        {
            std::lock_guard lock(mutex_);
            if (!closed_ && WriteClock::now() < job.deadline) {
                jobs_.push_back(std::move(job));
                ready_.notify_one();
                return;
            }
        }
        finish(key, charge, {EmbWriteState::Expired});
    } catch (...) {
        if (reserved) {
            finish(key, job.charge, {EmbWriteState::ProducerFailure});
        } else {
            std::lock_guard lock(mutex_);
            ++stats_.skipped;
        }
    }
}

bool RemoteWriteQueue::reserveExternal(int64_t bytes, bool check_task_limit) {
    std::lock_guard lock(mutex_);
    if (closed_ || bytes > max_bytes_ - stats_.bytes - external_bytes_
        || (check_task_limit && stats_.pending >= max_pending_ - external_tasks_)) {
        return false;
    }
    external_bytes_ += bytes;
    ++external_tasks_;
    return true;
}

void RemoteWriteQueue::releaseExternal(int64_t bytes) {
    std::lock_guard lock(mutex_);
    external_bytes_ -= bytes;
    --external_tasks_;
}

RemoteWriteQueue::Stats RemoteWriteQueue::stats() const {
    std::lock_guard lock(mutex_);
    return stats_;
}

void RemoteWriteQueue::finish(const std::string& key, int64_t bytes, const EmbWriteResult& result) {
    std::lock_guard lock(mutex_);
    keys_.erase(key);
    --stats_.pending;
    if (arena_ || !result.inflight) {
        stats_.bytes -= bytes;
    }
    switch (result.state) {
        case EmbWriteState::Success:
            ++stats_.success;
            break;
        case EmbWriteState::Expired:
        case EmbWriteState::LeaseUnavailable:
            ++stats_.skipped;
            break;
        default:
            ++stats_.error;
            break;
    }
    if (result.inflight) {
        ++stats_.inflight_failure;
    }
}

EmbWriteResult RemoteWriteQueue::writeStaged(Job& job, const std::string& trace) {
    auto output = at::empty({job.bytes}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    packEmbTensorSlices(output, std::move(job.prefix), job.payload_offset, job.tensors);
    if (WriteClock::now() >= job.deadline) {
        return {EmbWriteState::Expired};
    }
    const KvcmObjectNativeSaveRequestV2 request{
        job.key.data(),
        job.key.size(),
        trace.data(),
        trace.size(),
        output.data_ptr(),
        static_cast<uint64_t>(job.bytes),
        0,
        0,
        0,
        0,
        nullptr,
        0,
    };
    std::list<at::Tensor>::iterator buffer;
    {
        std::lock_guard lock(mutex_);
        if (closed_ || stats_.inflight_failure) {
            return {EmbWriteState::Expired};
        }
        buffer = staging_buffers_.insert(staging_buffers_.end(), std::move(output));
    }
    const auto result = api_.save(api_.context, &request);
    if (!result.data_transfer_may_be_inflight) {
        std::lock_guard lock(mutex_);
        staging_buffers_.erase(buffer);
    }
    return {!result.code && !result.data_transfer_may_be_inflight ? EmbWriteState::Success : EmbWriteState::SdkFailure,
            result.code,
            result.data_transfer_may_be_inflight != 0};
}

void RemoteWriteQueue::run() {
    static std::atomic<uint64_t> sequence{0};
    const auto                   trace_prefix = "emb-write-" + std::to_string(::getpid()) + "-"
                              + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + "-";
    for (;;) {
        Job  job;
        bool cancelled;
        {
            std::unique_lock lock(mutex_);
            ready_.wait(lock, [this] { return closed_ || !jobs_.empty(); });
            if (jobs_.empty()) {
                return;
            }
            cancelled = closed_;
            job       = std::move(jobs_.front());
            jobs_.pop_front();
        }
        EmbWriteResult result{EmbWriteState::Expired};
        try {
            if (!cancelled && WriteClock::now() < job.deadline) {
                std::shared_lock gate(graphGate(), std::defer_lock);
                if (gate.try_lock_until(job.deadline)) {
                    // Both staging and GDR wait before they allocate or lease.
                    for (const auto& event : job.events) {
                        event->synchronize();
                    }
                    const auto remaining =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(job.deadline - WriteClock::now()).count();
                    if (remaining > 0) {
                        if (job.trace.empty()) {
                            job.trace = trace_prefix + std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
                        }
                        result = arena_ ? writeEmbObject(*arena_,
                                                         root_,
                                                         job.bytes,
                                                         std::move(job.prefix),
                                                         job.payload_offset,
                                                         job.tensors,
                                                         {},
                                                         api_,
                                                         job.key,
                                                         job.trace,
                                                         remaining,
                                                         false,
                                                         true) :
                                          writeStaged(job, job.trace);
                    }
                }
            }
        } catch (...) {
            result.state = EmbWriteState::ProducerFailure;
        }
        // Release the independently owned snapshot before returning capacity.
        job.tensors.clear();
        finish(job.key, job.charge, result);
    }
}

void RemoteWriteQueue::close() {
    std::lock_guard closing(close_mutex_);
    {
        std::lock_guard lock(mutex_);
        closed_ = true;
    }
    ready_.notify_all();
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    // The registered transport still owns the root and handles quarantine.
    root_.reset();
}

}  // namespace rtp_llm::emb_local
#endif
