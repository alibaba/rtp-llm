#include "rtp_llm/cpp/utils/cuda_call_trace/CudaCallTrace.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if USING_CUDA
#include <cupti.h>
#endif

// Activity structures are versioned ABIs. Older toolkits still build a disabled
// stub; an enabled recording must never reinterpret a different CUPTI ABI.
#if USING_CUDA && CUPTI_API_VERSION >= 130000
#include "rtp_llm/cpp/utils/cuda_call_trace/RecordQueue.h"
#include "rtp_llm/cpp/utils/cuda_call_trace/ActiveRecords.h"
#include "rtp_llm/cpp/utils/cuda_call_trace/IncidentRetention.h"

#include <array>
#include <chrono>
#include <cinttypes>
#include <dlfcn.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <link.h>
#include <limits.h>
#include <map>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <thread>
#include <time.h>
#include <tuple>
#include <unistd.h>

namespace rtp_llm::cuda_trace {
namespace {

constexpr size_t   kCriticalCapacity = 16384;
constexpr size_t   kBulkCapacity     = 32768;
constexpr size_t   kBufferCount      = 8;
constexpr size_t   kBufferBytes      = 1 << 20;
constexpr size_t   kOpenCapacity     = 4096;
constexpr uint64_t kFileBytes        = 64ULL << 20;
constexpr size_t   kFileCount        = 16;

enum class Kind : uint8_t {
    ApiEnter,
    ApiExit,
    Activity,
    ScopeEnter,
    ScopeExit,
    Mark,
    Warning,
    CallbackUnavailable,
    FlushBegin,
    FlushEnd
};
const char* kindName(Kind kind) noexcept {
    switch (kind) {
        case Kind::ApiEnter:
            return "api_enter";
        case Kind::ApiExit:
            return "api_exit";
        case Kind::Activity:
            return "activity";
        case Kind::ScopeEnter:
            return "scope_enter";
        case Kind::ScopeExit:
            return "scope_exit";
        case Kind::Mark:
            return "mark";
        case Kind::Warning:
            return "warning";
        case Kind::CallbackUnavailable:
            return "callback_unavailable";
        case Kind::FlushBegin:
            return "flush_begin";
        case Kind::FlushEnd:
            return "flush_end";
    }
    return "unknown";
}

struct Record {
    Kind     kind           = Kind::Mark;
    bool     params_known   = false;
    bool     name_truncated = false;
    uint32_t tid = 0, domain = 0, cbid = 0, correlation_id = 0, context_uid = 0;
    uint32_t activity_kind = 0, device_id = 0, stream_id = 0, runtime_correlation_id = 0, event_id = 0, graph_id = 0;
    int32_t  result     = 0;
    uint64_t record_seq = 0, mono_ns = 0, call_id = 0, scope_id = 0, parent_scope_id = 0, step = 0;
    uint64_t active_handle = 0;
    int64_t  request_id    = -1;
    uint64_t raw_context = 0, stream = 0, event = 0, src = 0, dst = 0, bytes = 0;
    uint64_t value[8] = {};
    uint64_t start_ns = 0, end_ns = 0, event_sync_id = 0, graph_node_id = 0;
    char     name[160] = {};
};

uint64_t monotonicNs() noexcept {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ULL + uint64_t(ts.tv_nsec);
}
uint64_t wallNs() noexcept {
    timespec ts{};
    clock_gettime(CLOCK_REALTIME, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ULL + uint64_t(ts.tv_nsec);
}
uint32_t threadId() noexcept {
    return static_cast<uint32_t>(syscall(SYS_gettid));
}

void copyName(Record& record, const char* name) noexcept {
    if (name == nullptr) {
        return;
    }
    const size_t n    = strnlen(name, sizeof(record.name));
    const size_t copy = n < sizeof(record.name) ? n : sizeof(record.name) - 1;
    std::memcpy(record.name, name, copy);
    record.name[copy]     = '\0';
    record.name_truncated = n == sizeof(record.name);
}

thread_local Context tls_context{};
struct Recorder;
std::atomic<Recorder*> recorder{nullptr};
std::atomic<int>       start_pid{0};

#include "rtp_llm/cpp/utils/cuda_call_trace/ApiParams.inc"

struct Cupti {
    void* library = nullptr;
#define CUPTI_SYMBOL(NAME) decltype(&NAME) NAME##_fn = nullptr
    CUPTI_SYMBOL(cuptiGetVersion);
    CUPTI_SYMBOL(cuptiGetTimestamp);
    CUPTI_SYMBOL(cuptiSubscribe);
    CUPTI_SYMBOL(cuptiUnsubscribe);
    CUPTI_SYMBOL(cuptiEnableDomain);
    CUPTI_SYMBOL(cuptiEnableCallback);
    CUPTI_SYMBOL(cuptiGetCallbackName);
    CUPTI_SYMBOL(cuptiActivityRegisterCallbacks);
    CUPTI_SYMBOL(cuptiActivityEnable);
    CUPTI_SYMBOL(cuptiActivityEnableAndDump);
    CUPTI_SYMBOL(cuptiActivityDisable);
    CUPTI_SYMBOL(cuptiActivityGetNextRecord);
    CUPTI_SYMBOL(cuptiActivityGetNumDroppedRecords);
    CUPTI_SYMBOL(cuptiActivityFlushAll);
#undef CUPTI_SYMBOL

    bool load() noexcept {
        // Prefer the already-loaded CUPTI instance (including RTLD_LOCAL
        // dependencies of Torch/Kineto). Never load a second CUPTI ABI.
        const bool global_symbols = dlsym(RTLD_DEFAULT, "cuptiGetVersion") != nullptr;
        struct Loaded {
            unsigned count          = 0;
            char     path[PATH_MAX] = {};
        } loaded;
        dl_iterate_phdr(
            [](dl_phdr_info* info, size_t, void* argument) -> int {
                auto&       found = *static_cast<Loaded*>(argument);
                const char* base  = std::strrchr(info->dlpi_name, '/');
                base              = base ? base + 1 : info->dlpi_name;
                if (std::strncmp(base, "libcupti.so", 11) == 0) {
                    ++found.count;
                    std::strncpy(found.path, info->dlpi_name, sizeof(found.path) - 1);
                }
                return 0;
            },
            &loaded);
        if (loaded.count > 1) {
            return false;
        }
        if (global_symbols) {
            library = RTLD_DEFAULT;
        } else if (loaded.count == 1) {
            library = dlopen(loaded.path, RTLD_NOW | RTLD_NOLOAD);
            if (!library) {
                return false;
            }
        } else {
            library = dlopen("libcupti.so", RTLD_NOW | RTLD_LOCAL);
        }
        if (!global_symbols && !library && loaded.count == 0) {
            // CUDA wheels may expose only the versioned soname in the loader
            // path. No CUDA context is initialized by these dlopen attempts.
            library = dlopen("libcupti.so.13", RTLD_NOW | RTLD_LOCAL);
        }
        if (!global_symbols && !library && loaded.count == 0) {
            library = dlopen("libcupti.so.2026.1.1", RTLD_NOW | RTLD_LOCAL);
        }
        if (!global_symbols && !library) {
            return false;
        }
#define LOAD(NAME)                                                                                                     \
    NAME##_fn = reinterpret_cast<decltype(NAME##_fn)>(dlsym(library, #NAME));                                          \
    if (!NAME##_fn) {                                                                                                  \
        return false;                                                                                                  \
    }
        LOAD(cuptiGetVersion)
        LOAD(cuptiGetTimestamp)
        LOAD(cuptiSubscribe)
        LOAD(cuptiUnsubscribe)
        LOAD(cuptiEnableDomain)
        LOAD(cuptiEnableCallback)
        LOAD(cuptiGetCallbackName)
        LOAD(cuptiActivityRegisterCallbacks)
        LOAD(cuptiActivityEnable)
        LOAD(cuptiActivityEnableAndDump)
        LOAD(cuptiActivityDisable)
        LOAD(cuptiActivityGetNextRecord)
        LOAD(cuptiActivityGetNumDroppedRecords)
        LOAD(cuptiActivityFlushAll)
#undef LOAD
        return true;
    }
};

// All allocation and JSON formatting happens during start or on the CPU writer.
// This intentionally leaked process-lifetime object avoids calling CUDA/CUPTI
// finalization, joining a blocked flusher, or taking inherited locks at exit.
struct Recorder {
    static_assert(std::atomic<uint64_t>::is_always_lock_free, "callbacks require lock-free counters");
    int                               pid                = getpid();
    int                               rank               = -1;
    uint64_t                          process_startticks = 0;
    std::string                       hostname, prefix, trace_id;
    Cupti                             cupti{};
    uint32_t                          runtime_version = 0;
    CUpti_SubscriberHandle            subscriber{};
    std::atomic<bool>                 enabled{false}, writer_running{true}, collect_calls{true};
    std::atomic<bool>                 subscribed{false};
    std::array<CUpti_ActivityKind, 7> enabled_activities{};
    size_t                            enabled_activity_count = 0;
    std::atomic<uint64_t>             ids{1}, sequence{1};
    std::atomic<uint64_t>             host_dropped{0}, activity_dropped{0}, pool_exhausted{0}, callback_errors{0};
    std::atomic<uint64_t>             unknown_params{0};
    std::atomic<uint64_t>             critical_dropped{0}, bulk_dropped{0}, active_state_overflow{0};
    std::atomic<uint64_t>             untracked_api_calls{0}, close_mismatches{0};
    std::array<Record, 32>            unavailable_callbacks{};
    std::atomic<uint32_t>             optional_callback_unavailable{0};
    detail::RecordQueue<Record, kCriticalCapacity> critical_queue;
    detail::RecordQueue<Record, kBulkCapacity>     bulk_queue;
    using ActiveLedger = detail::ActiveRecords<Record, kOpenCapacity>;
    ActiveLedger active_calls, active_scopes;
    struct Buffer {
        std::atomic<bool> in_use{false};
        alignas(64) std::array<uint8_t, kBufferBytes> data{};
    };
    std::array<Buffer, kBufferCount> buffers{};
    // Startup metadata is assembled before the writer starts and never mutated.
    std::string metadata;
    uint64_t    writer_seq = 0, file_generation = 0, file_bytes = 0;
    uint64_t    snapshot_missed_slots = 0, snapshot_begin_ns = 0, snapshot_end_ns = 0;
    FILE*       output = nullptr;
    // Rebuilt from immutable live slots; disk delivery never controls liveness.
    using Key = std::tuple<uint32_t, uint32_t, uint32_t, uint64_t>;
    struct OpenRecord {
        Record   record;
        uint64_t writer_seq;
    };
    std::map<Key, OpenRecord>        open;
    std::map<uint64_t, OpenRecord>   open_scopes;
    bool                             incident_active = false;
    detail::IncidentDirectory        incident_owned{};
    Key                              incident_key{};
    uint64_t                         incident_call_id = 0, incident_count = 0, incident_pin_errors = 0;
    uint64_t                         incident_handle = 0;
    std::array<uint64_t, kFileCount> segment_first_writer_seq{};

    bool sameProcess() const noexcept {
        return pid == getpid();
    }
    void emit(Record record) noexcept {
        if (!sameProcess() || !writer_running.load(std::memory_order_relaxed)) {
            return;
        }
        if (!record.record_seq) {
            record.record_seq = sequence.fetch_add(1, std::memory_order_relaxed);
        }
        const bool bulk =
            record.kind == Kind::Activity ?
                record.activity_kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL :
                ((record.kind == Kind::ApiEnter || record.kind == Kind::ApiExit) && isLaunchApi(record.name));
        if (!(bulk ? bulk_queue.push(record) : critical_queue.push(record))) {
            host_dropped.fetch_add(1, std::memory_order_relaxed);
            (bulk ? bulk_dropped : critical_dropped).fetch_add(1, std::memory_order_relaxed);
        }
    }
    void warning(const char* message, int result = 0, uint64_t value = 0) noexcept {
        Record r{};
        r.kind     = Kind::Warning;
        r.mono_ns  = monotonicNs();
        r.tid      = threadId();
        r.result   = result;
        r.value[0] = value;
        copyName(r, message);
        emit(r);
    }
    void        writer() noexcept;
    void        flushLoop() noexcept;
    void        rollbackStartup() noexcept;
    void        failure(const char* reason) noexcept;
    void        writeSnapshot();
    void        refreshActive();
    void        pinCandidate(uint64_t now);
    void        closeSegment();
    void        rotate();
    std::string json(const Record& r, uint64_t writer_sequence) const;
};

Record businessRecord(Kind kind, const char* name) noexcept {
    Record r{};
    r.kind       = kind;
    r.mono_ns    = monotonicNs();
    r.tid        = threadId();
    r.scope_id   = tls_context.scope_id;
    r.request_id = tls_context.request_id;
    r.step       = tls_context.step;
    copyName(r, name);
    return r;
}

void CUPTIAPI apiCallback(void*                userdata,
                          CUpti_CallbackDomain domain,
                          CUpti_CallbackId     cbid,
                          const void*          data) noexcept {
    auto* instance = static_cast<Recorder*>(userdata);
    if (!instance || !instance->collect_calls.load(std::memory_order_relaxed) || !instance->sameProcess()
        || (domain != CUPTI_CB_DOMAIN_DRIVER_API && domain != CUPTI_CB_DOMAIN_RUNTIME_API)) {
        return;
    }
    // No throwing operations, locks, allocations, CUDA API calls or file I/O.
    const auto& cb = *static_cast<const CUpti_CallbackData*>(data);
    if (!trackedApi(cb.functionName)) {
        return;
    }
    const bool exit       = cb.callbackSite == CUPTI_API_EXIT;
    Record     r          = businessRecord(exit ? Kind::ApiExit : Kind::ApiEnter, cb.functionName);
    r.domain              = domain;
    r.cbid                = cbid;
    r.correlation_id      = cb.correlationId;
    r.context_uid         = cb.contextUid;
    r.raw_context         = address(cb.context);
    const uint64_t handle = exit && cb.correlationData ? *cb.correlationData : 0;
    r.call_id = exit ? Recorder::ActiveLedger::id(handle) : instance->ids.fetch_add(1, std::memory_order_relaxed);
    if (exit && cb.functionReturnValue) {
        // CUDA runtime and driver results both use an integer enum.
        std::memcpy(&r.result, cb.functionReturnValue, sizeof(r.result));
    }
    decodeParams(r, cb, exit);
    if (!r.params_known && !exit) {
        instance->unknown_params.fetch_add(1, std::memory_order_relaxed);
    }
    r.record_seq = instance->sequence.fetch_add(1, std::memory_order_relaxed);
    if (!exit) {
        if (cb.correlationData) {
            const auto tracked  = instance->active_calls.begin(r, r.call_id);
            *cb.correlationData = tracked ? tracked : Recorder::ActiveLedger::untracked(r.call_id);
            if (!tracked) {
                instance->active_state_overflow.fetch_add(1, std::memory_order_relaxed);
            }
        } else {
            // Without callback-owned correlation storage there is no reliable
            // unique handle to close. Keep history, never create an immortal slot.
            instance->untracked_api_calls.fetch_add(1, std::memory_order_relaxed);
        }
    } else if ((handle & 8191) && !instance->active_calls.close(handle)) {
        instance->close_mismatches.fetch_add(1, std::memory_order_relaxed);
    }
    instance->emit(r);
}

void CUPTIAPI bufferRequested(uint8_t** buffer, size_t* size, size_t* max_records) noexcept {
    *buffer        = nullptr;
    *size          = 0;
    *max_records   = 0;
    auto* instance = recorder.load(std::memory_order_acquire);
    if (!instance || !instance->sameProcess()) {
        return;
    }
    if (!instance->writer_running.load(std::memory_order_relaxed)) {
        return;
    }
    for (auto& slot : instance->buffers) {
        bool expected = false;
        if (slot.in_use.compare_exchange_strong(expected, true, std::memory_order_acquire)) {
            *buffer = slot.data.data();
            *size   = slot.data.size();
            return;
        }
    }
    instance->pool_exhausted.fetch_add(1, std::memory_order_relaxed);
}

template<typename Kernel>
void kernelRecord(Record& r, const Kernel& a) noexcept {
    copyName(r, a.name);
    r.start_ns       = a.start;
    r.end_ns         = a.end;
    r.device_id      = a.deviceId;
    r.context_uid    = a.contextId;
    r.stream_id      = a.streamId;
    r.correlation_id = a.correlationId;
    r.graph_id       = a.graphId;
    r.graph_node_id  = a.graphNodeId;
    r.value[0]       = a.gridX;
    r.value[1]       = a.gridY;
    r.value[2]       = a.gridZ;
    r.value[3]       = a.blockX;
    r.value[4]       = a.blockY;
    r.value[5]       = a.blockZ;
    r.value[6]       = a.queued;
    r.value[7]       = a.submitted;
}

void activityRecord(Recorder& instance, CUpti_Activity* activity) noexcept {
    Record r{};
    r.kind          = Kind::Activity;
    r.mono_ns       = monotonicNs();  // Arrival time, not GPU/API execution time.
    r.tid           = threadId();     // Collector callback thread, not origin thread.
    r.activity_kind = activity->kind;
    r.params_known  = true;
    switch (activity->kind) {
        case CUPTI_ACTIVITY_KIND_DRIVER:
        case CUPTI_ACTIVITY_KIND_RUNTIME: {
            const auto& a = *reinterpret_cast<const CUpti_ActivityAPI*>(activity);
            copyName(r, activity->kind == CUPTI_ACTIVITY_KIND_DRIVER ? "driver_api" : "runtime_api");
            r.start_ns       = a.start;
            r.end_ns         = a.end;
            r.cbid           = a.cbid;
            r.correlation_id = a.correlationId;
            r.tid            = a.threadId;
            r.result         = a.returnValue;
            r.domain =
                activity->kind == CUPTI_ACTIVITY_KIND_DRIVER ? CUPTI_CB_DOMAIN_DRIVER_API : CUPTI_CB_DOMAIN_RUNTIME_API;
            break;
        }
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
#if CUPTI_API_VERSION >= 130200
            if (instance.runtime_version / 100 == 1302) {
                kernelRecord(r, *reinterpret_cast<const CUpti_ActivityKernel11*>(activity));
            } else
#endif
            {
                kernelRecord(r, *reinterpret_cast<const CUpti_ActivityKernel10*>(activity));
            }
            break;
        }
        case CUPTI_ACTIVITY_KIND_MEMCPY: {
            const auto& a = *reinterpret_cast<const CUpti_ActivityMemcpy6*>(activity);
            copyName(r, "memcpy");
            r.start_ns               = a.start;
            r.end_ns                 = a.end;
            r.device_id              = a.deviceId;
            r.context_uid            = a.contextId;
            r.stream_id              = a.streamId;
            r.correlation_id         = a.correlationId;
            r.runtime_correlation_id = a.runtimeCorrelationId;
            r.bytes                  = a.bytes;
            r.value[0]               = a.copyKind;
            r.value[1]               = a.srcKind;
            r.value[2]               = a.dstKind;
            r.value[3]               = a.flags;
            r.value[4]               = a.copyCount;
            r.graph_id               = a.graphId;
            r.graph_node_id          = a.graphNodeId;
            break;
        }
        case CUPTI_ACTIVITY_KIND_MEMSET: {
            const auto& a = *reinterpret_cast<const CUpti_ActivityMemset4*>(activity);
            copyName(r, "memset");
            r.start_ns       = a.start;
            r.end_ns         = a.end;
            r.device_id      = a.deviceId;
            r.context_uid    = a.contextId;
            r.stream_id      = a.streamId;
            r.correlation_id = a.correlationId;
            r.bytes          = a.bytes;
            r.value[0]       = a.value;
            r.graph_id       = a.graphId;
            r.graph_node_id  = a.graphNodeId;
            break;
        }
        case CUPTI_ACTIVITY_KIND_CUDA_EVENT: {
            const auto& a = *reinterpret_cast<const CUpti_ActivityCudaEvent2*>(activity);
            copyName(r, "event_record");
            r.device_id      = a.deviceId;
            r.context_uid    = a.contextId;
            r.stream_id      = a.streamId;
            r.correlation_id = a.correlationId;
            r.event_id       = a.eventId;
            r.event_sync_id  = a.cudaEventSyncId;
            // deviceTimestamp tracing is deliberately not enabled: it can add
            // dependencies and is unnecessary for record-generation matching.
            break;
        }
        case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION: {
            const auto& a = *reinterpret_cast<const CUpti_ActivitySynchronization2*>(activity);
            copyName(r, "synchronization");
            r.start_ns       = a.start;
            r.end_ns         = a.end;
            r.context_uid    = a.contextId;
            r.stream_id      = a.streamId;
            r.correlation_id = a.correlationId;
            r.event_id       = a.cudaEventId;
            r.event_sync_id  = a.cudaEventSyncId;
            r.result         = a.returnValue;
            r.value[0]       = a.type;
            break;
        }
        case CUPTI_ACTIVITY_KIND_STREAM: {
            const auto& a = *reinterpret_cast<const CUpti_ActivityStream*>(activity);
            copyName(r, "stream");
            r.context_uid    = a.contextId;
            r.stream_id      = a.streamId;
            r.correlation_id = a.correlationId;
            r.value[0]       = a.flag;
            r.value[1]       = a.priority;
            break;
        }
        case CUPTI_ACTIVITY_KIND_CONTEXT: {
            copyName(r, "context");
#if CUPTI_API_VERSION >= 130200
            if (instance.runtime_version / 100 == 1302) {
                const auto& a = *reinterpret_cast<const CUpti_ActivityContext4*>(activity);
                r.context_uid = a.contextId;
                r.device_id   = a.deviceId;
                r.value[0]    = a.computeApiKind;
            } else
#endif
            {
                const auto& a = *reinterpret_cast<const CUpti_ActivityContext3*>(activity);
                r.context_uid = a.contextId;
                r.device_id   = a.deviceId;
                r.value[0]    = a.computeApiKind;
            }
            break;
        }
        default:
            copyName(r, "unknown_activity");
            r.params_known = false;
            break;
    }
    instance.emit(r);
}

void CUPTIAPI
bufferCompleted(CUcontext context, uint32_t stream_id, uint8_t* buffer, size_t, size_t valid_size) noexcept {
    auto* instance = recorder.load(std::memory_order_acquire);
    if (!instance || !instance->sameProcess()) {
        return;
    }
    if (instance->writer_running.load(std::memory_order_relaxed) && valid_size && buffer) {
        CUpti_Activity* activity = nullptr;
        CUptiResult     result;
        while ((result = instance->cupti.cuptiActivityGetNextRecord_fn(buffer, valid_size, &activity))
               == CUPTI_SUCCESS) {
            activityRecord(*instance, activity);
        }
        if (result != CUPTI_ERROR_MAX_LIMIT_REACHED) {
            instance->callback_errors.fetch_add(1, std::memory_order_relaxed);
            instance->warning("activity_parse_error", result);
        }
    }
    if (instance->writer_running.load(std::memory_order_relaxed)) {
        size_t     dropped = 0;
        const auto result  = instance->cupti.cuptiActivityGetNumDroppedRecords_fn(context, stream_id, &dropped);
        if (result == CUPTI_SUCCESS) {
            instance->activity_dropped.fetch_add(dropped, std::memory_order_relaxed);
        } else {
            instance->callback_errors.fetch_add(1, std::memory_order_relaxed);
        }
    }
    for (auto& slot : instance->buffers) {
        if (slot.data.data() == buffer) {
            slot.in_use.store(false, std::memory_order_release);
            return;
        }
    }
    if (buffer) {
        instance->callback_errors.fetch_add(1, std::memory_order_relaxed);
    }
}

std::string jsonQuote(const std::string& value) {
    std::ostringstream out;
    out << '"';
    constexpr char hex[] = "0123456789abcdef";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') {
            out << '\\' << c;
        } else if (c < 0x20) {
            out << "\\u00" << hex[c >> 4] << hex[c & 15];
        } else {
            out << c;
        }
    }
    out << '"';
    return out.str();
}
std::string hexString(uint64_t value) {
    std::ostringstream out;
    out << "\"0x" << std::hex << value << '"';
    return out.str();
}

std::string Recorder::json(const Record& r, uint64_t ws) const {
    std::ostringstream out;
    out << "{\"type\":" << jsonQuote(kindName(r.kind)) << ",\"name\":" << jsonQuote(r.name) << ",\"pid\":" << pid
        << ",\"rank\":" << rank << ",\"trace_id\":" << jsonQuote(trace_id);
#define TEXT_FIELD(FIELD) out << ",\"" #FIELD "\":\"" << r.FIELD << '"'
#define NUM_FIELD(FIELD) out << ",\"" #FIELD "\":" << r.FIELD
#define HEX_FIELD(FIELD) out << ",\"" #FIELD "\":" << hexString(r.FIELD)
    TEXT_FIELD(record_seq);
    TEXT_FIELD(mono_ns);
    TEXT_FIELD(call_id);
    TEXT_FIELD(scope_id);
    TEXT_FIELD(parent_scope_id);
    TEXT_FIELD(request_id);
    TEXT_FIELD(step);
    NUM_FIELD(tid);
    NUM_FIELD(domain);
    NUM_FIELD(cbid);
    NUM_FIELD(correlation_id);
    NUM_FIELD(context_uid);
    HEX_FIELD(raw_context);
    HEX_FIELD(stream);
    HEX_FIELD(event);
    HEX_FIELD(src);
    HEX_FIELD(dst);
    TEXT_FIELD(bytes);
    TEXT_FIELD(start_ns);
    TEXT_FIELD(end_ns);
    TEXT_FIELD(event_sync_id);
    TEXT_FIELD(graph_node_id);
    NUM_FIELD(activity_kind);
    NUM_FIELD(device_id);
    NUM_FIELD(stream_id);
    NUM_FIELD(runtime_correlation_id);
    NUM_FIELD(event_id);
    NUM_FIELD(graph_id);
    NUM_FIELD(result);
#undef TEXT_FIELD
#undef NUM_FIELD
#undef HEX_FIELD
    for (unsigned i = 0; i != 8; ++i) {
        out << ",\"value" << i << "\":\"" << r.value[i] << '"';
    }
    out << ",\"writer_seq\":\"" << ws << "\",\"params_known\":" << (r.params_known ? "true" : "false")
        << ",\"name_truncated\":" << (r.name_truncated ? "true" : "false") << ",\"tid_source\":"
        << jsonQuote(r.kind != Kind::Activity ? "linux_gettid" :
                     (r.activity_kind == CUPTI_ACTIVITY_KIND_DRIVER || r.activity_kind == CUPTI_ACTIVITY_KIND_RUNTIME) ?
                                                "cupti_activity_thread_id" :
                                                "collector_linux_gettid")
        << "}";
    return out.str();
}

void Recorder::closeSegment() {
    if (output) {
        const int flush_result = std::fflush(output);
        const int close_result = std::fclose(output);
        output                 = nullptr;
        if (flush_result != 0 || close_result != 0) {
            throw std::runtime_error("cannot flush/close rotated trace segment; recording is incomplete");
        }
    }
}

void Recorder::rotate() {
    closeSegment();
    const auto path = prefix + "." + std::to_string(file_generation % kFileCount) + ".jsonl";
    segment_first_writer_seq[file_generation % kFileCount] = writer_seq + 1;
    // A sealed incident holds hard links to old segments. Unlink this name
    // before creating a new inode; fopen("w") alone would truncate the pin.
    std::filesystem::remove(path);
    output = std::fopen(path.c_str(), "w");
    if (!output) {
        throw std::runtime_error("cannot open trace segment " + path);
    }
    const auto header = metadata + ",\"file_generation\":\"" + std::to_string(file_generation++)
                        + "\",\"first_writer_seq\":\"" + std::to_string(writer_seq + 1) + "\"}\n";
    if (std::fwrite(header.data(), 1, header.size(), output) != header.size()) {
        throw std::runtime_error("cannot write trace segment metadata");
    }
    file_bytes = header.size();
}

void Recorder::refreshActive() {
    snapshot_begin_ns = monotonicNs();
    open.clear();
    open_scopes.clear();
    snapshot_missed_slots = active_calls.snapshot([this](Record record, uint64_t handle) {
        record.active_handle = handle;
        open.emplace(Key{record.domain, record.tid, record.correlation_id, record.call_id}, OpenRecord{record, 0});
    });
    snapshot_missed_slots += active_scopes.snapshot([this](Record record, uint64_t handle) {
        record.active_handle = handle;
        open_scopes.emplace(record.scope_id, OpenRecord{record, 0});
    });
    snapshot_end_ns = monotonicNs();
    if (incident_active && !active_calls.contains(incident_handle)) {
        incident_active = false;
    }
}

void Recorder::writeSnapshot() {
    const std::string temp   = prefix + ".open.json.tmp";
    const std::string target = prefix + ".open.json";
    std::ofstream     out(temp, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot open active-call snapshot");
    }
    out << "{\"type\":\"open_snapshot\"" << metadata.substr(std::strlen("{\"type\":\"metadata\"")) << ",\"mono_ns\":\""
        << monotonicNs() << "\",\"last_writer_seq\":\"" << writer_seq << "\",\"next_record_seq\":\"" << sequence.load()
        << "\",\"host_dropped\":\"" << host_dropped.load() << "\",\"activity_dropped\":\"" << activity_dropped.load()
        << "\",\"pool_exhausted\":\"" << pool_exhausted.load() << "\",\"callback_errors\":\"" << callback_errors.load()
        << "\",\"unknown_params\":\"" << unknown_params.load() << "\",\"critical_dropped\":\""
        << critical_dropped.load() << "\",\"bulk_dropped\":\"" << bulk_dropped.load()
        << "\",\"active_state_overflow\":\"" << active_state_overflow.load() << "\",\"untracked_api_calls\":\""
        << untracked_api_calls.load() << "\",\"close_mismatches\":\"" << close_mismatches.load()
        << "\",\"snapshot_missed_slots\":\"" << snapshot_missed_slots << "\",\"snapshot_begin_ns\":\""
        << snapshot_begin_ns << "\",\"snapshot_end_ns\":\"" << snapshot_end_ns
        << "\",\"enabled\":" << (enabled.load() ? "true" : "false")
        << ",\"incident_active\":" << (incident_active ? "true" : "false") << ",\"incident_count\":\"" << incident_count
        << "\",\"incident_pin_errors\":\"" << incident_pin_errors << '"';
    // Startup alone appends immutable entries, publishing each with release.
    const auto unavailable_count = optional_callback_unavailable.load(std::memory_order_acquire);
    out << ",\"optional_callback_unavailable\":\"" << unavailable_count << "\",\"unavailable_callbacks\":[";
    for (uint32_t i = 0; i != unavailable_count; ++i) {
        if (i)
            out << ',';
        out << json(unavailable_callbacks[i], 0);
    }
    out << "],\"open_calls\":[";
    bool comma = false;
    for (const auto& entry : open) {
        if (comma) {
            out << ',';
        }
        comma = true;
        out << json(entry.second.record, entry.second.writer_seq);
    }
    out << "],\"open_scopes\":[";
    comma = false;
    for (const auto& entry : open_scopes) {
        if (comma) {
            out << ',';
        }
        comma = true;
        out << json(entry.second.record, entry.second.writer_seq);
    }
    out << "]}\n";
    out.close();
    if (!out || std::rename(temp.c_str(), target.c_str()) != 0) {
        throw std::runtime_error("cannot atomically publish active-call snapshot");
    }
}

void Recorder::pinCandidate(uint64_t now) {
    if (incident_active) {
        return;
    }
    auto lookup = [this](uint64_t id, detail::ScopeEvidence& evidence) {
        const auto found = open_scopes.find(id);
        if (found == open_scopes.end()) {
            return false;
        }
        evidence.name            = found->second.record.name;
        evidence.parent_scope_id = found->second.record.parent_scope_id;
        evidence.batch_size      = found->second.record.value[1];
        return true;
    };
    auto candidate = open.end();
    for (auto it = open.begin(); it != open.end(); ++it) {
        const auto& r = it->second.record;
        if (detail::longDemandCall(r.mono_ns, now, r.request_id, r.scope_id, lookup)
            && (candidate == open.end() || r.mono_ns < candidate->second.record.mono_ns)) {
            candidate = it;
        }
    }
    if (candidate == open.end()) {
        return;
    }
    if (!active_calls.contains(candidate->second.record.active_handle)) {
        return;
    }
    incident_active  = true;
    incident_key     = candidate->first;
    incident_call_id = candidate->second.record.call_id;
    incident_handle  = candidate->second.record.active_handle;
    ++incident_count;
    writeSnapshot();
    closeSegment();
    try {
        std::vector<std::filesystem::path> files;
        for (size_t slot = 0; slot != kFileCount; ++slot) {
            const auto path = prefix + "." + std::to_string(slot) + ".jsonl";
            if (std::filesystem::exists(path)) {
                files.emplace_back(path);
            }
        }
        files.emplace_back(prefix + ".open.json");
        const uint64_t first_generation = file_generation > kFileCount ? file_generation - kFileCount : 0;
        uint64_t       first_writer     = writer_seq + 1;
        for (auto first : segment_first_writer_seq) {
            if (first && first < first_writer) {
                first_writer = first;
            }
        }
        std::ostringstream manifest;
        manifest << "{\"type\":\"long_call_candidate\",\"trace_id\":" << jsonQuote(trace_id) << ",\"pid\":" << pid
                 << ",\"rank\":" << rank << ",\"mono_ns\":\"" << now << "\",\"first_file_generation\":\""
                 << first_generation << "\",\"last_file_generation\":\"" << file_generation - 1
                 << "\",\"last_writer_seq\":\"" << writer_seq << "\",\"first_retained_writer_seq\":\"" << first_writer
                 << "\",\"trigger_enter_disk_coverage\":\"unknown; live ledger is independent of history delivery\""
                 << ",\"earlier_segments_rolled_out\":" << (first_generation ? "true" : "false")
                 << ",\"pin_complete\":true,\"is_hang_diagnosis\":false,\"trigger\":"
                 << json(candidate->second.record, candidate->second.writer_seq)
                 << ",\"retention\":\"one fixed slot; replaced only after trigger EXIT and another long demand call\"}";
        detail::pinClosedFiles(prefix + ".incident", files, manifest.str(), incident_owned);
        std::fprintf(stderr,
                     "CUDA_CALL_TRACE retained long-call candidate (not a hang diagnosis): %s.incident\n",
                     prefix.c_str());
    } catch (const detail::RetentionCleanupError&) {
        ++incident_pin_errors;
        throw;
    } catch (const std::exception& e) {
        ++incident_pin_errors;
        // A partial pin has no complete manifest. Preserve the error rather than
        // retrying I/O for the same still-open call every snapshot interval.
        std::fprintf(stderr, "CUDA_CALL_TRACE incident pin failed: %s\n", e.what());
    }
    rotate();
    warning(incident_pin_errors ? "long_call_candidate_pin_check_snapshot_errors" : "long_call_candidate_pinned",
            0,
            incident_call_id);
}

void Recorder::writer() noexcept {
    try {
        uint64_t last_snapshot = 0, last_drops = 0, last_drop_observation_seq = 0;
        while (writer_running.load(std::memory_order_relaxed)) {
            Record r{};
            // Bound each drain so a saturated queue cannot starve snapshots.
            for (size_t count = 0; count != 8192; ++count) {
                // Prefer dependency/scope records, but reserve one in four
                // opportunities for launch/kernel history under critical load.
                bool got = count % 4 == 3 ? bulk_queue.pop(r) : critical_queue.pop(r);
                if (!got) {
                    got = count % 4 == 3 ? critical_queue.pop(r) : bulk_queue.pop(r);
                }
                if (!got) {
                    break;
                }
                ++writer_seq;
                const std::string line = json(r, writer_seq) + '\n';
                if (std::fwrite(line.data(), 1, line.size(), output) != line.size()) {
                    throw std::runtime_error("trace write failed");
                }
                file_bytes += line.size();
                if (file_bytes >= kFileBytes) {
                    rotate();
                }
            }
            const uint64_t now = monotonicNs();
            if (now - last_snapshot >= 1000000000ULL) {
                const uint64_t dropped =
                    host_dropped.load() + activity_dropped.load() + pool_exhausted.load() + callback_errors.load();
                if (dropped != last_drops) {
                    ++writer_seq;
                    const std::string line =
                        "{\"type\":\"drop\",\"trace_id\":" + jsonQuote(trace_id) + ",\"writer_seq\":\""
                        + std::to_string(writer_seq) + "\",\"mono_ns\":\"" + std::to_string(now)
                        + "\",\"pid\":" + std::to_string(pid) + ",\"rank\":" + std::to_string(rank)
                        + ",\"host_dropped\":\"" + std::to_string(host_dropped.load()) + "\",\"critical_dropped\":\""
                        + std::to_string(critical_dropped.load()) + "\",\"bulk_dropped\":\""
                        + std::to_string(bulk_dropped.load()) + "\",\"producer_seq_at_previous_counter_sample\":\""
                        + std::to_string(last_drop_observation_seq) + "\",\"producer_seq_at_counter_sample\":\""
                        + std::to_string(sequence.load() - 1)
                        + "\",\"interval_semantics\":\"counter observations only; delayed producers may reference sequences before the previous sample\""
                        + ",\"activity_dropped\":\"" + std::to_string(activity_dropped.load())
                        + "\",\"pool_exhausted\":\"" + std::to_string(pool_exhausted.load())
                        + "\",\"callback_errors\":\"" + std::to_string(callback_errors.load()) + "\"}\n";
                    if (std::fwrite(line.data(), 1, line.size(), output) != line.size()) {
                        throw std::runtime_error("trace drop-counter write failed");
                    }
                    file_bytes += line.size();
                    last_drops = dropped;
                }
                last_drop_observation_seq = sequence.load();
                if (std::fflush(output) != 0) {
                    throw std::runtime_error("trace flush failed");
                }
                refreshActive();
                pinCandidate(now);
                writeSnapshot();
                // A slow snapshot must not immediately schedule another full
                // rewrite before the writer gets time to drain its queues.
                last_snapshot = monotonicNs();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        if (std::fflush(output) != 0) {
            throw std::runtime_error("cannot flush final host trace records; recording is incomplete");
        }
        refreshActive();
        writeSnapshot();
    } catch (const std::exception& e) {
        enabled.store(false);
        collect_calls.store(false);
        writer_running.store(false);
        failure(e.what());
        std::fprintf(stderr, "CUDA_CALL_TRACE writer failed: %s; recording is incomplete\n", e.what());
    } catch (...) {
        enabled.store(false);
        collect_calls.store(false);
        writer_running.store(false);
        failure("writer exception");
        std::fprintf(stderr, "CUDA_CALL_TRACE writer failed; recording is incomplete\n");
    }
}

void Recorder::rollbackStartup() noexcept {
    collect_calls.store(false);
    enabled.store(false);
    if (subscribed) {
        cupti.cuptiEnableDomain_fn(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
        cupti.cuptiEnableDomain_fn(0, subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
    }
    while (enabled_activity_count) {
        const auto kind   = enabled_activities[--enabled_activity_count];
        const auto result = cupti.cuptiActivityDisable_fn(kind);
        if (result != CUPTI_SUCCESS) {
            warning("startup_rollback_activity_disable_failed", result, kind);
        }
    }
    if (subscribed) {
        const auto result = cupti.cuptiUnsubscribe_fn(subscriber);
        if (result != CUPTI_SUCCESS) {
            warning("startup_rollback_unsubscribe_failed", result);
        } else {
            subscribed.store(false, std::memory_order_release);
        }
    }
}

void Recorder::failure(const char* reason) noexcept {
    try {
        if (prefix.empty()) {
            return;
        }
        std::ofstream out(prefix + ".failure.json", std::ios::trunc);
        out << "{\"type\":\"recording_failed\",\"trace_id\":" << jsonQuote(trace_id) << ",\"pid\":" << pid
            << ",\"rank\":" << rank << ",\"mono_ns\":\"" << monotonicNs() << "\",\"cupti_version\":" << runtime_version
            << ",\"reason\":" << jsonQuote(reason) << ",\"complete\":false}\n";
    } catch (...) {
        // stderr remains the fallback when the diagnostic filesystem itself fails.
    }
}

void Recorder::flushLoop() noexcept {
    while (writer_running.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        Record begin  = businessRecord(Kind::FlushBegin, "cuptiActivityFlushAll");
        begin.call_id = ids.fetch_add(1, std::memory_order_relaxed);
        emit(begin);
        // Normal flush collects completed activities only. It may block inside
        // CUPTI/driver; this separate thread never owns the host writer's state.
        const auto result = cupti.cuptiActivityFlushAll_fn(0);
        Record     end    = businessRecord(Kind::FlushEnd, "cuptiActivityFlushAll");
        end.call_id       = begin.call_id;
        end.result        = result;
        emit(end);
        if (result != CUPTI_SUCCESS) {
            warning("activity_flush_failed", result);
        }
    }
}

uint64_t readStartticks() {
    std::ifstream file("/proc/self/stat");
    std::string   line;
    std::getline(file, line);
    const auto close = line.rfind(')');
    if (close == std::string::npos) {
        return 0;
    }
    std::istringstream fields(line.substr(close + 2));
    std::string        field;
    for (unsigned i = 3; i <= 22; ++i) {
        if (!(fields >> field)) {
            return 0;
        }
    }
    return std::stoull(field);
}

}  // namespace

void start(int rank) noexcept {
    const char* enabled_env = std::getenv("RTP_LLM_GPU_STAGE_TRACE");
    if (!enabled_env || std::strcmp(enabled_env, "1") != 0) {
        return;
    }
    const int pid      = getpid();
    int       expected = 0;
    if (!start_pid.compare_exchange_strong(expected, pid)) {
        if (expected != pid) {
            std::fprintf(stderr,
                         "CUDA_CALL_TRACE disabled: inherited across fork; initialize in a fresh spawned backend\n");
        }
        return;
    }
    Recorder* instance = nullptr;
    try {
        instance                     = new Recorder();
        instance->rank               = rank;
        instance->process_startticks = readStartticks();
        char hostname[256]           = {};
        gethostname(hostname, sizeof(hostname) - 1);
        instance->hostname = hostname;
        instance->trace_id = instance->hostname + ":" + std::to_string(pid) + ":"
                             + std::to_string(instance->process_startticks) + ":" + std::to_string(monotonicNs());
        const char*       file_env = std::getenv("RTP_LLM_GPU_STAGE_TRACE_FILE");
        const std::string base     = file_env && *file_env ? file_env : "logs/cuda_call_trace";
        instance->prefix           = base + ".rank" + std::to_string(rank) + ".pid" + std::to_string(pid) + ".start"
                           + std::to_string(instance->process_startticks);
        const std::filesystem::path directory = std::filesystem::path(instance->prefix).parent_path();
        if (!directory.empty()) {
            std::filesystem::create_directories(directory);
        }
        if (!instance->cupti.load()) {
            throw std::runtime_error("cannot dynamically load CUPTI symbols");
        }
        uint32_t   version           = 0;
        const auto version_result    = instance->cupti.cuptiGetVersion_fn(&version);
        const bool supported_profile = version / 100 == 1300
#if CUPTI_API_VERSION >= 130200
                                       || version / 100 == 1302
#endif
            ;
        if (version_result != CUPTI_SUCCESS || !supported_profile) {
            throw std::runtime_error("CUPTI runtime/header ABI mismatch (runtime " + std::to_string(version)
                                     + ", compiled " + std::to_string(CUPTI_API_VERSION) + ")");
        }
        instance->runtime_version = version;
        Dl_info library_info{};
        dladdr(reinterpret_cast<void*>(instance->cupti.cuptiGetVersion_fn), &library_info);
        uint64_t       cupti_ns     = 0;
        const uint64_t before       = monotonicNs();
        const auto     clock_result = instance->cupti.cuptiGetTimestamp_fn(&cupti_ns);
        const uint64_t after        = monotonicNs();
        if (clock_result != CUPTI_SUCCESS) {
            throw std::runtime_error("CUPTI timestamp calibration failed");
        }
        std::ostringstream meta;
        meta
            << "{\"type\":\"metadata\",\"schema_version\":2,\"trace_id\":" << jsonQuote(instance->trace_id)
            << ",\"pid\":" << pid << ",\"rank\":" << rank << ",\"hostname\":" << jsonQuote(instance->hostname)
            << ",\"process_startticks\":\"" << instance->process_startticks << "\",\"cupti_version\":" << version
            << ",\"cupti_header_version\":" << CUPTI_API_VERSION
            << ",\"cupti_library\":" << jsonQuote(library_info.dli_fname ? library_info.dli_fname : "unknown")
            << ",\"monotonic_before_ns\":\"" << before << "\",\"cupti_ns\":\"" << cupti_ns
            << "\",\"monotonic_after_ns\":\"" << after << "\",\"wall_ns\":\"" << wallNs()
            << "\",\"preexisting_state\":\"CONTEXT/STREAM dump requested; earlier event records, graph construction and API history remain unknown\",\"api_key\":\"trace_id/domain/tid/correlation_id\""
            << ",\"activity_clock\":\"CUPTI\",\"host_clock\":\"CLOCK_MONOTONIC\",\"event_device_timestamps\":false"
            << ",\"kernel_latency_timestamps\":false,\"forced_flush\":false,\"critical_queue_capacity\":"
            << kCriticalCapacity << ",\"bulk_queue_capacity\":" << kBulkCapacity
            << ",\"open_capacity_per_kind\":" << kOpenCapacity << ",\"segment_bytes\":" << kFileBytes
            << ",\"active_state_source\":\"callback_slots\",\"active_snapshot_interval_ns\":\"1000000000\""
            << ",\"active_snapshot_semantics\":\"bounded non-atomic sweep; writer_seq=0 denotes live state independent of history delivery\""
            << ",\"api_filter\":\"copy/alloc/free/memset, launch, stream/event dependency and lifetime, context mutation, graph mutation; excludes getters and polling queries\""
            << ",\"callback_coverage\":\"selected callback enable marks plus optional_callback_unavailable count/list; unavailable means registration unsupported, not a GPU execution error or queue loss\""
            << ",\"history_orphan_semantics\":\"unmatched disk ENTER/EXIT is incomplete history, not current liveness\""
            << ",\"queue_classes\":\"critical=dependency APIs/activities and business scopes/marks; bulk=launch APIs and concurrent kernels\""
            << ",\"segment_count\":" << kFileCount << ",\"activity_pool_bytes\":" << kBufferCount * kBufferBytes
            << ",\"incident_slots\":1,\"incident_threshold_ns\":\"5000000000\",\"incident_requires_demand\":true"
            << ",\"nominal_segment_storage_bytes\":\"" << 2 * kFileCount * kFileBytes << '"'
            << ",\"retention_caveat\":\"ring may already have rolled before candidate detection; scope/API snapshots have separate bounded capacity\""
            << ",\"requested_activities\":[\"CONCURRENT_KERNEL\",\"MEMCPY\",\"MEMSET\",\"CONTEXT\",\"STREAM\",\"SYNCHRONIZATION\",\"CUDA_EVENT\"]"
            << ",\"activity_enable_results\":\"activity_enable mark records; enabled snapshot is true only after all succeeded\""
            << ",\"completeness\":\"callbacks cover selected attempted APIs; absent activity does not prove non-submission or GPU completion\"";
        instance->metadata = meta.str();
        instance->rotate();
        recorder.store(instance, std::memory_order_release);
        std::thread([instance] { instance->writer(); }).detach();
        auto result = instance->cupti.cuptiSubscribe_fn(&instance->subscriber, apiCallback, instance);
        if (result != CUPTI_SUCCESS) {
            instance->warning("cuptiSubscribe_failed_possible_profiler_conflict", result);
            throw std::runtime_error("cuptiSubscribe failed " + std::to_string(result)
                                     + "; profiler conflict or unsupported environment");
        }
        instance->subscribed.store(true, std::memory_order_release);
        result = instance->cupti.cuptiActivityRegisterCallbacks_fn(bufferRequested, bufferCompleted);
        if (result != CUPTI_SUCCESS) {
            instance->warning("activity_callback_registration_failed", result);
            throw std::runtime_error("activity callback registration failed");
        }
        const CUpti_ActivityKind activities[] = {CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
                                                 CUPTI_ACTIVITY_KIND_MEMCPY,
                                                 CUPTI_ACTIVITY_KIND_MEMSET,
                                                 CUPTI_ACTIVITY_KIND_CONTEXT,
                                                 CUPTI_ACTIVITY_KIND_STREAM,
                                                 CUPTI_ACTIVITY_KIND_SYNCHRONIZATION,
                                                 CUPTI_ACTIVITY_KIND_CUDA_EVENT};
        for (auto activity : activities) {
            const bool dump_existing =
                activity == CUPTI_ACTIVITY_KIND_CONTEXT || activity == CUPTI_ACTIVITY_KIND_STREAM;
            result               = dump_existing ? instance->cupti.cuptiActivityEnableAndDump_fn(activity) :
                                                   instance->cupti.cuptiActivityEnable_fn(activity);
            Record status        = businessRecord(Kind::Mark, "activity_enable");
            status.activity_kind = activity;
            status.value[0]      = dump_existing;
            status.result        = result;
            instance->emit(status);
            if (result != CUPTI_SUCCESS) {
                throw std::runtime_error("activity enable failed kind " + std::to_string(activity) + " result "
                                         + std::to_string(result));
            }
            instance->enabled_activities[instance->enabled_activity_count++] = activity;
        }
        for (auto domain : {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_CB_DOMAIN_RUNTIME_API}) {
            const uint32_t count    = domain == CUPTI_CB_DOMAIN_DRIVER_API ?
                                          static_cast<uint32_t>(CUPTI_DRIVER_TRACE_CBID_SIZE) :
                                          static_cast<uint32_t>(CUPTI_RUNTIME_TRACE_CBID_SIZE);
            uint64_t       selected = 0, unsupported = 0;
            for (uint32_t id = 1; id < count; ++id) {
                const char* name = nullptr;
                result           = instance->cupti.cuptiGetCallbackName_fn(domain, id, &name);
                if (result != CUPTI_SUCCESS || !name) {
                    if (requiredCallback(domain, id)) {
                        throw std::runtime_error("required callback name unavailable domain " + std::to_string(domain)
                                                 + " cbid " + std::to_string(id));
                    }
                    ++unsupported;  // Newer headers can contain IDs absent in an older supported runtime.
                    continue;
                }
                if (!trackedApi(name)) {
                    continue;
                }
                result = instance->cupti.cuptiEnableCallback_fn(1, instance->subscriber, domain, id);
                if (result != CUPTI_SUCCESS) {
                    if (result == CUPTI_ERROR_INVALID_PARAMETER && domain == CUPTI_CB_DOMAIN_DRIVER_API
                        && optionalNewCallback(name)) {
                        const auto slot = instance->optional_callback_unavailable.load(std::memory_order_relaxed);
                        if (slot == instance->unavailable_callbacks.size()) {
                            throw std::runtime_error("optional callback coverage table exhausted");
                        }
                        Record unavailable     = businessRecord(Kind::CallbackUnavailable, name);
                        unavailable.domain     = domain;
                        unavailable.cbid       = id;
                        unavailable.result     = result;
                        unavailable.record_seq = instance->sequence.fetch_add(1, std::memory_order_relaxed);
                        instance->unavailable_callbacks[slot] = unavailable;
                        instance->optional_callback_unavailable.store(slot + 1, std::memory_order_release);
                        instance->emit(unavailable);
                        continue;
                    }
                    throw std::runtime_error("selected callback enable failed domain " + std::to_string(domain)
                                             + " cbid " + std::to_string(id) + " name " + name + " result "
                                             + std::to_string(result));
                }
                ++selected;
            }
            if (!selected) {
                throw std::runtime_error("no selected callbacks enabled in domain " + std::to_string(domain));
            }
            Record status   = businessRecord(Kind::Mark, "selected_callbacks_enabled");
            status.domain   = domain;
            status.value[0] = selected;
            status.value[1] = unsupported;
            instance->emit(status);
        }
        if (!instance->writer_running.load(std::memory_order_acquire)) {
            throw std::runtime_error("CPU writer failed during CUPTI initialization");
        }
        instance->enabled.store(true, std::memory_order_release);
        if (!instance->writer_running.load(std::memory_order_acquire)) {
            instance->enabled.store(false, std::memory_order_release);
            throw std::runtime_error("CPU writer failed during recorder publication");
        }
        instance->warning("recording_started_preexisting_state_unknown");
        std::thread([instance] { instance->flushLoop(); }).detach();
        std::fprintf(stderr,
                     "CUDA_CALL_TRACE enabled rank=%d pid=%d prefix=%s (observational; instrumentation has overhead)\n",
                     rank,
                     pid,
                     instance->prefix.c_str());
    } catch (const std::exception& e) {
        if (instance) {
            instance->rollbackStartup();
            instance->warning("recording_start_failed");
            instance->failure(e.what());
            instance->writer_running.store(false);
        }
        std::fprintf(stderr, "CUDA_CALL_TRACE disabled: %s; no complete trace is available\n", e.what());
    } catch (...) {
        if (instance) {
            instance->rollbackStartup();
            instance->failure("initialization exception");
            instance->writer_running.store(false);
        }
        std::fprintf(stderr, "CUDA_CALL_TRACE disabled: initialization exception; no complete trace is available\n");
    }
}

bool active() noexcept {
    auto* instance = recorder.load(std::memory_order_acquire);
    return instance && instance->enabled.load(std::memory_order_relaxed) && instance->sameProcess();
}
bool ownsCupti() noexcept {
    auto* instance = recorder.load(std::memory_order_acquire);
    return instance && instance->sameProcess() && instance->subscribed.load(std::memory_order_acquire);
}
uint64_t newId() noexcept {
    return active() ? recorder.load(std::memory_order_relaxed)->ids.fetch_add(1, std::memory_order_relaxed) : 0;
}
Context capture() noexcept {
    return active() ? tls_context : Context{};
}
Scope::Scope(const char* name, int64_t request_id, uint64_t step, uint64_t value0, uint64_t value1) noexcept {
    if (!active()) {
        return;
    }
    previous_            = tls_context;
    id_                  = newId();
    tls_context.scope_id = id_;
    if (request_id != -1) {
        tls_context.request_id = request_id;
    }
    if (step != 0) {
        tls_context.step = step;
    }
    auto record            = businessRecord(Kind::ScopeEnter, name);
    record.parent_scope_id = previous_.scope_id;
    record.value[0]        = value0;
    record.value[1]        = value1;
    auto* instance         = recorder.load(std::memory_order_relaxed);
    record.record_seq      = instance->sequence.fetch_add(1, std::memory_order_relaxed);
    active_handle_         = instance->active_scopes.begin(record, id_);
    if (!active_handle_) {
        instance->active_state_overflow.fetch_add(1, std::memory_order_relaxed);
    }
    instance->emit(record);
}
Scope::~Scope() noexcept {
    if (!id_) {
        return;
    }
    auto* instance = recorder.load(std::memory_order_relaxed);
    if (instance && instance->sameProcess() && active_handle_) {
        if (!instance->active_scopes.close(active_handle_)) {
            instance->close_mismatches.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (active()) {
        auto record            = businessRecord(Kind::ScopeExit, "scope_exit");
        record.parent_scope_id = previous_.scope_id;
        recorder.load(std::memory_order_relaxed)->emit(record);
    }
    tls_context = previous_;
}
ContextGuard::ContextGuard(Context context) noexcept {
    if (active()) {
        previous_   = tls_context;
        tls_context = context;
        installed_  = true;
    }
}
ContextGuard::~ContextGuard() noexcept {
    if (installed_) {
        tls_context = previous_;
    }
}
void mark(const char* name, uint64_t value0, uint64_t value1, int64_t request_id) noexcept {
    if (!active()) {
        return;
    }
    auto record = businessRecord(Kind::Mark, name);
    if (request_id != -1) {
        record.request_id = request_id;
    }
    record.value[0] = value0;
    record.value[1] = value1;
    recorder.load(std::memory_order_relaxed)->emit(record);
}

}  // namespace rtp_llm::cuda_trace

#else
namespace rtp_llm::cuda_trace {
void start(int) noexcept {
    const char* value = std::getenv("RTP_LLM_GPU_STAGE_TRACE");
    if (value && std::strcmp(value, "1") == 0) {
        std::fprintf(stderr,
                     "CUDA_CALL_TRACE unavailable: build requires CUDA with supported CUPTI 13.0/13.2 headers\n");
    }
}
bool active() noexcept {
    return false;
}
bool ownsCupti() noexcept {
    return false;
}
uint64_t newId() noexcept {
    return 0;
}
Context capture() noexcept {
    return {};
}
Scope::Scope(const char*, int64_t, uint64_t, uint64_t, uint64_t) noexcept {}
Scope::~Scope() noexcept = default;
ContextGuard::ContextGuard(Context) noexcept {}
ContextGuard::~ContextGuard() noexcept = default;
void mark(const char*, uint64_t, uint64_t, int64_t) noexcept {}
}  // namespace rtp_llm::cuda_trace
#endif
