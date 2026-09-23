#include "rtp_llm/cpp/utils/CudacoreFlightRecorder.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>

namespace rtp_llm {
namespace {

constexpr size_t      kCapacity = kCudacoreFlightCapacity;
std::atomic<uint64_t> recording_failures{0};

struct StoredEvent {
    CudacoreFlightEvent event;
    uint64_t            sequence{0};
    uint64_t            monotonic_ns{0};
    int64_t             thread_id{0};
};

struct Recorder {
    std::mutex                                          mutex;
    std::array<StoredEvent, kCapacity>                  events{};
    uint64_t                                            next_sequence{1};
    std::array<StoredEvent, kCudacorePostFaultCapacity> post_fault_events{};
    uint64_t                                            freeze_sequence{0};
    bool                                                frozen{false};
};

Recorder& recorder() {
    static Recorder* value = new Recorder();
    return *value;
}

bool enabled() {
    static const bool value = [] {
        const char* setting = std::getenv("RTP_LLM_CUDACORE_FLIGHT_RECORDER");
        return setting == nullptr || setting[0] != '0' || setting[1] != '\0';
    }();
    return value;
}

uint64_t monotonicNs() {
    struct timespec ts = {};
    if (::clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
}

int64_t threadId() {
    static thread_local const int64_t tid = static_cast<int64_t>(::syscall(SYS_gettid));
    return tid;
}

const char* kindName(CudacoreFlightKind kind) {
    switch (kind) {
        case CudacoreFlightKind::BatchSubmit:
            return "batch_submit";
        case CudacoreFlightKind::BatchSubmitResult:
            return "batch_submit_result";
        case CudacoreFlightKind::BatchCompletion:
            return "batch_completion";
    }
    return "unknown";
}

std::string eventJson(const StoredEvent& stored) {
    const auto& e    = stored.event;
    std::string json = "{\"sequence\":" + std::to_string(stored.sequence);
    json += ",\"kind\":\"" + std::string(kindName(e.kind)) + "\"";
    json += ",\"copy_id\":" + std::to_string(e.copy_id);
    json += ",\"related_sequence\":" + std::to_string(e.related_sequence);
    json += ",\"monotonic_ns\":" + std::to_string(stored.monotonic_ns);
    json += ",\"thread_id\":" + std::to_string(stored.thread_id);
    json += ",\"device_index\":" + std::to_string(e.device_index);
    json += ",\"stream\":" + std::to_string(e.stream);
    json += ",\"tile_count\":" + std::to_string(e.tile_count);
    json += ",\"total_bytes\":" + std::to_string(e.total_bytes);
    json += ",\"first_dst\":" + std::to_string(e.first_dst);
    json += ",\"first_src\":" + std::to_string(e.first_src);
    json += ",\"first_bytes\":" + std::to_string(e.first_bytes);
    json += ",\"last_dst\":" + std::to_string(e.last_dst);
    json += ",\"last_src\":" + std::to_string(e.last_src);
    json += ",\"last_bytes\":" + std::to_string(e.last_bytes);
    json += ",\"cuda_error\":" + std::to_string(e.cuda_error);
    json += "}";
    return json;
}

}  // namespace

void prepareCudacoreFlightRecorder() noexcept {
    try {
        (void)enabled();
        (void)recorder();
    } catch (...) {}
}

uint64_t recordCudacoreFlightEvent(const CudacoreFlightEvent& event) noexcept {
    if (!enabled()) {
        return 0;
    }
    try {
        Recorder&                   state     = recorder();
        const auto                  timestamp = monotonicNs();
        const auto                  tid       = threadId();
        std::lock_guard<std::mutex> lock(state.mutex);
        const uint64_t              sequence = state.next_sequence++;
        StoredEvent&                stored =
            state.frozen ? state.post_fault_events[(sequence - state.freeze_sequence) % kCudacorePostFaultCapacity] :
                                          state.events[(sequence - 1) % kCapacity];
        stored.event        = event;
        stored.sequence     = sequence;
        stored.monotonic_ns = timestamp;
        stored.thread_id    = tid;
        return sequence;
    } catch (...) {
        recording_failures.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
}

void freezeCudacoreFlightRecorder() noexcept {
    try {
        Recorder&                   state = recorder();
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.frozen) {
            state.freeze_sequence = state.next_sequence;
            state.frozen          = true;
        }
    } catch (...) {}
}

std::string snapshotCudacoreFlightRecorderJson() noexcept {
    try {
        // Allocate before taking the writer lock; serialize only after releasing
        // it. A diagnostic reader must never make submitters wait on JSON or I/O.
        std::vector<StoredEvent> history(kCapacity);
        std::vector<StoredEvent> tail(kCudacorePostFaultCapacity);
        uint64_t                 end, freeze_sequence;
        bool                     frozen;
        {
            Recorder&                   state = recorder();
            std::lock_guard<std::mutex> lock(state.mutex);
            std::copy(state.events.begin(), state.events.end(), history.begin());
            std::copy(state.post_fault_events.begin(), state.post_fault_events.end(), tail.begin());
            end             = state.next_sequence;
            frozen          = state.frozen;
            freeze_sequence = state.freeze_sequence;
        }
        const uint64_t history_end = frozen ? freeze_sequence : end;
        const uint64_t begin       = history_end > kCapacity ? history_end - kCapacity : 1;
        const uint64_t tail_begin =
            frozen ?
                std::max(freeze_sequence, end > kCudacorePostFaultCapacity ? end - kCudacorePostFaultCapacity : 0) :
                end;
        std::string json = "{\"enabled\":" + std::string(enabled() ? "true" : "false");
        json += ",\"capacity\":" + std::to_string(kCapacity);
        json += ",\"post_fault_capacity\":" + std::to_string(kCudacorePostFaultCapacity);
        json += ",\"dropped\":" + std::to_string(recording_failures.load(std::memory_order_relaxed));
        json += ",\"overwritten\":" + std::to_string(begin - 1);
        json += ",\"post_fault_overwritten\":" + std::to_string(frozen ? tail_begin - freeze_sequence : 0);
        json += ",\"frozen\":" + std::string(frozen ? "true" : "false");
        json += ",\"freeze_sequence\":" + std::to_string(freeze_sequence);
        auto append_events =
            [&](const auto& records, uint64_t first_sequence, uint64_t last_sequence, uint64_t offset) {
                json += '[';
                bool first = true;
                for (uint64_t sequence = first_sequence; sequence < last_sequence; ++sequence) {
                    const StoredEvent& stored = records[(sequence - offset) % records.size()];
                    if (stored.sequence != sequence) {
                        continue;
                    }
                    if (!first) {
                        json += ',';
                    }
                    first = false;
                    json += eventJson(stored);
                }
                json += ']';
            };
        json += ",\"events\":";
        append_events(history, begin, history_end, 1);
        json += ",\"post_fault_events\":";
        append_events(tail, tail_begin, end, freeze_sequence);
        json += '}';
        return json;
    } catch (...) {
        return "null";
    }
}

namespace cudacore_test {
void resetFlightRecorder() noexcept {
    try {
        Recorder&                   state = recorder();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.events        = {};
        state.next_sequence = 1;
        recording_failures.store(0, std::memory_order_relaxed);
        state.post_fault_events = {};
        state.freeze_sequence   = 0;
        state.frozen            = false;
    } catch (...) {}
}
}  // namespace cudacore_test

}  // namespace rtp_llm
