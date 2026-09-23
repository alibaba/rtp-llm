#include "rtp_llm/cpp/utils/CudacoreFlightRecorder.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <string>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>

namespace rtp_llm {
namespace {

constexpr size_t kCapacity = 256;

struct StoredEvent {
    CudacoreFlightEvent event;
    uint64_t            sequence{0};
    uint64_t            monotonic_ns{0};
    int64_t             thread_id{0};
};

struct Recorder {
    std::mutex                         mutex;
    std::array<StoredEvent, kCapacity> events{};
    uint64_t                           next_sequence{1};
    std::atomic<uint64_t>              dropped{0};
    bool                               frozen{false};
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
        Recorder&                    state = recorder();
        std::unique_lock<std::mutex> lock(state.mutex, std::try_to_lock);
        if (!lock.owns_lock()) {
            // Dropping a record is preferable to pausing a copy submission.
            state.dropped.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
        if (state.frozen) {
            return 0;
        }
        const uint64_t sequence = state.next_sequence++;
        StoredEvent&   stored   = state.events[(sequence - 1) % kCapacity];
        stored.event            = event;
        stored.sequence         = sequence;
        stored.monotonic_ns     = monotonicNs();
        stored.thread_id        = threadId();
        return sequence;
    } catch (...) {
        return 0;
    }
}

void freezeCudacoreFlightRecorder() noexcept {
    try {
        Recorder&                   state = recorder();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.frozen = true;
    } catch (...) {}
}

std::string snapshotCudacoreFlightRecorderJson() noexcept {
    try {
        Recorder&                   state = recorder();
        std::lock_guard<std::mutex> lock(state.mutex);
        const uint64_t              end   = state.next_sequence;
        const uint64_t              begin = end > kCapacity ? end - kCapacity : 1;
        std::string                 json  = "{\"enabled\":" + std::string(enabled() ? "true" : "false");
        json += ",\"capacity\":" + std::to_string(kCapacity);
        json += ",\"dropped\":" + std::to_string(state.dropped.load(std::memory_order_relaxed));
        json += ",\"events\":[";
        bool first = true;
        for (uint64_t sequence = begin; sequence < end; ++sequence) {
            const StoredEvent& stored = state.events[(sequence - 1) % kCapacity];
            if (stored.sequence != sequence) {
                continue;
            }
            if (!first) {
                json += ',';
            }
            first = false;
            json += eventJson(stored);
        }
        json += "]}";
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
        state.dropped.store(0, std::memory_order_relaxed);
        state.frozen = false;
    } catch (...) {}
}
}  // namespace cudacore_test

}  // namespace rtp_llm
