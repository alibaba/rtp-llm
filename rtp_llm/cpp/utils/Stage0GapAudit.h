#pragma once

// Stage-0 inter-chunk gap diagnosis recorder (DSV41_STAGE0_GAP_AUDIT=off|cpu|cpu_gpu,
// default off). A bounded, silent, CPU-first span recorder at the pre-existing
// PPExecutor / admission / scheduler boundaries. When OFF: one predictable branch
// on a cached flag per probe — no sync, no allocation, no locks. When ON: fixed-size
// records append to a preallocated per-process ring; NO formatting/printing/JSON/
// file I/O/allocator growth/GPU value reads on the measured path; flush only outside
// measured blocks (request completion) via stage0_gap_flush().
//
// Identities are immutable and come from the admitted request/dispatch objects
// (request_id from plan.model_input.request_id, slot = current_slot_, generation =
// the slot's reuse count); chunk_id is the request's own prefill-chunk ordinal,
// maintained by the recorder per request_id (never forward_counter // 8).
//
// Clock: CLOCK_MONOTONIC via clock_gettime (C++) and time.monotonic_ns() (Python) —
// the same underlying Linux monotonic clock. No CPU<->GPU timestamp subtraction.
//
// Budget note: this recorder is the ONE recorder design for this task. It changes no
// wait/drain/admission/NCCL/result-pump behavior.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <unordered_map>
#include <sys/syscall.h>
#include <unistd.h>

namespace rtp_llm {

enum class Stage0GapEvent : uint32_t {
    PROCESS_ENTER = 0,
    PROCESS_EXIT,
    PLAN_READY,
    SLOT_WAIT_ENTER,
    SLOT_WAIT_EXIT,
    ADMIT_PREPARE_ENTER,
    ADMIT_PREPARE_EXIT,
    PY_PREPARE_ENTER,
    PY_PREPARE_EXIT,
    ADMIT_AGREE_ENTER,
    ADMIT_AGREE_EXIT,
    FRAME_RECV_ENTER,
    FRAME_RECV_EXIT,
    FORWARD_LAUNCH,
    FORWARD_LAUNCH_EXIT,
    FWD_DONE_WAIT_ENTER,
    FWD_DONE_WAIT_EXIT,
    COMMIT_DRAIN_ENTER,
    COMMIT_DRAIN_EXIT,
    COMPLETE_COMMIT,
    FRAME_SEND,
    RESULT_SEND,
    RESULT_RECV_ENTER,
    RESULT_RECV_EXIT,
    RETIRE,
    SCHED_SKIP_INFLIGHT,
    SCHED_SKIP_CREDIT,
    // cpu_gpu mode (plan §5): GPU-side stream markers + readiness, stage-0 only.
    GPU_FWD_SPAN,    // aux = µs of NEXT_HEAD(k)->MODEL_TAIL(k) on the model stream
    GPU_STREAM_GAP,  // aux = µs of MODEL_TAIL(k-1)->NEXT_HEAD(k) on the model stream
    GPU_QUERY,       // aux = anchor*8 + result (0=ready,1=not_ready,2=not_recorded,3=error)
    _COUNT
};

struct Stage0GapRecord {
    int64_t  ns;          // CLOCK_MONOTONIC nanoseconds
    int64_t  request_id;  // admitted request identity (-1 if unknown/empty round)
    int64_t  chunk_id;    // this request's prefill-chunk ordinal (-1 if none)
    int32_t  slot;        // current_slot_
    int32_t  generation;  // slot reuse generation
    uint32_t kind;        // Stage0GapEvent
    uint32_t aux;         // per-kind auxiliary (e.g. scheduler skip)
    int32_t  thread_lo;   // linux tid (low 32 bits)
    uint32_t stage;       // pp stage
};

// Session cap: DSV41_STAGE0_GAP_AUDIT=off|cpu|cpu_gpu (default off), read once.
// When off, nothing ever records regardless of the per-request latch.
inline int stage0_gap_session_cap() {
    static const int cap = [] {
        const char* v = std::getenv("DSV41_STAGE0_GAP_AUDIT");
        if (!v)
            return 0;
        if (std::strcmp(v, "cpu") == 0)
            return 1;
        if (std::strcmp(v, "cpu_gpu") == 0)
            return 2;
        return 0;
    }();
    return cap;
}

struct Stage0GapLatch {
    std::atomic<int64_t>  request_id{-1};
    std::atomic<int64_t>  chunk_id{-1};
    std::atomic<uint64_t> generation{0};
    std::atomic<int>      mode{0};  // 0 = off (default between requests)
};
inline Stage0GapLatch& stage0_gap_latch();

// Per-request mode map (repaired channel): the controller writes the frozen
// schedule's rid->mode map to DSV41_STAGE0_GAP_MODE_MAP (simple "rid mode" lines)
// BEFORE the session starts. Read ONCE (lazy, first use) and cached. This binds
// the mode to the native request id deterministically — no time-racy env read at
// the admission boundary (the next request's chunk-0 admission overlaps the
// previous request's drain, which is why the env channel leaked). Never read
// per chunk from a file at record time; the map is memory after the one load.
inline const std::unordered_map<int64_t, int>& stage0_gap_mode_map() {
    static const std::unordered_map<int64_t, int> map = [] {
        std::unordered_map<int64_t, int> m;
        const char*                      path = std::getenv("DSV41_STAGE0_GAP_MODE_MAP");
        if (!path || !*path)
            return m;
        FILE* f = std::fopen(path, "r");
        if (!f)
            return m;
        long long rid;
        int       mode;
        while (std::fscanf(f, "%lld %d", &rid, &mode) == 2)
            m[(int64_t)rid] = mode;
        std::fclose(f);
        return m;
    }();
    return map;
}

// Resolve the mode for a native request id: map first (deterministic), else the
// per-request env (fallback for non-schedule drivers). Min-ed with the cap.
inline int stage0_gap_mode_for_rid(int64_t rid) {
    int cap = stage0_gap_session_cap();
    if (cap == 0)
        return 0;
    const auto& m  = stage0_gap_mode_map();
    auto        it = m.find(rid);
    int         mode;
    if (it != m.end()) {
        mode = it->second;
    } else {
        const char* v = std::getenv("DSV41_STAGE0_GAP_AUDIT_NOW");
        mode          = 0;
        if (v) {
            if (std::strcmp(v, "cpu") == 0)
                mode = 1;
            else if (std::strcmp(v, "cpu_gpu") == 0)
                mode = 2;
        }
    }
    return mode < cap ? mode : cap;
}

// Legacy per-plan env read (kept for the pre-sync PLAN_READY gate only; the
// authoritative per-request binding is stage0_gap_mode_for_rid at post-sync).
inline int stage0_gap_read_request_mode() {
    int cap = stage0_gap_session_cap();
    if (cap == 0)
        return 0;
    const char* v = std::getenv("DSV41_STAGE0_GAP_AUDIT_NOW");
    int         m = 0;
    if (v) {
        if (std::strcmp(v, "cpu") == 0)
            m = 1;
        else if (std::strcmp(v, "cpu_gpu") == 0)
            m = 2;
    }
    return m < cap ? m : cap;
}

inline int stage0_gap_mode() {
    return stage0_gap_latch().mode.load(std::memory_order_relaxed);
}
inline bool stage0_gap_enabled() {
    return stage0_gap_mode() != 0;
}

struct Stage0GapState {
    static constexpr size_t RING = 1 << 16;  // 65536 records
    std::atomic<uint64_t>   write_idx{0};
    std::atomic<uint64_t>   overflow{0};
    Stage0GapRecord         ring[RING];

    // per-request chunk ordinal (batch-1 prefill; maintained by the recorder)
    std::atomic<int64_t>  active_request{-1};
    std::atomic<int64_t>  active_chunk{-1};
    std::atomic<uint64_t> request_generation{0};
};

inline Stage0GapState& stage0_gap_state() {
    static Stage0GapState state;
    return state;
}

inline int64_t stage0_gap_now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return int64_t(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

// Latched per-request mode/identity, set at admission by the controller and
// cleared after completion. Never changed mid-request.
inline Stage0GapLatch& stage0_gap_latch() {
    static Stage0GapLatch latch;
    return latch;
}

inline void stage0_gap_register_exit();

inline void stage0_gap_set_context(int64_t request_id, int64_t chunk_id, uint64_t generation, int mode) {
    auto& l = stage0_gap_latch();
    l.chunk_id.store(chunk_id, std::memory_order_relaxed);
    l.request_id.store(request_id, std::memory_order_relaxed);
    l.generation.store(generation, std::memory_order_relaxed);
    l.mode.store(mode, std::memory_order_relaxed);
}
inline void stage0_gap_clear_context() {
    stage0_gap_latch().request_id.store(-1, std::memory_order_relaxed);
}

inline void stage0_gap_record(Stage0GapEvent kind, int32_t slot, uint32_t aux, uint32_t stage) {
    if (!stage0_gap_enabled())
        return;
    stage0_gap_register_exit();  // one-time atexit registration (cheap static flag)
    auto&          st  = stage0_gap_state();
    auto&          l   = stage0_gap_latch();
    const int64_t  ns  = stage0_gap_now_ns();
    const uint64_t idx = st.write_idx.fetch_add(1, std::memory_order_relaxed);
    if (idx >= Stage0GapState::RING) {
        st.overflow.fetch_add(1, std::memory_order_relaxed);
        return;  // count overflow explicitly; never overwrite evidence
    }
    Stage0GapRecord* r = &st.ring[idx];
    r->ns              = ns;
    r->request_id      = l.request_id.load(std::memory_order_relaxed);
    r->chunk_id        = l.chunk_id.load(std::memory_order_relaxed);
    r->slot            = slot;
    r->generation      = static_cast<int32_t>(l.generation.load(std::memory_order_relaxed));
    r->kind            = static_cast<uint32_t>(kind);
    r->aux             = aux;
#if defined(__linux__) && defined(SYS_gettid)
    r->thread_lo = static_cast<int32_t>(::syscall(SYS_gettid));  // gettid, no allocation
#else
    r->thread_lo = static_cast<int32_t>(::getpid());  // portable fallback
#endif
    r->stage = stage;
}

// RAII span guard: records ENTER at construction, EXIT at destruction. Same call
// pattern as PPCadenceGuard. Off path = one predictable branch (x2 on the flag).
struct Stage0GapGuard {
    Stage0GapEvent enter_kind, exit_kind;
    int32_t        slot;
    uint32_t       aux, stage;
    bool           active;
    Stage0GapGuard(Stage0GapEvent e, Stage0GapEvent x, int32_t slot_, uint32_t aux_, uint32_t stage_):
        enter_kind(e), exit_kind(x), slot(slot_), aux(aux_), stage(stage_), active(stage0_gap_enabled()) {
        if (active)
            stage0_gap_record(enter_kind, slot, aux, stage);
    }
    ~Stage0GapGuard() {
        if (active)
            stage0_gap_record(exit_kind, slot, aux, stage);
    }
};

// Flush the accumulated ring to a binary file (raw Stage0GapRecord array).
// Called from NormalEngine::stop (engine stop, outside every measured block,
// before the teardown SIGTERM that skips atexit). Idempotent (writes once).
// The path comes from DSV41_STAGE0_GAP_AUDIT_DIR + pid.
inline void stage0_gap_flush() {
    if (stage0_gap_session_cap() == 0)
        return;
    const char* dir = std::getenv("DSV41_STAGE0_GAP_AUDIT_DIR");
    if (!dir || !*dir)
        return;
    auto&                    st = stage0_gap_state();
    static std::atomic<bool> flushed{false};
    if (flushed.exchange(true))
        return;
    const uint64_t n = std::min<uint64_t>(st.write_idx.load(std::memory_order_relaxed), Stage0GapState::RING);
    char           path[1024];
    std::snprintf(path, sizeof(path), "%s/stage0-gap-pid%d.bin", dir, ::getpid());
    FILE* f = std::fopen(path, "wb");
    if (!f)
        return;
    std::fwrite(&n, sizeof(n), 1, f);
    uint64_t ov = st.overflow.load(std::memory_order_relaxed);
    std::fwrite(&ov, sizeof(ov), 1, f);
    std::fwrite(st.ring, sizeof(Stage0GapRecord), n, f);
    std::fclose(f);
}

inline void stage0_gap_flush_at_exit() {
    stage0_gap_flush();
}

inline void stage0_gap_register_exit() {
    if (stage0_gap_session_cap() != 0) {
        static const bool registered = (std::atexit(stage0_gap_flush_at_exit), true);
        (void)registered;
    }
}

}  // namespace rtp_llm
