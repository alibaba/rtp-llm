#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <unordered_map>

#include <unistd.h>
#if defined(__linux__) || defined(__APPLE__)
#include <pthread.h>
#endif
#if defined(__linux__)
#include <sys/syscall.h>
#endif

namespace rtp_llm {

enum class AsyncProfileKind {
    Execution,
    Lifecycle,
    Wait,
    Derived
};

inline int64_t asyncProfileThreadId() noexcept {
#if defined(__linux__)
    return syscall(SYS_gettid);
#elif defined(__APPLE__)
    uint64_t tid = 0;
    pthread_threadid_np(nullptr, &tid);
    return static_cast<int64_t>(tid);
#else
    return 0;  // Unknown is preferable to a fabricated OS thread identity.
#endif
}

struct AsyncProfileRecord {
    const char*          stage{"unknown"};
    const char*          path{"disk_to_device"};
    const char*          literal_name{nullptr};
    int64_t              request_id{-1};
    size_t               begin{0};
    size_t               end{0};
    int64_t              rank{-1};
    uint64_t             transfer_id{0};
    int64_t              start_us{0};
    const char*          scope{nullptr};
    const char*          operation{nullptr};
    AsyncProfileKind     kind{AsyncProfileKind::Execution};
    int64_t              start_tid{0};
    std::array<char, 64> start_thread_name{};
    uint64_t             workflow_id{0};
    uint64_t             parent_transfer_id{0};
    uint64_t             related_workflow_id{0};
    int64_t              client_request_id{-1};
};

// Only the bounded worker buffer borrows names. Exported snapshots outlive the
// session and must own their metadata, including callers' temporary strings.
struct AsyncProfileMetadata {
    std::string      stage{"unknown"};
    std::string      path{"disk_to_device"};
    int64_t          request_id{-1};
    size_t           begin{0};
    size_t           end{0};
    int64_t          rank{-1};
    uint64_t         transfer_id{0};
    int64_t          start_us{0};
    std::string      scope;
    std::string      operation;
    AsyncProfileKind kind{AsyncProfileKind::Execution};
    int64_t          start_tid{0};
    std::string      start_thread_name;
    uint64_t         workflow_id{0};
    uint64_t         parent_transfer_id{0};
    uint64_t         related_workflow_id{0};
    int64_t          client_request_id{-1};

    AsyncProfileMetadata() = default;
    AsyncProfileMetadata(const AsyncProfileRecord& r):
        stage(r.stage),
        path(r.path),
        request_id(r.request_id),
        begin(r.begin),
        end(r.end),
        rank(r.rank),
        transfer_id(r.transfer_id),
        start_us(r.start_us),
        scope(r.scope ? r.scope : ""),
        operation(r.operation ? r.operation : ""),
        kind(r.kind),
        start_tid(r.start_tid),
        start_thread_name(r.start_thread_name.data(),
                          std::find(r.start_thread_name.begin(), r.start_thread_name.end(), '\0')
                              - r.start_thread_name.begin()),
        workflow_id(r.workflow_id),
        parent_transfer_id(r.parent_transfer_id),
        related_workflow_id(r.related_workflow_id),
        client_request_id(r.client_request_id) {}
};

struct AsyncProfileEvent {
    std::string          name;
    int64_t              start_us;
    int64_t              end_us;
    AsyncProfileMetadata record{};
    int64_t              pid{0};
    int64_t              end_tid{0};
    uint64_t             session_id{0};
    uint64_t             span_id{0};
    bool                 start_clipped{false};
    bool                 end_clipped{false};
    bool                 thread_mismatch{false};
};

// The owner allocates a bounded buffer once. Workers publish POD records without
// allocations, session mutexes or Kineto TLS; only the owner formats names.
class AsyncProfileSession {
public:
    static int64_t nowUs() noexcept {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
            .count();
    }
    static int64_t steadyNowUs() noexcept {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }
    using Record = AsyncProfileRecord;
    struct Entry {
        Record               record;
        int64_t              end_tid{0};  // Published by end_us's release store.
        std::atomic<int64_t> end_us{0};
        std::atomic<bool>    finishing{false};
        std::atomic<bool>    ready{false};
    };

    explicit AsyncProfileSession(int64_t start_us = nowUs(), size_t limit = 100000):
        start_us_(start_us),
        steady_start_us_(steadyNowUs()),
        limit_(limit),
        generation_(nextGeneration().fetch_add(1, std::memory_order_relaxed)),
        entries_(new Entry[limit]) {}

    static std::shared_ptr<AsyncProfileSession> start(size_t limit = 100000) noexcept {
        try {
            auto                        session = std::make_shared<AsyncProfileSession>(nowUs(), limit);
            std::lock_guard<std::mutex> lock(activationMutex());
            std::atomic_store(&active(), session);
            activeGeneration().store(session->generation_, std::memory_order_release);
            return session;
        } catch (...) {
            return {};
        }
    }
    static bool enabled() noexcept {
        return activeGeneration().load(std::memory_order_acquire) != 0;
    }
    static std::shared_ptr<AsyncProfileSession> current() noexcept {
        // Avoid libstdc++'s shared_ptr atomic lock while profiling is off.
        return enabled() ? std::atomic_load(&active()) : std::shared_ptr<AsyncProfileSession>{};
    }
    // Worker names are stable for a profiling session. Read once per thread and
    // session, while the thread is alive; keep the worker buffer allocation-free.
    const std::array<char, 64>& currentThreadName() const noexcept {
        struct CachedName {
            uint64_t             generation{0};
            std::array<char, 64> name{};
        };
        thread_local CachedName cached;
        if (cached.generation != generation_) {
            cached.name.fill('\0');
#if defined(__linux__) || defined(__APPLE__)
            if (pthread_getname_np(pthread_self(), cached.name.data(), cached.name.size()) != 0)
                cached.name.fill('\0');
#endif
            cached.name.back() = '\0';
            cached.generation  = generation_;
        }
        return cached.name;
    }
    static uint64_t currentGeneration() noexcept {
        return activeGeneration().load(std::memory_order_acquire);
    }
    uint64_t generation() const noexcept {
        return generation_;
    }

    // Bounded thread-local deduplication: no task registry, heap allocation or
    // locking on the transfer hot path. Links are owned by the capture buffer.
    static void taskLink(uint64_t    task,
                         uint64_t    workflow,
                         uint64_t    parent,
                         int64_t     request,
                         const char* operation = "transfer",
                         uint64_t    related   = 0) noexcept {
        const auto generation = currentGeneration();
        if (!generation)
            return;
        struct Key {
            uint64_t generation{}, task{}, workflow{}, parent{}, related{};
        };
        thread_local std::array<Key, 64> seen{};
        auto&                            key = seen[(task ^ related) % seen.size()];
        if (key.generation == generation && key.task == task && key.workflow == workflow && key.parent == parent
            && key.related == related)
            return;
        auto session = current();
        if (!session)
            return;
        Record r;
        r.stage               = related ? "workflow_dependency" : "task_link";
        r.path                = "association";
        r.kind                = AsyncProfileKind::Derived;
        r.transfer_id         = task;
        r.workflow_id         = workflow;
        r.parent_transfer_id  = parent;
        r.related_workflow_id = related;
        r.request_id          = request;
        r.operation           = operation;
        r.start_us            = session->toWallUs(steadyNowUs());
        if (auto entry = session->add(r)) {
            entry->end_us.store(r.start_us, std::memory_order_release);
            key = {session->generation_, task, workflow, parent, related};
        }
    }
    int64_t toWallUs(int64_t steady_us) const noexcept {
        return start_us_ + (steady_us - steady_start_us_);
    }

    Entry* add(Record record) noexcept {
        if (closed_.load(std::memory_order_acquire))
            return nullptr;
        if (saturated_.load(std::memory_order_relaxed)) {
            dropped_.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }
        auto index = size_.load(std::memory_order_relaxed);
        for (size_t attempt = 0; index < limit_ && attempt < 16; ++attempt) {
            if (size_.compare_exchange_weak(index, index + 1, std::memory_order_acq_rel)) {
                auto& entry  = entries_[index];
                entry.record = record;
                entry.ready.store(true, std::memory_order_release);
                return &entry;
            }
        }
        if (index >= limit_)
            saturated_.store(true, std::memory_order_relaxed);
        dropped_.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    // As with stage/path, this borrowed name must outlive the session.
    Entry* add(const char* literal_name, int64_t start_us) noexcept {
        Record r;
        r.literal_name = literal_name;
        r.start_us     = start_us;
        return add(r);
    }
    std::vector<AsyncProfileEvent> finish(int64_t stop_us = nowUs()) {
        if (closed_.exchange(true, std::memory_order_acq_rel))
            return {};
        {
            std::lock_guard<std::mutex> lock(activationMutex());
            auto                        expected = std::atomic_load(&active());
            if (expected.get() == this
                && std::atomic_compare_exchange_strong(&active(), &expected, std::shared_ptr<AsyncProfileSession>{})) {
                auto generation = generation_;
                activeGeneration().compare_exchange_strong(generation, 0, std::memory_order_acq_rel);
            }
        }
        const auto                     count = std::min(size_.load(std::memory_order_acquire), limit_);
        std::vector<AsyncProfileEvent> events;
        events.reserve(count + 1);
        auto dropped = dropped_.load(std::memory_order_relaxed);
        for (size_t index = 0; index < count; ++index) {
            const auto& entry = entries_[index];
            // A concurrent unpublished reservation is skipped, never waited on.
            if (!entry.ready.load(std::memory_order_acquire)) {
                ++dropped;
                continue;
            }
            const auto end         = entry.end_us.load(std::memory_order_acquire);
            const auto begin       = std::max(start_us_, entry.record.start_us);
            const auto clipped_end = end == 0 ? stop_us : std::min(end, stop_us);
            if (clipped_end >= begin) {
                auto name = formatName(entry.record);
                if (end == 0 || end > stop_us)
                    name += " incomplete=1";
                AsyncProfileEvent event{std::move(name), begin, clipped_end, entry.record};
                event.pid             = getpid();
                event.end_tid         = end == 0 ? 0 : entry.end_tid;
                event.session_id      = generation_;
                event.span_id         = index + 1;
                event.start_clipped   = entry.record.start_us < start_us_;
                event.end_clipped     = end == 0 || end > stop_us;
                event.thread_mismatch = end != 0 && entry.record.start_tid != event.end_tid;
                events.push_back(std::move(event));
            }
        }
        // Association records are metadata, never additional timeline ranges.
        // Resolve them on the owner so existing stage counts/semantics survive.
        std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> links;
        for (const auto& event : events) {
            if (event.record.stage == "task_link")
                links[event.record.transfer_id] = {event.record.workflow_id, event.record.parent_transfer_id};
        }
        for (auto& event : events) {
            auto found = links.find(event.record.transfer_id);
            if (!event.record.workflow_id && found != links.end()) {
                event.record.workflow_id        = found->second.first;
                event.record.parent_transfer_id = found->second.second;
            }
        }
        events.erase(std::remove_if(events.begin(),
                                    events.end(),
                                    [](const auto& event) { return event.record.stage == "task_link"; }),
                     events.end());
        if (dropped) {
            AsyncProfileEvent event{"cache.transfer.profiler.dropped(count=" + std::to_string(dropped) + ")",
                                    start_us_,
                                    std::max(start_us_, stop_us),
                                    {}};
            event.record.kind  = AsyncProfileKind::Derived;
            event.record.stage = "dropped";
            event.record.path  = "profiler";
            event.pid          = getpid();
            event.session_id   = generation_;
            event.span_id      = count + 1;
            events.push_back(std::move(event));
        }
        return events;
    }

private:
    static std::string formatName(const Record& r) {
        if (r.literal_name)
            return r.literal_name;
        auto name = std::string("cache.transfer.") + r.path + "." + r.stage + "(request_id="
                    + std::to_string(r.request_id) + ",batch=" + std::to_string(r.begin) + ":" + std::to_string(r.end)
                    + ",rank=" + std::to_string(r.rank) + ",transfer_id=" + std::to_string(r.transfer_id);
        if (r.scope)
            name += std::string(",scope=") + r.scope;
        if (r.operation)
            name += std::string(",operation=") + r.operation;
        return name + ")";
    }
    static std::shared_ptr<AsyncProfileSession>& active() {
        static std::shared_ptr<AsyncProfileSession> s;
        return s;
    }
    static std::mutex& activationMutex() {
        static std::mutex mutex;
        return mutex;
    }
    static std::atomic<uint64_t>& activeGeneration() {
        static std::atomic<uint64_t> g{0};
        return g;
    }
    static std::atomic<uint64_t>& nextGeneration() {
        static std::atomic<uint64_t> g{1};
        return g;
    }
    const int64_t            start_us_, steady_start_us_;
    const size_t             limit_;
    const uint64_t           generation_;
    std::unique_ptr<Entry[]> entries_;
    std::atomic<size_t>      size_{0};
    std::atomic<uint64_t>    dropped_{0};
    std::atomic<bool>        saturated_{false}, closed_{false};
};

class AsyncProfileSpan {
public:
    AsyncProfileSpan() = default;
    AsyncProfileSpan(const char*      stage,
                     int64_t          request_id,
                     size_t           begin,
                     size_t           end,
                     int64_t          start_us    = 0,
                     int64_t          rank        = -1,
                     uint64_t         transfer_id = 0,
                     const char*      path        = "disk_to_device",
                     AsyncProfileKind kind              = AsyncProfileKind::Execution,
                     int64_t          client_request_id = -1) noexcept:
        record_{stage, path, nullptr, request_id, begin, end, rank, transfer_id, start_us, nullptr, nullptr, kind},
        steady_start_us_(start_us == 0 ? AsyncProfileSession::steadyNowUs() : 0),
        captured_(false) {
        record_.client_request_id = client_request_id;
        capture();
    }
    static AsyncProfileSpan fromSteady(const char*      stage,
                                       int64_t          request_id,
                                       size_t           begin,
                                       size_t           end,
                                       int64_t          start,
                                       int64_t          rank,
                                       uint64_t         id,
                                       const char*      path,
                                       const char*      scope     = nullptr,
                                       const char*      operation = nullptr,
                                       AsyncProfileKind kind      = AsyncProfileKind::Execution) noexcept {
        AsyncProfileSpan span;
        span.record_          = {stage, path, nullptr, request_id, begin, end, rank, id, 0, scope, operation, kind};
        span.steady_start_us_ = start;
        span.captured_        = false;
        span.capture();
        return span;
    }
    void finish(int64_t end_us = 0) const noexcept {
        if (entry_)
            finishEntry(entry_, end_us == 0 ? owner_->toWallUs(AsyncProfileSession::steadyNowUs()) : end_us);
        else if (!captured_ && AsyncProfileSession::enabled()) {
            if (auto session = AsyncProfileSession::current()) {
                auto r = record_;
                if (r.start_us == 0)
                    r.start_us = session->toWallUs(steady_start_us_);
                finishEntry(session->add(r),
                            end_us == 0 ? session->toWallUs(AsyncProfileSession::steadyNowUs()) : end_us);
            }
        }
    }
    void finishAtSteady(int64_t end_us) const noexcept {
        if (entry_)
            finishEntry(entry_, owner_->toWallUs(end_us));
        else if (!captured_ && AsyncProfileSession::enabled()) {
            if (auto session = AsyncProfileSession::current()) {
                auto r = record_;
                if (r.start_us == 0)
                    r.start_us = session->toWallUs(steady_start_us_);
                finishEntry(session->add(r), session->toWallUs(end_us));
            }
        }
    }

private:
    void capture() noexcept {
        if (auto session = AsyncProfileSession::current()) {
            captured_         = true;
            record_.start_tid = asyncProfileThreadId();
            if (record_.kind == AsyncProfileKind::Execution)
                record_.start_thread_name = session->currentThreadName();
            if (record_.start_us == 0)
                record_.start_us = session->toWallUs(steady_start_us_);
            entry_ = session->add(record_);
            if (entry_)
                owner_ = std::move(session);
        }
    }
    static void finishEntry(AsyncProfileSession::Entry* entry, int64_t end_us) noexcept {
        if (entry && !entry->finishing.exchange(true, std::memory_order_acq_rel)) {
            entry->end_tid = asyncProfileThreadId();
            entry->end_us.store(end_us, std::memory_order_release);
        }
    }
    AsyncProfileSession::Record          record_;
    int64_t                              steady_start_us_{0};
    bool                                 captured_{true};
    std::shared_ptr<AsyncProfileSession> owner_;
    AsyncProfileSession::Entry*          entry_{nullptr};
};
// Owned by GenerateStream and observed under its existing mutex. This tracks
// engine acceptance through committed FINISHED, rather than HTTP/socket life.
class AsyncProfileRequest {
public:
    void begin(int64_t request, int64_t start_us, int64_t client_request = -1) noexcept {
        request_id_        = request;
        client_request_id_ = client_request;
        start_us_          = start_us > 0 ? start_us : AsyncProfileSession::nowUs();
        phase_start_us_    = start_us_;
        observe("waiting");
    }
    void observe(const char* phase, const char* terminal = nullptr) noexcept {
        if (finished_)
            return;
        const auto generation = AsyncProfileSession::currentGeneration();
        if (!terminal && std::string_view(phase_) == phase && (!generation || generation == generation_))
            return;
        const auto now     = AsyncProfileSession::nowUs();
        auto       session = AsyncProfileSession::current();
        if (session && generation_ != session->generation()) {
            generation_ = session->generation();
            root_       = makeSpan("request_lifetime", start_us_);
            phase_span_ = makeSpan(phase_, phase_start_us_);
        }
        if (!terminal && std::string_view(phase_) != phase) {
            phase_span_.finish(now);
            phase_          = phase;
            phase_start_us_ = now;
            phase_span_     = session ? makeSpan(phase_, now) : AsyncProfileSpan{};
        }
        if (terminal) {
            phase_span_.finish(now);
            root_.finish(now);
            if (session) {
                auto end = makeSpan(terminal, now);
                end.finish(now);
            }
            finished_ = true;
        }
    }
    ~AsyncProfileRequest() {
        if (start_us_ && !finished_)
            observe(phase_, "released_without_terminal");
    }

private:
    AsyncProfileSpan makeSpan(const char* stage, int64_t start) const noexcept {
        // Request/client IDs are immutable; stage strings must be literals.
        return AsyncProfileSpan(
            stage, request_id_, 0, 0, start, -1, 0, "request", AsyncProfileKind::Lifecycle, client_request_id_);
    }
    int64_t          request_id_{-1}, client_request_id_{-1}, start_us_{0}, phase_start_us_{0};
    uint64_t         generation_{0};
    const char*      phase_{"waiting"};
    bool             finished_{false};
    AsyncProfileSpan root_, phase_span_;
};
}  // namespace rtp_llm
