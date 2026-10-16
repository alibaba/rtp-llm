#include "rtp_llm/cpp/engine_base/schedulers/PDFusionScheduleCoordinator.h"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <poll.h>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
namespace rtp_llm {
namespace {
int64_t monotonicMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
[[noreturn]] void fail(const std::string& reason) {
    throw std::runtime_error("PDFUSION_COORD group failure: " + reason);
}
}  // namespace
PDFusionCadenceConfig PDFusionCadenceConfig::fromEnv() {
    auto read = [](const char* name, int64_t fallback) {
        const auto* value = std::getenv(name);
        if (!value) {
            return fallback;
        }
        const std::string text(value);
        if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
            fail(std::string("invalid ") + name);
        }
        try {
            return static_cast<int64_t>(std::stoll(text));
        } catch (const std::exception&) {
            fail(std::string("invalid ") + name);
        }
    };
    PDFusionCadenceConfig config;
    config.adaptive = read("PDFUSION_CADENCE_ADAPTIVE", 0);
    if (config.adaptive) {
        config.decode_window        = read("PDFUSION_CADENCE_DECODE_WINDOW", 64);
        config.low_water_pct        = read("PDFUSION_CADENCE_LOW_WATER_PCT", 50);
        config.prefill_max_wait_ms  = read("PDFUSION_CADENCE_PREFILL_MAX_WAIT_MS", 0);
        config.min_decode_share_pct = read("PDFUSION_CADENCE_MIN_DECODE_SHARE_PCT", 25);
    }
    config.validate();
    return config;
}
void PDFusionCadenceConfig::validate() const {
    if ((adaptive != 0 && adaptive != 1) || decode_window < 1 || decode_window > 1000000 || low_water_pct < 1
        || low_water_pct > 100 || prefill_max_wait_ms < 0 || prefill_max_wait_ms > 600000 || min_decode_share_pct < 0
        || min_decode_share_pct > 99) {
        fail("invalid adaptive cadence configuration");
    }
}
bool PDFusionCadenceConfig::usesCoordinator(const std::string& coord_mode, const std::string& scheduler_mode) const {
    validate();
    if (coord_mode != "off" && coord_mode != "cadence") {
        fail("unknown pdfusion_coord_mode [" + coord_mode + "]");
    }
    if (adaptive) {
        return true;
    }
    if (coord_mode == "cadence" && scheduler_mode != "ratio") {
        fail("legacy global cadence requires pdfusion_scheduler_mode=ratio");
    }
    return coord_mode == "cadence";
}
int64_t PDFusionCadenceConfig::decodeSteps(const std::string& legacy_ratio) const {
    if (adaptive) {
        return 1;
    }
    if (legacy_ratio.empty() || legacy_ratio.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("global cadence requires an integer decode_prefill_ratio >= 1");
    }
    const auto n = std::stoll(legacy_ratio);
    if (n < 1 || n > 1000000) {
        throw std::invalid_argument("global cadence ratio outside [1,1000000]");
    }
    return n;
}
bool PDFusionCadenceConfig::operator==(const PDFusionCadenceConfig& other) const {
    return adaptive == other.adaptive && decode_window == other.decode_window && low_water_pct == other.low_water_pct
           && prefill_max_wait_ms == other.prefill_max_wait_ms && min_decode_share_pct == other.min_decode_share_pct;
}
PDFusionGlobalCadence::PDFusionGlobalCadence(int64_t decode_steps, int rank_count, PDFusionCadenceConfig config):
    decode_steps_(config.adaptive ? 1 : decode_steps),
    decode_since_prefill_(decode_steps_),
    rank_count_(rank_count),
    config_(config) {
    config_.validate();
    if (rank_count < 1 || rank_count > 4) {
        fail("invalid cadence rank count");
    }
    if (decode_steps_ < 1 || decode_steps_ > 1000000) {
        fail("cadence must be in [1,1000000]");
    }
}
PDFusionPlan PDFusionGlobalCadence::choose(const std::array<PDFusionPreparedState, 4>& states) {
    decision_                            = {};
    decision_.debt_before_scaled         = decode_debt_scaled_;
    decision_.admit_before               = prefillAdmissionAllowed();
    decision_.admit_credit_before_scaled = admission_credit_scaled_;
    bool    prefill = false, decode = false;
    int64_t ready_mask = 0, oldest_ready_us = 0;
    for (int rank = 0; rank < rank_count_; ++rank) {
        const auto& state = states[rank];
        if (state.stopped) {
            fail("scheduler stopped");
        }
        if (state.draining != 0 && state.draining != 1) {
            fail("invalid draining state");
        }
        if (state.draining) {
            decision_.drain_mask |= 1 << rank;
        }
        if (state.last_model_us < 0 || state.last_model_us > 600000000) {
            fail("invalid completed model duration");
        }
        decision_.last_model_us = std::max(decision_.last_model_us, state.last_model_us);
        prefill |= state.ready_prefill > 0;
        decode |= state.ready_decode > 0;
        if (state.ready_prefill > 0) {
            ready_mask |= 1 << rank;
            oldest_ready_us = std::max(oldest_ready_us, state.oldest_ready_us);
        }
        if (config_.adaptive) {
            auto& peak = decision_.decode_peak[rank];
            if (decode_history_.empty()) {
                peak = state.ready_decode;
            } else {
                for (const auto& sample : decode_history_) {
                    peak = std::max(peak, sample[rank]);
                }
            }
            // Exact floor(peak * pct / 100), without overflowing the product.
            const auto low_water = (peak / 100) * config_.low_water_pct + (peak % 100) * config_.low_water_pct / 100;
            if (state.ready_decode == 0 || state.ready_decode <= low_water) {
                decision_.low_decode_mask |= 1 << rank;
            }
        }
    }
    if (config_.adaptive) {
        admission_credit_scaled_ = 0;
        if (config_.min_decode_share_pct && awaiting_model_cost_) {
            decision_.charged_plan = static_cast<int64_t>(completed_plan_);
            const auto cost        = decision_.last_model_us;
            if (completed_plan_ == PDFusionPlan::PREFILL) {
                decode_debt_scaled_ += cost * config_.min_decode_share_pct;
            } else {
                const auto credit = cost * (100 - config_.min_decode_share_pct);
                // At most one Decode quantum of credit: an idle/Decode-only
                // period must not finance another unbounded Prefill burst.
                decode_debt_scaled_ = std::max(decode_debt_scaled_ - credit, -credit);
                // Admission happens BEFORE the next PREPARE delivers the cost
                // of the Decode about to run. Allow one globally observed
                // Decode quantum of lookahead to hide that feedback delay.
                // Actual debt is still charged/repaid without approximation;
                // a slower/faster next round carries its real residual forward.
                admission_credit_scaled_ = credit;
            }
        }
        awaiting_model_cost_ = false;
        if (!decode || !config_.min_decode_share_pct) {
            decode_debt_scaled_      = 0;
            admission_credit_scaled_ = 0;
        }
        admit_prefill_                      = decode_debt_scaled_ <= admission_credit_scaled_;
        decision_.debt_after_scaled         = decode_debt_scaled_;
        decision_.admit_credit_after_scaled = admission_credit_scaled_;
        decision_.decode_debt_us = std::max<int64_t>(0, decode_debt_scaled_) / (100 - config_.min_decode_share_pct);
        if (ready_mask == (1 << rank_count_) - 1) {
            decision_.reason = "all_ready";
            return PDFusionPlan::PREFILL;
        }
        if (prefill && decision_.low_decode_mask) {
            decision_.reason = "decode_low";
            return PDFusionPlan::PREFILL;
        }
        if (prefill && decision_.drain_mask) {
            decision_.reason = "drain";
            return PDFusionPlan::PREFILL;
        }
        if (prefill && config_.prefill_max_wait_ms > 0 && oldest_ready_us >= config_.prefill_max_wait_ms * 1000) {
            decision_.reason = "prefill_timeout";
            return PDFusionPlan::PREFILL;
        }
        decision_.reason = decode ? (prefill ? "accumulate" : "decode_only") : "idle";
        return decode ? PDFusionPlan::DECODE : PDFusionPlan::IDLE;
    }
    if (prefill && (!decode || decode_since_prefill_ >= decode_steps_)) {
        decision_.reason = "ratio";
        return PDFusionPlan::PREFILL;
    }
    decision_.reason = decode ? "ratio" : "idle";
    return decode ? PDFusionPlan::DECODE : PDFusionPlan::IDLE;
}
void PDFusionGlobalCadence::finish(PDFusionPlan plan, const std::array<PDFusionPreparedState, 4>& committed) {
    int64_t                total = 0;
    std::array<int64_t, 4> sample{};
    for (int rank = 0; rank < rank_count_; ++rank) {
        const auto& state = committed[rank];
        if (state.ready_prefill < 0 || state.ready_decode < 0 || (plan != PDFusionPlan::PREFILL && state.ready_prefill)
            || (plan != PDFusionPlan::DECODE && state.ready_decode)) {
            fail("committed count violates plan");
        }
        total += state.ready_prefill + state.ready_decode;
        sample[rank] = state.ready_decode;
    }
    finish(plan, total);
    if (config_.adaptive && plan == PDFusionPlan::DECODE && total > 0) {
        decode_history_.push_back(sample);
        if (decode_history_.size() > static_cast<size_t>(config_.decode_window)) {
            decode_history_.pop_front();
        }
    }
}
void PDFusionGlobalCadence::finish(PDFusionPlan plan, int64_t count) {
    if (count < 0 || (plan == PDFusionPlan::IDLE && count != 0)) {
        fail("invalid committed count");
    }
    if (!count) {
        return;
    }
    if (config_.adaptive && config_.min_decode_share_pct) {
        completed_plan_      = plan;
        awaiting_model_cost_ = true;
        if (plan == PDFusionPlan::PREFILL) {
            // All ranks learn the SAME execution cost in the next PREPARE.
            // Stop admitting more work immediately; do not use local clocks
            // for a distributed admission decision.
            admit_prefill_           = false;
            admission_credit_scaled_ = 0;
        }
    }
    if (plan == PDFusionPlan::PREFILL) {
        decode_since_prefill_ = 0;
    } else if (plan == PDFusionPlan::DECODE) {
        decode_since_prefill_ = std::min(decode_since_prefill_ + 1, decode_steps_);
    } else {
        fail("unknown plan");
    }
}
PDFusionScheduleCoordinator::Deadline PDFusionScheduleCoordinator::deadlineAfter(int timeout_ms) {
    return monotonicMs() + timeout_ms;
}
void PDFusionScheduleCoordinator::waitReady(int fd, short events, Deadline deadline) {
    for (;;) {
        const auto left = deadline - monotonicMs();
        if (left <= 0) {
            fail("control deadline exceeded");
        }
        pollfd    item{fd, events, 0};
        const int result = ::poll(&item, 1, static_cast<int>(left));
        if (result < 0 && errno == EINTR) {
            continue;
        }
        if (result <= 0) {
            fail(result == 0 ? "control deadline exceeded" : "poll failed");
        }
        if (item.revents & (POLLERR | POLLNVAL)) {
            fail("peer socket error");
        }
        if (item.revents & events) {
            return;
        }
        if (item.revents & POLLHUP) {
            fail("peer disconnected");
        }
    }
}
void PDFusionScheduleCoordinator::sendPacket(int fd, const void* value, size_t bytes, Deadline deadline) {
    for (;;) {
        waitReady(fd, POLLOUT, deadline);
        const auto n = ::send(fd, value, bytes, MSG_NOSIGNAL | MSG_DONTWAIT);
        if (n < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
            continue;
        }
        if (n != static_cast<ssize_t>(bytes)) {
            fail("control send failed");
        }
        return;
    }
}
void PDFusionScheduleCoordinator::receivePacket(int fd, void* value, size_t bytes, Deadline deadline) {
    for (;;) {
        waitReady(fd, POLLIN, deadline);
        const auto n = ::recv(fd, value, bytes, MSG_DONTWAIT | MSG_TRUNC);
        if (n < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
            continue;
        }
        if (n != static_cast<ssize_t>(bytes)) {
            fail(n == 0 ? "peer EOF" : "invalid control packet size");
        }
        return;
    }
}
void PDFusionScheduleCoordinator::checkPeer(int fd) {
    ucred     peer{};
    socklen_t size = sizeof(peer);
    if (::getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &peer, &size) != 0 || peer.uid != ::geteuid()) {
        fail("control peer uid mismatch");
    }
}
PDFusionScheduleCoordinator::Packet
PDFusionScheduleCoordinator::packet(int64_t epoch, int64_t phase, const PDFusionPreparedState& state) const {
    Packet msg;
    msg.rank           = rank_;
    msg.epoch          = epoch;
    msg.phase          = phase;
    msg.decode_steps   = decode_steps_;
    msg.rank_count     = rank_count_;
    msg.mtp_tokens     = mtp_tokens_;
    msg.cadence_config = cadence_config_;
    std::memcpy(msg.run_id, run_id_.c_str(), run_id_.size() + 1);
    msg.state = state;
    return msg;
}
void PDFusionScheduleCoordinator::validate(const Packet& msg, int rank, int64_t epoch, int64_t phase) const {
    if (msg.magic != Packet{}.magic || msg.rank != rank || msg.epoch != epoch || msg.phase != phase
        || msg.decode_steps != decode_steps_ || msg.rank_count != rank_count_ || msg.mtp_tokens != mtp_tokens_
        || !(msg.cadence_config == cadence_config_) || std::memcmp(msg.run_id, packet(0, 0, {}).run_id, 64) != 0) {
        fail("run/rank/epoch/phase/config mismatch");
    }
    const auto& s = msg.state;
    if (s.ready_prefill < 0 || s.ready_decode < 0 || s.input_tokens < 0 || s.oldest_ready_us < 0
        || s.decode_unserved_us < 0 || s.waiting < 0 || s.loading < 0 || s.kv_available < 0 || s.last_model_us < 0
        || s.last_model_us > 600000000 || (s.stopped != 0 && s.stopped != 1)) {
        fail("invalid prepared state");
    }
}
PDFusionScheduleCoordinator::PDFusionScheduleCoordinator(std::string           run_id,
                                                         int                   rank,
                                                         int64_t               decode_steps,
                                                         int                   timeout_ms,
                                                         int                   startup_timeout_ms,
                                                         int                   rank_count,
                                                         int                   mtp_tokens,
                                                         PDFusionCadenceConfig cadence_config):
    run_id_(std::move(run_id)),
    rank_(rank),
    rank_count_(rank_count),
    mtp_tokens_(mtp_tokens),
    decode_steps_(cadence_config.adaptive ? 1 : decode_steps),
    cadence_config_(cadence_config),
    timeout_ms_(timeout_ms) {
    cadence_config_.validate();
    if ((rank_count != 2 && rank_count != 4) || rank < 0 || rank >= rank_count || mtp_tokens < 0 || mtp_tokens > 4
        || run_id_.empty() || run_id_.size() > 63
        || run_id_.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")
               != std::string::npos
        || decode_steps_ < 1 || decode_steps_ > 1000000 || timeout_ms < 1 || timeout_ms > 600000
        || startup_timeout_ms < 1 || startup_timeout_ms > 600000) {
        fail("invalid channel configuration");
    }
    try {
        connectGroup(startup_timeout_ms);
    } catch (...) {
        abort();
        throw;
    }
}
PDFusionScheduleCoordinator::~PDFusionScheduleCoordinator() {
    abort();
}
void PDFusionScheduleCoordinator::abort() noexcept {
    failed_ = true;
    for (auto& fd : sockets_) {
        if (fd >= 0) {
            ::shutdown(fd, SHUT_RDWR);
            ::close(fd);
            fd = -1;
        }
    }
    if (listener_ >= 0) {
        ::close(listener_);
        listener_ = -1;
    }
}
void PDFusionScheduleCoordinator::connectGroup(int startup_timeout_ms) {
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    const auto name    = "rtp-pdfusion-" + std::to_string(::geteuid()) + "-" + run_id_;
    if (name.size() + 1 > sizeof(address.sun_path)) {
        fail("control address too long");
    }
    std::memcpy(address.sun_path + 1, name.data(), name.size());
    const auto size     = static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + 1 + name.size());
    const auto deadline = deadlineAfter(startup_timeout_ms);
    if (rank_ == 0) {
        listener_ = ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
        if (listener_ < 0 || ::bind(listener_, reinterpret_cast<sockaddr*>(&address), size) != 0
            || ::listen(listener_, rank_count_) != 0) {
            fail("leader bind/listen failed; run identity must be unique");
        }
        for (int count = 0; count < rank_count_ - 1; ++count) {
            waitReady(listener_, POLLIN, deadline);
            const int fd = ::accept4(listener_, nullptr, nullptr, SOCK_CLOEXEC | SOCK_NONBLOCK);
            if (fd < 0) {
                fail("accept failed");
            }
            try {
                checkPeer(fd);
                Packet hello;
                receivePacket(fd, &hello, sizeof(hello), deadline);
                if (hello.rank < 1 || hello.rank >= rank_count_ || sockets_[hello.rank] >= 0) {
                    fail("invalid or duplicate rank");
                }
                validate(hello, hello.rank, 0, 0);
                sockets_[hello.rank] = fd;
            } catch (...) {
                ::close(fd);
                throw;
            }
        }
        const auto ack = packet(0, 0, {});
        for (int rank = 1; rank < rank_count_; ++rank) {
            sendPacket(sockets_[rank], &ack, sizeof(ack), deadline);
        }
    } else {
        for (;;) {
            if (monotonicMs() >= deadline) {
                fail("startup connect deadline exceeded");
            }
            sockets_[0] = ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
            if (sockets_[0] < 0) {
                fail("peer socket failed");
            }
            if (::connect(sockets_[0], reinterpret_cast<sockaddr*>(&address), size) == 0) {
                break;
            }
            const int error = errno;
            ::close(sockets_[0]);
            sockets_[0] = -1;
            if (error != ECONNREFUSED && error != ENOENT && error != EAGAIN && error != EINTR) {
                fail("peer connect failed");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        checkPeer(sockets_[0]);
        const auto hello = packet(0, 0, {});
        sendPacket(sockets_[0], &hello, sizeof(hello), deadline);
        Packet ack;
        receivePacket(sockets_[0], &ack, sizeof(ack), deadline);
        validate(ack, 0, 0, 0);
    }
}
std::array<PDFusionPreparedState, 4>
PDFusionScheduleCoordinator::exchange(int64_t epoch, Phase phase, const PDFusionPreparedState& state) {
    try {
        const int64_t expected_epoch = next_phase_ == Phase::PREPARE ? last_epoch_ + 1 : last_epoch_;
        if (failed_ || phase != next_phase_ || epoch != expected_epoch) {
            fail("local control sequence mismatch");
        }
        const auto deadline = deadlineAfter(timeout_ms_);
        Packets    messages{};
        messages[rank_] = packet(epoch, static_cast<int64_t>(phase), state);
        validate(messages[rank_], rank_, epoch, static_cast<int64_t>(phase));
        if (rank_ == 0) {
            for (int rank = 1; rank < rank_count_; ++rank) {
                receivePacket(sockets_[rank], &messages[rank], sizeof(Packet), deadline);
                validate(messages[rank], rank, epoch, static_cast<int64_t>(phase));
            }
            for (int rank = 1; rank < rank_count_; ++rank) {
                sendPacket(sockets_[rank], &messages, sizeof(messages), deadline);
            }
        } else {
            sendPacket(sockets_[0], &messages[rank_], sizeof(Packet), deadline);
            receivePacket(sockets_[0], &messages, sizeof(messages), deadline);
        }
        std::array<PDFusionPreparedState, 4> result{};
        for (int rank = 0; rank < rank_count_; ++rank) {
            validate(messages[rank], rank, epoch, static_cast<int64_t>(phase));
            if (messages[rank].state.stopped) {
                fail("peer scheduler stopped");
            }
            result[rank] = messages[rank].state;
        }
        last_epoch_ = epoch;
        next_phase_ = phase == Phase::PREPARE ? Phase::COMMIT : Phase::PREPARE;
        return result;
    } catch (...) {
        abort();
        throw;
    }
}
}  // namespace rtp_llm
