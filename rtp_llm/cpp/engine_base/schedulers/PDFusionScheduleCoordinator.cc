#include "rtp_llm/cpp/engine_base/schedulers/PDFusionScheduleCoordinator.h"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstring>
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
PDFusionGlobalCadence::PDFusionGlobalCadence(int64_t decode_steps):
    decode_steps_(decode_steps), decode_since_prefill_(decode_steps) {
    if (decode_steps < 1 || decode_steps > 1000000) {
        fail("cadence must be in [1,1000000]");
    }
}
PDFusionPlan PDFusionGlobalCadence::choose(const std::array<PDFusionPreparedState, 4>& states) const {
    bool prefill = false, decode = false;
    for (const auto& state : states) {
        if (state.stopped) {
            fail("scheduler stopped");
        }
        prefill |= state.ready_prefill > 0;
        decode |= state.ready_decode > 0;
    }
    if (prefill && (!decode || decode_since_prefill_ >= decode_steps_)) {
        return PDFusionPlan::PREFILL;
    }
    return decode ? PDFusionPlan::DECODE : PDFusionPlan::IDLE;
}
void PDFusionGlobalCadence::finish(PDFusionPlan plan, int64_t count) {
    if (count < 0 || (plan == PDFusionPlan::IDLE && count != 0)) {
        fail("invalid committed count");
    }
    if (!count) {
        return;
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
    msg.rank         = rank_;
    msg.epoch        = epoch;
    msg.phase        = phase;
    msg.decode_steps = decode_steps_;
    std::memcpy(msg.run_id, run_id_.c_str(), run_id_.size() + 1);
    msg.state = state;
    return msg;
}
void PDFusionScheduleCoordinator::validate(const Packet& msg, int rank, int64_t epoch, int64_t phase) const {
    if (msg.magic != Packet{}.magic || msg.rank != rank || msg.epoch != epoch || msg.phase != phase
        || msg.decode_steps != decode_steps_ || std::memcmp(msg.run_id, packet(0, 0, {}).run_id, 64) != 0) {
        fail("run/rank/epoch/phase/config mismatch");
    }
    const auto& s = msg.state;
    if (s.ready_prefill < 0 || s.ready_decode < 0 || s.input_tokens < 0 || s.oldest_ready_us < 0
        || s.decode_unserved_us < 0 || s.waiting < 0 || s.loading < 0 || s.kv_available < 0
        || (s.stopped != 0 && s.stopped != 1)) {
        fail("invalid prepared state");
    }
}
PDFusionScheduleCoordinator::PDFusionScheduleCoordinator(
    std::string run_id, int rank, int64_t decode_steps, int timeout_ms, int startup_timeout_ms):
    run_id_(std::move(run_id)), rank_(rank), decode_steps_(decode_steps), timeout_ms_(timeout_ms) {
    if (rank < 0 || rank >= 4 || run_id_.empty() || run_id_.size() > 63
        || run_id_.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")
               != std::string::npos
        || decode_steps < 1 || decode_steps > 1000000 || timeout_ms < 1 || timeout_ms > 600000 || startup_timeout_ms < 1
        || startup_timeout_ms > 600000) {
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
            || ::listen(listener_, 4) != 0) {
            fail("leader bind/listen failed; run identity must be unique");
        }
        for (int count = 0; count < 3; ++count) {
            waitReady(listener_, POLLIN, deadline);
            const int fd = ::accept4(listener_, nullptr, nullptr, SOCK_CLOEXEC | SOCK_NONBLOCK);
            if (fd < 0) {
                fail("accept failed");
            }
            try {
                checkPeer(fd);
                Packet hello;
                receivePacket(fd, &hello, sizeof(hello), deadline);
                if (hello.rank < 1 || hello.rank > 3 || sockets_[hello.rank] >= 0) {
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
        for (int rank = 1; rank < 4; ++rank) {
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
        Packets    messages;
        messages[rank_] = packet(epoch, static_cast<int64_t>(phase), state);
        validate(messages[rank_], rank_, epoch, static_cast<int64_t>(phase));
        if (rank_ == 0) {
            for (int rank = 1; rank < 4; ++rank) {
                receivePacket(sockets_[rank], &messages[rank], sizeof(Packet), deadline);
                validate(messages[rank], rank, epoch, static_cast<int64_t>(phase));
            }
            for (int rank = 1; rank < 4; ++rank) {
                sendPacket(sockets_[rank], &messages, sizeof(messages), deadline);
            }
        } else {
            sendPacket(sockets_[0], &messages[rank_], sizeof(Packet), deadline);
            receivePacket(sockets_[0], &messages, sizeof(messages), deadline);
        }
        std::array<PDFusionPreparedState, 4> result;
        for (int rank = 0; rank < 4; ++rank) {
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
