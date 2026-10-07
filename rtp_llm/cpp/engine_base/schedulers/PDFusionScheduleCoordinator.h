#pragma once
#include <array>
#include <cstdint>
#include <string>
namespace rtp_llm {
struct PDFusionPreparedState {
    int64_t ready_prefill = 0, ready_decode = 0, input_tokens = 0;
    int64_t oldest_ready_us = 0, decode_unserved_us = 0;
    int64_t waiting = 0, loading = 0, kv_available = 0, stopped = 0;
};
enum class PDFusionPlan : int64_t {
    IDLE    = 0,
    PREFILL = 1,
    DECODE  = 2
};
class PDFusionGlobalCadence {
public:
    explicit PDFusionGlobalCadence(int64_t decode_steps);
    PDFusionPlan choose(const std::array<PDFusionPreparedState, 4>& states) const;
    void         finish(PDFusionPlan plan, int64_t committed_real_count);
    int64_t      decodeSincePrefill() const {
        return decode_since_prefill_;
    }

private:
    int64_t decode_steps_, decode_since_prefill_;
};
// Single-host four-rank dedicated control channel. Never call under a scheduler lock.
// Any exception requires whole-run failure, never fallback to local scheduling.
class PDFusionScheduleCoordinator {
public:
    enum class Phase : int64_t {
        PREPARE = 1,
        COMMIT  = 2
    };
    PDFusionScheduleCoordinator(
        std::string run_id, int rank, int64_t decode_steps, int timeout_ms = 30000, int startup_timeout_ms = 300000);
    ~PDFusionScheduleCoordinator();
    PDFusionScheduleCoordinator(const PDFusionScheduleCoordinator&)                    = delete;
    PDFusionScheduleCoordinator&         operator=(const PDFusionScheduleCoordinator&) = delete;
    std::array<PDFusionPreparedState, 4> exchange(int64_t epoch, Phase phase, const PDFusionPreparedState& state);
    void                                 abort() noexcept;

private:
    struct Packet {
        uint64_t              magic = 0x455044434f4f5231ULL;
        int64_t               rank = 0, epoch = 0, phase = 0, decode_steps = 0;
        char                  run_id[64] = {};
        PDFusionPreparedState state;
    };
    using Packets  = std::array<Packet, 4>;
    using Deadline = int64_t;
    void               connectGroup(int startup_timeout_ms);
    Packet             packet(int64_t epoch, int64_t phase, const PDFusionPreparedState& state) const;
    void               validate(const Packet& msg, int rank, int64_t epoch, int64_t phase) const;
    void               sendPacket(int fd, const void* value, size_t bytes, Deadline deadline);
    void               receivePacket(int fd, void* value, size_t bytes, Deadline deadline);
    void               checkPeer(int fd);
    static Deadline    deadlineAfter(int timeout_ms);
    static void        waitReady(int fd, short events, Deadline deadline);
    std::string        run_id_;
    int                rank_;
    int64_t            decode_steps_;
    int                timeout_ms_;
    int                listener_ = -1;
    std::array<int, 4> sockets_{{-1, -1, -1, -1}};
    int64_t            last_epoch_ = 0;
    Phase              next_phase_ = Phase::PREPARE;
    bool               failed_     = false;
};
}  // namespace rtp_llm
