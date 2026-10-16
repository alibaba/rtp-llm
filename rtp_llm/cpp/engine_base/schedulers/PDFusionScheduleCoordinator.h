#pragma once
#include <array>
#include <cstdint>
#include <deque>
#include <string>
namespace rtp_llm {
struct PDFusionPreparedState {
    int64_t ready_prefill = 0, ready_decode = 0, input_tokens = 0;
    int64_t oldest_ready_us = 0, decode_unserved_us = 0;
    int64_t waiting = 0, loading = 0, kv_available = 0, stopped = 0;
    int64_t last_model_us = 0;
    int64_t draining      = 0;
};
enum class PDFusionPlan : int64_t {
    IDLE    = 0,
    PREFILL = 1,
    DECODE  = 2
};
struct PDFusionCadenceConfig {
    int64_t                      adaptive = 0, decode_window = 64, low_water_pct = 50, prefill_max_wait_ms = 0;
    int64_t                      min_decode_share_pct = 25;
    static PDFusionCadenceConfig fromEnv();
    void                         validate() const;
    bool usesCoordinator(const std::string& coord_mode, const std::string& scheduler_mode) const;
    // Adaptive mode has no legacy N: use a canonical value for the handshake.
    int64_t decodeSteps(const std::string& legacy_ratio) const;
    bool    operator==(const PDFusionCadenceConfig& other) const;
};
struct PDFusionCadenceDecision {
    std::array<int64_t, 4> decode_peak{};
    int64_t                low_decode_mask = 0, drain_mask = 0;
    int64_t                decode_debt_us = 0, last_model_us = 0;
    int64_t                debt_before_scaled = 0, debt_after_scaled = 0, charged_plan = 0, admit_before = 1;
    int64_t                admit_credit_before_scaled = 0, admit_credit_after_scaled = 0;
    const char*            reason = "idle";
};
class PDFusionGlobalCadence {
public:
    explicit PDFusionGlobalCadence(int64_t decode_steps, int rank_count = 4, PDFusionCadenceConfig config = {});
    PDFusionPlan                   choose(const std::array<PDFusionPreparedState, 4>& states);
    void                           finish(PDFusionPlan plan, int64_t committed_real_count);
    void                           finish(PDFusionPlan plan, const std::array<PDFusionPreparedState, 4>& committed);
    const PDFusionCadenceDecision& decision() const {
        return decision_;
    }
    int64_t decodeSincePrefill() const {
        return decode_since_prefill_;
    }
    // Gates only NEW admission. Existing loaded/loading reservations keep their
    // resources and remain eligible for the ordinary global Prefill policy.
    bool prefillAdmissionAllowed() const {
        return !config_.adaptive || !config_.min_decode_share_pct || admit_prefill_;
    }

private:
    int64_t                            decode_steps_, decode_since_prefill_;
    int                                rank_count_;
    PDFusionCadenceConfig              config_;
    PDFusionCadenceDecision            decision_;
    std::deque<std::array<int64_t, 4>> decode_history_;
    int64_t                            decode_debt_scaled_      = 0;
    int64_t                            admission_credit_scaled_ = 0;
    PDFusionPlan                       completed_plan_          = PDFusionPlan::IDLE;
    bool                               awaiting_model_cost_ = false, admit_prefill_ = true;
};
// Single-host two/four-rank dedicated control channel. Never call under a scheduler lock.
// Any exception requires whole-run failure, never fallback to local scheduling.
class PDFusionScheduleCoordinator {
public:
    enum class Phase : int64_t {
        PREPARE = 1,
        COMMIT  = 2
    };
    PDFusionScheduleCoordinator(std::string           run_id,
                                int                   rank,
                                int64_t               decode_steps,
                                int                   timeout_ms         = 30000,
                                int                   startup_timeout_ms = 300000,
                                int                   rank_count         = 4,
                                int                   mtp_tokens         = 0,
                                PDFusionCadenceConfig cadence_config     = {});
    ~PDFusionScheduleCoordinator();
    PDFusionScheduleCoordinator(const PDFusionScheduleCoordinator&)                    = delete;
    PDFusionScheduleCoordinator&         operator=(const PDFusionScheduleCoordinator&) = delete;
    std::array<PDFusionPreparedState, 4> exchange(int64_t epoch, Phase phase, const PDFusionPreparedState& state);
    void                                 abort() noexcept;

private:
    struct Packet {
        uint64_t              magic = 0x455044434f4f5237ULL;
        int64_t               rank = 0, epoch = 0, phase = 0, decode_steps = 0;
        int64_t               rank_count = 4, mtp_tokens = 0;
        PDFusionCadenceConfig cadence_config;
        char                  run_id[64] = {};
        PDFusionPreparedState state;
    };
    using Packets  = std::array<Packet, 4>;
    using Deadline = int64_t;
    void                  connectGroup(int startup_timeout_ms);
    Packet                packet(int64_t epoch, int64_t phase, const PDFusionPreparedState& state) const;
    void                  validate(const Packet& msg, int rank, int64_t epoch, int64_t phase) const;
    void                  sendPacket(int fd, const void* value, size_t bytes, Deadline deadline);
    void                  receivePacket(int fd, void* value, size_t bytes, Deadline deadline);
    void                  checkPeer(int fd);
    static Deadline       deadlineAfter(int timeout_ms);
    static void           waitReady(int fd, short events, Deadline deadline);
    std::string           run_id_;
    int                   rank_;
    int                   rank_count_, mtp_tokens_;
    int64_t               decode_steps_;
    PDFusionCadenceConfig cadence_config_;
    int                   timeout_ms_;
    int                   listener_ = -1;
    std::array<int, 4>    sockets_{{-1, -1, -1, -1}};
    int64_t               last_epoch_ = 0;
    Phase                 next_phase_ = Phase::PREPARE;
    bool                  failed_     = false;
};
}  // namespace rtp_llm
