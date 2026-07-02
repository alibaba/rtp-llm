#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "torch/all.h"
#include <c10/core/Event.h>

#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

// Shared grammar-matcher mask/commit/spec-verify mechanics for GrammarLogitsProcessor and
// ReasoningGrammarLogitsProcessor. Callers hold their own mutex; methods suffixed Locked
// must be invoked under that lock.
class GrammarMaskCore {
public:
    // Per-row outcome of a single spec-verify bitmask fill. fillGrammarRowLocked never
    // returns an "allow-all" state (callers start each row pre-filled with kBitmaskAllowAll),
    // so there is intentionally no AllowAll member here.
    enum class RowState {
        Active,
        Finished,
        Terminated,
        Failed,
    };

    GrammarMaskCore(std::shared_ptr<RtpGrammarMatcher> matcher, int64_t eos_token_id);

    const std::shared_ptr<RtpGrammarMatcher>& matcher() const {
        return matcher_;
    }
    int64_t eosTokenId() const {
        return eos_token_id_;
    }

    int64_t acceptedTokenLen() const {
        return accepted_token_len_;
    }
    void setAcceptedTokenLen(int64_t len) {
        accepted_token_len_ = len;
    }

    bool    finished() const;
    bool    isTerminated() const;
    bool    isPassthroughForMask() const;
    int64_t numAcceptedTokens() const;
    int32_t vocabSize() const;
    void    markFinished();

    RtpGrammarMatcher::ReasonerSnapshot reasonerSnapshot() const;
    void                                restoreReasoner(const RtpGrammarMatcher::ReasonerSnapshot& snap);
    void                                rollback(int n);
    bool                                acceptToken(int32_t token_id);

    // Normal decode: cached GPU/CPU mask from current matcher state.
    void applyMaskLocked(const torch::Tensor& logits, ErrorInfo& out_err);

    // Commit path after sampling.
    void acceptCommittedLocked(const int32_t* tokens, size_t n, ErrorInfo& out_err);

    // Spec verify: full grammar-only walk (used by GrammarLogitsProcessor).
    int runSpecVerifyLocked(const SpecLogitsProcessorRequest& request, ErrorInfo& out_err);

    // Shared spec-verify scaffolding for both the grammar-only walk (above) and the
    // reasoning-grammar think-routed walk. Snapshots the reasoner, runs `walk` under
    // try/catch, then rolls back the provisional grammar accepts (`grammar_accepted_prefix`,
    // updated by `walk`) and restores the reasoner. Contract:
    //   - walk(grammar_accepted_prefix, walk_err) performs the per-offset fill+accept loop,
    //     returns the accept cap, advances the matcher, and increments grammar_accepted_prefix
    //     for each provisionally accepted grammar token. It may set walk_err for a non-exception
    //     failure (e.g. fillBitmask) and is responsible for any markFinished/force-eos it needs
    //     in that case.
    //   - On a thrown exception: matcher is finished, eos is forced into bitmask row 0, and
    //     GRAMMAR_VERIFY_EXCEPTION is returned via out_err (message prefixed with `who`).
    //   - Else if walk set walk_err: that error is propagated and 0 returned.
    //   - Else the cap from walk is returned.
    int runSpecVerifyGuarded(int32_t*    bitmask_cpu_out,
                             size_t      W,
                             const char* who,
                             const std::function<int(int& grammar_accepted_prefix, ErrorInfo& walk_err)>& walk,
                             ErrorInfo&                                                                   out_err);

    // Spec verify building blocks (used by ReasoningGrammarLogitsProcessor think routing).
    ErrorInfo preflightSpecRequest(const SpecLogitsProcessorRequest& request) const;
    ErrorInfo validateMatcherInvariantsLocked(const SpecLogitsProcessorRequest& request,
                                              const std::vector<int>&           extra_token_ids = {});
    RowState  fillGrammarRowLocked(int32_t* row, size_t W, size_t model_vocab_size);

    static void forceToken(const torch::Tensor& logits, int64_t token_id);
    static void maskToken(const torch::Tensor& logits, int64_t token_id);

private:
    enum class DeviceMaskMode {
        UNSET,
        NOOP,
        MASK,
        PASSTHROUGH,
        TERMINATED,
        FINISHED,
    };

    struct DeviceMaskState {
        DeviceMaskMode              mode      = DeviceMaskMode::UNSET;
        int64_t                     token_len = -1;
        c10::Device                 device    = c10::Device(c10::DeviceType::CPU);
        torch::Tensor               vocab_mask;
        int32_t                     grammar_vocab_size = 0;
        std::shared_ptr<c10::Event> mask_ready;
    };

    DeviceMaskState buildDeviceMaskStateLocked(const c10::Device& device, ErrorInfo& out_err);
    void            publishMaskToDevice(DeviceMaskState& state, torch::Tensor vocab_mask, const c10::Device& device);
    void            applyDeviceMaskState(const torch::Tensor& logits, const DeviceMaskState& state);

    std::shared_ptr<RtpGrammarMatcher> matcher_;
    int64_t                            eos_token_id_       = 0;
    int64_t                            accepted_token_len_ = 0;
    std::optional<c10::Device>         last_mask_device_;
    DeviceMaskState                    device_mask_state_{};
    torch::Tensor                      reusable_bitmask_cpu_;
    torch::Tensor                      reusable_vocab_mask_cpu_;
    int32_t                            reusable_mask_words_ = 0;
};

}  // namespace rtp_llm
