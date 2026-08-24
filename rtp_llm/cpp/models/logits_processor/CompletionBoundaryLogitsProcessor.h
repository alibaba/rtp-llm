#pragma once

#include <initializer_list>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <vector>

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"

namespace rtp_llm {

struct CompletionBoundarySpec {
    std::vector<int32_t> think_close_token_ids;
    std::vector<int32_t> response_open_token_ids;
    std::vector<int32_t> response_close_token_ids;
    std::vector<int32_t> tools_open_token_ids;
    std::vector<int32_t> tools_close_token_ids;
    std::vector<int32_t> message_close_token_ids;
    std::vector<int32_t> whitespace_token_ids;
    bool                 starts_in_think = false;

    bool hasStatefulCompletionFields() const;
    bool isStatefulCompletionGuard() const;
};

struct CompletionTokenSequenceState {
    std::vector<int32_t> token_ids;
    std::vector<size_t>  prefix_table;
    size_t               status = 0;

    CompletionTokenSequenceState() = default;
    explicit CompletionTokenSequenceState(std::vector<int32_t> tokens);

    bool advance(int32_t token_id);
    void reset();
};

struct CompletionBoundaryState {
    enum class Phase : uint8_t {
        LEGACY_BOUNDARY = 0,
        THINK_BODY,
        EXPECT_RESPONSE_OPEN,
        RESPONSE_BODY,
        AFTER_RESPONSE,
        TOOLS_BODY,
        EXPECT_MESSAGE_CLOSE,
        COMPLETE,
        INVALID,
    };

    std::vector<int32_t>         boundary_token_ids;
    std::vector<size_t>          prefix_table;
    size_t                       boundary_status       = 0;
    int32_t                      input_length          = 0;
    int32_t                      current_output_length = 0;
    bool                         is_beam_search        = false;
    Phase                        phase                 = Phase::LEGACY_BOUNDARY;
    CompletionTokenSequenceState think_close;
    CompletionTokenSequenceState response_open;
    CompletionTokenSequenceState response_close;
    CompletionTokenSequenceState tools_open;
    CompletionTokenSequenceState tools_close;
    CompletionTokenSequenceState message_close;
    std::unordered_set<int32_t>  whitespace_token_ids;
    bool                         response_has_content = false;
    bool                         tools_has_content    = false;
    bool                         tools_channel_opened = false;

    CompletionBoundaryState() = default;
    CompletionBoundaryState(std::vector<int32_t> boundary, int32_t input_length, bool is_beam_search);
    CompletionBoundaryState(CompletionBoundarySpec spec, int32_t input_length, bool is_beam_search);

    bool finished() const;
    bool shouldForceStop() const;
    void advance(int32_t token_id);

private:
    void advanceStatefulGuard(int32_t token_id);
    bool advanceUnexpectedBoundary(int32_t token_id,
                                   std::initializer_list<CompletionTokenSequenceState*> sequences);
    void resetStructuralSequences();
    void markInvalid();
    bool isWhitespace(int32_t token_id) const;
    static bool
    advancesBodyContent(const CompletionTokenSequenceState& sequence, size_t previous_status, bool completed);
};

// A compatibility guard for pre-tokenized chat RPCs that no longer carry the
// original tool schema. It leaves normal logits untouched and masks only EOS
// and single-token stop words until the declared chat state reaches a valid
// completion. Request-aware structural grammars remain the stronger path.
class CompletionBoundaryLogitsProcessor: public BaseLogitsProcessor, public SpecLogitsProcessor {
public:
    CompletionBoundaryLogitsProcessor(std::vector<CompletionBoundaryState> states,
                                      std::vector<int32_t>                 guarded_stop_token_ids);

    void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override;
    void updateMultiSeqStatus(const std::vector<int>& src_batch_indices) override;
    void updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) override;

    bool    isSpecVerifyEligible() const override;
    int     tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) override;
    bool    isStateful() const override;
    int64_t acceptedTokenLen() const override;

    std::vector<size_t> boundaryStatus() const;

private:
    void maskGuardedStops(const torch::Tensor& logits, size_t vocab_size) const;
    void forceGuardedStop(const torch::Tensor& logits, size_t vocab_size) const;

private:
    std::vector<CompletionBoundaryState> states_;
    std::vector<int32_t>                 guarded_stop_token_ids_;
    mutable std::mutex                   mutex_;
};

using CompletionBoundaryLogitsProcessorPtr = std::shared_ptr<CompletionBoundaryLogitsProcessor>;

}  // namespace rtp_llm
