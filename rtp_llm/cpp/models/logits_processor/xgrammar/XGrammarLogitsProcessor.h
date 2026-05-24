#pragma once

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/PdTransferableLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/xgrammar/XGrammarGpuState.h"

#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

struct XGrammarRuntimeState;

struct StreamXGrammarInfo {
    bool        active = false;
    bool        waiting_for_think_end = false;
    std::string grammar_kind;
    std::string canonical_schema;
    std::string schema_sha256;
    std::string tokenizer_fp;
    std::string cache_key;
    std::vector<int> end_think_token_ids;
    size_t      think_end_match_pos = 0;
    int64_t     consumed_seq_len = 0;
    bool        terminated = false;
    bool        dead = false;
    std::vector<int> accepted_tokens;

    StreamXGrammarInfo copy() const {
        return *this;
    }
};

class XGrammarLogitsProcessor: public BaseLogitsProcessor,
                                public SpecLogitsProcessor,
                                public PdTransferableLogitsProcessor {
public:
    XGrammarLogitsProcessor() = default;
    explicit XGrammarLogitsProcessor(std::vector<StreamXGrammarInfo> xgrammar_infos);
    ~XGrammarLogitsProcessor() override;

    static std::shared_ptr<XGrammarLogitsProcessor> fromGenerateInput(std::shared_ptr<GenerateInput> generate_input,
                                                                      int32_t                        num);

    // BaseLogitsProcessor
    void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override;
    void updateMultiSeqStatus(const std::vector<int>& src_batch_indices) override;
    void updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) override;

    // SpecLogitsProcessor (single-sequence; multi-seq returns false / cap = P)
    bool isSpecVerifyEligible() const override;
    int  tryAcceptAndFillBitmask(const int32_t* draft_tokens,
                                 int            propose_step,
                                 int32_t*       bitmask_cpu_out,
                                 size_t         bitmask_size_int32) override;

    // PdTransferableLogitsProcessor — wraps the existing token-replay path.
    std::string                          pdProcessorKind() const override;
    std::string                          pdSnapshotVersion() const override;
    PdTransferableLogitsProcessor::PdSnapshot
                                         exportPdSnapshot(bool exclude_last_token) const override;
    void                                 restorePdSnapshot(
        const PdTransferableLogitsProcessor::PdSnapshot& snapshot) override;

    // Legacy direct-replay entry points (kept for the existing PD wiring during
    // Stage A; the new envelope on the RPC layer routes through the PD mixin
    // methods above which delegate here).
    static const std::string& pdReplayStateVersion();
    std::vector<int>          exportPdReplayAcceptedTokens(size_t batch_idx, bool exclude_last_token) const;
    int64_t                   exportPdReplayConsumedSeqLen(size_t batch_idx, bool exclude_last_token) const;
    void                      restorePdReplayState(const std::vector<int>& accepted_tokens,
                                                   int64_t                 consumed_seq_len,
                                                   const std::string&      replay_state_version,
                                                   size_t                  batch_idx = 0);

    size_t size() const {
        return xgrammar_infos_.size();
    }

    const std::vector<StreamXGrammarInfo>& infosForTest() const {
        return xgrammar_infos_;
    }

private:
    std::vector<StreamXGrammarInfo> xgrammar_infos_;
    XGrammarDeviceState             device_state_;
    std::unique_ptr<XGrammarRuntimeState> runtime_state_;
};

using XGrammarLogitsProcessorPtr = std::shared_ptr<XGrammarLogitsProcessor>;

}  // namespace rtp_llm
