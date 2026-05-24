#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

// Capability mixin for LogitsProcessor backends whose internal state must
// travel from a Prefill node to a Decode node in the PD-disaggregated path.
//
// Two carry-over strategies are supported in the snapshot; backends fill
// whichever fits:
//   (A) accepted_tokens path  — decode side replays tokens through the
//                                backend's normal AcceptToken/feed_token
//                                interface (xgrammar's current behaviour)
//   (B) opaque_blob path      — decode side deserialises a backend-specific
//                                binary blob (e.g. some llguidance versions)
//
// Backends are discovered by RPC code via dynamic_pointer_cast on the
// stream's logits processor list; per-snapshot dispatch on the decode side is
// keyed by pdProcessorKind().
class PdTransferableLogitsProcessor {
public:
    virtual ~PdTransferableLogitsProcessor() = default;

    struct PdSnapshot {
        std::string          kind;             // backend tag, e.g. "xgrammar"
        std::string          version;          // backend snapshot version string
        std::vector<int32_t> accepted_tokens;  // path (A)
        int64_t              consumed_seq_len = 0;
        std::string          opaque_blob;      // path (B), optional
    };

    // Backend type tag — decode side routes a snapshot to the processor whose
    // pdProcessorKind() matches snapshot.kind.
    //
    // INVARIANT: pdProcessorKind() must be UNIQUE across all PdTransferable
    // processors attached to a single GenerateStream. The decode-side
    // dispatcher walks the stream's processor list and dispatches each
    // PdSnapshot to the FIRST processor whose kind matches, then breaks.
    // Two processors with the same kind would cause the second one to be
    // silently skipped, leaving its state un-restored.
    virtual std::string pdProcessorKind() const = 0;

    // Snapshot version string used by the decode side to validate
    // compatibility before applying the snapshot.
    virtual std::string pdSnapshotVersion() const = 0;

    // Prefill-side export. exclude_last_token=true drops the trailing accepted
    // token because the first decode token is carried separately as
    // first_generate_token_id in the request.
    virtual PdSnapshot exportPdSnapshot(bool exclude_last_token) const = 0;

    // Decode-side restore. Called after stream creation, before the first step.
    virtual void restorePdSnapshot(const PdSnapshot& snapshot) = 0;
};

using PdTransferableLogitsProcessorPtr = std::shared_ptr<PdTransferableLogitsProcessor>;

}  // namespace rtp_llm
