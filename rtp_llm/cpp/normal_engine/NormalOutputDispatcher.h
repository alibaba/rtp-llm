#pragma once

#include <functional>
#include <vector>

#include <pybind11/pybind11.h>
#include <torch/all.h>
#include "absl/status/status.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/models/SampleInfos.h"

namespace py = pybind11;

namespace rtp_llm {

class NormalOutputDispatcher {
public:
    NormalOutputDispatcher() = default;
    // The triton kernel module is forwarded from the parent processor so call
    // sites that need it (currently none in this class — every grammar op
    // here is pure C++ accept_token) can adopt it without re-importing.
    // Kept around for symmetry with NormalBatchStreamProcessor and to leave
    // an obvious extension point if a future hot-path step needs the kernel.
    explicit NormalOutputDispatcher(py::module_ triton_bitmask_ops): triton_bitmask_ops_(std::move(triton_bitmask_ops)) {}

    // Drops the Python module ref under the GIL. The default pybind11
    // destructor would call Py_DECREF on whichever thread destroyed the
    // object — UB unless that thread happens to hold the GIL. close() is
    // idempotent so parents may invoke it explicitly before releasing this
    // object (NormalBatchStreamProcessor::close does exactly that).
    ~NormalOutputDispatcher();
    void close() noexcept;

    absl::Status dispatch(const StreamGroups& stream_groups, const MergedOutput& merge_outputs) const;

    void batchAcceptGrammarTokens(const StreamGroups&  stream_groups,
                                  const torch::Tensor& token_ids_cpu) const;

    // Generic batch-accept driver. Pure C++ on the hot path:
    //   * `extract_tokens(stream)` returns the tokens to feed for that stream
    //     (empty vector ⇒ skip the stream). The closure is called for every
    //     stream in iteration order, so per-stream offset counters maintained
    //     in the caller stay in lock-step with stream_groups.
    //   * For each stream that has both a matcher and a non-empty token list,
    //     the helper drives `matcher->acceptToken` in a tight loop and reports
    //     the parser-rejected case via stream->reportError. Streams that have
    //     finished are also marked done so the per-step bitmask collection
    //     skips them.
    //   * `log_prefix` is used verbatim in warning messages so logs stay
    //     grep-able per call site.
    //
    // This method does NOT acquire the GIL. The previous Python-backed
    // implementation took GIL for ~the entire batch via py::list assembly +
    // grammar_batch_ops.batch_accept_tokens; that GIL acquire is gone after
    // Phase 3.
    void invokeBatchAcceptTokens(const StreamGroups&                                              stream_groups,
                                 const std::function<std::vector<int32_t>(const GenerateStreamPtr&)>& extract_tokens,
                                 const char*                                                      log_prefix) const;

private:
    void dispatchSingleStream(GenerateStreamPtr    stream,
                              const MergedOutput&  merge_outputs,
                              int                  batch_idx_in,
                              int                  batch_idx_out,
                              int                  token_offset,
                              bool                 return_all_probs,
                              const torch::Tensor& new_tokens_all,
                              const torch::Tensor& token_ids_cpu,
                              const torch::Tensor& success_cpu) const;

    py::module_ triton_bitmask_ops_;  // currently unused on this class's hot path
};

}  // namespace rtp_llm
