#pragma once

#include <list>
#include <memory>

#include <pybind11/pybind11.h>
#include <torch/all.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/normal_engine/NormalModelInputGatherer.h"
#include "rtp_llm/cpp/normal_engine/NormalOutputDispatcher.h"
#include "rtp_llm/cpp/normal_engine/NormalSamplerInputGatherer.h"

namespace py = pybind11;

namespace rtp_llm {

class NormalBatchStreamProcessor {
public:
    NormalBatchStreamProcessor(const ModelConfig&                 model_config,
                               const PDSepConfig&                 pd_sep_config,
                               const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
                               const CacheConfig&                 cache_config,
                               bool                               warm_up);

    // Virtual so deletion through a base pointer (e.g. MtpBatchStreamProcessor
    // held as NormalBatchStreamProcessor*) runs the derived dtor chain. The
    // dtor body is close(): we release Python handles while the owning thread
    // is known to be the one running the engine teardown, instead of relying
    // on pybind11's default Py_DECREF-from-wherever-dtor-lands behaviour.
    virtual ~NormalBatchStreamProcessor();

    // Idempotent shutdown hook. Releases the triton-bitmask module ref (and
    // the copy held by output_dispatcher_) under the GIL when the interpreter
    // is still alive, or leaks the PyObject* when it has already finalized.
    // Callable any number of times; further calls are no-ops.
    void close() noexcept;

    virtual absl::Status dispatch(const StreamGroups& stream_groups, const MergedOutput& merge_outputs) const;
    virtual absl::StatusOr<GptModelInputs> gatherModelInput(const StreamGroups& stream_groups) const;
    virtual absl::StatusOr<SamplerInputs>  gatherSamplerInput(const StreamGroups&    stream_groups,
                                                              const GptModelInputs&  model_inputs,
                                                              const GptModelOutputs& model_output) const;

    void applyGrammarConstraints(SamplerInputs& inputs, const StreamGroups& stream_groups) const;

    // Returns true if callers must skip kernel dispatch this batch.
    // - kernel module present  → returns false (proceed normally)
    // - no Python interpreter  → returns true silently (cc_test path)
    // - module missing in prod → marks every grammar-bearing stream errored
    //                            via reportError and returns true
    // Surfaces a missing-kernel failure to clients without throwing from
    // the inference hot path.
    bool reportGrammarUnavailableIfNeeded(const StreamGroups& stream_groups) const;

    // Exposes the underlying dispatcher so call sites outside dispatch() (e.g.
    // MtpExecutor's prefill bonus accept) can reuse batchAcceptGrammarTokens
    // and the shared invokeBatchAcceptTokens driver, instead of each adding
    // its own copy of the GIL/triples/error-report boilerplate.
    NormalOutputDispatcher* outputDispatcher() const { return output_dispatcher_.get(); }

protected:
    SamplerInputs allocateSamplerInputs(const StreamGroups& stream_groups,
                                        size_t              total_batch_size_in,
                                        size_t              total_batch_size_out,
                                        size_t              propose_step = 0) const;

    void setCommonSamplerInputs(SamplerInputs&                sampler_inputs,
                                std::list<GenerateStreamPtr>& all_streams,
                                bool                          score_batch  = false,
                                size_t                        propose_step = 0) const {
        fillSamplerCommonInputs(sampler_inputs, all_streams, score_batch, propose_step);
    }

    void fillSamplerCommonInputs(SamplerInputs&                sampler_inputs,
                                 std::list<GenerateStreamPtr>& all_streams,
                                 bool                          score_batch  = false,
                                 size_t                        propose_step = 0) const;

    void setLogitsProcessorInputs(SamplerInputs&                sampler_inputs,
                                  std::list<GenerateStreamPtr>& all_streams,
                                  bool                          score_batch = false) const;

protected:
    NormalModelInputGathererConfig              model_input_gatherer_config_;
    std::unique_ptr<NormalModelInputGatherer>   model_input_gatherer_;
    std::unique_ptr<NormalSamplerInputGatherer> sampler_input_gatherer_;
    std::unique_ptr<NormalOutputDispatcher>     output_dispatcher_;
    // Python module wrapping the triton bitmask kernel. Imported once at
    // construction; held empty in cc_test ctors with no Python interpreter.
    // The hot path acquires the GIL once per batch *only* to call into this
    // module — everything else (matcher collection, bitmask fill, device
    // copy) is native C++. Released via close() under a GIL-scoped acquire
    // so the Py_DECREF runs on a thread that actually holds the GIL; see
    // close()/~NormalBatchStreamProcessor.
    py::module_                                 triton_bitmask_ops_;
};

}  // namespace rtp_llm
