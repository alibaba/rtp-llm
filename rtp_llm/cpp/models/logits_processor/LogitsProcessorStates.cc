#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"

#include <exception>
#include <optional>
#include <string>
#include <utility>

#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorException.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/Logger.h"
#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

using namespace std;

namespace rtp_llm {

LogitsProcessorStates::LogitsProcessorStates() {};

void LogitsProcessorStates::batchProcess(const SamplerInputs& inputs) {
    const bool has_spec_mask = inputs.spec_vocab_mask_gpu.defined();
    if (has_spec_mask) {
        auto logits = inputs.logits;
        RTP_LLM_CHECK_WITH_INFO(inputs.spec_vocab_mask_gpu.device() == logits.device(),
                                "MTP verify spec mask device (%s) must match logits device (%s)",
                                inputs.spec_vocab_mask_gpu.device().str().c_str(),
                                logits.device().str().c_str());
        RTP_LLM_CHECK_WITH_INFO(logits.size(1) >= static_cast<int64_t>(inputs.vocab_size),
                                "MTP verify logits vocab dim (%lld) < vocab_size=%lld",
                                static_cast<long long>(logits.size(1)),
                                static_cast<long long>(inputs.vocab_size));
#if USING_CUDA
        // Spec mask was allocated on a copy stream; pin it to the compute stream.
        if (inputs.spec_vocab_mask_gpu.is_cuda()) {
            c10::cuda::CUDACachingAllocator::recordStream(
                inputs.spec_vocab_mask_gpu.storage().data_ptr(),
                at::cuda::getCurrentCUDAStream(inputs.spec_vocab_mask_gpu.device().index()));
        }
#endif
        const int64_t V = static_cast<int64_t>(inputs.vocab_size);
        logits.narrow(1, 0, V).masked_fill_(inputs.spec_vocab_mask_gpu.narrow(1, 0, V),
                                            BaseLogitsProcessor::neg_inf);
    }

    for (size_t i = 0; i < logits_processors_.size(); i++) {
        // MTP verify: SpecLogitsProcessor bitmask was applied above; skip duplicate work.
        if (has_spec_mask && std::dynamic_pointer_cast<SpecLogitsProcessor>(logits_processors_[i]) != nullptr
            && i < processor_ids_.size() && inputs.hasAppliedSpecProcessor(processor_ids_[i])) {
            continue;
        }
        // Per-processor try/catch: each processor maps 1:1 to a stream slice, so any
        // RTP_LLM_CHECK throw (or std::exception leaking through process()) belongs
        // to that stream alone. Sampler::forward and preprocessLogits have no
        // try/catch, so without this guard a single stream's check failure aborts
        // the whole batch. Route the failure to the owning stream so the offending
        // stream's error_info_ is set; other streams continue.
        std::optional<ErrorInfo> err;
        try {
            logits_processors_[i]->process(inputs, intervals_[i].first, intervals_[i].second);
        } catch (const LogitsProcessorException& e) {
            err = ErrorInfo(e.code(), e.what());
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING(
                "logits processor #%zu threw during process(): %s; isolating to its stream", i, e.what());
            err = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                            std::string("logits processor exception: ") + e.what());
        } catch (...) {
            RTP_LLM_LOG_WARNING("logits processor #%zu threw unknown exception during process(); "
                                "isolating to its stream",
                                i);
            err = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "logits processor unknown exception");
        }
        if (err && err->hasError() && i < error_sinks_.size() && error_sinks_[i]) {
            error_sinks_[i](err->code(), err->ToString());
        }
    }
}

void LogitsProcessorStates::insert(const BaseLogitsProcessorPtr& ptr,
                                   size_t                        start,
                                   size_t                        finish,
                                   uint64_t                      stream_id,
                                   size_t                        processor_idx,
                                   ErrorSink                     error_sink) {
    logits_processors_.push_back(ptr);
    intervals_.push_back(std::make_pair(start, finish));
    processor_ids_.push_back({stream_id, processor_idx});
    error_sinks_.push_back(std::move(error_sink));
}

}  // namespace rtp_llm
