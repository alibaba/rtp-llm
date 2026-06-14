#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"

#include <algorithm>
#include <exception>
#include <string>
#include <utility>

#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/Logger.h"
#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "rtp_llm/cpp/models/logits_processor/grammar_kernels/xgrammar_kernels.h"
#endif

using namespace std;

namespace rtp_llm {

LogitsProcessorStates::LogitsProcessorStates() {};

namespace {

bool isProcessorApplied(const SamplerInputs& inputs, const SpecLogitsProcessorId& processor_id) {
    if (!processor_id.valid()) {
        return false;
    }
    return std::find(inputs.spec_applied_processors.begin(), inputs.spec_applied_processors.end(), processor_id)
           != inputs.spec_applied_processors.end();
}

}  // namespace

void LogitsProcessorStates::batchProcess(const SamplerInputs& inputs) {
    const bool has_spec_mask = inputs.spec_vocab_mask_gpu.defined();
    if (has_spec_mask) {
#if USING_CUDA
        // Spec mask was allocated on a copy stream; pin it to the compute stream.
        if (inputs.spec_vocab_mask_gpu.is_cuda()) {
            c10::cuda::CUDACachingAllocator::recordStream(
                inputs.spec_vocab_mask_gpu.storage().data_ptr(),
                at::cuda::getCurrentCUDAStream(inputs.spec_vocab_mask_gpu.device().index()));
        }
        RTP_LLM_CHECK_WITH_INFO(inputs.spec_vocab_mask_gpu.scalar_type() == torch::kInt32,
                                "MTP verify spec mask must be int32 packed bitmask");
        auto logits = inputs.logits;
        // Producer (MtpExecutor::buildSpecGrammarMask) and consumer live in different
        // modules now; assert the cross-module contract loudly so a producer-side
        // shape/device drift fails here instead of silently masking the wrong row.
        RTP_LLM_CHECK_WITH_INFO(logits.defined() && logits.dim() == 2, "MTP verify logits must be defined 2-D tensor");
        RTP_LLM_CHECK_WITH_INFO(inputs.spec_vocab_mask_gpu.dim() == 2, "MTP verify spec mask must be 2-D");
        RTP_LLM_CHECK_WITH_INFO(inputs.spec_vocab_mask_gpu.is_contiguous(), "MTP verify spec mask must be contiguous");
        RTP_LLM_CHECK_WITH_INFO(inputs.spec_vocab_mask_gpu.device() == logits.device(),
                                "MTP verify spec mask device (%s) must match logits device (%s)",
                                inputs.spec_vocab_mask_gpu.device().str().c_str(),
                                logits.device().str().c_str());
        RTP_LLM_CHECK_WITH_INFO(inputs.spec_vocab_mask_gpu.size(0) == logits.size(0),
                                "MTP verify spec mask rows (%lld) must match logits rows (%lld)",
                                static_cast<long long>(inputs.spec_vocab_mask_gpu.size(0)),
                                static_cast<long long>(logits.size(0)));
        const int64_t expected_words = (static_cast<int64_t>(inputs.vocab_size) + 31) / 32;
        RTP_LLM_CHECK_WITH_INFO(inputs.spec_vocab_mask_gpu.size(1) >= expected_words,
                                "MTP verify spec mask word_count (%lld) < ceil(vocab_size=%lld / 32) = %lld",
                                static_cast<long long>(inputs.spec_vocab_mask_gpu.size(1)),
                                static_cast<long long>(inputs.vocab_size),
                                static_cast<long long>(expected_words));
        RTP_LLM_CHECK_WITH_INFO(logits.size(1) >= static_cast<int64_t>(inputs.vocab_size),
                                "MTP verify logits vocab dim (%lld) < vocab_size=%lld",
                                static_cast<long long>(logits.size(1)),
                                static_cast<long long>(inputs.vocab_size));
        invokeApplyXGrammarBitmaskInplace(logits,
                                          inputs.spec_vocab_mask_gpu,
                                          static_cast<int64_t>(inputs.vocab_size),
                                          cuda_graph::graphGetCurrentStream().stream());
#else
        inputs.logits.masked_fill_(inputs.spec_vocab_mask_gpu, BaseLogitsProcessor::neg_inf);
#endif
    }

    for (size_t i = 0; i < logits_processors_.size(); i++) {
        // MTP verify: SpecLogitsProcessor bitmask was applied above; skip duplicate work.
        if (has_spec_mask && std::dynamic_pointer_cast<SpecLogitsProcessor>(logits_processors_[i]) != nullptr
            && i < processor_ids_.size() && isProcessorApplied(inputs, processor_ids_[i])) {
            continue;
        }
        // Per-processor try/catch: each processor maps 1:1 to a stream slice, so any
        // RTP_LLM_CHECK throw (or std::exception leaking through process()) belongs
        // to that stream alone. Sampler::forward and preprocessLogits have no
        // try/catch, so without this guard a single stream's check failure aborts
        // the whole batch. Route the failure through the processor's own reporter
        // so the offending stream's error_info_ is set; other streams continue.
        try {
            logits_processors_[i]->process(inputs, intervals_[i].first, intervals_[i].second);
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("logits processor #%zu threw during process(): %s; isolating to its stream",
                                i,
                                e.what());
            logits_processors_[i]->reportErrorViaReporter(ErrorCode::EXECUTION_EXCEPTION,
                                                          std::string("logits processor exception: ") + e.what(),
                                                          /*stream_lock_held=*/false);
        } catch (...) {
            RTP_LLM_LOG_WARNING("logits processor #%zu threw unknown exception during process(); "
                                "isolating to its stream",
                                i);
            logits_processors_[i]->reportErrorViaReporter(ErrorCode::EXECUTION_EXCEPTION,
                                                          "logits processor unknown exception",
                                                          /*stream_lock_held=*/false);
        }
    }
}

void LogitsProcessorStates::insert(
    const BaseLogitsProcessorPtr& ptr, size_t start, size_t finish, uint64_t stream_id, size_t processor_idx) {
    logits_processors_.push_back(ptr);
    intervals_.push_back(std::make_pair(start, finish));
    processor_ids_.push_back({stream_id, processor_idx});
}

}  // namespace rtp_llm
