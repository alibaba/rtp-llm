#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/normal_engine/speculative/SpecGrammarHelpers.h"
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <dlpack/dlpack.h>

namespace rtp_llm {

NormalBatchStreamProcessor::NormalBatchStreamProcessor(
    const ModelConfig&                 model_config,
    const PDSepConfig&                 pd_sep_config,
    const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
    const CacheConfig&                 cache_config,
    bool                               warm_up) {
    model_input_gatherer_config_.num_layers              = model_config.num_layers;
    model_input_gatherer_config_.vocab_size              = model_config.vocab_size;
    model_input_gatherer_config_.input_vocab_size        = model_config.input_vocab_size;
    model_input_gatherer_config_.has_positional_encoding = model_config.has_positional_encoding;
    model_input_gatherer_config_.is_multimodal           = model_config.mm_model_config.is_multimodal;
    model_input_gatherer_config_.mm_position_ids_style =
        static_cast<PositionIdsStyle>(model_config.mm_model_config.mm_position_ids_style);
    model_input_gatherer_config_.position_id_len_factor     = model_config.attn_config.rope_config.index_factor;
    model_input_gatherer_config_.role_type                  = pd_sep_config.role_type;
    model_input_gatherer_config_.decode_entrance            = pd_sep_config.decode_entrance;
    model_input_gatherer_config_.block_stride_bytes         = cache_config.kv_block_stride_bytes;
    model_input_gatherer_config_.scale_stride_bytes         = cache_config.kv_scale_stride_bytes;
    model_input_gatherer_config_.seq_size_per_block         = cache_config.seq_size_per_block;
    model_input_gatherer_config_.kernel_seq_size_per_block  = cache_config.kernel_seq_size_per_block;
    model_input_gatherer_config_.kernel_blocks_per_kv_block = cache_config.kernelBlocksPerKvBlock();
    model_input_gatherer_config_.kv_cache_group_nums        = cache_config.groupNums();
    model_input_gatherer_config_.layer_to_kv_cache_group_id = cache_config.layer_to_group_id;
    model_input_gatherer_config_.kv_cache_group_types       = cache_config.group_types;
    model_input_gatherer_config_.warm_up                    = warm_up;
    model_input_gatherer_config_.enable_detail_log          = profiling_debug_logging_config.enable_detail_log;

    model_input_gatherer_   = std::make_unique<NormalModelInputGatherer>(model_input_gatherer_config_);
    sampler_input_gatherer_ = std::make_unique<NormalSamplerInputGatherer>();

    // Triton bitmask kernel lives in Python; it is the one remaining GIL
    // touchpoint on the grammar hot path. Skip the import in cc_test (no
    // interpreter); tolerate import failure so engine startup doesn't abort
    // when grammar isn't going to be used (kernel dispatch surfaces it later).
    if (Py_IsInitialized()) {
        // Import + child ctor under one GIL scope: NormalOutputDispatcher's
        // py::module_ member copy issues Py_INCREF that requires the GIL.
        py::gil_scoped_acquire acquire;
        try {
            triton_bitmask_ops_ = py::module_::import("rtp_llm.models_py.triton_kernels.grammar.bitmask_ops");
        } catch (const py::error_already_set& e) {
            RTP_LLM_LOG_WARNING(
                "failed to import rtp_llm.models_py.triton_kernels.grammar.bitmask_ops (%s); "
                "grammar requests will be rejected at kernel dispatch — verify triton install",
                e.what());
        }
        output_dispatcher_ = std::make_unique<NormalOutputDispatcher>(triton_bitmask_ops_);
    } else {
        output_dispatcher_ = std::make_unique<NormalOutputDispatcher>();
    }
}

NormalBatchStreamProcessor::~NormalBatchStreamProcessor() {
    close();
}

void NormalBatchStreamProcessor::close() noexcept {
    // Release child's Python handle first while the GIL is still safe to
    // acquire; after this the child's dtor finds an empty handle (no-op).
    if (output_dispatcher_) {
        output_dispatcher_->close();
    }
    if (!triton_bitmask_ops_) {
        return;
    }
    if (Py_IsInitialized()) {
        py::gil_scoped_acquire acquire;
        triton_bitmask_ops_ = py::module_();
    } else {
        // Post-finalize: GIL acquire is UB. Deliberately leak.
        (void)triton_bitmask_ops_.release();
    }
}

absl::Status NormalBatchStreamProcessor::dispatch(const StreamGroups& stream_groups,
                                                  const MergedOutput& merge_outputs) const {
    // Grammar accept_token rides on the dispatcher's existing token_ids_cpu copy.
    return output_dispatcher_->dispatch(stream_groups, merge_outputs);
}

absl::StatusOr<GptModelInputs> NormalBatchStreamProcessor::gatherModelInput(const StreamGroups& stream_groups) const {
    return model_input_gatherer_->gather(stream_groups);
}

absl::StatusOr<SamplerInputs> NormalBatchStreamProcessor::gatherSamplerInput(
    const StreamGroups& stream_groups, const GptModelInputs& model_inputs, const GptModelOutputs& model_output) const {
    return sampler_input_gatherer_->gather(stream_groups, model_inputs, model_output);
}

SamplerInputs NormalBatchStreamProcessor::allocateSamplerInputs(const StreamGroups& stream_groups,
                                                                size_t              total_batch_size_in,
                                                                size_t              total_batch_size_out,
                                                                size_t              propose_step) const {
    return sampler_input_gatherer_->allocateSamplerInputs(
        stream_groups, total_batch_size_in, total_batch_size_out, propose_step);
}

void NormalBatchStreamProcessor::fillSamplerCommonInputs(SamplerInputs&                sampler_inputs,
                                                         std::list<GenerateStreamPtr>& all_streams,
                                                         bool                          score_batch,
                                                         size_t                        propose_step) const {
    sampler_input_gatherer_->fillSamplerCommonInputs(sampler_inputs, all_streams, score_batch, propose_step);
}

void NormalBatchStreamProcessor::setLogitsProcessorInputs(SamplerInputs&                sampler_inputs,
                                                          std::list<GenerateStreamPtr>& all_streams,
                                                          bool                          score_batch) const {
    sampler_input_gatherer_->setLogitsProcessorInputs(sampler_inputs, all_streams, score_batch);
}

void NormalBatchStreamProcessor::applyGrammarConstraints(SamplerInputs&      inputs,
                                                         const StreamGroups& stream_groups) const {
    if (!spec_grammar::streamGroupsHaveAttachedGrammar(stream_groups)) {
        return;
    }

    // Collect (sampler_row_idx, matcher) for matchers that will actively
    // constrain their row. Passthrough / finished / terminated matchers leave
    // their row at the all-allow default (the kernel treats that as a no-op).
    std::vector<std::pair<int, RtpGrammarMatcher*>> active;
    int total_batch = 0;
    int vocab_size  = -1;
    for (auto& stream : stream_groups.allStreams()) {
        const int sampler_batch_size =
            stream->needTilingForSampling() ? stream->nextBatchSize() : stream->currentBatchSize();
        RtpGrammarMatcher* m = stream->tryGetGrammarMatcher();
        if (m && !m->isTerminated() && !m->finished() && !m->isPassthroughForMask()) {
            for (int i = 0; i < sampler_batch_size; ++i) {
                active.emplace_back(total_batch + i, m);
            }
            if (vocab_size < 0) {
                vocab_size = m->vocabSize();
            }
        }
        total_batch += sampler_batch_size;
    }
    if (active.empty()) {
        return;
    }
    if (reportGrammarUnavailableIfNeeded(stream_groups)) {
        return;
    }

    // Allocate [batch, ceil(vocab/32)] int32 bitmask on CPU, all-1s
    // (allow-all). Matches xgrammar.allocate_token_bitmask contract.
    const int words   = (vocab_size + 31) / 32;
    auto      bitmask = at::full({total_batch, words}, /*fill_value=*/-1, at::dtype(at::kInt));
    DLTensor  dl      = spec_grammar::makeBitmaskView(bitmask.data_ptr<int32_t>(), total_batch, words);

    for (auto& [row_idx, m] : active) {
        m->fillBitmask(&dl, row_idx);
    }

    auto bitmask_gpu = bitmask.to(inputs.logits.device(), /*non_blocking=*/true);
    // Logits may be vocab-padded for tensor parallelism; slice down before the kernel.
    auto target_logits =
        inputs.logits.size(1) > vocab_size ? inputs.logits.slice(/*dim=*/1, 0, vocab_size) : inputs.logits;

    // The one remaining GIL touch on the grammar hot path. Eliminating it
    // requires exposing apply_token_bitmask_inplace_triton as a TORCH_LIBRARY op.
    py::gil_scoped_acquire acquire;
    triton_bitmask_ops_.attr("apply_token_bitmask_inplace_triton")(
        convertTensorToObject(target_logits), convertTensorToObject(bitmask_gpu));
}

bool NormalBatchStreamProcessor::reportGrammarUnavailableIfNeeded(
    const StreamGroups& stream_groups) const {
    if (triton_bitmask_ops_) {
        return false;
    }
    if (!Py_IsInitialized()) {
        return true;  // cc_test: no interpreter, skip kernel.
    }
    // Triton import failed at init but grammar is being requested — surface
    // through reportError on each matcher-bearing stream; never throw on hot path.
    for (auto& stream : stream_groups.allStreams()) {
        if (stream->hasGrammarMatcher()) {
            stream->reportError(ErrorCode::EXECUTION_EXCEPTION,
                                "grammar bitmask kernel unavailable: triton import failed "
                                "at engine init (see prior WARNING). Verify triton install.");
        }
    }
    return true;
}

}  // namespace rtp_llm
