#include "rtp_llm/cpp/models/logits_processor/xgrammar/XGrammarLogitsProcessor.h"

#include "rtp_llm/cpp/models/logits_processor/xgrammar/XGrammarCompilerCache.h"
#include "rtp_llm/cpp/utils/Logger.h"
#if USING_CUDA
#include "rtp_llm/models_py/bindings/cuda/kernels/xgrammar_kernels.h"
#include <ATen/cuda/CUDAContext.h>
#endif

#include <cstring>

namespace rtp_llm {

#if RTP_LLM_ENABLE_XGRAMMAR_CPP
struct XGrammarRuntimeState {
    std::vector<xgrammar::GrammarMatcher> matchers;
    torch::Tensor                         bitmask_cpu;
    torch::Tensor                         bitmask_gpu;
};
#else
struct XGrammarRuntimeState {};
#endif

namespace {

const std::string kPdReplayStateVersion = "rtp-xgrammar-pd-replay-v1";

std::vector<int> effectiveThinkEndTokenIds(const std::vector<int>& end_think_token_ids) {
    constexpr int32_t kDeepSeekNewlineTokenId = 201;
    constexpr int32_t kQwenGlmNewlineTokenId  = 198;
    if (end_think_token_ids.size() > 1
        && (end_think_token_ids.front() == kDeepSeekNewlineTokenId
            || end_think_token_ids.front() == kQwenGlmNewlineTokenId)) {
        return std::vector<int>(end_think_token_ids.begin() + 1, end_think_token_ids.end());
    }
    return end_think_token_ids;
}

bool updateThinkEndMatch(StreamXGrammarInfo& info, int token_id) {
    if (!info.waiting_for_think_end || info.end_think_token_ids.empty()) {
        return true;
    }
    if (token_id == info.end_think_token_ids[info.think_end_match_pos]) {
        ++info.think_end_match_pos;
        if (info.think_end_match_pos == info.end_think_token_ids.size()) {
            info.waiting_for_think_end = false;
            info.active                = true;
            info.think_end_match_pos   = 0;
            return true;
        }
        return false;
    }
    info.think_end_match_pos = token_id == info.end_think_token_ids.front() ? 1 : 0;
    return false;
}

#if RTP_LLM_ENABLE_XGRAMMAR_CPP
DLTensor makeCpuInt32DLTensor(torch::Tensor& tensor, int64_t* shape) {
    DLTensor dl_tensor;
    dl_tensor.data               = tensor.data_ptr<int32_t>();
    dl_tensor.device             = DLDevice{kDLCPU, 0};
    dl_tensor.ndim               = static_cast<int32_t>(tensor.dim());
    dl_tensor.dtype              = DLDataType{kDLInt, 32, 1};
    dl_tensor.shape              = shape;
    dl_tensor.strides            = nullptr;
    dl_tensor.byte_offset        = 0;
    return dl_tensor;
}
#endif

}  // namespace

XGrammarLogitsProcessor::XGrammarLogitsProcessor(std::vector<StreamXGrammarInfo> xgrammar_infos):
    xgrammar_infos_(std::move(xgrammar_infos)) {}

XGrammarLogitsProcessor::~XGrammarLogitsProcessor() = default;

void XGrammarLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    RTP_LLM_CHECK(size() == finish_idx - start_idx);
#if RTP_LLM_ENABLE_XGRAMMAR_CPP
    RTP_LLM_CHECK_WITH_INFO(runtime_state_ != nullptr, "xgrammar runtime state is not initialized");
    RTP_LLM_CHECK_WITH_INFO(runtime_state_->matchers.size() == size(), "xgrammar matcher count mismatch");

    const auto batch_size = size();
    const auto vocab_size = static_cast<size_t>(inputs.vocab_size);
    const auto bitmask_size = (vocab_size + 31) / 32;

    bool need_fill = false;
    for (size_t i = 0; i < size(); ++i) {
        const auto& info = xgrammar_infos_[i];
        if (!info.active || info.terminated || info.dead) {
            continue;
        }
        need_fill = true;
        break;
    }
    if (!need_fill) {
        return;
    }

    runtime_state_->bitmask_cpu = torch::full({static_cast<int64_t>(batch_size), static_cast<int64_t>(bitmask_size)},
                                              static_cast<int32_t>(-1),
                                              torch::dtype(torch::kInt32).device(torch::kCPU));
    int64_t bitmask_shape[2]    = {static_cast<int64_t>(batch_size), static_cast<int64_t>(bitmask_size)};
    auto    bitmask_dl          = makeCpuInt32DLTensor(runtime_state_->bitmask_cpu, bitmask_shape);

    bool need_apply = false;
    for (size_t i = 0; i < size(); ++i) {
        auto& info = xgrammar_infos_[i];
        if (!info.active || info.terminated || info.dead) {
            continue;
        }
        need_apply = runtime_state_->matchers[i].FillNextTokenBitmask(&bitmask_dl, static_cast<int>(i))
                     || need_apply;
    }
    if (!need_apply) {
        return;
    }

    auto batch_logits = inputs.logits.narrow(0, start_idx, batch_size);
    if (inputs.logits.is_cuda()) {
#if USING_CUDA
        runtime_state_->bitmask_gpu = runtime_state_->bitmask_cpu.to(inputs.logits.device(), true);
        invokeApplyXGrammarBitmaskInplace(
            batch_logits, runtime_state_->bitmask_gpu, static_cast<int64_t>(vocab_size),
            at::cuda::getCurrentCUDAStream().stream());
#else
        RTP_LLM_CHECK_WITH_INFO(false, "xgrammar CUDA logits reached a non-CUDA build");
#endif
    } else {
        auto vocab_mask_cpu = torch::zeros({static_cast<int64_t>(batch_size), static_cast<int64_t>(vocab_size)},
                                           torch::dtype(torch::kUInt8).device(torch::kCPU));
        auto* bitmask_ptr = runtime_state_->bitmask_cpu.data_ptr<int32_t>();
        auto* vocab_mask_ptr = vocab_mask_cpu.data_ptr<uint8_t>();
        for (size_t row = 0; row < batch_size; ++row) {
            const auto& info = xgrammar_infos_[row];
            if (!info.active || info.terminated || info.dead) {
                continue;
            }
            auto* row_mask = vocab_mask_ptr + row * vocab_size;
            for (size_t token_id = 0; token_id < vocab_size; ++token_id) {
                const auto word = static_cast<uint32_t>(bitmask_ptr[row * bitmask_size + token_id / 32]);
                const bool allow = ((word >> (token_id % 32)) & 1U) != 0;
                row_mask[token_id] = allow ? 0 : 1;
            }
        }
        maskLogits(batch_logits, vocab_mask_cpu);
    }
#else
    for (size_t i = 0; i < size(); ++i) {
        const auto& info = xgrammar_infos_[i];
        if (!info.active || info.terminated || info.dead) {
            continue;
        }
        (void)inputs;
        RTP_LLM_CHECK_WITH_INFO(false,
                                "xgrammar response_format reached backend, but this build does not link "
                                "xgrammar C++ runtime; refusing silent CPU matcher fallback");
    }
#endif
}

void XGrammarLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& src_batch_indices) {
    std::vector<StreamXGrammarInfo> new_infos;
    new_infos.reserve(src_batch_indices.size());
    for (auto src_batch_idx : src_batch_indices) {
        new_infos.push_back(xgrammar_infos_[src_batch_idx].copy());
    }
    xgrammar_infos_ = std::move(new_infos);
#if RTP_LLM_ENABLE_XGRAMMAR_CPP
    if (runtime_state_) {
        auto new_runtime = std::make_unique<XGrammarRuntimeState>();
        new_runtime->matchers.reserve(src_batch_indices.size());
        for (auto src_batch_idx : src_batch_indices) {
            new_runtime->matchers.push_back(runtime_state_->matchers[src_batch_idx].Fork());
        }
        runtime_state_ = std::move(new_runtime);
    }
#endif
}

void XGrammarLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    RTP_LLM_CHECK(2 == new_tokens.dim());
    RTP_LLM_CHECK(size() == static_cast<size_t>(new_tokens.size(0)));
    auto* token_ptr = new_tokens.data_ptr<int>();
    for (size_t i = 0; i < size(); ++i) {
        auto& info = xgrammar_infos_[i];
        if (!info.active || info.terminated || info.dead) {
            if (!info.waiting_for_think_end) {
                continue;
            }
        }
        for (int32_t j = 0; j < num_new_tokens; ++j) {
            auto token_id = token_ptr[i * new_tokens.size(1) + j];
            if (info.waiting_for_think_end) {
                updateThinkEndMatch(info, token_id);
                continue;
            }
#if RTP_LLM_ENABLE_XGRAMMAR_CPP
            bool accepted = runtime_state_ && runtime_state_->matchers[i].AcceptToken(token_id);
            if (!accepted) {
                info.dead = true;
                RTP_LLM_LOG_WARNING("xgrammar matcher rejected backend token %d for row %lu", token_id, i);
                break;
            }
            info.terminated = runtime_state_->matchers[i].IsTerminated();
#endif
            info.consumed_seq_len += 1;
            info.accepted_tokens.push_back(token_id);
        }
    }
}

const std::string& XGrammarLogitsProcessor::pdReplayStateVersion() {
    return kPdReplayStateVersion;
}

std::vector<int> XGrammarLogitsProcessor::exportPdReplayAcceptedTokens(size_t batch_idx,
                                                                       bool   exclude_last_token) const {
    RTP_LLM_CHECK_WITH_INFO(batch_idx < xgrammar_infos_.size(),
                            "xgrammar export replay batch_idx out of range: %lu >= %lu",
                            batch_idx,
                            xgrammar_infos_.size());
    const auto& tokens = xgrammar_infos_[batch_idx].accepted_tokens;
    if (exclude_last_token && !tokens.empty()) {
        return std::vector<int>(tokens.begin(), tokens.end() - 1);
    }
    return tokens;
}

int64_t XGrammarLogitsProcessor::exportPdReplayConsumedSeqLen(size_t batch_idx, bool exclude_last_token) const {
    RTP_LLM_CHECK_WITH_INFO(batch_idx < xgrammar_infos_.size(),
                            "xgrammar export replay batch_idx out of range: %lu >= %lu",
                            batch_idx,
                            xgrammar_infos_.size());
    const auto& info = xgrammar_infos_[batch_idx];
    if (exclude_last_token && info.consumed_seq_len > 0) {
        return info.consumed_seq_len - 1;
    }
    return info.consumed_seq_len;
}

void XGrammarLogitsProcessor::restorePdReplayState(const std::vector<int>& accepted_tokens,
                                                   int64_t                 consumed_seq_len,
                                                   const std::string&      replay_state_version,
                                                   size_t                  batch_idx) {
    RTP_LLM_CHECK_WITH_INFO(batch_idx < xgrammar_infos_.size(),
                            "xgrammar restore replay batch_idx out of range: %lu >= %lu",
                            batch_idx,
                            xgrammar_infos_.size());
    RTP_LLM_CHECK_WITH_INFO(replay_state_version.empty() || replay_state_version == pdReplayStateVersion(),
                            "xgrammar replay state version mismatch: got [%s], expected [%s]",
                            replay_state_version.c_str(),
                            pdReplayStateVersion().c_str());
    RTP_LLM_CHECK_WITH_INFO(consumed_seq_len >= 0, "xgrammar consumed_seq_len must be non-negative");
    RTP_LLM_CHECK_WITH_INFO(static_cast<int64_t>(accepted_tokens.size()) == consumed_seq_len,
                            "xgrammar replay token count %lu does not match consumed_seq_len %ld",
                            accepted_tokens.size(),
                            consumed_seq_len);

    auto& info = xgrammar_infos_[batch_idx];
    if (consumed_seq_len <= info.consumed_seq_len) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(info.consumed_seq_len == 0,
                            "xgrammar replay restore must happen before backend tokens are accepted, current=%ld",
                            info.consumed_seq_len);

    if (!accepted_tokens.empty()) {
        info.waiting_for_think_end = false;
        info.active                = true;
        info.think_end_match_pos   = 0;
    }

    for (auto token_id : accepted_tokens) {
#if RTP_LLM_ENABLE_XGRAMMAR_CPP
        bool accepted = runtime_state_ && runtime_state_->matchers[batch_idx].AcceptToken(token_id);
        if (!accepted) {
            info.dead = true;
            RTP_LLM_CHECK_WITH_INFO(false, "xgrammar replay token %d rejected by matcher", token_id);
        }
        info.terminated = runtime_state_->matchers[batch_idx].IsTerminated();
#endif
        info.accepted_tokens.push_back(token_id);
        info.consumed_seq_len += 1;
    }
}

XGrammarLogitsProcessorPtr XGrammarLogitsProcessor::fromGenerateInput(std::shared_ptr<GenerateInput> generate_input,
                                                                      int32_t                        num) {
    const auto& generate_config = generate_input->generate_config;
    if (!generate_config || !generate_config->xgrammar_enabled) {
        return nullptr;
    }

    RTP_LLM_CHECK_WITH_INFO(!generate_config->xgrammar_compile_cache_key.empty(),
                            "xgrammar_compile_cache_key must be set when xgrammar_enabled is true");
    RTP_LLM_CHECK_WITH_INFO(!generate_config->xgrammar_canonical_schema.empty(),
                            "xgrammar_canonical_schema must be set when xgrammar_enabled is true");

    XGrammarCompilerCache::instance().init(generate_config->xgrammar_compile_cache_capacity);
    auto table = XGrammarCompilerCache::instance().getOrInsertDeviceTable(generate_config->xgrammar_compile_cache_key,
                                                                          generate_config->xgrammar_tokenizer_fp);

    std::vector<StreamXGrammarInfo> infos;
    infos.reserve(num);
    auto end_think_token_ids = effectiveThinkEndTokenIds(generate_config->end_think_token_ids);
    for (int32_t i = 0; i < num; ++i) {
        StreamXGrammarInfo info;
        info.active           = !generate_config->in_think_mode;
        info.waiting_for_think_end = generate_config->in_think_mode && !end_think_token_ids.empty();
        info.grammar_kind     = generate_config->xgrammar_grammar_kind;
        info.canonical_schema = generate_config->xgrammar_canonical_schema;
        info.schema_sha256    = generate_config->xgrammar_schema_sha256;
        info.tokenizer_fp     = generate_config->xgrammar_tokenizer_fp;
        info.cache_key        = generate_config->xgrammar_compile_cache_key;
        info.end_think_token_ids = end_think_token_ids;
        infos.push_back(std::move(info));
    }
    auto processor = std::make_shared<XGrammarLogitsProcessor>(std::move(infos));
#if RTP_LLM_ENABLE_XGRAMMAR_CPP
    RTP_LLM_CHECK_WITH_INFO(table.compiled_grammar != nullptr, "xgrammar compiled grammar missing from cache table");
    processor->runtime_state_ = std::make_unique<XGrammarRuntimeState>();
    processor->runtime_state_->matchers.reserve(num);
    for (int32_t i = 0; i < num; ++i) {
        processor->runtime_state_->matchers.emplace_back(*table.compiled_grammar);
    }
#else
    (void)table;
#endif
    return processor;
}

}  // namespace rtp_llm
