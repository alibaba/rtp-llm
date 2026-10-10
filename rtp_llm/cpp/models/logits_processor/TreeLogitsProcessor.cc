#include "rtp_llm/cpp/models/logits_processor/TreeLogitsProcessor.h"

#include <cmath>
#include <limits>
#include "autil/EnvUtil.h"
#if USING_CUDA || USING_ROCM
#include "rtp_llm/models_py/bindings/common/kernels/mask_logits_csr.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#if USING_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#elif USING_ROCM
#include <c10/hip/HIPCachingAllocator.h>
#endif
#endif
using namespace std;

namespace rtp_llm {

TreeLogitsProcessor::TreeLogitsProcessor(std::vector<StreamTreeInfo> tree_infos): tree_infos_(tree_infos) {}

std::optional<ErrorInfo>
TreeLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    auto batch_size = size();
    RTP_LLM_CHECK(batch_size == finish_idx - start_idx);
    if (batch_size == 0) { return std::nullopt; }
    const auto snapshot = tree_infos_.front().csr_snapshot;
    if (snapshot) {
        auto logits = inputs.logits.narrow(0, start_idx, batch_size);
        if (!csr_host_states_.defined() || csr_host_states_.numel() != static_cast<int64_t>(batch_size)) {
            csr_host_states_ = torch::empty({static_cast<int64_t>(batch_size)},
                torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
        }
        auto* states = csr_host_states_.data_ptr<int32_t>();
        for (size_t i = 0; i < batch_size; ++i) { states[i] = tree_infos_[i].maskState(); }
#if USING_CUDA || USING_ROCM
        if (logits.is_cuda()) {
            if (!snapshot->deviceReady() || snapshot->deviceRowPtr().device() != logits.device()) {
                return ErrorInfo(ErrorCode::INVALID_PARAMS, "runtime CSR snapshot has no matching GPU buffers");
            }
            auto device_states = csr_host_states_.to(logits.device());
            auto stream = cuda_graph::graphGetCurrentStream().stream();
#if USING_CUDA
            c10::cuda::CUDACachingAllocator::recordStream(snapshot->deviceRowPtr().storage().data_ptr(),
                                                         cuda_graph::graphGetCurrentStream());
            c10::cuda::CUDACachingAllocator::recordStream(snapshot->deviceColIdx().storage().data_ptr(),
                                                         cuda_graph::graphGetCurrentStream());
#elif USING_ROCM
            c10::hip::HIPCachingAllocator::recordStream(snapshot->deviceRowPtr().storage().data_ptr(),
                                                       cuda_graph::graphGetCurrentStream());
            c10::hip::HIPCachingAllocator::recordStream(snapshot->deviceColIdx().storage().data_ptr(),
                                                       cuda_graph::graphGetCurrentStream());
#endif
            CSRLogitsType type;
            switch (logits.scalar_type()) {
                case torch::kFloat32: type = CSRLogitsType::FLOAT32; break;
                case torch::kFloat16: type = CSRLogitsType::FLOAT16; break;
                case torch::kBFloat16: type = CSRLogitsType::BFLOAT16; break;
                default: return ErrorInfo(ErrorCode::INVALID_PARAMS, "unsupported CSR logits dtype");
            }
            if (!logits.is_contiguous()) {
                return ErrorInfo(ErrorCode::INVALID_PARAMS, "CSR sampling requires contiguous logits");
            }
            invokeCSRMaskLogitsByType(logits.data_ptr(), type, device_states.data_ptr<int32_t>(),
                snapshot->deviceRowPtr().data_ptr<int32_t>(), snapshot->deviceColIdx().data_ptr<int32_t>(),
                batch_size, logits.size(1), snapshot->stateCount(), stream);
            return std::nullopt;
        }
#endif
        auto mask = torch::ones(logits.sizes(), torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
        auto allowed = mask.accessor<bool, 2>();
        for (size_t i = 0; i < batch_size; ++i) {
            const auto state = states[i];
            if (state < 0 || static_cast<size_t>(state) >= snapshot->stateCount()) { continue; }
            for (int32_t edge = snapshot->rowPtr()[state]; edge < snapshot->rowPtr()[state + 1]; ++edge) {
                const auto token = snapshot->colIdx()[edge];
                if (token >= logits.size(1)) {
                    return ErrorInfo(ErrorCode::OUT_OF_VOCAB_RANGE, "CSR token exceeds model vocabulary");
                }
                allowed[i][token] = false;
            }
        }
        logits.masked_fill_(mask, -std::numeric_limits<float>::infinity());
        return std::nullopt;
    }
    bool                             need_process = false;
    std::vector<std::vector<size_t>> batch_candidate_token_ids(batch_size);

    for (size_t i = 0; i < size(); ++i) {
        auto& info = tree_infos_[i];
        if (!info.in_tree_mode) {
            continue;
        }
        const auto& candidate_token_ids = info.dfa_ptr->getCandidateTokenIds();
        batch_candidate_token_ids[i]    = candidate_token_ids;
        if (candidate_token_ids.size() > 0) {
            need_process = true;
        }
    }
    // If no beams need processing, return early
    if (!need_process) {
        return std::nullopt;
    }

    auto   batch_logits     = inputs.logits.narrow(0, start_idx, batch_size);
    size_t vocab_size       = batch_logits.size(1);
    auto   batch_vocab_mask = generateVocabMask(batch_size, vocab_size, batch_candidate_token_ids);
    maskLogits(batch_logits, batch_vocab_mask);
    return std::nullopt;
}

void TreeLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& src_batch_indices) {
    std::vector<StreamTreeInfo> new_tree_infos;
    for (auto src_batch_idx : src_batch_indices) {
        new_tree_infos.push_back(tree_infos_[src_batch_idx].copy());
    }
    tree_infos_ = std::move(new_tree_infos);
}

std::optional<ErrorInfo> TreeLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    RTP_LLM_CHECK(2 == new_tokens.dim());
    RTP_LLM_CHECK(size() == (size_t)new_tokens.size(0));

    for (size_t i = 0; i < size(); i++) {
        auto& info = tree_infos_[i];
        if (info.csr_snapshot) {
            const auto offset = info.is_beam_search ? info.current_output_length + info.input_length : 0;
            for (int32_t j = 0; j < num_new_tokens; ++j) {
                if (j + offset >= new_tokens.size(1)) {
                    return ErrorInfo(ErrorCode::INVALID_PARAMS, "CSR token buffer is shorter than the committed history");
                }
                const auto token = new_tokens.data_ptr<int32_t>()[i * new_tokens.size(1) + j + offset];
                if (info.isFinishedCsrBeam()) {
                    if (token != info.csr_snapshot->endTokenId()) {
                        return ErrorInfo(ErrorCode::INVALID_PARAMS, "finished CSR beam may only repeat EOS");
                    }
                    continue;
                }
                const auto next = info.csr_snapshot->transition(info.csr_state, token);
                if (next == ConstraintTreeCsrSnapshot::INVALID_TRANSITION) {
                    return ErrorInfo(ErrorCode::INVALID_PARAMS, "CSR constraint tree rejected an invalid token transition");
                }
                info.csr_state = next;
                info.in_tree_mode = next >= 0;
            }
            info.current_output_length += num_new_tokens;
            continue;
        }
        if (!info.in_tree_mode)
            continue;

        auto offset = info.is_beam_search ? (info.current_output_length + info.input_length) : 0;

        if (!info.is_beam_search) {
            RTP_LLM_CHECK(num_new_tokens == new_tokens.size(1));
        }

        for (size_t j = 0; j < num_new_tokens; ++j) {
            auto current_token_id = new_tokens.data_ptr<int>()[i * new_tokens.size(1) + j + offset];
            info.dfa_ptr->next(current_token_id);
            if (info.dfa_ptr->hasError()) {
                break;
            }
        }

        info.current_output_length += num_new_tokens;
    }
    return std::nullopt;
}

TreeLogitsProcessorPtr TreeLogitsProcessor::fromGenerateInput(std::shared_ptr<GenerateInput> generate_input,
                                                              int32_t                        num) {
    return fromGenerateInput(std::move(generate_input), num, ConstraintTreeCsrManager::instance()->snapshot());
}

TreeLogitsProcessorPtr TreeLogitsProcessor::fromGenerateInput(
    std::shared_ptr<GenerateInput> generate_input, int32_t num, ConstraintTreeCsrSnapshotPtr snapshot) {
    if (snapshot) {
        std::vector<StreamTreeInfo> states;
        for (int32_t i = 0; i < num; ++i) {
            states.emplace_back(true, generate_input->inputLength(), 0,
                generate_input->generate_config->hasNumBeams() || generate_input->generate_config->num_return_sequences > 1,
                snapshot);
        }
        return std::make_shared<TreeLogitsProcessor>(std::move(states));
    }
    if (!PrefixToCandidateTokens::instance()->initSuccess()) {
        return nullptr;
    }

    auto processor_ptr = std::make_shared<TreeLogitsProcessor>();
    for (size_t i = 0; i < num; i++) {
        StreamTreeInfo              tree_info(PrefixToCandidateTokens::instance()->initSuccess(),
                                 generate_input->inputLength(),
                                 0,
                                 generate_input->generate_config->hasNumBeams()
                                     || generate_input->generate_config->num_return_sequences > 1,
                                 std::make_shared<TreeDFA<std::string, int>>(PrefixToCandidateTokens::instance()));
        std::vector<StreamTreeInfo> tree_infos       = {tree_info};
        auto                        single_processor = std::make_shared<TreeLogitsProcessor>(tree_infos);

        processor_ptr->insert(single_processor, 1);
    }

    return processor_ptr;
}

std::vector<std::string> TreeLogitsProcessor::getStatus() {
    std::vector<std::string> status_list;
    for (const auto& tree_info : tree_infos_) {
        status_list.push_back(tree_info.csr_snapshot ?
            "csr:v" + std::to_string(tree_info.csr_snapshot->version()) + ":state:" + std::to_string(tree_info.csr_state) :
            tree_info.dfa_ptr->status());
    }
    return status_list;
}

std::string TreeLogitsProcessor::validateCsrRequest(const ConstraintTreeCsrSnapshotPtr& snapshot,
                                                    const GenerateConfig&               generate_config,
                                                    bool                                runtime_tree_required) {
    if (!snapshot) {
        return runtime_tree_required ? "runtime constraint tree is required but no CSR snapshot is active" :
                                       std::string();
    }
    if (generate_config.num_beams <= 0) {
        return "runtime constraint tree requires num_beams to be positive";
    }
    for (auto width : generate_config.variable_num_beams) {
        if (width <= 0) {
            return "runtime constraint tree requires every variable_num_beams entry to be positive";
        }
    }
    // Later steps select globally from all surviving parents, not just the
    // root. Their actual capacity is checked on the sampler's selected scores.
    const auto first_width = generate_config.variable_num_beams.empty() ? generate_config.num_beams :
                                                                          generate_config.variable_num_beams.front();
    if (snapshot->rootCandidateCount() < static_cast<size_t>(first_width)) {
        return "runtime constraint tree root candidate count [" + std::to_string(snapshot->rootCandidateCount())
               + "] is smaller than first-step num_beams [" + std::to_string(first_width) + "]";
    }
    return {};
}

std::optional<ErrorInfo> TreeLogitsProcessor::validateBeamScores(const torch::Tensor& scores, size_t count) const {
    if (tree_infos_.empty() || !tree_infos_.front().csr_snapshot || !tree_infos_.front().is_beam_search) {
        return std::nullopt;
    }
    if (!scores.defined() || scores.numel() != static_cast<int64_t>(count)) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "CSR beam search requires one cumulative score per output");
    }
    auto host = scores.to(torch::kCPU, torch::kFloat32).contiguous();
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(host.data_ptr<float>()[i])) {
            return ErrorInfo(ErrorCode::INVALID_PARAMS, "CSR beam search has insufficient valid candidates or non-finite scores");
        }
    }
    return std::nullopt;
}
}  // namespace rtp_llm
