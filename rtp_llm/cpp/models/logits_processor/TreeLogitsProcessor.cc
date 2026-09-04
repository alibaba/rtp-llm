#include "rtp_llm/cpp/models/logits_processor/TreeLogitsProcessor.h"
#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"

using namespace std;

namespace rtp_llm {

TreeLogitsProcessor::TreeLogitsProcessor(rtp_llm::DeviceBase* device): BaseLogitsProcessor(device){};

TreeLogitsProcessor::TreeLogitsProcessor(rtp_llm::DeviceBase* device, std::vector<StreamTreeInfo> tree_infos):
    BaseLogitsProcessor(device), tree_infos_(tree_infos) {}

void TreeLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    auto batch_size = size();
    RTP_LLM_CHECK(batch_size == finish_idx - start_idx);
    if (batch_size == 0) {
        return;
    }

    // A processor belongs to one generation request, so all its beams pin the
    // same immutable snapshot even if a newer version becomes active globally.
    const auto csr_snapshot = tree_infos_.front().csr_snapshot;
    if (csr_snapshot) {
        if (csr_snapshot->deviceReady() && device_->getDeviceProperties().type == DeviceType::Cuda) {
            ensureCsrStateBuffers(batch_size);
            auto* host_states = csr_host_states_->data<int32_t>();
            for (size_t index = 0; index < batch_size; ++index) {
                const auto& info = tree_infos_[index];
                RTP_LLM_CHECK(info.csr_snapshot == csr_snapshot);
                host_states[index] = info.csr_state;
            }
            device_->copy({*csr_device_states_, *csr_host_states_});
            auto batch_logits = inputs.logits->slice(start_idx, batch_size);
            device_->csrMaskLogits(
                *batch_logits, *csr_device_states_, *csr_snapshot->deviceRowPtr(), *csr_snapshot->deviceColIdx());
            return;
        }

        // CPU/device-test fallback still uses the CSR rows; production CUDA
        // workers take the direct GPU kernel path above.
        const auto& row_ptr = csr_snapshot->rowPtr();
        const auto& col_idx = csr_snapshot->colIdx();
        for (size_t index = 0; index < batch_size; ++index) {
            const auto&                      info = tree_infos_[index];
            std::vector<std::vector<size_t>> candidate_token_ids(1);
            if (info.csr_state >= 0 && static_cast<size_t>(info.csr_state) < csr_snapshot->stateCount()) {
                for (int32_t edge = row_ptr[info.csr_state]; edge < row_ptr[info.csr_state + 1]; ++edge) {
                    candidate_token_ids[0].push_back(static_cast<size_t>(col_idx[edge]));
                }
            }
            auto logits     = inputs.logits->slice(start_idx + index, 1);
            auto vocab_mask = generateVocabMask(1, logits->shape()[1], candidate_token_ids);
            maskLogits(logits, vocab_mask);
        }
        return;
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
        return;
    }

    auto   batch_logits     = inputs.logits->slice(start_idx, batch_size);
    size_t vocab_size       = batch_logits->shape()[1];
    auto   batch_vocab_mask = generateVocabMask(batch_size, vocab_size, batch_candidate_token_ids);
    maskLogits(batch_logits, batch_vocab_mask);
}

void TreeLogitsProcessor::ensureCsrStateBuffers(size_t count) {
    if (csr_host_states_ && csr_device_states_ && csr_host_states_->size() == count) {
        return;
    }
    csr_host_states_ =
        device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {count}, rtp_llm::AllocationType::HOST}, {});
    csr_device_states_ =
        device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {count}, rtp_llm::AllocationType::DEVICE}, {});
}

void TreeLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& src_batch_indices) {
    std::vector<StreamTreeInfo> new_tree_infos;
    for (auto src_batch_idx : src_batch_indices) {
        new_tree_infos.push_back(tree_infos_[src_batch_idx].copy());
    }
    tree_infos_ = std::move(new_tree_infos);
}

void TreeLogitsProcessor::updateStatus(const rtp_llm::BufferPtr& new_tokens, int32_t num_new_tokens) {
    RTP_LLM_CHECK(2 == new_tokens->shape().size());
    RTP_LLM_CHECK(size() == new_tokens->shape()[0]);

    for (size_t i = 0; i < size(); i++) {
        auto& info   = tree_infos_[i];
        auto  offset = info.is_beam_search ? (info.current_output_length + info.input_length) : 0;

        if (!info.is_beam_search) {
            RTP_LLM_CHECK(num_new_tokens == new_tokens->shape()[1]);
        }

        if (info.csr_snapshot) {
            for (size_t j = 0; j < num_new_tokens; ++j) {
                RTP_LLM_CHECK_WITH_INFO(j + offset < new_tokens->shape()[1],
                                        "CSR token offset [%zu] exceeds token buffer width [%zu]",
                                        j + offset,
                                        new_tokens->shape()[1]);
                const auto token      = *(*new_tokens)[i].dataWithOffset<int>(j + offset);
                const auto next_state = info.csr_snapshot->transition(info.csr_state, token);
                RTP_LLM_CHECK_WITH_INFO(next_state != ConstraintTreeCsrSnapshot::INVALID_TRANSITION,
                                        "CSR constraint tree version [%llu] rejected token [%d] at state [%d]",
                                        static_cast<unsigned long long>(info.csr_snapshot->version()),
                                        token,
                                        info.csr_state);
                if (token == info.csr_snapshot->endTokenId()) {
                    RTP_LLM_CHECK_WITH_INFO(
                        next_state == -1, "CSR end token [%d] did not lead to the terminal state", token);
                    info.csr_state    = -1;
                    info.in_tree_mode = false;
                    break;
                }
                RTP_LLM_CHECK_WITH_INFO(
                    next_state >= 0, "CSR non-end token [%d] led to an invalid state [%d]", token, next_state);
                info.csr_state = next_state;
            }
            info.current_output_length += num_new_tokens;
            continue;
        }

        if (!info.in_tree_mode)
            continue;

        for (size_t j = 0; j < num_new_tokens; ++j) {
            RTP_LLM_CHECK_WITH_INFO(j + offset < new_tokens->shape()[1],
                                    "tree token offset [%zu] exceeds token buffer width [%zu]",
                                    j + offset,
                                    new_tokens->shape()[1]);
            auto current_token_id = *(*new_tokens)[i].dataWithOffset<int>(j + offset);
            info.dfa_ptr->next(current_token_id);
        }

        info.current_output_length += num_new_tokens;
    }
}

TreeLogitsProcessorPtr TreeLogitsProcessor::fromGenerateInput(rtp_llm::DeviceBase*           device,
                                                              std::shared_ptr<GenerateInput> generate_input,
                                                              int32_t                        num) {
    const auto csr_snapshot          = ConstraintTreeCsrManager::instance()->snapshot();
    const bool runtime_tree_required = autil::EnvUtil::getEnv("CONSTRAINT_TREE_REQUIRED", false);
    const auto admission_error =
        validateCsrRequest(csr_snapshot, *generate_input->generate_config, runtime_tree_required);
    RTP_LLM_CHECK_WITH_INFO(admission_error.empty(), "%s", admission_error.c_str());
    if (csr_snapshot) {
        RTP_LLM_CHECK_WITH_INFO(device->getDeviceProperties().type != DeviceType::Cuda || csr_snapshot->deviceReady(),
                                "runtime constraint tree version [%llu] has no GPU buffers",
                                static_cast<unsigned long long>(csr_snapshot->version()));
        auto processor_ptr = std::make_shared<TreeLogitsProcessor>(device);
        for (size_t i = 0; i < num; ++i) {
            StreamTreeInfo tree_info(
                true, generate_input->inputLength(), 0, generate_input->generate_config->hasNumBeams(), csr_snapshot);
            auto single_processor =
                std::make_shared<TreeLogitsProcessor>(device, std::vector<StreamTreeInfo>{std::move(tree_info)});
            processor_ptr->insert(single_processor, 1);
        }
        return processor_ptr;
    }

    if (!PrefixToCandidateTokens::instance()->initSuccess()) {
        return nullptr;
    }

    auto processor_ptr = std::make_shared<TreeLogitsProcessor>(rtp_llm::DeviceFactory::getDefaultDevice());
    for (size_t i = 0; i < num; i++) {
        StreamTreeInfo              tree_info(PrefixToCandidateTokens::instance()->initSuccess(),
                                 generate_input->inputLength(),
                                 0,
                                 generate_input->generate_config->hasNumBeams(),
                                 std::make_shared<TreeDFA<std::string, int>>(PrefixToCandidateTokens::instance()));
        std::vector<StreamTreeInfo> tree_infos       = {tree_info};
        auto                        single_processor = std::make_shared<TreeLogitsProcessor>(device, tree_infos);

        processor_ptr->insert(single_processor, 1);
    }

    return processor_ptr;
}

std::string TreeLogitsProcessor::validateCsrRequest(const ConstraintTreeCsrSnapshotPtr& snapshot,
                                                    const GenerateConfig&               generate_config,
                                                    bool                                runtime_tree_required) {
    if (!snapshot) {
        return runtime_tree_required ? "runtime constraint tree is required but no CSR snapshot is active" :
                                       std::string();
    }
    if (!generate_config.variable_num_beams.empty()) {
        return "runtime constraint tree does not support variable_num_beams in this release";
    }
    if (generate_config.num_beams <= 0) {
        return "runtime constraint tree requires num_beams to be positive";
    }
    if (snapshot->rootCandidateCount() < static_cast<size_t>(generate_config.num_beams)) {
        return "runtime constraint tree root candidate count [" + std::to_string(snapshot->rootCandidateCount())
               + "] is smaller than num_beams [" + std::to_string(generate_config.num_beams) + "]";
    }
    return {};
}

std::vector<std::string> TreeLogitsProcessor::getStatus() {
    std::vector<std::string> status_list;
    for (const auto& tree_info : tree_infos_) {
        if (tree_info.csr_snapshot) {
            status_list.push_back("csr:v" + std::to_string(tree_info.csr_snapshot->version())
                                  + ":state:" + std::to_string(tree_info.csr_state));
        } else {
            status_list.push_back(tree_info.dfa_ptr->status());
        }
    }
    return status_list;
}

}  // namespace rtp_llm
