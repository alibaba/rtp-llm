#pragma once

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/DFAUtil.h"

namespace rtp_llm {

struct StreamTreeInfo {
    bool                                       in_tree_mode;
    int32_t                                    input_length;
    int32_t                                    current_output_length;
    bool                                       is_beam_search;
    std::shared_ptr<TreeDFA<std::string, int>> dfa_ptr;
    StreamTreeInfo() = default;
    StreamTreeInfo(bool                                       in_tree_mode,
                   int32_t                                    input_length,
                   int32_t                                    output_length,
                   bool                                       is_beam_search,
                   std::shared_ptr<TreeDFA<std::string, int>> dfa_ptr):
        in_tree_mode(in_tree_mode),
        input_length(input_length),
        current_output_length(output_length),
        is_beam_search(is_beam_search),
        dfa_ptr(dfa_ptr) {}
    StreamTreeInfo copy() {
        StreamTreeInfo tree_info;
        tree_info.in_tree_mode          = in_tree_mode;
        tree_info.input_length          = input_length;
        tree_info.current_output_length = current_output_length;
        tree_info.is_beam_search        = is_beam_search;
        if (dfa_ptr) {
            tree_info.dfa_ptr = std::make_shared<TreeDFA<std::string, int>>(*dfa_ptr);
        }
        return tree_info;
    }
};

class TreeLogitsProcessor: public BaseLogitsProcessor {
public:
    TreeLogitsProcessor(rtp_llm::DeviceBase* device);
    TreeLogitsProcessor(rtp_llm::DeviceBase* device, std::vector<StreamTreeInfo> tree_infos);
    virtual ~TreeLogitsProcessor() {}

public:
    static std::shared_ptr<TreeLogitsProcessor>
    fromGenerateInput(rtp_llm::DeviceBase* device, std::shared_ptr<GenerateInput> generate_input, int32_t num);

public:
    void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override;
    void updateMultiSeqStatus(const std::vector<int>& src_batch_indices) override;
    void updateStatus(const rtp_llm::BufferPtr& new_tokens, int32_t num_new_tokens) override;

public:
    std::vector<std::string> getStatus();
    size_t                   size() {
        return tree_infos_.size();
    }
    void insert(std::shared_ptr<TreeLogitsProcessor> others, size_t num) {
        if (others != nullptr) {
            tree_infos_.insert(tree_infos_.end(), others->tree_infos_.begin(), others->tree_infos_.end());
        }
    }

private:
    std::vector<StreamTreeInfo> tree_infos_;
};
typedef std::shared_ptr<TreeLogitsProcessor> TreeLogitsProcessorPtr;

}  // namespace rtp_llm