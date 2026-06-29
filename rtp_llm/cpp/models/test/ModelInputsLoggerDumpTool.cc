#include "rtp_llm/cpp/models/ModelInputsLogger.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace rtp_llm {
namespace {

GptModelInputs makeInputs() {
    GptModelInputs inputs;
    inputs.trace_ids                 = {"trace-a", "trace-b"};
    inputs.combo_tokens              = torch::tensor({11, 12, 13}, torch::kInt32);
    inputs.input_lengths             = torch::tensor({3}, torch::kInt32);
    inputs.sequence_lengths          = torch::tensor({2}, torch::kInt32);
    inputs.lm_output_indexes         = torch::tensor({2}, torch::kInt32);
    inputs.prefix_lengths            = torch::tensor({1}, torch::kInt32);
    inputs.sequence_lengths_plus_1   = torch::tensor({3}, torch::kInt32);
    inputs.combo_tokens_type_ids     = torch::tensor({0, 0, 1}, torch::kInt32);
    inputs.combo_position_ids        = torch::tensor({0, 1, 2}, torch::kInt32);
    inputs.kv_cache_block_id         = torch::tensor({{{7, 8}}}, torch::kInt32);
    inputs.kv_cache_layer_to_group   = torch::tensor({0}, torch::kInt32);
    inputs.kv_cache_group_types      = torch::tensor({1}, torch::kInt32);
    inputs.kv_cache_update_mapping   = torch::tensor({{1, 2}}, torch::kInt32);
    inputs.request_id                = torch::tensor({12345}, torch::kInt64);
    inputs.request_pd_separation     = torch::tensor({false}, torch::kBool);
    inputs.kv_block_stride_bytes     = 4096;
    inputs.kv_scale_stride_bytes     = 128;
    inputs.seq_size_per_block        = 64;
    inputs.kernel_seq_size_per_block = 32;
    inputs.decode_entrance           = true;
    inputs.use_opaque_kv_cache_store = true;
    inputs.need_all_logits           = true;
    inputs.need_all_hidden_states    = true;
    inputs.need_moe_gating           = true;
    return inputs;
}

}  // namespace
}  // namespace rtp_llm

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::cerr << "usage: " << argv[0] << " <log_dir> [dump_count]" << std::endl;
        return 2;
    }

    const auto dump_count = argc == 3 ? std::stoi(argv[2]) : 1;
    setenv("LOG_PATH", argv[1], 1);
    setenv("FRONTEND_SERVER_ID", "3", 1);
    rtp_llm::ModelInputsLogger logger(/*rank_id=*/2, /*backup_count=*/1, nullptr);
    for (int i = 0; i < dump_count; ++i) {
        logger.log(rtp_llm::makeInputs());
    }
    return 0;
}
