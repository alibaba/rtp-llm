#include <cstdint>
#include <iostream>

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

int main() {
    constexpr int64_t long_duration_us = (int64_t{1} << 31) + 12345;

    GenerateOutputsPB outputs;
    outputs.set_request_id(123);

    auto* output = outputs.mutable_flatten_output();
    output->add_finished(true);

    auto* output_ids = output->mutable_output_ids();
    output_ids->set_data_type(TensorPB::INT32);
    output_ids->add_shape(1);
    output_ids->add_shape(1);
    const int32_t token_id = 0;
    output_ids->set_int32_data(&token_id, sizeof(token_id));

    auto* aux_info = output->add_aux_info();
    aux_info->set_cost_time_us(long_duration_us);
    aux_info->set_first_token_cost_time_us(long_duration_us - 1);
    aux_info->set_wait_time_us(long_duration_us - 2);
    aux_info->set_iter_count(1);
    aux_info->set_input_len(3);
    aux_info->set_output_len(1);

    return outputs.SerializeToOstream(&std::cout) ? 0 : 1;
}
