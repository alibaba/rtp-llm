
#include "gtest/gtest.h"

#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/models/Sampler.h"

using namespace std;

namespace rtp_llm {

class SamplerDataBuilder {
public:
    SamplerDataBuilder() = default;

    struct Config {
        size_t            batch_size;
        size_t            vocab_size;
        size_t            max_length;
        rtp_llm::DataType logits_type = rtp_llm::DataType::TYPE_FP32;
    };

    SamplerInputs allocate(Config config) {
        SamplerInputs sampler_inputs;
        sampler_inputs.step           = config.max_length;
        sampler_inputs.batch_size     = config.batch_size;
        sampler_inputs.batch_size_out = config.batch_size;
        auto bs                       = (int64_t)config.batch_size;
        sampler_inputs.logits         = torch::empty(
            {bs, (int64_t)config.vocab_size},
            torch::TensorOptions().dtype(rtp_llm::dataTypeToTorchType(config.logits_type)).device(torch::kCUDA));
        sampler_inputs.sequence_lengths   = torch::empty({bs}, torch::kInt32);
        sampler_inputs.input_lengths      = torch::empty({bs}, torch::kInt32);
        sampler_inputs.num_beams_in       = torch::empty({bs}, torch::kLong);
        sampler_inputs.num_beams_out      = torch::empty({bs}, torch::kLong);
        sampler_inputs.top_k              = torch::empty({bs}, torch::kInt32);
        sampler_inputs.top_p              = torch::empty({bs}, torch::kFloat32);
        sampler_inputs.temperature        = torch::empty({bs}, torch::kFloat32);
        sampler_inputs.repetition_penalty = torch::empty({bs}, torch::kFloat32);
        sampler_inputs.cum_log_probs      = torch::empty({bs}, torch::kFloat32);
        sampler_inputs.token_ids          = torch::empty({bs, (int64_t)(sampler_inputs.step + 1)}, torch::kInt32);
        return sampler_inputs;
    };

    void setSequenceLengths(SamplerInputs& sampler_inputs, std::vector<int>& sequence_lengths) {
        RTP_LLM_CHECK(sequence_lengths.size() == sampler_inputs.batch_size);
        sampler_inputs.sequence_lengths = torch::tensor(sequence_lengths, torch::kInt32);
    };
};

class ModelDataTest: public DeviceTestBase {};

TEST(LinearReplayInputsTest, RequestSlicePreservesPhysicalGroupsAndInt64Epochs) {
    auto inputs = LinearReplayInputs::allocate(8, 3, torch::kCPU);
    inputs.slot_ids.copy_(torch::arange(8, torch::kInt32));
    inputs.active_block_ids.copy_(torch::arange(24, torch::kInt32).reshape({3, 8}));
    inputs.state_read_block_ids.copy_(inputs.active_block_ids + 100);
    const int64_t epoch = int64_t{1} << 40;
    inputs.verify_epochs.fill_(epoch);
    inputs.slot_generations.fill_(epoch + 1);

    auto rows = inputs.slice(2, 3);
    EXPECT_EQ(rows.slot_ids.sizes(), torch::IntArrayRef({3}));
    EXPECT_EQ(rows.active_block_ids.sizes(), torch::IntArrayRef({3, 3}));
    EXPECT_EQ(rows.active_block_ids.stride(0), 8);
    EXPECT_FALSE(rows.active_block_ids.is_contiguous());
    EXPECT_TRUE(torch::equal(rows.slot_ids, torch::tensor({2, 3, 4}, torch::kInt32)));
    EXPECT_TRUE(
        torch::equal(rows.active_block_ids, torch::tensor({{2, 3, 4}, {10, 11, 12}, {18, 19, 20}}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(rows.state_read_block_ids, rows.active_block_ids + 100));
    EXPECT_EQ(rows.verify_epochs.scalar_type(), torch::kInt64);
    EXPECT_EQ(rows.verify_epochs[0].item<int64_t>(), epoch);
    EXPECT_EQ(rows.slot_generations[0].item<int64_t>(), epoch + 1);

    // Graph slices must keep the same pool backing, including after the parent is released.
    auto* backing = rows.active_block_ids.data_ptr<int>();
    inputs        = LinearReplayInputs{};
    EXPECT_EQ(rows.active_block_ids.data_ptr<int>(), backing);
    EXPECT_EQ(rows.active_block_ids[2][2].item<int>(), 20);
}

TEST_F(ModelDataTest, testConstruct) {
    SamplerDataBuilder builder;
    SamplerInputs      sampler_inputs   = builder.allocate({4, 1024, 1024});
    std::vector<int>   sequence_lengths = {1, 2, 3, 4};
    builder.setSequenceLengths(sampler_inputs, sequence_lengths);
    auto sl = sampler_inputs.sequence_lengths;
    EXPECT_EQ(std::vector<int>(sl.data_ptr<int>(), sl.data_ptr<int>() + sl.numel()), std::vector<int>({1, 2, 3, 4}));
}

TEST_F(ModelDataTest, testTensorHolderReleasesOnThirdRound) {
    TensorHolder holder;
    auto         t0 = torch::empty({1}, torch::kFloat32);
    auto         t1 = torch::empty({1}, torch::kFloat32);
    auto         t2 = torch::empty({1}, torch::kFloat32);

    holder.hold(t0);
    holder.release();
    ASSERT_EQ(holder.clear_tensors.size(), 1);
    EXPECT_EQ(holder.clear_tensors.front().front().data_ptr(), t0.data_ptr());

    holder.hold(t1);
    holder.release();
    ASSERT_EQ(holder.clear_tensors.size(), 2);
    EXPECT_EQ(holder.clear_tensors.front().front().data_ptr(), t0.data_ptr());

    holder.hold(t2);
    holder.release();
    ASSERT_EQ(holder.clear_tensors.size(), 2);
    EXPECT_EQ(holder.clear_tensors.front().front().data_ptr(), t1.data_ptr());
}

TEST_F(ModelDataTest, MtpDraftUpdatePaddingPreservesQueryWidthAndPrefixMetadata) {
    py::scoped_interpreter interpreter;
    py::class_<torch_ext::PyModelInitResources>(py::module_::import("__main__"), "PyModelInitResources");
    auto py_model = py::module_::import("types").attr("SimpleNamespace")();
    py_model.attr("requires_sequence_parallel_padding") = true;
    py_model.attr("initialize") = py::cpp_function([](py::object) { return true; });
    GptModelInitParams params{};
    params.parallelism_config.tp_size = 16;
    params.hw_kernel_config.enable_cuda_graph = false;
    PyWrappedModel model(params, py_model);
    const auto cuda_i32 = torch::TensorOptions(torch::kInt32).device(torch::kCUDA);

    for (int64_t batch_size : {1, 2, 3, 4}) {
        SCOPED_TRACE(batch_size);
        torch_ext::PyModelInputs inputs;
        inputs.input_ids = torch::arange(batch_size * 4, cuda_i32);
        auto& attention = inputs.attention_inputs;
        attention.is_prefill = true;
        attention.is_mtp_draft_update = true;
        attention.total_tokens = batch_size * 4;
        attention.input_lengths = torch::full({batch_size}, 4, cuda_i32);
        attention.input_lengths_host = attention.input_lengths.cpu();
        attention.prefix_lengths = torch::arange(100, 100 + batch_size, cuda_i32);
        attention.prefix_lengths_host = attention.prefix_lengths.cpu();
        attention.sequence_lengths = torch::empty({0}, cuda_i32);
        attention.sequence_lengths_host = torch::empty({0}, torch::kInt32);
        attention.sequence_lengths_plus_1_d = attention.prefix_lengths + 1;
        attention.kv_cache_kernel_block_id_device = torch::full({batch_size, 2}, 7, cuda_i32);
        attention.kv_cache_kernel_block_id_device_by_group = {attention.kv_cache_kernel_block_id_device};

        model.padTensorParallelInputs(inputs);

        const auto expected_lengths = torch::tensor({4, 4, 4, 4}, torch::kInt32);
        EXPECT_TRUE(torch::equal(attention.input_lengths.cpu(), expected_lengths));
        EXPECT_TRUE(torch::equal(attention.input_lengths_host, expected_lengths));
        ASSERT_EQ(attention.prefix_lengths.numel(), 4);
        auto expected_prefix = torch::zeros({4}, torch::kInt32);
        expected_prefix.narrow(0, 0, batch_size).copy_(torch::arange(100, 100 + batch_size, torch::kInt32));
        EXPECT_TRUE(torch::equal(attention.prefix_lengths.cpu(), expected_prefix));
        EXPECT_TRUE(torch::equal(attention.prefix_lengths_host, expected_prefix));
        EXPECT_EQ(attention.sequence_lengths.numel(), 0);
        EXPECT_EQ(attention.sequence_lengths_host.numel(), 0);
        EXPECT_TRUE(torch::equal(attention.sequence_lengths_plus_1_d.cpu(), expected_prefix + 1));
        EXPECT_EQ(attention.logical_request_count, batch_size);
        EXPECT_EQ(attention.physical_request_count, 4);
        EXPECT_EQ(attention.logical_token_count, batch_size * 4);
        EXPECT_EQ(attention.physical_token_count, 16);
        EXPECT_EQ(attention.total_tokens, 16);
        EXPECT_EQ(inputs.input_ids.numel(), 16);
        EXPECT_TRUE(torch::equal(inputs.input_ids.narrow(0, 0, batch_size * 4),
                                 torch::arange(batch_size * 4, cuda_i32)));
        auto expected_table = torch::zeros({4, 2}, torch::kInt32);
        expected_table.narrow(0, 0, batch_size).fill_(7);
        EXPECT_TRUE(torch::equal(attention.kv_cache_kernel_block_id_device.cpu(), expected_table));
        EXPECT_TRUE(torch::equal(attention.kv_cache_kernel_block_id_device_by_group[0].cpu(), expected_table));
        if (batch_size < 4) {
            const auto expected_cu = torch::tensor({0, 4, 8, 12, 16}, torch::kInt32);
            EXPECT_TRUE(torch::equal(attention.cu_seqlens.cpu(), expected_cu));
            EXPECT_TRUE(torch::equal(attention.cu_seqlens_host, expected_cu));
        }
    }
}

}  // namespace rtp_llm
