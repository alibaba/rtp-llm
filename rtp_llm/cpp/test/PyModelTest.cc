#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"

#include "rtp_llm/cpp/test/ModelTestUtil.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/devices/torch_impl/GptModel.hpp"
#include "rtp_llm/cpp/models/PyWrappedModel.h"

using namespace std;
using namespace rtp_llm;

class PyModelTest: public DeviceTestBase {};

TEST_F(PyModelTest, testSimple) {
    const auto path    = test_data_path_ + "../../test/model_test/fake_test/testdata/qwen_0.5b";
    auto       weights = loadWeightsFromDir(path);
    RTP_LLM_CHECK(weights->lm_head->kernel != nullptr);
    RTP_LLM_CHECK(weights->embedding != nullptr);
    RTP_LLM_CHECK(weights->layers.size() == 4);

    const auto py_path = test_data_path_ + "python/fake_gpt_model.py";
    RTP_LLM_LOG_INFO("Using python model path: %s", py_path.c_str());

    GptModelDescription description;
    description.ffn_conf.activation_type = ActivationType::Swiglu;
    description.norm_type                = NormType::rmsnorm;
    auto& attention_conf                 = description.attention_conf;
    attention_conf.head_num              = 16;
    attention_conf.kv_head_num           = 16;
    attention_conf.size_per_head         = 64;
    attention_conf.tokens_per_block      = 16;
    attention_conf.rope_config.style     = RopeStyle::Base;
    attention_conf.rope_config.dim       = 64;
    attention_conf.rope_config.base      = 1000000;
    attention_conf.is_causal             = true;

    const auto  cache_block_num = 128;
    CacheConfig cache_config(KVCacheParam({static_cast<uint>(weights->layers.size()),
                                           cache_block_num,
                                           static_cast<uint>(attention_conf.kv_head_num),
                                           static_cast<uint>(attention_conf.size_per_head),
                                           static_cast<uint>(attention_conf.tokens_per_block),
                                           DataType::TYPE_FP16}));

    const std::vector<int32_t> input_lengths_vec = {3};

    auto       combo_tokens      = createBuffer<int32_t>({3}, {13048, 11, 220}, AllocationType::HOST);
    auto       input_lengths     = createBuffer<int32_t>({1}, input_lengths_vec, AllocationType::HOST);
    auto       sequence_lengths  = createBuffer<int32_t>({0}, {}, AllocationType::HOST);
    auto       kv_cache          = torch::empty(0);
    auto       kv_cache_block_id = allocateKVBlocks(cache_config, input_lengths_vec, kv_cache);
    const auto mask_tensor       = create_context_mask(input_lengths_vec).to(torch::kFloat16);
    const auto mask_buf          = tensorToBuffer(mask_tensor);
    // auto model = make_unique<PyWrappedModel>(
    //     GptModelInitParams({device_, *weights, description, cache_manager_->kvCacheBuffer()}), py_path, "GptModel");

    // GptModelInputs inputs;
    // inputs.combo_tokens =  std::move(combo_tokens);
    // inputs.input_lengths = std::move(input_lengths);
    // inputs.sequence_lengths = std::move(sequence_lengths);

    // inputs.prefix_lengths = createBuffer<int32_t>({1}, {0}, AllocationType::HOST);
    // inputs.lm_output_indexes = createBuffer<int32_t>({1}, {2}, AllocationType::HOST);
    // inputs.attention_mask = mask_buf;
    // inputs.kv_cache_block_id = kv_cache_block_id;
    // device_->syncAndCheck();

    // auto outputs = model->forward(inputs);
    // device_->syncAndCheck();
}
