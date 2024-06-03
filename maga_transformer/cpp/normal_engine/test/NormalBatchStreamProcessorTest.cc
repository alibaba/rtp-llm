#include "torch/all.h"
#include "gtest/gtest.h"
#include <memory>

#define private public
#include "maga_transformer/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "maga_transformer/cpp/dataclass/Query.h"
#include "src/fastertransformer/devices/testing/TestBase.h"
#include "src/fastertransformer/core/BufferHelper.h"

using namespace std;

namespace rtp_llm {

class NormalBatchStreamProcessorTest: public DeviceTestBase {
};

TEST_F(NormalBatchStreamProcessorTest, testSimpleAssemble) {
    ResourceContext resource_context;
    GptInitParameter param;
    param.max_seq_len_   = 2048;
    param.num_layers_    = 2;
    param.int8_kv_cache_ = true;
    NormalBatchStreamProcessor     processor(param, true);
    std::shared_ptr<GenerateInput> query1 = make_shared<GenerateInput>();
    query1->input_ids                     = createBuffer<int32_t>({2}, {1, 2}, AllocationType::HOST);
    query1->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream1             = make_shared<GenerateStream>(query1, param, resource_context, nullptr);
    query1->input_ids                     = createBuffer<int32_t>({1}, {1}, AllocationType::HOST);
    BatchKVCacheBlockAddr addr1;
    addr1.k_ptr       = {{{(void*)1, (void*)(2)}, {(void*)3, (void*)(4)}}};
    addr1.v_ptr       = {{{(void*)5, (void*)(6)}, {(void*)7, (void*)(8)}}};
    addr1.k_scale_ptr = {{{(void*)11, (void*)(12)}, {(void*)13, (void*)(14)}}};
    addr1.v_scale_ptr = {{{(void*)15, (void*)(16)}, {(void*)17, (void*)(18)}}};
    stream1->setKVCache(addr1);
    stream1->setIsContextStream(false);

    std::shared_ptr<GenerateInput> query2 = make_shared<GenerateInput>();
    query2->input_ids                     = createBuffer<int32_t>({3}, {1, 2, 3}, AllocationType::HOST);
    query2->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream2             = make_shared<GenerateStream>(query2, param, resource_context, nullptr);
    query2->input_ids                     = createBuffer<int32_t>({2}, {1, 2}, AllocationType::HOST);
    BatchKVCacheBlockAddr addr2;
    addr2.k_ptr       = {{{(void*)10, (void*)(20)}, {(void*)30, (void*)(40)}}};
    addr2.v_ptr       = {{{(void*)50, (void*)(60)}, {(void*)70, (void*)(80)}}};
    addr2.k_scale_ptr = {{{(void*)110, (void*)(120)}, {(void*)130, (void*)(140)}}};
    addr2.v_scale_ptr = {{{(void*)150, (void*)(160)}, {(void*)170, (void*)(180)}}};
    stream2->setKVCache(addr2);
    stream2->setIsContextStream(false);

    std::shared_ptr<GenerateInput> query3 = make_shared<GenerateInput>();
    query3->input_ids                     = createBuffer<int32_t>({3}, {1, 2, 3}, AllocationType::HOST);
    query3->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr     stream3         = make_shared<GenerateStream>(query3, param, resource_context, nullptr);
    BatchKVCacheBlockAddr addr3;
    addr3.k_ptr       = {{{(void*)100}, {(void*)300}}};
    addr3.v_ptr       = {{{(void*)500}, {(void*)700}}};
    addr3.k_scale_ptr = {{{(void*)1100}, {(void*)1300}}};
    addr3.v_scale_ptr = {{{(void*)1500}, {(void*)1700}}};
    stream3->setKVCache(addr3);

    std::shared_ptr<GenerateInput> query4 = make_shared<GenerateInput>();
    query4->input_ids                     = createBuffer<int32_t>({4}, {1, 2, 3, 4}, AllocationType::HOST);
    query4->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr     stream4         = make_shared<GenerateStream>(query4, param, resource_context, nullptr);
    BatchKVCacheBlockAddr addr4;
    addr4.k_ptr       = {{{(void*)1000, (void*)(2000)}, {(void*)3000, (void*)(4000)}}};
    addr4.v_ptr       = {{{(void*)5000, (void*)(6000)}, {(void*)7000, (void*)(8000)}}};
    addr4.k_scale_ptr = {{{(void*)11000, (void*)(12000)}, {(void*)13000, (void*)(14000)}}};
    addr4.v_scale_ptr = {{{(void*)15000, (void*)(16000)}, {(void*)17000, (void*)(18000)}}};
    stream4->setKVCache(addr4);
    stream4->setReuseLength(1);

    std::list<GenerateStreamPtr> streams;
    streams.emplace_back(stream1);
    streams.emplace_back(stream2);
    streams.emplace_back(stream3);
    streams.emplace_back(stream4);

    for (const auto& stream: streams) {
        stream->setRunning();
    }

    {
        StreamGroups stream_groups(streams);

        auto merge_input_status = processor.gatherModelInput(stream_groups);

        EXPECT_TRUE(merge_input_status.ok());
        auto&            model_input      = merge_input_status.value();
        vector<int>      combo_tokens     = {2, 3, 1, 2, 3, 2, 3, 4};
        vector<int>      input_lengths    = {1, 2, 3, 3};
        vector<int>      sequence_lengths = {1, 2};
        vector<int>      prefix_lengths   = {0, 0, 0, 1};
        vector<uint64_t> kv_cache_blocks  = {1, 2, 5, 6, 10, 20, 50, 60, 100, 0, 500, 0, 1000, 2000, 5000, 6000,
                                             3, 4, 7, 8, 30, 40, 70, 80, 300, 0, 700, 0, 3000, 4000, 7000, 8000};
        vector<uint64_t> kv_cache_scales  = {11,  12,    15,    16,    110,   120, 150,   160,   1100,  0,    1500,
                                             0,   11000, 12000, 15000, 16000, 13,  14,    17,    18,    130,  140,
                                             170, 180,   1300,  0,     1700,  0,   13000, 14000, 17000, 18000};
        EXPECT_EQ(combo_tokens, buffer2vector<int>(*model_input.combo_tokens));
        EXPECT_EQ(input_lengths, buffer2vector<int>(*model_input.input_lengths));
        EXPECT_EQ(sequence_lengths, buffer2vector<int>(*model_input.sequence_lengths));
        EXPECT_EQ(prefix_lengths, buffer2vector<int>(*model_input.prefix_lengths));
        EXPECT_EQ(kv_cache_blocks, buffer2vector<uint64_t>(*model_input.kv_cache_blocks));
        EXPECT_EQ(kv_cache_scales, buffer2vector<uint64_t>(*model_input.kv_cache_scales));
        EXPECT_EQ(model_input.attention_mask->size(), 2 * 3 * 4);
    }

    {
        NormalBatchStreamProcessor     processor(param, false);
        StreamGroups stream_groups(streams);
        auto merge_input_status = processor.gatherModelInput(stream_groups);
        EXPECT_TRUE(merge_input_status.ok());
        auto&            model_input      = merge_input_status.value();
        EXPECT_EQ(model_input.attention_mask.get(), nullptr);
    }
}

}  // namespace rtp_llm
