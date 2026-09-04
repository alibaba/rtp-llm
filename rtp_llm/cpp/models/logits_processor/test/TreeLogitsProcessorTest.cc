
#include "gtest/gtest.h"

#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/models/logits_processor/TreeLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/models/logits_processor/PrefixToCandidateTokens.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <unistd.h>

using namespace std;

namespace rtp_llm {

namespace {

void appendU32(std::string& output, uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        output.push_back(static_cast<char>((value >> shift) & 0xff));
    }
}

void appendU64(std::string& output, uint64_t value) {
    appendU32(output, static_cast<uint32_t>(value));
    appendU32(output, static_cast<uint32_t>(value >> 32));
}

std::string makeCsrArtifact(uint64_t                    version,
                            int32_t                     start_token_id,
                            int32_t                     end_token_id,
                            uint64_t                    sid_count,
                            const std::vector<int32_t>& row_ptr,
                            const std::vector<int32_t>& col_idx,
                            const std::vector<int32_t>& next_state) {
    std::string output("RTPCSR01", 8);
    appendU32(output, 1);
    appendU32(output, 48);
    appendU64(output, version);
    appendU32(output, static_cast<uint32_t>(start_token_id));
    appendU32(output, static_cast<uint32_t>(end_token_id));
    appendU32(output, static_cast<uint32_t>(row_ptr.size() - 1));
    appendU32(output, static_cast<uint32_t>(col_idx.size()));
    appendU64(output, sid_count);
    for (const int32_t value : row_ptr) {
        appendU32(output, static_cast<uint32_t>(value));
    }
    for (const int32_t value : col_idx) {
        appendU32(output, static_cast<uint32_t>(value));
    }
    for (const int32_t value : next_state) {
        appendU32(output, static_cast<uint32_t>(value));
    }
    return output;
}

}  // namespace

class SamplerDataBuilder {
public:
    SamplerDataBuilder(): device_(rtp_llm::DeviceFactory::getDefaultDevice()){};

    struct Config {
        size_t            batch_size;
        size_t            vocab_size;
        size_t            max_length;
        rtp_llm::DataType logits_type = rtp_llm::DataType::TYPE_FP32;
    };

    BaseLogitsProcessorPtr generateLogitsProcessor(bool in_tree_mode, size_t batch_size, std::string file_path) {
        std::vector<StreamTreeInfo> tree_infos;

        PrefixToCandidateTokens::instance()->reloadPrefixDict(file_path);

        for (size_t i = 0; i < batch_size; i++) {
            auto tree_info =
                StreamTreeInfo(in_tree_mode,
                               0,
                               0,
                               0,
                               std::make_shared<TreeDFA<std::string, int>>(PrefixToCandidateTokens::instance()));
            tree_infos.push_back(tree_info);
        }

        BaseLogitsProcessorPtr processor_ptr = std::make_shared<TreeLogitsProcessor>(device_, tree_infos);
        return processor_ptr;
    }

    SamplerInputs allocate(Config config, std::vector<BaseLogitsProcessorPtr> processors, std::vector<size_t> nums) {
        SamplerInputs sampler_inputs;

        sampler_inputs.step                = config.max_length;
        sampler_inputs.batch_size          = config.batch_size;
        sampler_inputs.batch_size_out      = config.batch_size;
        sampler_inputs.vocab_size          = config.vocab_size;
        LogitsProcessorStatesPtr state_ptr = std::make_shared<LogitsProcessorStates>();
        for (size_t i = 0, idx = 0; i < processors.size(); i++) {
            state_ptr->insert(processors[i], idx, idx + nums[i]);
            idx += nums[i];
        }
        sampler_inputs.logits_processor_states_ptr = state_ptr;
        sampler_inputs.logits                      = device_->allocateBuffer(
            {config.logits_type, {config.batch_size, config.vocab_size}, rtp_llm::AllocationType::DEVICE}, {});
        sampler_inputs.sequence_lengths = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_INT32, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.input_lengths = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_INT32, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.num_beams_in = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_UINT64, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.num_beams_out = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_UINT64, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.top_k = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_UINT32, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.top_p = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_FP32, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.temperature = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_FP32, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.random_seeds = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_UINT64, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.repetition_penalty = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_FP32, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.min_lengths = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_INT32, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.cum_log_probs = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_FP32, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.token_ids = device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32,
                                                            {config.batch_size, sampler_inputs.step + 1},
                                                            rtp_llm::AllocationType::HOST},
                                                           {});
        device_->bufMemset(*sampler_inputs.logits, 0);
        device_->bufMemset(*sampler_inputs.token_ids, 0);
        return sampler_inputs;
    };

    void setTokenIds(SamplerInputs& sampler_inputs, std::vector<std::vector<int>>& token_ids) {
        RTP_LLM_CHECK(token_ids.size() == sampler_inputs.batch_size);
        RTP_LLM_CHECK(token_ids[0].size() == sampler_inputs.step + 1);
        for (auto i = 0; i < sampler_inputs.batch_size; i++) {
            auto tensor = Buffer2torchTensor(*sampler_inputs.token_ids->index(i), false);
            for (auto j = 0; j < sampler_inputs.step + 1; j++) {
                tensor[j] = token_ids[i][j];
            }
        }
    }

    rtp_llm::DeviceBase* device_;
};

class TreeLogitsProcessorTest: public DeviceTestBase {
protected:
    void SetUp() override {
        DeviceTestBase::SetUp();
    }

    void TearDown() override {
        DeviceTestBase::TearDown();
    }

    rtp_llm::BufferPtr randint(int start, int end, std::vector<int64_t> shape, bool is_host) {
        auto tensor  = torch::randint(start, end, shape, at::TensorOptions().dtype(at::ScalarType::Int));
        auto alloc_t = is_host ? AllocationType::HOST : AllocationType::DEVICE;
        return tensorToBuffer(tensor, alloc_t);
    }

    rtp_llm::BufferPtr rand(std::vector<int64_t> shape, bool is_host) {
        auto tensor  = torch::rand(torch::IntArrayRef(shape));
        auto alloc_t = is_host ? AllocationType::HOST : AllocationType::DEVICE;
        return tensorToBuffer(tensor, alloc_t);
    }
};

#define EXPECT_SIMILAR(vec1, vec2, eps)                                                                                \
    do {                                                                                                               \
        bool similar = true;                                                                                           \
        if (vec1.size() != vec2.size()) {                                                                              \
            similar = false;                                                                                           \
        } else {                                                                                                       \
            for (size_t i = 0; i < vec1.size(); ++i) {                                                                 \
                if (std::fabs(vec1[i] - vec2[i]) >= eps) {                                                             \
                    similar = false;                                                                                   \
                    break;                                                                                             \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        EXPECT_TRUE(similar) << "Vectors are not similar";                                                             \
    } while (0)

TEST_F(TreeLogitsProcessorTest, testGenerateVocabMask) {
    SamplerDataBuilder     builder;
    std::string            file_path  = "./rtp_llm/cpp/models/logits_processor/test/gir_prefix_dict.json";
    size_t                 batch_size = 4;
    size_t                 vocab_size = 1024;
    size_t                 max_length = 1024;
    BaseLogitsProcessorPtr processor  = builder.generateLogitsProcessor(true, batch_size, file_path);
    SamplerInputs sampler_inputs = builder.allocate({batch_size, vocab_size, max_length}, {processor}, {batch_size});
    std::vector<std::vector<size_t>> batch_candidate_token_ids = {{}, {1}, {2, 3, 4}, {1, 3, 5}};
    rtp_llm::BufferPtr vocab_mask = processor->generateVocabMask(batch_size, vocab_size, batch_candidate_token_ids);

    std::vector<std::vector<int32_t>> expect_vocab_mask(batch_size, std::vector<int32_t>(vocab_size, 1));
    expect_vocab_mask[1][1] = 0;
    expect_vocab_mask[2][2] = 0;
    expect_vocab_mask[2][3] = 0;
    expect_vocab_mask[2][4] = 0;
    expect_vocab_mask[3][1] = 0;
    expect_vocab_mask[3][3] = 0;
    expect_vocab_mask[3][5] = 0;

    auto vocab_mask_hosts = getBufferValues<uint8_t>(*vocab_mask);
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < vocab_size; j++) {
            ASSERT_TRUE(vocab_mask_hosts[i * vocab_size + j] == expect_vocab_mask[i][j]);
        }
    }
}

template<typename Dtype>
void setBuffer(rtp_llm::BufferPtr buf, std::vector<std::vector<Dtype>> content) {
    RTP_LLM_CHECK(buf->shape().size() == 2);
    RTP_LLM_CHECK(buf->shape()[0] == content.size());
    RTP_LLM_CHECK(buf->shape()[1] == content[0].size());
    for (auto i = 0; i < buf->shape()[0]; i++) {
        auto tensor = Buffer2torchTensor(*buf->index(i), false);
        for (auto j = 0; j < buf->shape()[1]; j++) {
            tensor[j] = content[i][j];
        }
    }
}

TEST_F(TreeLogitsProcessorTest, testUpdateStatus) {
    {
        SamplerDataBuilder     builder;
        std::string            file_path  = "./rtp_llm/cpp/models/logits_processor/test/gir_prefix_dict.json";
        size_t                 batch_size = 4;
        size_t                 vocab_size = 1024;
        size_t                 max_length = 10;
        BaseLogitsProcessorPtr processor  = builder.generateLogitsProcessor(true, batch_size, file_path);
        SamplerInputs          sampler_inputs =
            builder.allocate({batch_size, vocab_size, max_length}, {processor}, {batch_size});

        rtp_llm::BufferPtr new_token = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_INT32, {batch_size, 1}, rtp_llm::AllocationType::HOST}, {});
        std::vector<std::vector<int>> new_token_ids = {{64000}, {64003}, {64006}, {64008}};
        setBuffer(new_token, new_token_ids);

        processor->updateStatus(new_token, 1);

        auto                     proc        = std::dynamic_pointer_cast<TreeLogitsProcessor>(processor);
        std::vector<std::string> status_list = proc->getStatus();
        EXPECT_EQ("225_64000", status_list[0]);
        EXPECT_EQ("225_64003", status_list[1]);
        EXPECT_EQ("225_64006", status_list[2]);
        EXPECT_EQ("225_64008", status_list[3]);

        std::vector<std::vector<int>> token_ids_2 = {{64001}, {64001}, {64004}, {64001}};
        setBuffer(new_token, token_ids_2);

        processor->updateStatus(new_token, 1);

        status_list = proc->getStatus();
        EXPECT_EQ("225_64000_64001", status_list[0]);
        EXPECT_EQ("225_64003_64001", status_list[1]);
        EXPECT_EQ("225_64006_64004", status_list[2]);
        EXPECT_EQ("225_64008_64001", status_list[3]);

        std::vector<std::vector<int>> token_ids_3 = {{2}, {2}, {2}, {2}};
        setBuffer(new_token, token_ids_3);

        processor->updateStatus(new_token, 1);

        status_list = proc->getStatus();
        EXPECT_EQ("225_64000_64001_2", status_list[0]);
        EXPECT_EQ("225_64003_64001_2", status_list[1]);
        EXPECT_EQ("225_64006_64004_2", status_list[2]);
        EXPECT_EQ("225_64008_64001_2", status_list[3]);

        std::vector<std::vector<int>> token_ids_4 = {{1}, {1}, {1}, {1}};
        setBuffer(new_token, token_ids_4);

        processor->updateStatus(new_token, 1);

        status_list = proc->getStatus();
        EXPECT_EQ("225_64000_64001_2", status_list[0]);
        EXPECT_EQ("225_64003_64001_2", status_list[1]);
        EXPECT_EQ("225_64006_64004_2", status_list[2]);
        EXPECT_EQ("225_64008_64001_2", status_list[3]);
    }
}

TEST_F(TreeLogitsProcessorTest, testProcess) {
    {
        SamplerDataBuilder     builder;
        std::string            file_path  = "./rtp_llm/cpp/models/logits_processor/test/gir_prefix_dict.json";
        size_t                 batch_size = 4;
        size_t                 vocab_size = 100000;
        size_t                 max_length = 10;
        BaseLogitsProcessorPtr processor  = builder.generateLogitsProcessor(true, batch_size, file_path);
        SamplerInputs          sampler_inputs =
            builder.allocate({batch_size, vocab_size, max_length}, {processor}, {batch_size});

        std::vector<std::vector<int>> token_ids = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                   {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                                                   {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51},
                                                   {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9}};
        builder.setTokenIds(sampler_inputs, token_ids);

        std::vector<std::vector<float>> logits_list;
        std::vector<std::vector<float>> logits_index_list = {{64000}, {64003, 64006, 64008}, {64011}, {64001}};
        for (size_t i = 0; i < batch_size; i++) {
            auto logits = sampler_inputs.logits->index(i);
            auto tensor = Buffer2torchTensor(*logits, false);
            tensor.fill_(0);
            for (auto index : logits_index_list[i]) {
                tensor[index] = 1;
            }
        }
        processor->process(sampler_inputs, 0, batch_size);

        auto logits       = sampler_inputs.logits->index(0);
        auto logits_hosts = getBufferValues<float>(*logits);
        ASSERT_EQ(logits_hosts[64000], 1);
        ASSERT_EQ(logits_hosts[64003], 0);
        ASSERT_EQ(logits_hosts[64011], 0);
        ASSERT_TRUE(logits_hosts[64001] == -INFINITY);
    }
}

TEST_F(TreeLogitsProcessorTest, testWideCsrRootUsesGpuMaskAndPinnedSnapshot) {
    constexpr int32_t kFanout    = 20000;
    constexpr int32_t kVocabSize = 25001;
    constexpr int32_t kStart     = 24999;
    constexpr int32_t kEnd       = 25000;

    std::vector<int32_t> row_ptr(kFanout + 2);
    std::vector<int32_t> col_idx(kFanout * 2);
    std::vector<int32_t> next_state(kFanout * 2);
    row_ptr[0] = 0;
    row_ptr[1] = kFanout;
    for (int32_t index = 0; index < kFanout; ++index) {
        col_idx[index]              = index;
        next_state[index]           = index + 1;
        col_idx[kFanout + index]    = kEnd;
        next_state[kFanout + index] = -1;
        row_ptr[index + 2]          = kFanout + index + 1;
    }

    auto           manager = ConstraintTreeCsrManager::instance();
    const uint64_t version = manager->currentVersion() + 1;
    const auto     update  = manager->updateFromBinary(
        makeCsrArtifact(version, kStart, kEnd, kFanout, row_ptr, col_idx, next_state), device_);
    ASSERT_EQ(ConstraintTreeCsrUpdateCode::UPDATED, update.code) << update.message;
    const auto pinned = manager->snapshot();
    ASSERT_TRUE(pinned->deviceReady());

    TreeLogitsProcessor        invalid_request(device_, {StreamTreeInfo(true, 0, 0, false, pinned)});
    const std::vector<int32_t> invalid_token     = {kFanout};
    auto                       invalid_new_token = createHostBuffer<int32_t>({1, 1}, invalid_token.data());
    EXPECT_ANY_THROW(invalid_request.updateStatus(invalid_new_token, 1));
    EXPECT_EQ("csr:v" + std::to_string(version) + ":state:0", invalid_request.getStatus().front());

    TreeLogitsProcessor        premature_eos_request(device_, {StreamTreeInfo(true, 0, 0, false, pinned)});
    const std::vector<int32_t> premature_eos     = {kEnd};
    auto                       premature_eos_buf = createHostBuffer<int32_t>({1, 1}, premature_eos.data());
    EXPECT_ANY_THROW(premature_eos_request.updateStatus(premature_eos_buf, 1));

    TreeLogitsProcessor old_request(device_, {StreamTreeInfo(true, 0, 0, false, pinned)});
    SamplerDataBuilder  builder;
    auto                inputs = builder.allocate({1, kVocabSize, 1}, {}, {});
    Buffer2torchTensor(*inputs.logits, false).fill_(1.0f);
    old_request.process(inputs, 0, 1);
    auto root_logits = getBufferValues<float>(*inputs.logits);
    EXPECT_EQ(1.0f, root_logits[0]);
    EXPECT_EQ(1.0f, root_logits[kFanout - 1]);
    EXPECT_EQ(-INFINITY, root_logits[kFanout]);
    EXPECT_EQ(-INFINITY, root_logits[kEnd]);

    const std::vector<int32_t> selected_token = {12345};
    auto                       new_token      = createHostBuffer<int32_t>({1, 1}, selected_token.data());
    old_request.updateStatus(new_token, 1);
    EXPECT_EQ("csr:v" + std::to_string(version) + ":state:12346", old_request.getStatus().front());
    Buffer2torchTensor(*inputs.logits, false).fill_(1.0f);
    old_request.process(inputs, 0, 1);
    auto leaf_logits = getBufferValues<float>(*inputs.logits);
    EXPECT_EQ(1.0f, leaf_logits[kEnd]);
    EXPECT_EQ(-INFINITY, leaf_logits[12345]);

    const std::vector<int32_t> end_token     = {kEnd};
    auto                       end_token_buf = createHostBuffer<int32_t>({1, 1}, end_token.data());
    old_request.updateStatus(end_token_buf, 1);
    EXPECT_EQ("csr:v" + std::to_string(version) + ":state:-1", old_request.getStatus().front());
    Buffer2torchTensor(*inputs.logits, false).fill_(1.0f);
    old_request.process(inputs, 0, 1);
    const auto terminal_logits = getBufferValues<float>(*inputs.logits);
    EXPECT_TRUE(std::all_of(
        terminal_logits.begin(), terminal_logits.end(), [](float value) { return std::isinf(value) && value < 0; }));
    EXPECT_ANY_THROW(old_request.updateStatus(new_token, 1));

    StreamTreeInfo invalid_state_info(true, 0, 0, false, pinned);
    invalid_state_info.csr_state = static_cast<int32_t>(pinned->stateCount());
    TreeLogitsProcessor invalid_state_request(device_, {invalid_state_info});
    Buffer2torchTensor(*inputs.logits, false).fill_(1.0f);
    invalid_state_request.process(inputs, 0, 1);
    const auto invalid_state_logits = getBufferValues<float>(*inputs.logits);
    EXPECT_TRUE(std::all_of(invalid_state_logits.begin(), invalid_state_logits.end(), [](float value) {
        return std::isinf(value) && value < 0;
    }));

    const uint64_t next_version = version + 1;
    const auto     next_update  = manager->updateFromBinary(
        makeCsrArtifact(next_version, kStart, kEnd, 1, {0, 1, 2}, {24000, kEnd}, {1, -1}), device_);
    ASSERT_EQ(ConstraintTreeCsrUpdateCode::UPDATED, next_update.code) << next_update.message;

    // Simulate one old in-flight request and one request created after the swap.
    TreeLogitsProcessor pinned_request(device_, {StreamTreeInfo(true, 0, 0, false, pinned)});
    TreeLogitsProcessor new_request(device_, {StreamTreeInfo(true, 0, 0, false, manager->snapshot())});
    auto                two_rows = builder.allocate({2, kVocabSize, 1}, {}, {});
    Buffer2torchTensor(*two_rows.logits, false).fill_(1.0f);
    pinned_request.process(two_rows, 0, 1);
    new_request.process(two_rows, 1, 2);
    auto swapped_logits = getBufferValues<float>(*two_rows.logits);
    EXPECT_EQ(1.0f, swapped_logits[0]);
    EXPECT_EQ(-INFINITY, swapped_logits[24000]);
    EXPECT_EQ(-INFINITY, swapped_logits[kVocabSize]);
    EXPECT_EQ(1.0f, swapped_logits[kVocabSize + 24000]);
    EXPECT_NE(std::string::npos, pinned_request.getStatus().front().find("csr:v" + std::to_string(version)));
    EXPECT_NE(std::string::npos, new_request.getStatus().front().find("csr:v" + std::to_string(next_version)));
}

TEST_F(TreeLogitsProcessorTest, testCsrRequestAdmissionIsFailClosedAndSupportsOnlyFixedBeam) {
    GenerateConfig config;
    EXPECT_TRUE(TreeLogitsProcessor::validateCsrRequest(nullptr, config, false).empty());
    EXPECT_NE(std::string::npos,
              TreeLogitsProcessor::validateCsrRequest(nullptr, config, true).find("no CSR snapshot is active"));

    constexpr int32_t kStart  = 100;
    constexpr int32_t kEnd    = 101;
    auto              manager = ConstraintTreeCsrManager::instance();
    const auto        update  = manager->updateFromBinary(
        makeCsrArtifact(
            manager->currentVersion() + 1, kStart, kEnd, 2, {0, 2, 3, 4}, {10, 11, kEnd, kEnd}, {1, 2, -1, -1}),
        device_);
    ASSERT_EQ(ConstraintTreeCsrUpdateCode::UPDATED, update.code) << update.message;
    const auto snapshot = manager->snapshot();
    ASSERT_EQ(2, snapshot->rootCandidateCount());

    config.variable_num_beams = {1, 2};
    EXPECT_NE(std::string::npos,
              TreeLogitsProcessor::validateCsrRequest(snapshot, config, true).find("variable_num_beams"));

    config.variable_num_beams.clear();
    config.num_beams = 3;
    EXPECT_NE(std::string::npos,
              TreeLogitsProcessor::validateCsrRequest(snapshot, config, true).find("smaller than num_beams"));

    config.num_beams = 2;
    EXPECT_TRUE(TreeLogitsProcessor::validateCsrRequest(snapshot, config, true).empty());
}

TEST_F(TreeLogitsProcessorTest, testCsrGpuMaskLatencyRootMiddleAndEos) {
    if (std::getenv("CONSTRAINT_TREE_RUN_GPU_BENCHMARK") == nullptr) {
        GTEST_SKIP() << "set CONSTRAINT_TREE_RUN_GPU_BENCHMARK=1 to run latency measurements";
    }

    constexpr int32_t kVocabSize    = 220000;
    constexpr int32_t kRootFanout   = 30000;
    constexpr int32_t kMiddleFanout = 64;
    constexpr int32_t kMiddleBegin  = 100000;
    constexpr int32_t kEndToken     = 151645;
    constexpr int     kWarmups      = 20;
    constexpr int     kIterations   = 300;

    const std::vector<int32_t> row_ptr = {0, kRootFanout, kRootFanout + kMiddleFanout, kRootFanout + kMiddleFanout + 1};
    std::vector<int32_t>       col_idx;
    col_idx.reserve(row_ptr.back());
    for (int32_t token = 0; token < kRootFanout; ++token) {
        col_idx.push_back(token);
    }
    for (int32_t token = kMiddleBegin; token < kMiddleBegin + kMiddleFanout; ++token) {
        col_idx.push_back(token);
    }
    col_idx.push_back(kEndToken);

    auto row_ptr_host = vector2Buffer(row_ptr);
    auto col_idx_host = vector2Buffer(col_idx);
    auto row_ptr_gpu  = device_->clone({*row_ptr_host, AllocationType::DEVICE});
    auto col_idx_gpu  = device_->clone({*col_idx_host, AllocationType::DEVICE});
    auto logits       = device_->allocateBuffer({DataType::TYPE_FP32, {1, kVocabSize}, AllocationType::DEVICE},
                                          {"csr_mask_benchmark_logits"});
    device_->syncDeviceStream(DeviceStream::DEFAULT);

    const auto measure = [&](const char* scenario, int32_t state, int32_t allowed, int32_t disallowed) {
        const std::vector<int32_t> host_state_values = {state};
        auto                       host_states       = vector2Buffer(host_state_values);
        auto device_states = device_->clone({*host_states, AllocationType::DEVICE, {"csr_mask_benchmark_state"}});
        device_->bufMemset(*logits, 0);
        for (int iteration = 0; iteration < kWarmups; ++iteration) {
            device_->csrMaskLogits(*logits, *device_states, *row_ptr_gpu, *col_idx_gpu);
        }
        device_->syncDeviceStream(DeviceStream::DEFAULT);

        std::vector<double> latency_us;
        latency_us.reserve(kIterations);
        for (int iteration = 0; iteration < kIterations; ++iteration) {
            const auto begin = std::chrono::steady_clock::now();
            device_->csrMaskLogits(*logits, *device_states, *row_ptr_gpu, *col_idx_gpu);
            device_->syncDeviceStream(DeviceStream::DEFAULT);
            const auto end = std::chrono::steady_clock::now();
            latency_us.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
        }
        std::sort(latency_us.begin(), latency_us.end());
        const double p50 = latency_us[(latency_us.size() * 50 + 99) / 100 - 1];
        const double p99 = latency_us[(latency_us.size() * 99 + 99) / 100 - 1];
        std::cout << "CSR_GPU_MASK scenario=" << scenario << " iterations=" << kIterations << " p50_us=" << p50
                  << " p99_us=" << p99 << std::endl;
        EXPECT_GT(p50, 0.0);
        EXPECT_GE(p99, p50);

        const auto masked_logits = getBufferValues<float>(*logits);
        EXPECT_EQ(0.0f, masked_logits[allowed]);
        EXPECT_TRUE(std::isinf(masked_logits[disallowed]) && masked_logits[disallowed] < 0);
    };

    measure("root", 0, 0, kMiddleBegin);
    measure("middle", 1, kMiddleBegin, 0);
    measure("eos", 2, kEndToken, 0);
}

#undef EXPECT_SIMILAR

}  // namespace rtp_llm
