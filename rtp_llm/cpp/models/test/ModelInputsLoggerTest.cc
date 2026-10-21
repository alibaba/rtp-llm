#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <unistd.h>
#include "gtest/gtest.h"
#include "torch/all.h"
#include "torch/serialize.h"
#include "rtp_llm/cpp/models/ModelInputsLogger.h"

namespace rtp_llm {
namespace {
class DumpDirectory {
public:
    DumpDirectory():
        root_(
            std::filesystem::temp_directory_path()
            / ("rtp_llm_model_inputs_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {
        setenv("LOG_PATH", root_.c_str(), 1);
    }
    ~DumpDirectory() {
        if (std::filesystem::exists(output())) {
            for (const auto& entry : std::filesystem::directory_iterator(output())) {
                ::unlink(entry.path().c_str());
            }
            ::rmdir(output().c_str());
        }
        ::rmdir(root_.c_str());
        unsetenv("LOG_PATH");
    }
    std::filesystem::path output() const {
        return root_ / "model_inputs";
    }

private:
    std::filesystem::path root_;
};
TEST(ModelInputsLoggerTest, DumpsLoadableSnapshot) {
    ASSERT_TRUE(torch::cuda::is_available());
    DumpDirectory dump;
    {
        GptModelInputs inputs{};
        inputs.combo_tokens          = torch::tensor({1, 2, 3}, torch::kInt32).cuda();
        inputs.prefix_lengths        = torch::tensor({2, 1}, torch::kInt32);
        inputs.request_id            = torch::tensor({30, 40}, torch::kInt64);
        inputs.all_input_token_ids   = std::vector<torch::Tensor>{torch::tensor({8, 9, 1, 2}, torch::kInt32),
                                                                  torch::tensor({7, 3}, torch::kInt32)};
        inputs.request_is_multimodal = std::vector<bool>{true, false};
        ModelInputsLogger logger(0, 1, nullptr);
        logger.log(inputs, ModelInputsModelRole::NORMAL, 7);
        inputs.all_input_token_ids->front().fill_(99);
        inputs.all_input_token_ids.reset();
        inputs.request_is_multimodal.reset();
        inputs.combo_tokens = torch::tensor({4}, torch::kInt32).cuda();
        logger.log(inputs, ModelInputsModelRole::NORMAL, 8);
    }
    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(dump.output())) {
        if (entry.path().extension() == ".pt") {
            paths.push_back(entry.path());
        }
    }
    ASSERT_EQ(paths.size(), 1);
    std::ifstream     input(paths.front(), std::ios::binary);
    std::vector<char> bytes(std::istreambuf_iterator<char>(input), {});
    const auto        chunk = torch::pickle_load(bytes).toGenericDict();
    EXPECT_EQ(chunk.at("record_type").toStringRef(), "model_inputs_chunk");
    const auto records = chunk.at("records").toList();
    ASSERT_EQ(records.size(), 2);
    const auto payload = records.get(0).toGenericDict();
    EXPECT_EQ(payload.at("model_role").toStringRef(), "normal");
    EXPECT_EQ(payload.at("model_id").toInt(), 7);
    EXPECT_EQ(payload.at("execution_stage").toStringRef(), "prefill");
    EXPECT_TRUE(torch::equal(payload.at("combo_tokens").toTensor(), torch::tensor({1, 2, 3}, torch::kInt32)));
    const auto full_tokens = payload.at("all_input_token_ids").toTensorList();
    ASSERT_EQ(full_tokens.size(), 2);
    EXPECT_TRUE(torch::equal(full_tokens.get(0), torch::tensor({8, 9, 1, 2}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(full_tokens.get(1), torch::tensor({7, 3}, torch::kInt32)));
    const auto modalities = payload.at("request_is_multimodal").toBoolList();
    ASSERT_EQ(modalities.size(), 2);
    EXPECT_TRUE(modalities.get(0));
    EXPECT_FALSE(modalities.get(1));
    EXPECT_TRUE(torch::equal(payload.at("request_id").toTensor(), torch::tensor({30, 40}, torch::kInt64)));
    EXPECT_TRUE(records.get(1).toGenericDict().at("all_input_token_ids").isNone());
    EXPECT_TRUE(records.get(1).toGenericDict().at("request_is_multimodal").isNone());
    EXPECT_EQ(records.get(1).toGenericDict().at("model_id").toInt(), 8);
}
}  // namespace
}  // namespace rtp_llm
