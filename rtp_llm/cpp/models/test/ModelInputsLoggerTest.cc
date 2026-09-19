#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
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
        std::filesystem::remove_all(root_);
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
        inputs.combo_tokens        = torch::tensor({1, 2, 3}, torch::kInt32).cuda();
        inputs.prefix_lengths      = torch::tensor({0}, torch::kInt32);
        inputs.kv_cache_group_tags = {"linear", "full"};
        ModelInputsLogger logger(0, 1, nullptr);
        logger.log(inputs, ModelInputsModelRole::NORMAL, 7);
        GptModelInputs oversized{};
        oversized.kv_cache_group_tags = {std::string(64ULL * 1024ULL * 1024ULL, 'x')};
        logger.log(oversized, ModelInputsModelRole::NORMAL, 9);
        inputs.combo_tokens        = torch::tensor({4}, torch::kInt32).cuda();
        inputs.kv_cache_group_tags = {"changed"};
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
    const auto group_tags = payload.at("kv_cache_group_tags").toList();
    ASSERT_EQ(group_tags.size(), 2);
    EXPECT_EQ(group_tags.get(0).toStringRef(), "linear");
    EXPECT_EQ(group_tags.get(1).toStringRef(), "full");
    EXPECT_EQ(records.get(1).toGenericDict().at("kv_cache_group_tags").toList().get(0).toStringRef(), "changed");
    EXPECT_TRUE(torch::equal(payload.at("combo_tokens").toTensor(), torch::tensor({1, 2, 3}, torch::kInt32)));
    EXPECT_EQ(records.get(1).toGenericDict().at("model_id").toInt(), 8);
    EXPECT_EQ(records.get(1).toGenericDict().at("dropped_before").toInt(), 1);
}
}  // namespace
}  // namespace rtp_llm
