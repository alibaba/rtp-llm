#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <dirent.h>
#include <unistd.h>
#include "gtest/gtest.h"
#include "torch/all.h"
#include "torch/serialize.h"
#include "rtp_llm/cpp/models/ModelInputsLogger.h"

namespace rtp_llm {
namespace {
constexpr const char* kRecordCapacityEnv = "RTP_LLM_MODEL_INPUTS_LOG_MAX_RECORD_BYTES";
constexpr const char* kQueueCapacityEnv  = "RTP_LLM_MODEL_INPUTS_LOG_MAX_QUEUE_BYTES";

class CapacityEnvironment {
public:
    CapacityEnvironment(const char* record = nullptr, const char* queue = nullptr):
        old_record_(previous(kRecordCapacityEnv)), old_queue_(previous(kQueueCapacityEnv)) {
        set(kRecordCapacityEnv, record);
        set(kQueueCapacityEnv, queue);
    }
    ~CapacityEnvironment() {
        set(kRecordCapacityEnv, old_record_ ? old_record_->c_str() : nullptr);
        set(kQueueCapacityEnv, old_queue_ ? old_queue_->c_str() : nullptr);
    }

private:
    static std::optional<std::string> previous(const char* name) {
        const auto* value = std::getenv(name);
        return value ? std::optional<std::string>(value) : std::nullopt;
    }
    static void set(const char* name, const char* value) {
        if (value) {
            ::setenv(name, value, 1);
        } else {
            ::unsetenv(name);
        }
    }
    const std::optional<std::string> old_record_;
    const std::optional<std::string> old_queue_;
};

class DumpDirectory {
public:
    DumpDirectory():
        root_(
            std::filesystem::temp_directory_path()
            / ("rtp_llm_model_inputs_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {
        setenv("LOG_PATH", root_.c_str(), 1);
    }
    ~DumpDirectory() {
        // Raw POSIX cleanup on purpose: std::filesystem::remove_all segfaults
        // on the CUDA 13 A10 test workers.
        const auto output_dir = output();
        if (auto* dir = ::opendir(output_dir.c_str())) {
            while (auto* entry = ::readdir(dir)) {
                if (std::strcmp(entry->d_name, ".") != 0 && std::strcmp(entry->d_name, "..") != 0) {
                    ::unlinkat(::dirfd(dir), entry->d_name, 0);
                }
            }
            ::closedir(dir);
        }
        ::rmdir(output_dir.c_str());
        ::rmdir(root_.c_str());
        unsetenv("LOG_PATH");
    }
    std::filesystem::path output() const {
        return root_ / "model_inputs";
    }

private:
    std::filesystem::path root_;
};

std::vector<std::filesystem::path> dumpPaths(const DumpDirectory& dump) {
    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(dump.output())) {
        if (entry.path().extension() == ".pt") {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}
std::vector<c10::impl::GenericDict> dumpRecords(const DumpDirectory& dump) {
    std::vector<c10::impl::GenericDict> result;
    for (const auto& path : dumpPaths(dump)) {
        std::ifstream     input(path, std::ios::binary);
        std::vector<char> bytes(std::istreambuf_iterator<char>(input), {});
        const auto        chunk = torch::pickle_load(bytes).toGenericDict();
        EXPECT_EQ(chunk.at("schema_version").toInt(), 1);
        EXPECT_EQ(chunk.at("record_type").toStringRef(), "model_inputs_chunk");
        const auto records = chunk.at("records").toList();
        for (size_t row = 0; row < records.size(); ++row) {
            result.push_back(records.get(row).toGenericDict());
        }
    }
    return result;
}

TEST(ModelInputsLoggerTest, DumpsLoadableSnapshot) {
    ASSERT_TRUE(torch::cuda::is_available());
    CapacityEnvironment capacities;
    DumpDirectory dump;
    {
        GptModelInputs inputs{};
        inputs.combo_tokens          = torch::tensor({1, 2, 3}, torch::kInt32).cuda();
        inputs.prefix_lengths        = torch::tensor({2, 1}, torch::kInt32);
        inputs.kv_cache_group_tags   = {"linear", "full"};
        inputs.request_id            = torch::tensor({30, 40}, torch::kInt64);
        inputs.all_input_token_ids   = std::vector<torch::Tensor>{torch::tensor({8, 9, 1, 2}, torch::kInt32),
                                                                  torch::tensor({7, 3}, torch::kInt32)};
        inputs.request_is_multimodal = std::vector<bool>{true, false};
        ModelInputsLogger logger(0, 1, nullptr);
        logger.log(inputs, ModelInputsModelRole::NORMAL, 7);
        GptModelInputs oversized{};
        oversized.kv_cache_group_tags = {std::string(64ULL * 1024ULL * 1024ULL, 'x')};
        logger.log(oversized, ModelInputsModelRole::NORMAL, 9);
        inputs.all_input_token_ids->front().fill_(99);
        inputs.all_input_token_ids.reset();
        inputs.request_is_multimodal.reset();
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
    EXPECT_EQ(records.get(1).toGenericDict().at("dropped_before").toInt(), 1);
}

TEST(ModelInputsLoggerTest, OptInRoundTripsFull120MiBBFloat16FeaturesAndChunks) {
    ASSERT_TRUE(torch::cuda::is_available());
    CapacityEnvironment capacities("134217728", "536870912");
    DumpDirectory       dump;
    {
        GptModelInputs inputs{};
        inputs.combo_tokens   = torch::arange(32768, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        inputs.prefix_lengths = torch::tensor({0}, torch::kInt32);
        inputs.request_id     = torch::tensor({int64_t{123456789012345}}, torch::kInt64);
        inputs.trace_ids      = {"cp8-32k-draft-full-feature"};
        inputs.kv_cache_group_tags = {"swa_kv"};
        inputs.all_input_token_ids = std::vector<torch::Tensor>{torch::arange(32768, torch::kInt32)};
        inputs.last_hidden_states =
            torch::full({4096, 15360}, 1.5, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        ASSERT_EQ(inputs.last_hidden_states.numel() * inputs.last_hidden_states.element_size(),
                  120ULL * 1024ULL * 1024ULL);
        ModelInputsLogger logger(0, 8, nullptr);
        for (int64_t index = 0; index < 2; ++index) {
            inputs.last_hidden_states.fill_(1.5 + index);
            logger.log(inputs, ModelInputsModelRole::DRAFT_PREFILL, 300 + index);
        }
        inputs.last_hidden_states.fill_(-1);
    }
    // Two full records cannot share the configured 128MiB chunk.
    ASSERT_EQ(dumpPaths(dump).size(), 2);
    const auto records = dumpRecords(dump);
    ASSERT_EQ(records.size(), 2);
    for (size_t index = 0; index < records.size(); ++index) {
        const auto& payload = records[index];
        EXPECT_EQ(payload.at("record_type").toStringRef(), "gpt_model_inputs_snapshot");
        EXPECT_EQ(payload.at("model_role").toStringRef(), "draft_prefill");
        EXPECT_EQ(payload.at("record_sequence").toInt(), static_cast<int64_t>(index));
        EXPECT_EQ(payload.at("dropped_before").toInt(), 0);
        EXPECT_EQ(payload.at("model_id").toInt(), 300 + static_cast<int64_t>(index));
        const auto feature = payload.at("last_hidden_states").toTensor();
        EXPECT_TRUE(feature.device().is_cpu());
        EXPECT_EQ(feature.scalar_type(), torch::kBFloat16);
        EXPECT_EQ(feature.sizes().vec(), std::vector<int64_t>({4096, 15360}));
        EXPECT_EQ(feature.numel() * feature.element_size(), 120ULL * 1024ULL * 1024ULL);
        EXPECT_TRUE(feature.eq(1.5 + index).all().item<bool>());
        EXPECT_TRUE(torch::equal(payload.at("combo_tokens").toTensor(), torch::arange(32768, torch::kInt32)));
        EXPECT_TRUE(
            torch::equal(payload.at("all_input_token_ids").toTensorList().get(0), torch::arange(32768, torch::kInt32)));
        EXPECT_EQ(payload.at("trace_ids").toList().get(0).toStringRef(), "cp8-32k-draft-full-feature");
        EXPECT_TRUE(torch::equal(payload.at("request_id").toTensor(),
                                 torch::tensor({int64_t{123456789012345}}, torch::kInt64)));
    }
}

TEST(ModelInputsLoggerTest, DefaultCapacityStillDropsFull120MiBFeature) {
    ASSERT_TRUE(torch::cuda::is_available());
    CapacityEnvironment capacities;
    DumpDirectory       dump;
    {
        ModelInputsLogger logger(0, 8, nullptr);
        GptModelInputs    oversized{};
        oversized.last_hidden_states =
            torch::full({4096, 15360}, 1.5, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        logger.log(oversized, ModelInputsModelRole::DRAFT_PREFILL, 300);
        GptModelInputs small{};
        small.combo_tokens = torch::tensor({7}, torch::kInt32);
        logger.log(small, ModelInputsModelRole::TARGET, 301);
    }
    int     snapshots = 0;
    int64_t dropped   = 0;
    for (const auto& payload : dumpRecords(dump)) {
        if (payload.at("record_type").toStringRef() == "model_inputs_gap") {
            dropped += payload.at("dropped_count").toInt();
            continue;
        }
        ++snapshots;
        EXPECT_EQ(payload.at("model_id").toInt(), 301);
        EXPECT_EQ(payload.at("record_sequence").toInt(), 1);
        EXPECT_TRUE(payload.at("last_hidden_states").isNone());
        dropped += payload.at("dropped_before").toInt();
    }
    EXPECT_EQ(snapshots, 1);
    EXPECT_EQ(dropped, 1);
}

TEST(ModelInputsLoggerTest, CapacityOptionsAreCapturedPerInstance) {
    CapacityEnvironment capacities("65536", "262144");
    DumpDirectory       dump;
    {
        ModelInputsLogger low(0, 8, nullptr);
        ::setenv(kRecordCapacityEnv, "2097152", 1);
        ::setenv(kQueueCapacityEnv, "4194304", 1);
        ModelInputsLogger high(1, 8, nullptr);
        ::setenv(kRecordCapacityEnv, "65536", 1);
        ::setenv(kQueueCapacityEnv, "262144", 1);
        GptModelInputs large{};
        large.last_hidden_states = torch::ones({256 * 1024}, torch::kFloat32);
        low.log(large, ModelInputsModelRole::NORMAL, 10);
        high.log(large, ModelInputsModelRole::NORMAL, 20);
        GptModelInputs small{};
        small.combo_tokens = torch::tensor({7}, torch::kInt32);
        low.log(small, ModelInputsModelRole::NORMAL, 11);
    }
    bool large_saved = false, small_saved = false;
    for (const auto& payload : dumpRecords(dump)) {
        if (payload.at("record_type").toStringRef() != "gpt_model_inputs_snapshot") {
            continue;
        }
        const auto model = payload.at("model_id").toInt();
        EXPECT_NE(model, 10);
        if (model == 20) {
            large_saved        = true;
            const auto feature = payload.at("last_hidden_states").toTensor();
            EXPECT_EQ(feature.scalar_type(), torch::kFloat32);
            EXPECT_EQ(feature.numel(), 256 * 1024);
            EXPECT_TRUE(feature.eq(1).all().item<bool>());
            EXPECT_EQ(payload.at("record_sequence").toInt(), 0);
        } else if (model == 11) {
            small_saved = true;
            EXPECT_EQ(payload.at("record_sequence").toInt(), 1);
        }
    }
    EXPECT_TRUE(large_saved);
    EXPECT_TRUE(small_saved);
}

TEST(ModelInputsLoggerTest, InvalidOrInconsistentCapacityOptionsFailClosed) {
    DumpDirectory                                          dump;
    const std::vector<std::pair<const char*, const char*>> invalid = {
        {"", nullptr},
        {"0", nullptr},
        {"-1", nullptr},
        {"+67108864", nullptr},
        {" 67108864", nullptr},
        {"1.0", nullptr},
        {"NaN", nullptr},
        {"inf", nullptr},
        {"67108864junk", nullptr},
        {"18446744073709551616", nullptr},
        {"65535", nullptr},
        {"134217729", nullptr},
        {nullptr, ""},
        {nullptr, "0"},
        {nullptr, "-1"},
        {nullptr, "NaN"},
        {nullptr, "1e8"},
        {nullptr, "18446744073709551616"},
        {nullptr, "536870913"},
        {"134217728", "67108864"},
    };
    for (const auto& values : invalid) {
        SCOPED_TRACE(values.first ? values.first : "default-record");
        SCOPED_TRACE(values.second ? values.second : "default-queue");
        CapacityEnvironment capacities(values.first, values.second);
        EXPECT_THROW({ ModelInputsLogger logger(0, 8, nullptr); }, std::invalid_argument);
    }
    for (const auto& values :
         std::vector<std::pair<const char*, const char*>>{{"65536", "65536"}, {"134217728", "536870912"}}) {
        CapacityEnvironment capacities(values.first, values.second);
        EXPECT_NO_THROW({ ModelInputsLogger logger(0, 8, nullptr); });
    }
}

TEST(ModelInputsLoggerTest, ConfiguredChunkCapacityPreservesRotationAndSequences) {
    CapacityEnvironment capacities("65536", "262144");
    DumpDirectory       dump;
    {
        ModelInputsLogger logger(0, 2, nullptr);
        GptModelInputs    inputs{};
        inputs.combo_tokens = torch::arange(8192, torch::kInt32);
        for (int64_t model = 0; model < 3; ++model) {
            logger.log(inputs, ModelInputsModelRole::NORMAL, model);
        }
    }
    ASSERT_EQ(dumpPaths(dump).size(), 2);
    const auto records = dumpRecords(dump);
    ASSERT_EQ(records.size(), 2);
    for (size_t index = 0; index < records.size(); ++index) {
        EXPECT_EQ(records[index].at("record_sequence").toInt(), static_cast<int64_t>(index) + 1);
        EXPECT_EQ(records[index].at("model_id").toInt(), static_cast<int64_t>(index) + 1);
        EXPECT_EQ(records[index].at("dropped_before").toInt(), 0);
        EXPECT_TRUE(torch::equal(records[index].at("combo_tokens").toTensor(), torch::arange(8192, torch::kInt32)));
    }
}
}  // namespace
}  // namespace rtp_llm
