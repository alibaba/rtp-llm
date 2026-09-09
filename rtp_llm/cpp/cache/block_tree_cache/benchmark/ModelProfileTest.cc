#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/ModelProfile.h"

#include <cstdlib>
#include <functional>
#include <limits>
#include <stdexcept>
#include <gtest/gtest.h>
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

namespace rtp_llm::benchmark {
namespace {
const char* kProfile = R"({"profile_id":"test","tokens_per_block":64,
 "groups":[
  {"tag":"a","type":"SWA","sliding_window_size":80,"layer_count":2,"layer_stride_bytes":64,"group_payload_bytes":128},
  {"tag":"b","type":"SWA","sliding_window_size":80,"layer_count":3,"layer_stride_bytes":16,"group_payload_bytes":48}],
 "group_sets":[{"name":"swa","members":["a","b"],"payload_bytes":176}]})";

std::string mutateProfile(const std::function<void(rapidjson::Document&)>& mutate) {
    rapidjson::Document document;
    document.Parse(kProfile);
    mutate(document);
    rapidjson::StringBuffer                    buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    document.Accept(writer);
    return buffer.GetString();
}

TEST(ModelProfileTest, DerivesPayloadAndPreservesTokenGeometry) {
    auto profile = ModelProfile::fromString(kProfile);
    EXPECT_EQ(profile.tokens_per_block, 64u);
    ASSERT_EQ(profile.group_sets.size(), 1u);
    EXPECT_EQ(profile.group_sets[0].sliding_window_size, 80u);
    EXPECT_EQ(profile.group_sets[0].payload_bytes, 176u);
    EXPECT_EQ(profile.computeGroupSetPayloadBytes("swa"), 176u);
    auto no_payload = mutateProfile([](auto& d) { d["group_sets"][0].RemoveMember("payload_bytes"); });
    EXPECT_EQ(ModelProfile::fromString(no_payload).group_sets[0].payload_bytes, 176u);
}

TEST(ModelProfileTest, RejectsInvalidPayloadContracts) {
    const std::vector<std::pair<const char*, std::function<void(rapidjson::Document&)>>> cases = {
        {"payload type", [](auto& d) { d["group_sets"][0]["payload_bytes"].SetString("176", d.GetAllocator()); }},
        {"zero layer count", [](auto& d) { d["groups"][0]["layer_count"].SetUint64(0); }},
        {"zero block size", [](auto& d) { d["tokens_per_block"].SetUint64(0); }},
        {"group product", [](auto& d) { d["groups"][0]["group_payload_bytes"].SetUint64(127); }},
        {"group-set sum", [](auto& d) { d["group_sets"][0]["payload_bytes"].SetUint64(175); }},
        {"empty group set", [](auto& d) { d["group_sets"][0]["members"].Clear(); }},
        {"product overflow", [](auto& d) { d["groups"][0]["layer_stride_bytes"].SetUint64(UINT64_MAX); }},
        {"window overflow",
         [](auto& d) {
             d["groups"][0]["sliding_window_size"].SetUint64(static_cast<uint64_t>(std::numeric_limits<int>::max())
                                                             + 1);
         }},
        {"sum overflow",
         [](auto& d) {
             for (auto& group : d["groups"].GetArray()) {
                 group["layer_count"].SetUint64(1);
                 group["layer_stride_bytes"].SetUint64(UINT64_MAX);
                 group["group_payload_bytes"].SetUint64(UINT64_MAX);
             }
         }},
    };
    for (const auto& [name, mutate] : cases) {
        SCOPED_TRACE(name);
        EXPECT_THROW(ModelProfile::fromString(mutateProfile(mutate)), std::runtime_error);
    }
}

TEST(ModelProfileTest, LoadsShippedProfilesWithConsistentPayloads) {
    const char* runfiles  = std::getenv("TEST_SRCDIR");
    const char* workspace = std::getenv("TEST_WORKSPACE");
    ASSERT_NE(runfiles, nullptr);
    ASSERT_NE(workspace, nullptr);
    const std::string root =
        std::string(runfiles) + "/" + workspace + "/rtp_llm/cpp/cache/block_tree_cache/benchmark/profiles/";
    for (const auto& entry : std::vector<std::pair<std::string, size_t>>{
             {"deepseek_v4_pro_fp8_tp1_cp1.json", 128}, {"deepseek_v4_flash_fp8_tp1_cp1_tpb1024.json", 1024}}) {
        const auto profile = ModelProfile::load(root + entry.first);
        EXPECT_EQ(profile.tokens_per_block, entry.second);
        for (const auto& group_set : profile.group_sets) {
            EXPECT_EQ(group_set.payload_bytes, profile.computeGroupSetPayloadBytes(group_set.name));
        }
    }
}
}  // namespace
}  // namespace rtp_llm::benchmark
