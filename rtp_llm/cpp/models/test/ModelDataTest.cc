
#include "gtest/gtest.h"

#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/GenerationPrefillCudaGraphEligibility.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/distribute/CpuTpBroadcaster.h"

#include <functional>
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <type_traits>

extern char** environ;

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

TEST_F(ModelDataTest, testPrefillCPExecutionFollowsRoleConfig) {
    ParallelismConfig prefill_config;
    prefill_config.prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
    EXPECT_TRUE(buildExecProperties(prefill_config, DeviceResourceConfig{}).enable_prefill_cp);

    ParallelismConfig decode_config;
    decode_config.prefill_cp_config.method = CPRotateMethod::PREFILL_CP;
    EXPECT_FALSE(buildExecProperties(decode_config, DeviceResourceConfig{}).enable_prefill_cp);
}

TEST_F(ModelDataTest, testDSparkLongPrefillShapeHintsStayInt64) {
    GptModelInputs inputs;
    // expand() preserves the logical DSpARK aux shape without allocating the
    // 6 GiB/24 GiB backing storage used by real 256K/1M prefills.
    auto backing              = torch::empty({1, 1}, torch::kBFloat16);
    inputs.last_hidden_states = backing.expand({262144, 12288});

    auto shape_hints = getModelInputShapeHints(inputs);
    static_assert(std::is_same_v<GptModelInputShapeHints::value_type, int64_t>);
    EXPECT_EQ(shape_hints[GptModelInputIndex::mtpHiddenStates], 3221225472LL);
    EXPECT_EQ(shape_hints[GptModelInputIndex::mtpHiddenStatesRows], 262144LL);
    const auto wire_hints = makeModelInputShapeHintsTensor(inputs);
    EXPECT_EQ(wire_hints.scalar_type(), torch::kInt64);
    EXPECT_EQ(wire_hints.data_ptr<int64_t>()[GptModelInputIndex::mtpHiddenStates], 3221225472LL);
    EXPECT_EQ(decodeMtpHiddenStatesShape(shape_hints[GptModelInputIndex::mtpHiddenStates],
                                         shape_hints[GptModelInputIndex::mtpHiddenStatesRows]),
              (std::array<int64_t, 2>{262144, 12288}));

    inputs.last_hidden_states = backing.expand({1048576, 12288});
    shape_hints               = getModelInputShapeHints(inputs);
    EXPECT_EQ(shape_hints[GptModelInputIndex::mtpHiddenStates], 12884901888LL);
    EXPECT_EQ(decodeMtpHiddenStatesShape(shape_hints[GptModelInputIndex::mtpHiddenStates],
                                         shape_hints[GptModelInputIndex::mtpHiddenStatesRows]),
              (std::array<int64_t, 2>{1048576, 12288}));
}

TEST_F(ModelDataTest, testMtpHiddenShapeRejectsInvalidMetadataBeforeAllocation) {
    EXPECT_THROW((void)decodeMtpHiddenStatesShape(-1, 1), RTPException);
    EXPECT_THROW((void)decodeMtpHiddenStatesShape(1, 0), RTPException);
    EXPECT_THROW((void)decodeMtpHiddenStatesShape(5, 2), RTPException);
    EXPECT_THROW((void)decodeMtpHiddenStatesShape(0, 1), RTPException);
}

TEST_F(ModelDataTest, testShapeHintsCarryExactPackedBroadcastPlacement) {
    GptModelInputs inputs;
    inputs.combo_tokens             = torch::zeros({8}, torch::kInt32).cuda();
    inputs.input_lengths            = torch::zeros({1}, torch::kInt32).cuda();
    inputs.sequence_lengths         = torch::zeros({1}, torch::kInt32).cuda();
    inputs.prefix_lengths           = torch::zeros({1}, torch::kInt32).cuda();
    inputs.kv_cache_kernel_block_id = torch::zeros({4, 1, 93}, torch::kInt32).cuda();
    inputs.kv_cache_block_id        = torch::zeros({4, 1, 93}, torch::kInt32);
    inputs.kv_cache_group_types     = torch::zeros({4}, torch::kInt32);
    inputs.kv_cache_update_mapping  = torch::zeros({2, 3}, torch::kInt32).cuda();
    inputs.request_id               = torch::zeros({1}, torch::kInt64);
    inputs.request_pd_separation    = torch::zeros({1}, torch::kBool);
    inputs.lm_output_indexes        = torch::zeros({7}, torch::kInt32).cuda();
    // This is the exact mixed-device DSpARK TP2 case that previously made
    // rank 0 send 1504 CPU bytes while rank 1 waited for 1600.
    inputs.combo_position_ids = torch::zeros({24}, torch::kInt32).cuda();

    const auto hints     = getModelInputShapeHints(inputs);
    const auto map       = static_cast<uint32_t>(hints[GptModelInputIndex::tensorDeviceMap]);
    auto       is_device = [&](GptModelInputDeviceBit bit) { return (map & bit) != 0; };

    EXPECT_TRUE(is_device(GptModelInputDeviceBit::kDeviceBitComboTokens));
    EXPECT_TRUE(is_device(GptModelInputDeviceBit::kDeviceBitInputLengths));
    EXPECT_TRUE(is_device(GptModelInputDeviceBit::kDeviceBitSequenceLengths));
    EXPECT_TRUE(is_device(GptModelInputDeviceBit::kDeviceBitPrefixLengths));
    EXPECT_TRUE(is_device(GptModelInputDeviceBit::kDeviceBitKernelBlockId));
    EXPECT_FALSE(is_device(GptModelInputDeviceBit::kDeviceBitBlockId));
    EXPECT_FALSE(is_device(GptModelInputDeviceBit::kDeviceBitCacheGroupTypes));
    EXPECT_TRUE(is_device(GptModelInputDeviceBit::kDeviceBitCacheUpdateMapping));
    EXPECT_FALSE(is_device(GptModelInputDeviceBit::kDeviceBitRequestId));
    EXPECT_FALSE(is_device(GptModelInputDeviceBit::kDeviceBitRequestPdSeparation));
    EXPECT_TRUE(is_device(GptModelInputDeviceBit::kDeviceBitLmOutputIndexes));
    EXPECT_TRUE(is_device(GptModelInputDeviceBit::kDeviceBitComboPositionIds));
}

TEST_F(ModelDataTest, testTpSyncBlockTableShapesPreserve2DAnd3D) {
    GptModelInputs inputs;
    inputs.input_lengths            = torch::zeros({2}, torch::kInt32);
    inputs.kv_cache_kernel_block_id = torch::zeros({2, 11}, torch::kInt32);
    inputs.kv_cache_block_id        = torch::zeros({2, 7}, torch::kInt32);

    auto hints = getModelInputShapeHints(inputs);
    EXPECT_EQ(hints[GptModelInputIndex::kvCacheKernelBlockIdRank], 2);
    EXPECT_EQ(hints[GptModelInputIndex::kvCacheBlockIdRank], 2);
    EXPECT_EQ(hints[GptModelInputIndex::kvCacheGroupNum], 1);
    EXPECT_EQ(hints[GptModelInputIndex::maxKernelBlocksPerBatch], 11);
    EXPECT_EQ(hints[GptModelInputIndex::maxBlocksPerBatch], 7);
    EXPECT_EQ(decodeKvBlockTableShape(hints[GptModelInputIndex::kvCacheKernelBlockIdRank],
                                      hints[GptModelInputIndex::kvCacheGroupNum],
                                      hints[GptModelInputIndex::inputLengths],
                                      hints[GptModelInputIndex::maxKernelBlocksPerBatch]),
              (std::vector<int64_t>{2, 11}));
    EXPECT_EQ(decodeKvBlockTableShape(hints[GptModelInputIndex::kvCacheBlockIdRank],
                                      hints[GptModelInputIndex::kvCacheGroupNum],
                                      hints[GptModelInputIndex::inputLengths],
                                      hints[GptModelInputIndex::maxBlocksPerBatch]),
              (std::vector<int64_t>{2, 7}));

    inputs.kv_cache_kernel_block_id = torch::zeros({4, 2, 11}, torch::kInt32);
    inputs.kv_cache_block_id        = torch::zeros({4, 2, 7}, torch::kInt32);
    hints                           = getModelInputShapeHints(inputs);
    EXPECT_EQ(hints[GptModelInputIndex::kvCacheKernelBlockIdRank], 3);
    EXPECT_EQ(hints[GptModelInputIndex::kvCacheBlockIdRank], 3);
    EXPECT_EQ(hints[GptModelInputIndex::kvCacheGroupNum], 4);
    EXPECT_EQ(decodeKvBlockTableShape(hints[GptModelInputIndex::kvCacheKernelBlockIdRank],
                                      hints[GptModelInputIndex::kvCacheGroupNum],
                                      hints[GptModelInputIndex::inputLengths],
                                      hints[GptModelInputIndex::maxKernelBlocksPerBatch]),
              (std::vector<int64_t>{4, 2, 11}));
    EXPECT_EQ(decodeKvBlockTableShape(hints[GptModelInputIndex::kvCacheBlockIdRank],
                                      hints[GptModelInputIndex::kvCacheGroupNum],
                                      hints[GptModelInputIndex::inputLengths],
                                      hints[GptModelInputIndex::maxBlocksPerBatch]),
              (std::vector<int64_t>{4, 2, 7}));
}

TEST_F(ModelDataTest, testShapeHintsCarryTpControlPlaneAndCacheGeometry) {
    GptModelInputs inputs;
    inputs.need_all_logits           = true;
    inputs.need_all_hidden_states    = true;
    inputs.need_moe_gating           = true;
    inputs.warmup                    = true;
    inputs.skip_run                  = true;
    inputs.is_fake_stream            = true;
    inputs.is_target_verify          = true;
    inputs.pd_separation             = true;
    inputs.decode_entrance           = true;
    inputs.use_opaque_kv_cache_store = true;
    inputs.kv_block_stride_bytes     = 4096;
    inputs.kv_scale_stride_bytes     = 256;
    inputs.seq_size_per_block        = 64;
    inputs.kernel_seq_size_per_block = 16;

    const auto hints = getModelInputShapeHints(inputs);
    const auto flags = static_cast<uint32_t>(hints[GptModelInputIndex::modelControlFlags]);
    auto       has   = [&](GptModelInputControlFlag flag) { return (flags & flag) != 0; };

    EXPECT_TRUE(has(GptModelInputControlFlag::kControlNeedAllLogits));
    EXPECT_TRUE(has(GptModelInputControlFlag::kControlNeedAllHiddenStates));
    EXPECT_TRUE(has(GptModelInputControlFlag::kControlNeedMoeGating));
    EXPECT_TRUE(has(GptModelInputControlFlag::kControlWarmup));
    EXPECT_TRUE(has(GptModelInputControlFlag::kControlSkipRun));
    EXPECT_TRUE(has(GptModelInputControlFlag::kControlFakeStream));
    EXPECT_TRUE(has(GptModelInputControlFlag::kControlTargetVerify));
    EXPECT_TRUE(has(GptModelInputControlFlag::kControlPdSeparation));
    EXPECT_TRUE(has(GptModelInputControlFlag::kControlDecodeEntrance));
    EXPECT_TRUE(has(GptModelInputControlFlag::kControlOpaqueKvCacheStore));
    EXPECT_EQ(hints[GptModelInputIndex::kvBlockStrideBytes], 4096);
    EXPECT_EQ(hints[GptModelInputIndex::kvScaleStrideBytes], 256);
    EXPECT_EQ(hints[GptModelInputIndex::seqSizePerBlock], 64);
    EXPECT_EQ(hints[GptModelInputIndex::kernelSeqSizePerBlock], 16);
}

namespace {

void expectIntTensor(const torch::Tensor& actual, const torch::Tensor& expected) {
    ASSERT_TRUE(actual.defined());
    EXPECT_EQ(actual.device(), expected.device());
    EXPECT_TRUE(torch::equal(actual, expected));
}

void checkGroupedTpRound(GptModelInputs&                 inputs,
                         int                             rank,
                         const std::vector<std::string>& root_tags,
                         bool                            empty_tables) {
    auto local_tags = root_tags;
    if (rank != 0 && local_tags.size() > 1) {
        std::rotate(local_tags.begin(), local_tags.begin() + 1, local_tags.end());
    }
    const auto group_count    = static_cast<int64_t>(root_tags.size());
    const auto physical_slots = empty_tables ? 0 : 2;
    const auto kernel_slots   = empty_tables ? 0 : 3;
    auto       physical =
        torch::arange(group_count * 2 * physical_slots, torch::kInt32).reshape({group_count, 2, physical_slots});
    auto kernel = torch::arange(group_count * 2 * kernel_slots, torch::kInt32).reshape({group_count, 2, kernel_slots});
    auto types  = torch::arange(group_count, torch::kInt32);
    auto copies =
        empty_tables ?
            torch::empty({0, 3}, torch::kInt32) :
            torch::tensor(
                {static_cast<int>(group_count - 1), 11, 21, 0, 12, 22, static_cast<int>(group_count - 1), 13, 23},
                torch::kInt32)
                .reshape({3, 3});

    if (rank == 0) {
        inputs = GptModelInputs{};
    }
    inputs.kv_cache_group_tags = local_tags;
    if (rank == 0) {
        inputs.combo_tokens             = torch::tensor({7, 8}, torch::kInt32);
        inputs.input_lengths            = torch::tensor({1, 1}, torch::kInt32);
        inputs.sequence_lengths         = torch::tensor({4, 5}, torch::kInt32);
        inputs.kv_cache_block_id        = physical.clone();
        inputs.kv_cache_kernel_block_id = kernel.clone();
        inputs.kv_cache_group_types     = types.clone();
        inputs.kv_cache_update_mapping  = copies.clone();
    }
    const auto        original_physical = inputs.kv_cache_block_id;
    const auto        original_kernel   = inputs.kv_cache_kernel_block_id;
    const auto        original_types    = inputs.kv_cache_group_types;
    const auto        original_copies   = inputs.kv_cache_update_mapping;
    ParallelismConfig config;
    config.tp_size = 2;
    config.tp_rank = rank;
    tpSyncModelInputs(inputs, config);

    EXPECT_EQ(inputs.kv_cache_group_tags, local_tags);
    std::vector<int64_t> local_to_root;
    for (const auto& tag : local_tags) {
        local_to_root.push_back(std::find(root_tags.begin(), root_tags.end(), tag) - root_tags.begin());
    }
    const auto order = torch::tensor(local_to_root, torch::kInt64);
    if (empty_tables && rank != 0) {
        EXPECT_TRUE(!inputs.kv_cache_block_id.defined() || inputs.kv_cache_block_id.numel() == 0);
        EXPECT_TRUE(!inputs.kv_cache_kernel_block_id.defined() || inputs.kv_cache_kernel_block_id.numel() == 0);
    } else {
        expectIntTensor(inputs.kv_cache_block_id, physical.index_select(0, order));
        expectIntTensor(inputs.kv_cache_kernel_block_id, kernel.index_select(0, order));
    }
    expectIntTensor(inputs.kv_cache_group_types, types.index_select(0, order));
    auto expected_copies = copies.clone();
    for (int64_t row = 0; row < expected_copies.size(0); ++row) {
        auto& group_row = expected_copies.data_ptr<int>()[row * 3];
        group_row       = std::find(local_tags.begin(), local_tags.end(), root_tags[group_row]) - local_tags.begin();
    }
    if (empty_tables && rank != 0) {
        EXPECT_TRUE(!inputs.kv_cache_update_mapping.defined() || inputs.kv_cache_update_mapping.numel() == 0);
    } else {
        expectIntTensor(inputs.kv_cache_update_mapping, expected_copies);
    }
    expectIntTensor(inputs.combo_tokens, torch::tensor({7, 8}, torch::kInt32));
    if (rank == 0) {
        EXPECT_TRUE(inputs.kv_cache_block_id.is_same(original_physical));
        EXPECT_TRUE(inputs.kv_cache_kernel_block_id.is_same(original_kernel));
        EXPECT_TRUE(inputs.kv_cache_group_types.is_same(original_types));
        EXPECT_TRUE(inputs.kv_cache_update_mapping.is_same(original_copies));
    }
}

}  // namespace

// Re-exec gives each TP rank a fresh CUDA runtime even when other tests have already initialized CUDA.
TEST(ModelInputTpSyncChild, DISABLED_RunRank) {
    const auto rank_env = std::getenv("RTP_MODEL_TP_TEST_RANK");
    const auto base_env = std::getenv("RTP_MODEL_TP_TEST_BASE");
    ASSERT_NE(rank_env, nullptr);
    ASSERT_NE(base_env, nullptr);
    ::alarm(90);
    const int rank        = std::atoi(rank_env);
    auto&     broadcaster = CpuTpBroadcaster::instance();
    broadcaster.initialize(rank, 2, base_env);
    GptModelInputs inputs;
    checkGroupedTpRound(inputs, rank, {"z_group", "a_group", std::string("\x80") + "_group"}, false);
    checkGroupedTpRound(inputs, rank, {"single_group"}, false);
    checkGroupedTpRound(inputs, rank, {"z_group", "a_group"}, true);

    if (rank == 0) {
        inputs = GptModelInputs{};
    } else {
        inputs.kv_cache_block_id        = torch::ones({1, 2, 2}, torch::kInt32);
        inputs.kv_cache_kernel_block_id = torch::ones({1, 2, 3}, torch::kInt32);
        inputs.kv_cache_group_types     = torch::ones({1}, torch::kInt32);
        inputs.kv_cache_update_mapping  = torch::zeros({1, 3}, torch::kInt32);
    }
    inputs.kv_cache_group_tags = {"stale_group"};
    ParallelismConfig config;
    config.tp_size = 2;
    config.tp_rank = rank;
    tpSyncModelInputs(inputs, config);
    EXPECT_TRUE(inputs.kv_cache_group_tags.empty());
    EXPECT_TRUE(!inputs.kv_cache_block_id.defined() || inputs.kv_cache_block_id.numel() == 0);
    EXPECT_TRUE(!inputs.kv_cache_kernel_block_id.defined() || inputs.kv_cache_kernel_block_id.numel() == 0);
    EXPECT_TRUE(!inputs.kv_cache_group_types.defined() || inputs.kv_cache_group_types.numel() == 0);
    EXPECT_TRUE(!inputs.kv_cache_update_mapping.defined() || inputs.kv_cache_update_mapping.numel() == 0);

    // Both children exercise sender-side rejection, so malformed metadata cannot strand a peer in a collective.
    config.tp_rank = 0;
    GptModelInputs invalid;
    invalid.kv_cache_block_id   = torch::zeros({2, 1, 1}, torch::kInt32);
    invalid.kv_cache_group_tags = {"group", "group"};
    EXPECT_THROW(tpSyncModelInputs(invalid, config), RTPException);
    invalid.kv_cache_group_tags = {"group", ""};
    EXPECT_THROW(tpSyncModelInputs(invalid, config), RTPException);
    invalid.kv_cache_group_tags = {"group"};
    EXPECT_THROW(tpSyncModelInputs(invalid, config), RTPException);
    invalid.kv_cache_group_tags.clear();
    EXPECT_THROW(tpSyncModelInputs(invalid, config), RTPException);
    invalid.kv_cache_group_tags      = {"first", "second"};
    invalid.kv_cache_block_id        = torch::zeros({2, 1, 1}, torch::kInt32);
    invalid.kv_cache_update_mapping  = torch::tensor({2, 1, 2}, torch::kInt32).reshape({1, 3});
    EXPECT_THROW(tpSyncModelInputs(invalid, config), RTPException);
    broadcaster.reset();
    ::alarm(0);
}

TEST(ModelInputTpSyncTest, PreservesIdentityAcrossRankOrderings) {
    const auto        tmpdir  = std::getenv("TMPDIR");
    std::string       pattern = std::string(tmpdir ? tmpdir : "/tmp") + "/model_tp.XXXXXX";
    std::vector<char> temp_path(pattern.begin(), pattern.end());
    temp_path.push_back('\0');
    ASSERT_NE(::mkdtemp(temp_path.data()), nullptr);
    const std::string  base = std::string(temp_path.data()) + "/b";
    std::vector<pid_t> children;
    for (int rank = 0; rank < 2; ++rank) {
        std::vector<std::string> env_strings;
        for (char** entry = environ; *entry != nullptr; ++entry) {
            const std::string value(*entry);
            if (value.rfind("GTEST_", 0) != 0 && value.rfind("RTP_MODEL_TP_TEST_", 0) != 0) {
                env_strings.push_back(value);
            }
        }
        env_strings.push_back("RTP_MODEL_TP_TEST_RANK=" + std::to_string(rank));
        env_strings.push_back("RTP_MODEL_TP_TEST_BASE=" + base);
        std::vector<char*> child_env;
        for (auto& value : env_strings) {
            child_env.push_back(value.data());
        }
        child_env.push_back(nullptr);
        std::vector<std::string> args = {"/proc/self/exe",
                                         "--gtest_filter=ModelInputTpSyncChild.DISABLED_RunRank",
                                         "--gtest_also_run_disabled_tests",
                                         "--gtest_output="};
        std::vector<char*>       child_args;
        for (auto& arg : args) {
            child_args.push_back(arg.data());
        }
        child_args.push_back(nullptr);
        pid_t     pid    = -1;
        const int result = ::posix_spawn(&pid, "/proc/self/exe", nullptr, nullptr, child_args.data(), child_env.data());
        EXPECT_EQ(result, 0);
        if (result == 0) {
            children.push_back(pid);
        }
    }
    for (const auto pid : children) {
        int   status = 0;
        pid_t waited;
        do {
            waited = ::waitpid(pid, &status, 0);
        } while (waited < 0 && errno == EINTR);
        EXPECT_EQ(waited, pid);
        if (waited == pid) {
            EXPECT_TRUE(WIFEXITED(status)) << "TP test child failed: " << status;
            if (WIFEXITED(status)) {
                EXPECT_EQ(WEXITSTATUS(status), 0);
            }
        }
    }
    ::unlink((base + "_0.sock").c_str());
    ::rmdir(temp_path.data());
}

namespace {

GptModelDescription makeGenerationPrefillCudaGraphMoeDescription() {
    GptModelDescription description;
    description.data_type   = DataType::TYPE_BF16;
    description.act_qscheme = QScheme::Qfp8PerTokenBlock;
    MoeConfigs model_moe_config;
    model_moe_config.expert_num     = 96;
    model_moe_config.top_k          = 8;
    model_moe_config.use_all_gather = true;
    description.ffn_conf.moe_configs.emplace(model_moe_config);
    return description;
}

MoeConfig makeGenerationPrefillCudaGraphMoeRuntimeConfig() {
    MoeConfig config;
    config.moe_strategy           = "fp8_per_block_no_dp_masked";
    config.use_all_gather         = true;
    config.use_deepep_moe         = false;
    config.use_deepep_internode   = false;
    config.use_deepep_low_latency = false;
    return config;
}

constexpr bool kMaskedMoeBackendSupported = true;

}  // namespace

TEST_F(ModelDataTest, testGenerationPrefillCudaGraphRunnerOwnershipIsLimitedToNormalMainGeneration) {
    HWKernelConfig config;
    config.enable_cuda_graph                        = true;
    config.generation_prefill_capture_token_buckets = {64, 128};

    EXPECT_TRUE(isGenerationPrefillCudaGraphRequested(config, RoleType::PDFUSION));
    EXPECT_TRUE(supportsGenerationPrefillCudaGraphExecutionMode(SP_TYPE_NONE, false));
    EXPECT_FALSE(supportsGenerationPrefillCudaGraphExecutionMode(SP_TYPE_NONE, true));
    for (const auto speculative_type :
         {SP_TYPE_VANILLA, SP_TYPE_MTP, SP_TYPE_EAGLE3, SP_TYPE_EAGLE, SP_TYPE_DETERMINISTIC, SP_TYPE_DSPARK}) {
        EXPECT_FALSE(supportsGenerationPrefillCudaGraphExecutionMode(speculative_type, true));
        EXPECT_FALSE(shouldCreateGenerationPrefillCudaGraph(config, true, false, RoleType::PDFUSION, speculative_type));
    }
    EXPECT_TRUE(shouldCreateGenerationPrefillCudaGraph(config, true, false, RoleType::PDFUSION, SP_TYPE_NONE));
    // P/D roles ignore retained generation-prefill configuration regardless of
    // speculative mode, both at engine validation and at runner creation.
    for (const auto role_type : {RoleType::PREFILL, RoleType::DECODE}) {
        EXPECT_FALSE(isGenerationPrefillCudaGraphRequested(config, role_type));
        for (const auto speculative_type : {SP_TYPE_NONE, SP_TYPE_MTP, SP_TYPE_DSPARK}) {
            EXPECT_FALSE(shouldCreateGenerationPrefillCudaGraph(config, true, false, role_type, speculative_type));
            EXPECT_FALSE(shouldCreateGenerationPrefillCudaGraph(config, true, true, role_type, speculative_type));
        }
    }
    // Speculative target/draft wrappers do not own the normal-generation
    // prefill runner in the first implementation.
    EXPECT_FALSE(shouldCreateGenerationPrefillCudaGraph(config, true, true, RoleType::PDFUSION, SP_TYPE_NONE));

    config.enable_cuda_graph = false;
    EXPECT_FALSE(isGenerationPrefillCudaGraphRequested(config, RoleType::PDFUSION));
    EXPECT_FALSE(shouldCreateGenerationPrefillCudaGraph(config, true, false, RoleType::PDFUSION, SP_TYPE_NONE));
    config.enable_cuda_graph = true;
    config.generation_prefill_capture_token_buckets.clear();
    EXPECT_FALSE(isGenerationPrefillCudaGraphRequested(config, RoleType::PDFUSION));
    EXPECT_FALSE(shouldCreateGenerationPrefillCudaGraph(config, true, false, RoleType::PDFUSION, SP_TYPE_NONE));
    EXPECT_FALSE(shouldCreateGenerationPrefillCudaGraph(config, false, false, RoleType::PDFUSION, SP_TYPE_NONE));
}

TEST_F(ModelDataTest, testGenerationPrefillCudaGraphPaddedTokenIndexUsesCompleteSentinelRange) {
    EXPECT_TRUE(generationPrefillCudaGraphPaddedTokenIndexFitsInt32(
        HWKernelConfig::kGenerationPrefillCudaGraphMaxRequests,
        HWKernelConfig::kGenerationPrefillCudaGraphMaxCaptureTokens));
    // Keep the overflow helper defensive even though the public request-count
    // limit now rejects this unreachable configuration much earlier.
    EXPECT_FALSE(generationPrefillCudaGraphPaddedTokenIndexFitsInt32(715827882, 3));
    EXPECT_FALSE(generationPrefillCudaGraphPaddedTokenIndexFitsInt32(0, 3));
    EXPECT_FALSE(generationPrefillCudaGraphPaddedTokenIndexFitsInt32(1, 0));
}

TEST_F(ModelDataTest, testGenerationPrefillCudaGraphRequestCapacityUsesReachableContextBatchLimit) {
    EXPECT_EQ(generationPrefillCudaGraphReachableRequestCapacity(8, 32), 8);
    EXPECT_EQ(generationPrefillCudaGraphReachableRequestCapacity(64, 16), 16);
    EXPECT_EQ(generationPrefillCudaGraphReachableRequestCapacity(1, 1), 1);
    EXPECT_EQ(generationPrefillCudaGraphReachableRequestCapacity(0, 32), 0);
    EXPECT_EQ(generationPrefillCudaGraphReachableRequestCapacity(8, 0), 0);
    EXPECT_TRUE(generationPrefillCudaGraphMaxRequestsFitsCapacity(8, 8, 32));
    EXPECT_FALSE(generationPrefillCudaGraphMaxRequestsFitsCapacity(9, 8, 32));
    EXPECT_TRUE(generationPrefillCudaGraphMaxRequestsFitsCapacity(16, 64, 16));
    EXPECT_FALSE(generationPrefillCudaGraphMaxRequestsFitsCapacity(17, 64, 16));
    EXPECT_TRUE(
        generationPrefillCudaGraphMaxRequestsFitsCapacity(HWKernelConfig::kGenerationPrefillCudaGraphMaxRequests,
                                                          HWKernelConfig::kGenerationPrefillCudaGraphMaxRequests,
                                                          HWKernelConfig::kGenerationPrefillCudaGraphMaxRequests));
    EXPECT_FALSE(
        generationPrefillCudaGraphMaxRequestsFitsCapacity(HWKernelConfig::kGenerationPrefillCudaGraphMaxRequests + 1,
                                                          HWKernelConfig::kGenerationPrefillCudaGraphMaxRequests + 1,
                                                          HWKernelConfig::kGenerationPrefillCudaGraphMaxRequests + 1));
    EXPECT_FALSE(generationPrefillCudaGraphMaxRequestsFitsCapacity(0, 8, 32));
}

TEST_F(ModelDataTest, testGenerationPrefillCudaGraphSupportsDenseModel) {
    GptModelDescription description;
    EXPECT_TRUE(supportsGenerationPrefillCudaGraphMoe(
        description, ParallelismConfig{}, MoeConfig{}, /*masked_moe_backend_supported=*/false));
}

TEST_F(ModelDataTest, testGenerationPrefillCudaGraphMaskedMoeBackendRequiresSm90) {
    EXPECT_TRUE(supportsGenerationPrefillCudaGraphMaskedMoeBackend(9));
    EXPECT_FALSE(supportsGenerationPrefillCudaGraphMaskedMoeBackend(10));
    EXPECT_FALSE(supportsGenerationPrefillCudaGraphMaskedMoeBackend(12));
    EXPECT_FALSE(supportsGenerationPrefillCudaGraphMaskedMoeBackend(-1));
}

TEST_F(ModelDataTest, testGenerationPrefillCudaGraphRequiresSingleFullCacheGroup) {
    EXPECT_TRUE(supportsGenerationPrefillCudaGraphCacheTopology({CacheGroupType::FULL}));
    EXPECT_FALSE(supportsGenerationPrefillCudaGraphCacheTopology({}));
    EXPECT_FALSE(supportsGenerationPrefillCudaGraphCacheTopology({CacheGroupType::LINEAR}));
    EXPECT_FALSE(supportsGenerationPrefillCudaGraphCacheTopology({CacheGroupType::SWA}));
    EXPECT_FALSE(supportsGenerationPrefillCudaGraphCacheTopology({CacheGroupType::FULL, CacheGroupType::LINEAR}));
    EXPECT_FALSE(supportsGenerationPrefillCudaGraphCacheTopology({CacheGroupType::FULL, CacheGroupType::SWA}));
}

TEST_F(ModelDataTest, testGenerationPrefillCudaGraphSupportsSingleGpuFp8MaskedMoe) {
    EXPECT_TRUE(supportsGenerationPrefillCudaGraphMoe(makeGenerationPrefillCudaGraphMoeDescription(),
                                                      ParallelismConfig{},
                                                      makeGenerationPrefillCudaGraphMoeRuntimeConfig(),
                                                      kMaskedMoeBackendSupported));

    EXPECT_FALSE(supportsGenerationPrefillCudaGraphMoe(makeGenerationPrefillCudaGraphMoeDescription(),
                                                       ParallelismConfig{},
                                                       makeGenerationPrefillCudaGraphMoeRuntimeConfig(),
                                                       /*masked_moe_backend_supported=*/false));
}

TEST_F(ModelDataTest, testGenerationPrefillCudaGraphRequiresSingleDeviceParallelism) {
    const auto expect_rejected = [](const std::function<void(ParallelismConfig&)>& mutate) {
        ParallelismConfig config;
        mutate(config);
        EXPECT_FALSE(isSingleDeviceGenerationPrefillCudaGraphConfig(config));
    };

    EXPECT_TRUE(isSingleDeviceGenerationPrefillCudaGraphConfig(ParallelismConfig{}));
    expect_rejected([](auto& c) { c.world_size = 2; });
    expect_rejected([](auto& c) { c.tp_size = 2; });
    expect_rejected([](auto& c) { c.dp_size = 2; });
    expect_rejected([](auto& c) { c.ep_size = 2; });
    expect_rejected([](auto& c) { c.pp_size = 2; });
    expect_rejected([](auto& c) { c.ffn_sp_size = 2; });
    expect_rejected([](auto& c) { c.ffn_tp_size = 2; });
    expect_rejected([](auto& c) { c.enable_sp = true; });
    expect_rejected([](auto& c) { c.prefill_cp_config.method = CPRotateMethod::ALL_GATHER; });
    expect_rejected([](auto& c) { c.prefill_cp_config.method = CPRotateMethod::PREFILL_CP; });
    expect_rejected([](auto& c) { c.ffn_disaggregate_config.enable_ffn_disaggregate = true; });
}

TEST_F(ModelDataTest, testGenerationPrefillCudaGraphRejectsAutoMoeStrategy) {
    auto config         = makeGenerationPrefillCudaGraphMoeRuntimeConfig();
    config.moe_strategy = "auto";
    EXPECT_FALSE(supportsGenerationPrefillCudaGraphMoe(
        makeGenerationPrefillCudaGraphMoeDescription(), ParallelismConfig{}, config, kMaskedMoeBackendSupported));
}

TEST_F(ModelDataTest, testGenerationPrefillCudaGraphRejectsNonFp8PerBlockMoe) {
    auto description        = makeGenerationPrefillCudaGraphMoeDescription();
    description.act_qscheme = QScheme::NoQuantize;
    EXPECT_FALSE(supportsGenerationPrefillCudaGraphMoe(description,
                                                       ParallelismConfig{},
                                                       makeGenerationPrefillCudaGraphMoeRuntimeConfig(),
                                                       kMaskedMoeBackendSupported));
}

TEST_F(ModelDataTest, testGenerationPrefillCudaGraphRejectsGraphUnsafeMoeTransport) {
    auto config                   = makeGenerationPrefillCudaGraphMoeRuntimeConfig();
    config.use_deepep_low_latency = true;
    EXPECT_FALSE(supportsGenerationPrefillCudaGraphMoe(
        makeGenerationPrefillCudaGraphMoeDescription(), ParallelismConfig{}, config, kMaskedMoeBackendSupported));

    config                = makeGenerationPrefillCudaGraphMoeRuntimeConfig();
    config.use_all_gather = false;
    EXPECT_FALSE(supportsGenerationPrefillCudaGraphMoe(
        makeGenerationPrefillCudaGraphMoeDescription(), ParallelismConfig{}, config, kMaskedMoeBackendSupported));
}

TEST_F(ModelDataTest, testGenerationPrefillCudaGraphMoeGateCoversEveryRuntimeConstraint) {
    const auto expect_rejected = [](const std::function<void(MoeConfig&)>& mutate) {
        auto config = makeGenerationPrefillCudaGraphMoeRuntimeConfig();
        mutate(config);
        EXPECT_FALSE(supportsGenerationPrefillCudaGraphMoe(
            makeGenerationPrefillCudaGraphMoeDescription(), ParallelismConfig{}, config, kMaskedMoeBackendSupported));
    };

    expect_rejected([](auto& c) { c.use_deepep_moe = true; });
    expect_rejected([](auto& c) { c.use_deepep_internode = true; });
    expect_rejected([](auto& c) { c.use_deepep_low_latency = true; });
    expect_rejected([](auto& c) { c.use_deepep_p2p_low_latency = true; });
    expect_rejected([](auto& c) { c.use_mori_ep = true; });
    expect_rejected([](auto& c) { c.fake_balance_expert = true; });
    expect_rejected([](auto& c) { c.hack_moe_expert = true; });
    expect_rejected([](auto& c) { c.use_all_gather = false; });
}

TEST_F(ModelDataTest, testGenerationPrefillCudaGraphMoeGateCoversEveryModelConstraint) {
    const auto expect_rejected = [](const std::function<void(MoeConfigs&)>& mutate) {
        auto description = makeGenerationPrefillCudaGraphMoeDescription();
        mutate(description.ffn_conf.moe_configs.value());
        EXPECT_FALSE(supportsGenerationPrefillCudaGraphMoe(description,
                                                           ParallelismConfig{},
                                                           makeGenerationPrefillCudaGraphMoeRuntimeConfig(),
                                                           kMaskedMoeBackendSupported));
    };

    expect_rejected([](auto& c) { c.tp_size = 2; });
    expect_rejected([](auto& c) { c.dp_size = 2; });
    expect_rejected([](auto& c) { c.ep_size = 2; });
    expect_rejected([](auto& c) { c.use_all_gather = false; });
    expect_rejected([](auto& c) { c.expert_num = 0; });
    expect_rejected([](auto& c) { c.top_k = 0; });
    expect_rejected([](auto& c) { c.top_k = c.expert_num + 1; });
    expect_rejected([](auto& c) { c.extra_expert_num = 1; });
    expect_rejected([](auto& c) { c.enable_eplb = true; });
}

TEST_F(ModelDataTest, testGenerationPrefillCudaGraphRejectsDistributedOrEplbMoe) {
    auto parallelism       = ParallelismConfig{};
    parallelism.ep_size    = 2;
    parallelism.dp_size    = 2;
    parallelism.world_size = 2;
    EXPECT_FALSE(supportsGenerationPrefillCudaGraphMoe(makeGenerationPrefillCudaGraphMoeDescription(),
                                                       parallelism,
                                                       makeGenerationPrefillCudaGraphMoeRuntimeConfig(),
                                                       kMaskedMoeBackendSupported));

    auto description                              = makeGenerationPrefillCudaGraphMoeDescription();
    description.ffn_conf.moe_configs->enable_eplb = true;
    EXPECT_FALSE(supportsGenerationPrefillCudaGraphMoe(description,
                                                       ParallelismConfig{},
                                                       makeGenerationPrefillCudaGraphMoeRuntimeConfig(),
                                                       kMaskedMoeBackendSupported));
}

}  // namespace rtp_llm
