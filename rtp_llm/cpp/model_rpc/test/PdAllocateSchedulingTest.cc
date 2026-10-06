#include "rtp_llm/cpp/model_rpc/PdAllocateScheduling.h"

#include <gtest/gtest.h>
#include <google/protobuf/unknown_field_set.h>
#include <initializer_list>

namespace rtp_llm {
namespace {

GenerateInputPB populatedPrefillInput() {
    GenerateInputPB input;
    input.set_request_id(9001);
    input.set_client_id("prefill-client");
    input.set_start_time(123456789);
    input.add_token_ids(101);
    input.add_token_ids(202);
    input.set_batch_group_size(20);
    input.mutable_batch_group_id()->set_value(7001);
    auto* info = input.mutable_request_info();
    info->set_frontend_ip("10.0.0.1");
    info->set_dash_ip("10.0.0.2");
    info->set_trace_id("request-trace");
    info->set_request_id("external-request");
    info->set_source_role("frontend");

    auto* config = input.mutable_generate_config();
    config->mutable_force_batch()->set_value(1);
    config->mutable_batch_group_timeout()->set_value(900000);
    config->set_timeout_ms(180000);
    config->set_max_new_tokens(2);
    config->set_min_new_tokens(1);
    config->set_num_beams(2);
    config->set_num_return_sequences(2);
    config->add_variable_num_beams(1);
    config->set_do_sample(true);
    config->set_top_k(7);
    config->set_top_p(0.9f);
    config->set_temperature(0.7f);
    config->set_repetition_penalty(1.1f);
    config->mutable_random_seed()->set_value(20261004);
    config->mutable_task_id()->set_value("task-id");
    config->mutable_adapter_name()->set_value("adapter");
    config->mutable_trace_id()->set_value("config-trace");
    config->set_can_use_pd_separation(true);
    config->set_reuse_cache(true);
    config->set_enable_device_cache(true);
    config->set_enable_memory_cache(true);
    config->set_enable_remote_cache(true);
    config->set_unique_key("cache-key");
    config->set_gen_timeline(true);
    config->set_profile_step(2);
    config->set_profile_trace_name("prefill-profile");
    config->set_global_request_id(123);
    config->set_sp_edit(true);
    config->set_sp_input_lookup(true);
    config->add_sp_advice_prompt_token_ids(303);
    config->mutable_stop_words_list()->add_rows()->add_values(404);
    auto* route = config->add_role_addrs();
    route->set_role(RoleAddrPB::DECODE);
    route->set_ip("127.0.0.1");
    route->set_http_port(24100);
    route->set_grpc_port(24101);

    auto* mm = input.add_multimodal_inputs();
    mm->set_multimodal_url("fixture://image");
    mm->set_multimodal_type(1);
    mm->mutable_multimodal_tensor()->set_data_type(TensorPB::BF16);
    mm->mutable_multimodal_tensor()->add_shape(1);
    mm->mutable_multimodal_tensor()->set_bf16_data("\x01\x02", 2);
    mm->mutable_mm_preprocess_config()->set_width(32);
    mm->mutable_mm_preprocess_config()->set_height(16);
    mm->mutable_mm_preprocess_config()->set_mm_timeout_ms(1000);
    auto* span = input.mutable_multimodal_token_layout()->add_spans();
    span->set_offset(1);
    span->set_length(1);
    return input;
}

TEST(PdAllocateSchedulingTest, ForcedCloneChangesOnlyFourFieldsAndKeepsOriginal) {
    const auto      original       = populatedPrefillInput();
    const auto      original_bytes = original.SerializeAsString();
    GenerateInputPB clone(original);
    GenerateInputPB expected(original);
    expected.set_batch_group_size(1);
    expected.clear_batch_group_id();
    expected.mutable_generate_config()->clear_force_batch();
    expected.mutable_generate_config()->clear_batch_group_timeout();

    normalizePdAllocateScheduling(clone);

    EXPECT_EQ(original.SerializeAsString(), original_bytes);
    EXPECT_EQ(clone.SerializeAsString(), expected.SerializeAsString());
    EXPECT_EQ(clone.batch_group_size(), 1);
    EXPECT_FALSE(clone.has_batch_group_id());
    EXPECT_FALSE(clone.generate_config().has_force_batch());
    EXPECT_FALSE(clone.generate_config().has_batch_group_timeout());
    // A protobuf copy owns its repeated fields and nested messages independently.
    clone.set_token_ids(0, 999);
    clone.mutable_generate_config()->mutable_role_addrs(0)->set_ip("changed");
    EXPECT_EQ(original.SerializeAsString(), original_bytes);
}

TEST(PdAllocateSchedulingTest, MissingConfigIsByteIdenticalAndStaysAbsent) {
    auto input = populatedPrefillInput();
    input.clear_generate_config();
    const auto before = input.SerializeAsString();
    normalizePdAllocateScheduling(input);
    EXPECT_EQ(input.SerializeAsString(), before);
    EXPECT_FALSE(input.has_generate_config());
}

TEST(PdAllocateSchedulingTest, MissingForcePreservesInertGroupMetadata) {
    auto input = populatedPrefillInput();
    input.mutable_generate_config()->clear_force_batch();
    const auto before = input.SerializeAsString();
    normalizePdAllocateScheduling(input);
    EXPECT_EQ(input.SerializeAsString(), before);
    EXPECT_FALSE(input.generate_config().has_force_batch());
}

TEST(PdAllocateSchedulingTest, ExplicitZeroForcePreservesWrapperPresenceAndBytes) {
    auto input = populatedPrefillInput();
    input.mutable_generate_config()->mutable_force_batch()->set_value(0);
    const auto before = input.SerializeAsString();
    normalizePdAllocateScheduling(input);
    EXPECT_EQ(input.SerializeAsString(), before);
    EXPECT_TRUE(input.generate_config().has_force_batch());
    EXPECT_EQ(input.generate_config().force_batch().value(), 0);
}

TEST(PdAllocateSchedulingTest, AllNonzeroForcesNormalizeGroupAndOptionalPresence) {
    // ID variants: absent, explicit zero, negative. Timeout variants: absent,
    // explicit zero, populated. These also cover proto3 default-value presence.
    for (const int force : {1, -1, 2}) {
        for (const int group_size : {0, 1, 20}) {
            for (const int id_variant : {0, 1, 2}) {
                for (const int timeout_variant : {0, 1, 2}) {
                    SCOPED_TRACE(::testing::Message() << "force=" << force << " group=" << group_size
                                                      << " id=" << id_variant << " timeout=" << timeout_variant);
                    auto input = populatedPrefillInput();
                    input.set_batch_group_size(group_size);
                    input.clear_batch_group_id();
                    if (id_variant != 0) {
                        input.mutable_batch_group_id()->set_value(id_variant == 1 ? 0 : -7);
                    }
                    auto* config = input.mutable_generate_config();
                    config->mutable_force_batch()->set_value(force);
                    config->clear_batch_group_timeout();
                    if (timeout_variant != 0) {
                        config->mutable_batch_group_timeout()->set_value(timeout_variant == 1 ? 0 : 900000);
                    }
                    GenerateInputPB expected(input);
                    expected.set_batch_group_size(1);
                    expected.clear_batch_group_id();
                    expected.mutable_generate_config()->clear_force_batch();
                    expected.mutable_generate_config()->clear_batch_group_timeout();
                    normalizePdAllocateScheduling(input);
                    EXPECT_EQ(input.SerializeAsString(), expected.SerializeAsString());
                }
            }
        }
    }
}

TEST(PdAllocateSchedulingTest, NormalizationIsIdempotent) {
    auto input = populatedPrefillInput();
    normalizePdAllocateScheduling(input);
    const auto once = input.SerializeAsString();
    normalizePdAllocateScheduling(input);
    EXPECT_EQ(input.SerializeAsString(), once);
}

TEST(PdAllocateSchedulingTest, AllocateEnvelopeAndUnknownFieldsArePreserved) {
    auto original = populatedPrefillInput();
    original.GetReflection()->MutableUnknownFields(&original)->AddVarint(1000, 12345);
    auto* config = original.mutable_generate_config();
    config->GetReflection()->MutableUnknownFields(config)->AddLengthDelimited(1001, "future-config");
    const auto        original_bytes = original.SerializeAsString();
    GenerateRequestPB allocate;
    allocate.set_stage(RemoteStage::ALLOCATE);
    allocate.set_request_id(9001);
    allocate.set_client_id("allocate-client");
    allocate.set_start_time(123456789);
    allocate.set_prefill_cp_size(4);
    allocate.add_peer_addrs("127.0.0.1:23102");
    allocate.add_peer_addrs("127.0.0.1:23103");
    allocate.set_allocated_input(new GenerateInputPB(original));
    allocate.GetReflection()->MutableUnknownFields(&allocate)->AddVarint(1002, 54321);
    GenerateRequestPB expected(allocate);
    expected.mutable_input()->set_batch_group_size(1);
    expected.mutable_input()->clear_batch_group_id();
    expected.mutable_input()->mutable_generate_config()->clear_force_batch();
    expected.mutable_input()->mutable_generate_config()->clear_batch_group_timeout();

    normalizePdAllocateScheduling(*allocate.mutable_input());

    EXPECT_EQ(allocate.SerializeAsString(), expected.SerializeAsString());
    EXPECT_EQ(original.SerializeAsString(), original_bytes);
    EXPECT_EQ(allocate.stage(), RemoteStage::ALLOCATE);
    EXPECT_EQ(allocate.prefill_cp_size(), 4);
    EXPECT_EQ(allocate.peer_addrs_size(), 2);
}

}  // namespace
}  // namespace rtp_llm
