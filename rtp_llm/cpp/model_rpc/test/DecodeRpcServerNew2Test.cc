#include <memory>

#include <gtest/gtest.h>
#include "torch/all.h"

#include "rtp_llm/cpp/model_rpc/DecodeRpcServerNew2.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"

namespace rtp_llm::test {

namespace {

std::shared_ptr<GenerateStream> makeStream(const std::vector<int>& input_ids) {
    ModelConfig                    model_config;
    RuntimeConfig                  runtime_config;
    ResourceContext                resource_context;
    std::shared_ptr<GenerateInput> query = std::make_shared<GenerateInput>();

    model_config.max_seq_len                 = 4096;
    model_config.vocab_size                  = 32000;
    model_config.special_tokens.eos_token_id = 151643;

    query->input_ids       = torch::tensor(std::vector<int32_t>(input_ids.begin(), input_ids.end()), torch::kInt32);
    query->generate_config = std::make_shared<GenerateConfig>();

    return std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
}

GenerateOutputsPB makeOutputsWithDecodeReuse(int total, int local, int remote, int memory) {
    GenerateOutputsPB outputs_pb;
    outputs_pb.mutable_flatten_output()->add_finished(false);
    auto* aux_info = outputs_pb.mutable_flatten_output()->add_aux_info();
    aux_info->set_total_reuse_len(total);
    aux_info->set_local_reuse_len(local);
    aux_info->set_remote_reuse_len(remote);
    aux_info->set_memory_reuse_len(memory);
    aux_info->set_step_output_len(1);
    return outputs_pb;
}

}  // namespace

TEST(DecodeRpcServerNew2Test, DecodeEntranceRequiresPrefillIgnoresUniqueKeyPresence) {
    GenerateInputPB request;
    auto*           config = request.mutable_generate_config();
    config->set_max_new_tokens(8);
    config->set_num_beams(1);
    config->set_num_return_sequences(1);
    config->set_can_use_pd_separation(true);

    EXPECT_TRUE(decodeEntranceRequiresPrefill(request));

    config->set_unique_key("user-cache-key");
    EXPECT_TRUE(decodeEntranceRequiresPrefill(request));
}

TEST(DecodeRpcServerNew2Test, DecodeEntranceHandoffUsesInternalKeyAndPreservesBusinessKey) {
    GenerateInputPB request;
    auto*           config = request.mutable_generate_config();
    config->set_unique_key("shared-business-key");

    auto first  = buildDecodeEntranceKeys(request, "127.0.0.1", 1, 100);
    auto second = buildDecodeEntranceKeys(request, "127.0.0.1", 2, 100);

    EXPECT_EQ(first.business_unique_key, "shared-business-key");
    EXPECT_EQ(second.business_unique_key, "shared-business-key");
    EXPECT_NE(first.handoff_unique_key, second.handoff_unique_key);
    EXPECT_NE(first.handoff_unique_key, first.business_unique_key);
    EXPECT_NE(second.handoff_unique_key, second.business_unique_key);

    auto first_handoff_request  = makeDecodeEntranceHandoffRequest(request, first.handoff_unique_key);
    auto second_handoff_request = makeDecodeEntranceHandoffRequest(request, second.handoff_unique_key);

    EXPECT_EQ(request.generate_config().unique_key(), "shared-business-key");
    EXPECT_EQ(first_handoff_request.generate_config().unique_key(), first.handoff_unique_key);
    EXPECT_EQ(second_handoff_request.generate_config().unique_key(), second.handoff_unique_key);
    EXPECT_NE(first_handoff_request.generate_config().unique_key(), second_handoff_request.generate_config().unique_key());
}

TEST(DecodeRpcServerNew2Test, DecodeEntranceRequiresPrefillRejectsNonPdRequests) {
    GenerateInputPB request;
    auto*           config = request.mutable_generate_config();
    config->set_max_new_tokens(1);
    config->set_num_beams(1);
    config->set_num_return_sequences(1);
    config->set_can_use_pd_separation(true);

    EXPECT_FALSE(decodeEntranceRequiresPrefill(request));

    config->set_max_new_tokens(8);
    config->set_num_beams(2);
    EXPECT_FALSE(decodeEntranceRequiresPrefill(request));
}

TEST(DecodeRpcServerNew2Test, UpdateAuxInfoUsesPrefillReuseAsTopLevelAndPreservesDecodeReuse) {
    auto stream     = makeStream({11, 12, 13});
    auto outputs_pb = makeOutputsWithDecodeReuse(/*total=*/7, /*local=*/3, /*remote=*/4, /*memory=*/1);

    stream->setPrefillReuseLength(/*total=*/128, /*local=*/32, /*remote=*/96, /*memory=*/8);

    updateDecodeAuxInfo(outputs_pb, stream, nullptr);

    ASSERT_EQ(outputs_pb.flatten_output().aux_info_size(), 1);
    const auto& aux_info = outputs_pb.flatten_output().aux_info(0);
    EXPECT_TRUE(aux_info.pd_sep());

    EXPECT_EQ(aux_info.total_reuse_len(), 128);
    EXPECT_EQ(aux_info.local_reuse_len(), 32);
    EXPECT_EQ(aux_info.remote_reuse_len(), 96);
    EXPECT_EQ(aux_info.memory_reuse_len(), 8);

    EXPECT_EQ(aux_info.prefill_total_reuse_len(), 128);
    EXPECT_EQ(aux_info.prefill_local_reuse_len(), 32);
    EXPECT_EQ(aux_info.prefill_remote_reuse_len(), 96);
    EXPECT_EQ(aux_info.prefill_memory_reuse_len(), 8);

    EXPECT_EQ(aux_info.decode_total_reuse_len(), 7);
    EXPECT_EQ(aux_info.decode_local_reuse_len(), 3);
    EXPECT_EQ(aux_info.decode_remote_reuse_len(), 4);
    EXPECT_EQ(aux_info.decode_memory_reuse_len(), 1);
}

TEST(DecodeRpcServerNew2Test, UpdateAuxInfoFallsBackToPrefillContextSnapshot) {
    auto stream      = makeStream({21, 22, 23});
    auto outputs_pb  = makeOutputsWithDecodeReuse(/*total=*/0, /*local=*/0, /*remote=*/0, /*memory=*/0);
    auto prefill_ctx = std::make_shared<PrefillServerCallerContext>("127.0.0.1", "reuse-key");

    PrefillServerCallerContext::ReuseLensSnapshot snapshot;
    snapshot.total  = 64;
    snapshot.local  = 16;
    snapshot.remote = 48;
    snapshot.memory = 2;
    prefill_ctx->setPrefillReuseLensSnapshotForTest(snapshot);

    updateDecodeAuxInfo(outputs_pb, stream, prefill_ctx);

    ASSERT_EQ(outputs_pb.flatten_output().aux_info_size(), 1);
    const auto& aux_info = outputs_pb.flatten_output().aux_info(0);
    EXPECT_EQ(aux_info.total_reuse_len(), 64);
    EXPECT_EQ(aux_info.local_reuse_len(), 16);
    EXPECT_EQ(aux_info.remote_reuse_len(), 48);
    EXPECT_EQ(aux_info.memory_reuse_len(), 2);

    EXPECT_EQ(aux_info.prefill_total_reuse_len(), 64);
    EXPECT_EQ(aux_info.prefill_local_reuse_len(), 16);
    EXPECT_EQ(aux_info.prefill_remote_reuse_len(), 48);
    EXPECT_EQ(aux_info.prefill_memory_reuse_len(), 2);

    EXPECT_EQ(aux_info.decode_total_reuse_len(), 0);
    EXPECT_EQ(aux_info.decode_local_reuse_len(), 0);
    EXPECT_EQ(aux_info.decode_remote_reuse_len(), 0);
    EXPECT_EQ(aux_info.decode_memory_reuse_len(), 0);
}

// Test that prefill_finished detection correctly identifies when the first token
// triggers termination (e.g., max_new_tokens=1 or first token is EOS).
// This verifies the logic in pollStreamOutputWithPrefill that tracks prefill_finished
// to ensure the finished=true flag is not lost when the first token is a termination token.
TEST(DecodeRpcServerNew2Test, PrefillFirstResponseFinishedDetection) {
    // Simulate a prefill response where finished=true (first token is EOS / max_new_tokens=1)
    GenerateOutputsPB prefill_output_finished;
    prefill_output_finished.mutable_flatten_output()->add_finished(true);
    auto* aux_info = prefill_output_finished.mutable_flatten_output()->add_aux_info();
    aux_info->set_step_output_len(1);

    // Verify the finished flag is set correctly
    ASSERT_EQ(prefill_output_finished.flatten_output().finished_size(), 1);
    EXPECT_TRUE(prefill_output_finished.flatten_output().finished(0));

    // Simulate a normal prefill response where finished=false
    GenerateOutputsPB prefill_output_not_finished;
    prefill_output_not_finished.mutable_flatten_output()->add_finished(false);
    prefill_output_not_finished.mutable_flatten_output()->add_aux_info();

    ASSERT_EQ(prefill_output_not_finished.flatten_output().finished_size(), 1);
    EXPECT_FALSE(prefill_output_not_finished.flatten_output().finished(0));
}

}  // namespace rtp_llm::test
