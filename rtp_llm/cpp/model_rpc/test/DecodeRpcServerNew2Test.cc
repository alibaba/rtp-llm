#include <memory>

#include <gtest/gtest.h>
#include "torch/all.h"

#define private public
#define protected public
#include "rtp_llm/cpp/model_rpc/DecodeRpcServerNew2.h"
#undef private
#undef protected
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

TEST(DecodeRpcServerNew2Test, SelectDecodeEntranceDpIndexUsesHandoffSequence) {
    EXPECT_EQ(selectDecodeEntranceDpIndex(3, 0), 0);
    EXPECT_EQ(selectDecodeEntranceDpIndex(3, 1), 1);
    EXPECT_EQ(selectDecodeEntranceDpIndex(3, 2), 2);
    EXPECT_EQ(selectDecodeEntranceDpIndex(3, 3), 0);
    EXPECT_EQ(selectDecodeEntranceDpIndex(0, 10), 0);
}

TEST(DecodeRpcServerNew2Test, ParsePrefillDpAddrSupportsIpv4HostAndBracketIpv6) {
    std::string ip;
    uint32_t    port = 0;

    ASSERT_TRUE(DecodeRpcServerNew2::parsePrefillDpAddr("127.0.0.1:9000", &ip, &port).ok());
    EXPECT_EQ(ip, "127.0.0.1");
    EXPECT_EQ(port, 9000);

    ASSERT_TRUE(DecodeRpcServerNew2::parsePrefillDpAddr("prefill-0.service:9001", &ip, &port).ok());
    EXPECT_EQ(ip, "prefill-0.service");
    EXPECT_EQ(port, 9001);

    ASSERT_TRUE(DecodeRpcServerNew2::parsePrefillDpAddr("[::1]:9002", &ip, &port).ok());
    EXPECT_EQ(ip, "[::1]");
    EXPECT_EQ(port, 9002);

    ASSERT_TRUE(DecodeRpcServerNew2::parsePrefillDpAddr("fe80::1:9003", &ip, &port).ok());
    EXPECT_EQ(ip, "[fe80::1]");
    EXPECT_EQ(port, 9003);
}

TEST(DecodeRpcServerNew2Test, ParsePrefillDpAddrRejectsMalformedAddressOrPort) {
    std::string ip;
    uint32_t    port = 0;

    EXPECT_FALSE(DecodeRpcServerNew2::parsePrefillDpAddr("", &ip, &port).ok());
    EXPECT_FALSE(DecodeRpcServerNew2::parsePrefillDpAddr("127.0.0.1", &ip, &port).ok());
    EXPECT_FALSE(DecodeRpcServerNew2::parsePrefillDpAddr("fe80::1", &ip, &port).ok());
    EXPECT_FALSE(DecodeRpcServerNew2::parsePrefillDpAddr("[::1]9000", &ip, &port).ok());
    EXPECT_FALSE(DecodeRpcServerNew2::parsePrefillDpAddr("127.0.0.1:0", &ip, &port).ok());
    EXPECT_FALSE(DecodeRpcServerNew2::parsePrefillDpAddr("127.0.0.1:65536", &ip, &port).ok());
    EXPECT_FALSE(DecodeRpcServerNew2::parsePrefillDpAddr("127.0.0.1:not-a-port", &ip, &port).ok());
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
    DecodeRpcServerNew2 server;
    auto stream     = makeStream({11, 12, 13});
    auto outputs_pb = makeOutputsWithDecodeReuse(/*total=*/7, /*local=*/3, /*remote=*/4, /*memory=*/1);

    stream->setPrefillReuseLength(/*total=*/128, /*local=*/32, /*remote=*/96, /*memory=*/8);

    server.updateAuxInfo(outputs_pb, stream);

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

}  // namespace rtp_llm::test
