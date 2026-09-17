#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/model_rpc/DSV41RpcState.h"
#include "rtp_llm/cpp/utils/HashUtil.h"

namespace rtp_llm {
namespace {

ModelConfig modelConfig(bool draft = false) {
    ModelConfig model;
    model.num_layers = draft ? 3 : 40;
    model.hidden_size = 5120;
    model.max_seq_len = 1048576;
    model.dsv41_model_revision = std::string(40, 'a');
    auto& attn = model.attn_config;
    attn.dsv41_cache_layout_version = 1;
    attn.head_num = 64;
    attn.kv_head_num = 1;
    attn.size_per_head = 512;
    attn.sliding_window = 128;
    attn.indexer_head_dim = 128;
    attn.indexer_head_num = 32;
    attn.indexer_topk = 512;
    for (int layer = 0; layer < model.num_layers; ++layer)
        attn.layer_compress_ratios.push_back(draft || layer < 2 ? 0 : (layer < 20 ? 2 : 1));
    return model;
}

CacheConfig cacheConfig(RoleType role, bool draft = false) {
    ParallelismConfig parallelism;
    parallelism.role_type = role;
    parallelism.tp_size = role == RoleType::PREFILL ? 8 : 1;
    parallelism.prefill_cp_config.method = CPRotateMethod::PREFILL_CP;
    parallelism.prefill_cp_config.kv_cache_sharded = true;
    parallelism.prefill_cp_config.prefill_cp_size = 8;
    KVCacheConfig kv;
    kv.seq_size_per_block = 128;
    kv.dsv4_fixed_pool_blocks = 8;
    kv.test_block_num = 4;
    if (!draft)
        return CacheConfigCreator::createBasicConfig(modelConfig(), parallelism, kv, false, 5);
    SpeculativeExecutionConfig speculative;
    speculative.type = SP_TYPE_DSPARK;
    speculative.gen_num_per_cycle = 5;
    return CacheConfigCreator::createSpConfig(modelConfig(), modelConfig(true), parallelism, RuntimeConfig(), kv,
                                              speculative, std::nullopt, true, false);
}

DSV41ExecutionState publication(int64_t end, uint32_t drafts = 0) {
    DSV41ExecutionState state;
    state.request_id = 19;
    state.materialized_end = state.encoder_materialized_end = state.decoder_checkpoint_end = end;
    state.draft_layers = drafts;
    state.global_entries = state.index_entries = {end / 2, end / 2, end / 2, end};
    state.swa_valid_start.assign(40 + drafts, std::max<int64_t>(0, end - 128));
    state.swa_valid_end.assign(40 + drafts, end);
    state.swa_replay_floor.assign(40 + drafts, 0);
    state.pair_positions.assign(3, end % 2 ? end - 1 : -1);
    state.pair_valid.assign(3, end % 2);
    state.history_token_ids = {31, 32, 33};
    state.history_image_mask = {0, 0, 0};
    state.history_ready = true;
    if (drafts) {
        state.draft_committed = true;
        state.aux_valid_start = std::max<int64_t>(0, end - 128);
        state.aux_valid_end = end;
    }
    return state;
}

TEST(DSV41RpcStateTest, IdentityAcceptsPhysicalByteSlicesButRejectsExecutionContractChanges) {
    for (bool draft : {false, true}) {
        const auto prefill = cacheConfig(RoleType::PREFILL, draft);
        const auto decode = cacheConfig(RoleType::DECODE, draft);
        const auto identity = dsv41TransferIdentity(prefill, 8);
        EXPECT_NE(prefill.dsv41LayoutFingerprint(), decode.dsv41LayoutFingerprint());
        EXPECT_NO_THROW(validateDSV41TransferIdentity(decode, identity, 8));
        auto changed = identity;
        changed.set_replay_mode("bounded_checkpoint_v1");
        EXPECT_THROW(validateDSV41TransferIdentity(decode, changed, 8), std::invalid_argument);
        changed = identity;
        changed.set_tail_policy_version(2);
        EXPECT_THROW(validateDSV41TransferIdentity(decode, changed, 8), std::invalid_argument);
        changed = identity;
        changed.set_model_revision(std::string(40, 'b'));
        EXPECT_THROW(validateDSV41TransferIdentity(decode, changed, 8), std::invalid_argument);
        auto wrong_sources = decode;
        wrong_sources.dsv41_topk_owner.at(28) = 24;
        EXPECT_THROW(validateDSV41TransferIdentity(wrong_sources, identity, 8), std::invalid_argument);
        EXPECT_THROW(validateDSV41TransferIdentity(decode, V41TransferIdentityPB{}, 8), std::invalid_argument);
    }
}

TEST(DSV41RpcStateTest, LoadUsesProducerKeysWithoutChangingLocalMemoryIdentity) {
    const auto prefill = cacheConfig(RoleType::PREFILL, true);
    const auto decode = cacheConfig(RoleType::DECODE, true);
    const auto producer_seed = dsv41CacheIdentity(prefill).cacheKeySeed();
    const auto local_seed = dsv41CacheIdentity(decode).cacheKeySeed();
    ASSERT_NE(producer_seed, local_seed);
    ASSERT_NO_THROW(validateDSV41TransferIdentity(decode, dsv41TransferIdentity(prefill, 8), 8));
    for (int64_t length : {18, 128, 129, 1024, 1025, 1048576}) {
        std::vector<int32_t> tokens(length, 31);
        CacheKeysType source_keys, local_keys;
        auto source_hash = producer_seed;
        auto local_hash = local_seed;
        for (size_t start = 0; start < tokens.size(); start += 128) {
            const auto end = std::min(start + 128, tokens.size());
            source_hash = hashInt64Array(source_hash, tokens.data() + start, tokens.data() + end);
            local_hash = hashInt64Array(local_hash, tokens.data() + start, tokens.data() + end);
            source_keys.push_back(source_hash);
            local_keys.push_back(local_hash);
        }
        ASSERT_NE(source_keys, local_keys);
        auto input = std::make_shared<GenerateInput>();
        input->input_ids = torch::tensor(tokens, torch::kInt32);
        auto complete = std::make_shared<CompleteTokenIds>(1, 1, length, 128);
        complete->init(input);
        const auto snapshot = dsv41PrefillPromptCacheKeys(complete, prefill);
        EXPECT_EQ(snapshot, source_keys);
        auto live_resource = std::make_shared<BatchKVCacheResource>();
        live_resource->resetBatchSize(1);
        live_resource->cacheResource().setDsv41CacheState(
            std::make_shared<DSV41CacheState>(dsv41CacheIdentity(prefill)));
        initCacheKeys(live_resource, complete, 128);
        dropLastPartialBlock(live_resource);  // P may finish before the RPC thread sends LOAD.
        EXPECT_EQ(live_resource->cacheKeys().size(), length / 128);
        EXPECT_EQ(snapshot, source_keys);
        const auto saved_local_keys = local_keys;
        GenerateRequestPB load;
        load.set_request_id(19);
        load.set_stage(RemoteStage::LOAD);
        for (auto key : snapshot) load.add_prefill_cache_keys(key);
        GenerateRequestPB received;
        ASSERT_TRUE(received.ParseFromString(load.SerializeAsString()));
        EXPECT_EQ(dsv41PrefillCacheKeysForLoad(received, 19, length, 128), source_keys);
        EXPECT_EQ(local_keys, saved_local_keys);
        EXPECT_EQ(dsv41CacheIdentity(decode).cacheKeySeed(), local_seed);
        EXPECT_THROW(dsv41PrefillCacheKeysForLoad(received, 20, length, 128), std::invalid_argument);
        received.mutable_prefill_cache_keys()->RemoveLast();
        EXPECT_THROW(dsv41PrefillCacheKeysForLoad(received, 19, length, 128), std::invalid_argument);
        received.CopyFrom(load);
        received.add_prefill_cache_keys(0);
        EXPECT_THROW(dsv41PrefillCacheKeysForLoad(received, 19, length, 128), std::invalid_argument);
        received.CopyFrom(load);
        received.set_stage(RemoteStage::ALLOCATE);
        EXPECT_THROW(dsv41PrefillCacheKeysForLoad(received, 19, length, 128), std::invalid_argument);
    }
}

TEST(DSV41RpcStateTest, ProducerSnapshotRetainsCanonicalImageContentIdentityAcrossPages) {
    const auto prefill = cacheConfig(RoleType::PREFILL);
    auto input = std::make_shared<GenerateInput>();
    input->input_ids = torch::full({133}, 31, torch::kInt32);
    input->input_ids.slice(0, 127, 131).fill_(129264);
    auto prepared = std::make_shared<V41RequestInputs>();
    prepared->token_types = torch::full({133}, -1, torch::kInt32);
    prepared->token_types.slice(0, 127, 131).copy_(torch::arange(4, torch::kInt32));
    prepared->image_mask = prepared->token_types.ne(-1);
    V41ImageInput image;
    image.start = 127;
    image.n_vit_h = image.n_vit_w = 1;
    image.patches = torch::ones({1, 3, 14, 14}, torch::kBFloat16);
    image.types = torch::arange(4, torch::kInt32);
    image.content_sha256 = std::string(64, 'a');
    image.processor_identity = std::string(64, 'b');
    prepared->images.push_back(image);
    input->v41_inputs = prepared;
    auto complete = std::make_shared<CompleteTokenIds>(1, 1, 133, 128);
    complete->init(input);
    const auto snapshot = dsv41PrefillPromptCacheKeys(complete, prefill);
    ASSERT_EQ(snapshot.size(), 2u);
    prepared->images[0].content_sha256 = std::string(64, 'c');
    const auto other_content = dsv41PrefillPromptCacheKeys(complete, prefill);
    EXPECT_NE(snapshot[0], other_content[0]);
    EXPECT_NE(snapshot[1], other_content[1]);
    prepared->images[0].content_sha256 = image.content_sha256;
    prepared->images[0].processor_identity = std::string(64, 'd');
    EXPECT_NE(snapshot, dsv41PrefillPromptCacheKeys(complete, prefill));
    prepared->images[0].processor_identity = image.processor_identity;
    EXPECT_EQ(snapshot, dsv41PrefillPromptCacheKeys(complete, prefill));
}

TEST(DSV41RpcStateTest, CompleteEightShardInventoryAndPhysicalSliceSizes) {
    std::vector<std::string> peers;
    for (int rank = 0; rank < 8; ++rank) peers.push_back("host:" + std::to_string(9000 + rank) + ":0");
    EXPECT_NO_THROW(validateDSV41Peers(peers, 8));
    peers.back() = peers.front();
    EXPECT_THROW(validateDSV41Peers(peers, 8), std::invalid_argument);
    peers.pop_back();
    EXPECT_THROW(validateDSV41Peers(peers, 8), std::invalid_argument);
    const auto prefill = cacheConfig(RoleType::PREFILL, true);
    const auto decode = cacheConfig(RoleType::DECODE, true);
    for (size_t group : {4u, 5u}) {
        EXPECT_EQ(dsv41FixedDestinationSliceBytes(decode, group, 8), prefill.cache_specs[group]->block_size_bytes());
        EXPECT_EQ(dsv41FixedDestinationSliceBytes(decode, group, 8) * 8, decode.cache_specs[group]->block_size_bytes());
        EXPECT_THROW(dsv41FixedDestinationSliceBytes(prefill, group, 8), std::invalid_argument);
        EXPECT_THROW(dsv41FixedDestinationSliceBytes(decode, group, 4), std::invalid_argument);
    }
    EXPECT_EQ(dsv41FixedDestinationSliceBytes(decode, 4, 8), 3648u);
    EXPECT_EQ(dsv41FixedDestinationSliceBytes(decode, 5, 8), 9024u);
    EXPECT_THROW(dsv41FixedDestinationSliceBytes(decode, 0, 8), std::invalid_argument);
}

TEST(DSV41RpcStateTest, OddPublicationRestoresReadyOnlyWithCompleteTargetAndPairState) {
    const auto local_identity = dsv41CacheIdentity(cacheConfig(RoleType::DECODE));
    auto recovered = publication(1025);
    DSV41CacheState state(local_identity);
    EXPECT_EQ(state.view().target_ready_end, 0);
    auto missing = recovered;
    missing.pair_valid[1] = 0;
    EXPECT_THROW(state.publishExecution(missing, 0), std::invalid_argument);
    EXPECT_EQ(state.view().target_ready_end, 0);
    missing = recovered;
    missing.swa_valid_end[39]--;
    EXPECT_THROW(state.publishExecution(missing, 0), std::invalid_argument);
    missing = recovered;
    missing.index_entries[2]--;
    EXPECT_THROW(state.publishExecution(missing, 0), std::invalid_argument);
    EXPECT_NO_THROW(state.publishExecution(recovered, 0));
    EXPECT_EQ(state.view().target_ready_end, 1025);
    EXPECT_THROW(recovered.checkpoint(local_identity, 1024), std::invalid_argument);
}

TEST(DSV41RpcStateTest, DraftAndMemoryCheckpointRemainCompleteAndAlignedContracts) {
    const auto decode = cacheConfig(RoleType::DECODE, true);
    const auto identity = dsv41CacheIdentity(decode);
    auto state = publication(1024, 3);
    EXPECT_NO_THROW(state.checkpoint(identity, 1024));
    state.draft_committed = false;
    EXPECT_THROW(state.validate(identity, 3), std::invalid_argument);
    state = publication(1024);
    EXPECT_THROW(state.validate(identity, 3), std::invalid_argument);
    state = publication(1025, 3);
    EXPECT_NO_THROW(state.validate(identity, 3));
    EXPECT_THROW(state.checkpoint(identity, 1024), std::invalid_argument);
}

}  // namespace
}  // namespace rtp_llm
