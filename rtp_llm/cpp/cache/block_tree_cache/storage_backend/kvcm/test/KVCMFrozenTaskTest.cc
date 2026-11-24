#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeReclaimer.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"

#include <set>


namespace rtp_llm {
namespace {

TEST(KVCMFrozenTaskTest, TargetsExcludeLocalPrefixFilteredLinearStateAndUnsentTail) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    auto environment = makeMultiGroupBackendEnvironment("kvcm_frozen_target_boundary", 1, 1);
    auto client = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    ScopedReferencedBlocks allocation(environment, 4);
    const auto& blocks = allocation.get();
    const kv_cache_manager::Locations locations{
        {{"tp0_Ffull0", "full-102"}, {"tp0_Llinear0", "linear-102-filtered"}},
        {{"tp0_Ffull0", "full-103"}, {"tp0_Llinear0", "linear-103"}},
        {{"tp0_Ffull0", "full-104-unsent"}}};
    EXPECT_CALL(*client, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, locations)));
    auto request = makeGroupedStorageRequest(environment, {101, 102, 103, 104}, 1, blocks);
    auto matched = match(*backend.backend, request);
    ASSERT_TRUE(matched.success);
    ASSERT_EQ(matched.matched_blocks_num, 3u);
    ASSERT_NE(matched.match_meta, nullptr);
    request.keys = std::make_shared<const CacheKeysType>(CacheKeysType{101, 102, 103});
    request.handles.resize(3);
    std::vector<std::string> trace{"block_tree_read_1"};
    size_t tags = 0;
    EXPECT_CALL(*client, loadKvCachesForTag(_, _, _, _)).Times(4)
        .WillRepeatedly(Invoke([&](const auto&, const auto&, auto&, const auto&) {
            EXPECT_EQ(backend->reclaimer_->localState(trace.front()), TransferTaskRegistry::State::RUNNING);
            return ++tags < 4;
        }));
    EXPECT_TRUE(readResult(*backend.backend, request, matched.match_meta).ok());
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
    EXPECT_EQ(backend->reclaimer_->localState(trace.front()), TransferTaskRegistry::State::BUFFER_SAFE);
    EXPECT_TRUE(backend->reclaimer_->frozen_.empty());
    trace.front() = "block_tree_read_2";
    std::promise<ErrorInfo> done;
    auto completed = done.get_future();
    backend->read(std::move(request), std::move(matched.match_meta), [&](ErrorInfo error) {
        EXPECT_EQ(backend->reclaimer_->frozen_.count(trace.front()), 1u);
        done.set_value(std::move(error));
    });
    EXPECT_FALSE(await(completed).ok());
    EXPECT_EQ(backend->reclaimer_->localState(trace.front()), TransferTaskRegistry::State::RUNNING);
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
    const auto& record = backend->reclaimer_->frozen_.at(trace.front());
    std::set<std::pair<IBlockPool*, BlockIdxType>> targets;
    for (const auto& target : record.targets)
        targets.emplace(target.pool.get(), target.block);
    EXPECT_EQ(targets, (std::set<std::pair<IBlockPool*, BlockIdxType>>{
        {environment.pools_by_tag.at("full0").get(), blocks[1]},
        {environment.pools_by_tag.at("full0").get(), blocks[2]},
        {environment.pools_by_tag.at("linear0").get(), blocks[2]}}));
    for (const auto& tag : environment.cache_config.groupTags()) {
        const auto& pool = environment.pools_by_tag.at(tag);
        for (size_t i = 0; i < blocks.size(); ++i) {
            const bool frozen = i == 2 || (tag == "full0" && i == 1);
            EXPECT_EQ(pool->isFrozen(blocks[i]), frozen) << tag << " index=" << i;
            EXPECT_EQ(pool->treeRefCount(blocks[i]), frozen ? 1u : 0u);
        }
    }
}

}  // namespace
}  // namespace rtp_llm
