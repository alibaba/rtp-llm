#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/cache/SharedBlockCache.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/utils/HashUtil.h"

namespace rtp_llm::test {
namespace {

constexpr int kBlockSize = 64;

CompleteTokenIdsPtr makeTokens(int length, int capacity = 0, int batch_size = 1) {
    capacity               = std::max(length, capacity);
    auto input             = std::make_shared<GenerateInput>();
    input->input_ids       = torch::arange(100, 100 + capacity, torch::kInt32);
    input->generate_config = std::make_shared<GenerateConfig>();
    auto tokens            = std::make_shared<CompleteTokenIds>(batch_size, batch_size, capacity + 16, kBlockSize);
    tokens->init(input);
    tokens->setSeqLength(length);
    return tokens;
}

BatchKVCacheResourcePtr makeKeys(const CompleteTokenIdsPtr& tokens, bool mtp_lookahead = true) {
    auto resource = std::make_shared<BatchKVCacheResource>();
    resource->resetBatchSize(tokens->batchSize());
    initCacheKeys(resource, tokens, kBlockSize, mtp_lookahead);
    return resource;
}

void expectDependencies(const KVCacheResource& resource) {
    const auto& keys = resource.cacheKeys();
    const auto& deps = resource.blockDependencies();
    ASSERT_EQ(deps.size(), keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        EXPECT_EQ(deps[i].ordinal, i);
        EXPECT_EQ(deps[i].has_parent, i > 0);
        if (i > 0) {
            EXPECT_EQ(deps[i].parent_key, keys[i - 1]);
            EXPECT_NE(keys[i], keys[i - 1]);
        }
    }
}

}  // namespace

TEST(KVCacheHashUtilTest, DefaultModePreservesLegacyKeysAndPublication) {
    for (int length : {0, 1, 63, 64, 65, 127, 128, 129}) {
        SCOPED_TRACE(length);
        auto tokens   = makeTokens(length);
        auto resource = std::make_shared<BatchKVCacheResource>();
        resource->resetBatchSize(1);
        initCacheKeys(resource, tokens, kBlockSize);
        CacheKeysType expected;
        int64_t       hash = 0;
        for (int begin = 0; begin < length; begin += kBlockSize) {
            hash =
                hashInt64Array(hash, tokens->data(0) + begin, tokens->data(0) + std::min(begin + kBlockSize, length));
            expected.push_back(hash);
        }
        EXPECT_EQ(resource->cacheKeys(), expected);
        EXPECT_EQ(resource->lastBlockAligned(), length % kBlockSize == 0);
        updateCacheKeys(resource, tokens, kBlockSize);
        expected.resize(length / kBlockSize);
        EXPECT_EQ(resource->cacheKeys(), expected);
        dropLastPartialBlock(resource);
        EXPECT_EQ(resource->cacheKeys(), expected);
    }
}

TEST(KVCacheHashUtilTest, MtpIncludesTheLookaheadAtEveryBlockBoundary) {
    for (int boundary : {64, 512, 1024}) {
        SCOPED_TRACE(boundary);
        auto original = makeTokens(boundary + kBlockSize + 1);
        auto changed  = makeTokens(boundary + kBlockSize + 1);
        changed->data(0)[boundary] += 10000;
        auto a = makeKeys(original);
        auto b = makeKeys(changed);
        dropLastPartialBlock(a);
        dropLastPartialBlock(b);
        const size_t fork_block = boundary / kBlockSize - 1;
        ASSERT_EQ(a->cacheKeys().size(), b->cacheKeys().size());
        for (size_t i = 0; i < a->cacheKeys().size(); ++i) {
            if (i < fork_block) {
                EXPECT_EQ(a->cacheKeys()[i], b->cacheKeys()[i]);
            } else {
                EXPECT_NE(a->cacheKeys()[i], b->cacheKeys()[i]);
            }
        }
        changed->data(0)[boundary] = original->data(0)[boundary];
        changed->data(0)[boundary + 1] += 10000;
        b = makeKeys(changed);
        EXPECT_EQ(a->cacheKeys()[fork_block], b->cacheKeys()[fork_block]);
        EXPECT_NE(a->cacheKeys()[fork_block + 1], b->cacheKeys()[fork_block + 1]);
    }
}

TEST(KVCacheHashUtilTest, MtpTailPreservesPageCardinalityWithoutPublishingUnknownLookahead) {
    for (int length : {0, 1, 63, 64, 65, 127, 128, 129, 512, 513}) {
        SCOPED_TRACE(length);
        auto resource = makeKeys(makeTokens(length));
        EXPECT_EQ(resource->cacheKeys().size(), (length + kBlockSize - 1) / kBlockSize);
        EXPECT_EQ(resource->lastBlockAligned(), length == 0);
        expectDependencies(resource->cacheResource());
        dropLastPartialBlock(resource);
        EXPECT_EQ(resource->cacheKeys().size(), std::max(0, length - 1) / kBlockSize);
        EXPECT_TRUE(resource->lastBlockAligned());
    }
}

TEST(KVCacheHashUtilTest, MtpIncrementalKeysMatchFreshKeysForEveryBatchRow) {
    for (int initial_length : {0, 1, 63, 64, 65, 127, 128, 129}) {
        SCOPED_TRACE(initial_length);
        auto tokens = makeTokens(initial_length, 257, 2);
        tokens->data(1)[64] += 10000;
        auto incremental = makeKeys(tokens);
        for (int length : {initial_length, 64, 65, 128, 129, 192, 193, 257}) {
            if (length < initial_length) {
                continue;
            }
            SCOPED_TRACE(length);
            tokens->setSeqLength(length);
            updateCacheKeys(incremental, tokens, kBlockSize, true);
            auto fresh = makeKeys(tokens);
            dropLastPartialBlock(fresh);
            for (int row = 0; row < 2; ++row) {
                EXPECT_EQ(incremental->cacheKeys(row), fresh->cacheKeys(row));
                expectDependencies(incremental->cacheResource(row));
            }
            EXPECT_TRUE(incremental->lastBlockAligned());
            const auto once = incremental->cacheKeys();
            updateCacheKeys(incremental, tokens, kBlockSize, true);
            EXPECT_EQ(incremental->cacheKeys(), once);
        }
    }
}

TEST(KVCacheHashUtilTest, MtpKeysAreIsolatedFromLegacyCacheEntries) {
    auto tokens = makeTokens(3 * kBlockSize + 1);
    auto mtp    = makeKeys(tokens);
    auto legacy = makeKeys(tokens, false);
    dropLastPartialBlock(mtp);
    dropLastPartialBlock(legacy);
    SharedBlockCache cache;
    for (size_t i = 0; i < legacy->cacheKeys().size(); ++i) {
        cache.put(legacy->cacheKeys()[i], {static_cast<BlockIdxType>(100 + i)}, false);
    }
    for (auto key : mtp->cacheKeys()) {
        EXPECT_FALSE(cache.match(key).found);
    }
}

TEST(KVCacheHashUtilTest, MtpFirstInsertionDoesNotPoisonAnotherContinuationOrItsDescendants) {
    auto tokens_a                 = makeTokens(3 * kBlockSize + 1);
    auto tokens_b                 = makeTokens(3 * kBlockSize + 1);
    tokens_a->data(0)[kBlockSize] = 7;
    tokens_b->data(0)[kBlockSize] = 11;
    auto a                        = makeKeys(tokens_a);
    auto b                        = makeKeys(tokens_b);
    dropLastPartialBlock(a);
    dropLastPartialBlock(b);
    SharedBlockCache cache;
    for (size_t i = 0; i < a->cacheKeys().size(); ++i) {
        cache.put(a->cacheKeys()[i], {static_cast<BlockIdxType>(100 + i), static_cast<BlockIdxType>(200 + i)}, false);
        cache.put(b->cacheKeys()[i], {static_cast<BlockIdxType>(300 + i), static_cast<BlockIdxType>(400 + i)}, false);
        // Dedup keeps the first block for the same continuation, but must not
        // merge A's shifted draft KV with B's FULL or LINEAR cache entries.
        cache.put(b->cacheKeys()[i], {static_cast<BlockIdxType>(500 + i), static_cast<BlockIdxType>(600 + i)}, false);
        EXPECT_EQ(cache.matchGroup(a->cacheKeys()[i], 0), 100 + i);
        EXPECT_EQ(cache.matchGroup(b->cacheKeys()[i], 0), 300 + i);
        EXPECT_EQ(cache.matchGroup(b->cacheKeys()[i], 1), 400 + i);
    }
    auto descendant_tokens                 = makeTokens(4 * kBlockSize + 1);
    descendant_tokens->data(0)[kBlockSize] = 11;
    auto descendant                        = makeKeys(descendant_tokens);
    for (size_t i = 0; i < b->cacheKeys().size(); ++i) {
        EXPECT_EQ(descendant->cacheKeys()[i], b->cacheKeys()[i]);
        EXPECT_EQ(cache.matchGroup(descendant->cacheKeys()[i], 0), 300 + i);
    }
}

TEST(KVCacheHashUtilTest, AlignedMtpTailIsTransferredByP2PAndExcludedFromCpPrefixCache) {
    const auto config = makeSimpleMhaCacheConfig(1, 8, kBlockSize, DataType::TYPE_FP16);
    for (int length : {128, 192}) {
        SCOPED_TRACE(length);
        auto resource = makeKeys(makeTokens(length));
        resource->initGroups(config.topologyPtr());
        BlockIndicesType blocks;
        for (size_t i = 0; i < resource->cacheKeys().size(); ++i) {
            blocks.push_back(static_cast<BlockIdxType>(100 + i));
        }
        resource->setBatchBlocks(0, 0, blocks);
        auto& row      = resource->cacheResource();
        auto  transfer = LayerCacheBufferUtil::convertLayer(row, 0, 0, 0, -1);
        ASSERT_NE(transfer, nullptr);
        EXPECT_EQ(transfer->blockIdMap().size(), blocks.size());
        EXPECT_EQ(transfer->getBlockId(row.cacheKeys().back()), blocks.back());

        CPSlotMapper mapper(0, 2, kBlockSize);
        auto         canonical = mapper.canonicalCacheKeys(row.cacheKeys());
        auto         projected = mapper.projectConnectorResource(row, config, canonical);
        EXPECT_FALSE(projected.lastBlockAligned());
        auto reusable = projected.cacheKeys();
        ASSERT_FALSE(reusable.empty());
        reusable.pop_back();  // memory/remote prefix-cache publication contract
        if (length == 128) {
            EXPECT_TRUE(reusable.empty());
        } else {
            ASSERT_EQ(reusable.size(), 1);
            EXPECT_EQ(reusable.front(), row.cacheKeys()[1]);
            EXPECT_EQ(projected.cacheKeys().back(), row.cacheKeys().back());
        }
    }
}

}  // namespace rtp_llm::test
