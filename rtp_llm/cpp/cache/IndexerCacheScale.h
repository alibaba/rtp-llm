#pragma once

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include <memory>
#include <utility>
#include <vector>

namespace rtp_llm {

// Destination still contains the source bytes. No-op for identical encodings;
// MX -> FP32 restores scales; FP32 -> MX requantizes payload and scales.
// Conversion completes synchronously before cache readiness.
using IndexerScaleConversion = std::pair<BlockInfo, IndexerCacheLayout>;
// Pre-metadata releases use FP32 scales. Infer only an entirely absent layout,
// and only from a local descriptor that explicitly identifies this indexer ABI.
IndexerCacheLayout indexerSourceLayout(const BlockInfo& destination, IndexerCacheLayout source);
struct IndexerTransferBuffer {
    BlockInfo             block;
    size_t                allocation_bytes = 0;
    std::shared_ptr<void> owner;
};
// Old receivers cannot convert MX. Produce a private FP32-scale transfer copy;
// an already compatible source is returned without allocating or copying.
IndexerTransferBuffer legacyIndexerTransfer(const BlockInfo& source, size_t alignment = 1);
void validateIndexerScaleTransfer(const BlockInfo& destination, const IndexerCacheLayout& source);
void convertIndexerCacheScales(const std::vector<IndexerScaleConversion>& blocks);

}  // namespace rtp_llm
