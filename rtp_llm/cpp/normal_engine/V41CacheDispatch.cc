#include "rtp_llm/cpp/normal_engine/V41CacheDispatch.h"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>

namespace rtp_llm {

void V41CacheDispatch::prepare(const StreamGroups& streams, GptModelInputs& inputs) {
    for (auto it = requests_.begin(); it != requests_.end();) {
        const auto owner = it->second.stream.lock();
        if (!owner || owner->streamCacheResource().isResourceReleased()
            || owner->streamCacheResource().cacheIncarnation() != it->second.incarnation) {
            it = requests_.erase(it);
        } else {
            ++it;
        }
    }
    const auto batch_size = streams.totalModelBatchSize();
    inputs.v41_cache_spans =
        torch::full({static_cast<int64_t>(batch_size), 6}, -1, torch::TensorOptions(torch::kInt64));
    inputs.v41_cache_leases.clear();
    inputs.v41_cache_leases.reserve(batch_size);
    std::set<int64_t> seen;
    int64_t           row    = 0;
    const auto        append = [&](const GenerateStreamPtr& stream) {
        if (stream->isFakeStream()) {
            inputs.v41_cache_leases.insert(inputs.v41_cache_leases.end(), stream->currentBatchSize(), nullptr);
            row += stream->currentBatchSize();
            return;
        }
        const auto id = stream->streamId();
        if (id < 0 || !seen.insert(id).second || stream->currentBatchSize() != 1 || stream->maxBatchSize() != 1
            || stream->hasPendingAsyncBookkeeping()) {
            throw std::invalid_argument("V4.1 dispatch requires distinct single-sequence committed requests");
        }
        const int64_t start = stream->isContextStream() ? stream->prefixLength() : stream->seqLength() - 1;
        const int64_t count = stream->isContextStream() ? stream->contextLength() : 1;
        if (start < 0 || count <= 0 || start > 32768 || count > 32768 - start) {
            throw std::invalid_argument("V4.1 dispatch span exceeds the 32K context bound");
        }
        auto&      cache       = stream->streamCacheResource();
        const auto incarnation = cache.cacheIncarnation();
        auto       found       = requests_.find(id);
        if (found == requests_.end()) {
            if (start != 0) {
                throw std::invalid_argument(
                    "V4.1 cache admission requires token zero; prefix/PD restoration is pending");
            }
            if (generation_ == std::numeric_limits<int64_t>::max()) {
                throw std::overflow_error("V4.1 dispatch generations exhausted");
            }
            found = requests_.emplace(id, Request{stream, incarnation, ++generation_, 0, 0, {}}).first;
        }
        auto& request = found->second;
        if (request.stream.lock() != stream || request.incarnation != incarnation || request.next_position != start) {
            throw std::invalid_argument("V4.1 dispatch has a stale identity, replayed span or token gap");
        }
        auto lease = cache.retainAllocation();
        for (const auto& [tag, pages] : request.tables) {
            const auto next = lease->blockTables().find(tag);
            if (next == lease->blockTables().end() || next->second.size() < pages.size()
                || !std::equal(pages.begin(), pages.end(), next->second.begin())) {
                throw std::invalid_argument("V4.1 dispatch only allows append-only cache growth");
            }
        }
        auto* values          = inputs.v41_cache_spans.data_ptr<int64_t>() + row * 6;
        values[0]             = id;
        values[1]             = request.generation;
        values[2]             = incarnation;
        values[3]             = request.next_chunk++;
        values[4]             = start;
        values[5]             = count;
        request.next_position = start + count;
        request.tables        = lease->blockTables();
        inputs.v41_cache_leases.push_back(std::move(lease));
        ++row;
    };
    for (const auto& stream : streams.decodeStreams()) {
        append(stream);
    }
    for (const auto& stream : streams.contextStreams()) {
        append(stream);
    }
}

}  // namespace rtp_llm
