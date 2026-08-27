#include "rtp_llm/cpp/cache/KdaShadowCache.h"

#include <stdexcept>
#include <utility>

namespace rtp_llm {
namespace {

std::string invalidTransition(KdaShadowState state, KdaShadowCommandType command) {
    return "invalid KDA shadow transition: state=" + std::to_string(static_cast<int>(state))
           + " command=" + std::to_string(static_cast<int>(command));
}

}  // namespace

size_t KdaShadowKeyHash::operator()(const KdaShadowKey& key) const {
    const size_t h1 = std::hash<int64_t>{}(key.request_id);
    const size_t h2 = std::hash<uint64_t>{}(key.generation_epoch);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
}

HybridKdaShadowBlockAllocator::HybridKdaShadowBlockAllocator(HybridKVCacheAllocatorPtr allocator, int group_id):
    allocator_(std::move(allocator)), group_id_(group_id) {
    if (!allocator_) {
        throw std::invalid_argument("KDA shadow allocator must not be null");
    }
}

bool HybridKdaShadowBlockAllocator::reserve(int               seq_len,
                                            BlockIndicesType& blocks,
                                            BlockIndicesType& kernel_blocks) {
    BlockIds block_ids;
    if (!allocator_->mallocLinearGroupBlocks(group_id_, block_ids, seq_len)) {
        return false;
    }
    blocks        = block_ids.blocks();
    kernel_blocks = block_ids.kernelBlocks();
    return true;
}

bool HybridKdaShadowBlockAllocator::release(const BlockIndicesType& blocks) {
    allocator_->freeLinearGroupBlocks(group_id_, blocks);
    return true;
}

KdaShadowRegistry::KdaShadowRegistry(std::shared_ptr<KdaShadowBlockAllocator> allocator):
    allocator_(std::move(allocator)) {
    if (!allocator_) {
        throw std::invalid_argument("KDA shadow registry allocator must not be null");
    }
}

KdaShadowRegistry::~KdaShadowRegistry() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [_, record] : records_) {
        if (isLive(record.state) && record.owns_blocks && !record.blocks.empty()) {
            allocator_->release(record.blocks);
            record.blocks.clear();
            record.state = KdaShadowState::RELEASED;
        }
    }
}

bool KdaShadowRegistry::isLive(KdaShadowState state) {
    return state == KdaShadowState::RESERVED || state == KdaShadowState::LOADING
           || state == KdaShadowState::READY || state == KdaShadowState::RELEASING
           || state == KdaShadowState::ERROR;
}

KdaShadowResult KdaShadowRegistry::result(const KdaShadowRecord& record, bool success, bool idempotent) const {
    return {success, idempotent, record.state, record.blocks, record.kernel_blocks, record.error};
}

bool KdaShadowRegistry::releaseBlocks(KdaShadowRecord& record, std::string& error) {
    if (record.blocks.empty() || !record.owns_blocks) {
        record.blocks.clear();
        record.kernel_blocks.clear();
        return true;
    }
    if (!allocator_->release(record.blocks)) {
        error = "failed to release KDA shadow blocks";
        return false;
    }
    record.blocks.clear();
    record.kernel_blocks.clear();
    return true;
}

KdaShadowResult KdaShadowRegistry::apply(const KdaShadowCommand& command) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                       it = records_.find(command.key);

    if (command.type == KdaShadowCommandType::RESERVE) {
        if (command.seq_len <= 0) {
            return {false, false, KdaShadowState::ABSENT, {}, {}, "KDA shadow seq_len must be positive"};
        }
        if (it != records_.end()) {
            auto& existing = it->second;
            if ((existing.state == KdaShadowState::RESERVED || existing.state == KdaShadowState::LOADING
                 || existing.state == KdaShadowState::READY)
                && existing.seq_len == command.seq_len) {
                return result(existing, true, true);
            }
            auto failed  = result(existing, false);
            failed.error = existing.state == KdaShadowState::RELEASED ?
                               "released KDA shadow epoch cannot be reserved again" :
                               "duplicate KDA shadow reserve has incompatible seq_len or state";
            return failed;
        }

        KdaShadowRecord record;
        record.key     = command.key;
        record.seq_len = command.seq_len;
        if (!allocator_->reserve(command.seq_len, record.blocks, record.kernel_blocks)) {
            record.state = KdaShadowState::ERROR;
            record.error = "failed to reserve KDA shadow blocks";
        } else {
            record.state = KdaShadowState::RESERVED;
        }
        auto [inserted, _] = records_.emplace(command.key, std::move(record));
        return result(inserted->second, inserted->second.state == KdaShadowState::RESERVED);
    }

    if (command.type == KdaShadowCommandType::ADOPT) {
        if (command.seq_len <= 0 || command.adopted_blocks.empty()
            || command.adopted_kernel_blocks.empty()) {
            return {false,
                    false,
                    KdaShadowState::ABSENT,
                    {},
                    {},
                    "adopted KDA shadow blocks and kernel blocks must be non-empty"};
        }
        if (it != records_.end()) {
            auto& existing = it->second;
            if (!existing.owns_blocks && existing.seq_len == command.seq_len
                && existing.blocks == command.adopted_blocks
                && existing.kernel_blocks == command.adopted_kernel_blocks
                && (existing.state == KdaShadowState::RESERVED
                    || existing.state == KdaShadowState::LOADING
                    || existing.state == KdaShadowState::READY)) {
                return result(existing, true, true);
            }
            auto failed  = result(existing, false);
            failed.error = "duplicate KDA shadow adopt has incompatible blocks or state";
            return failed;
        }
        KdaShadowRecord record;
        record.key           = command.key;
        record.seq_len       = command.seq_len;
        record.blocks        = command.adopted_blocks;
        record.kernel_blocks = command.adopted_kernel_blocks;
        record.owns_blocks   = false;
        record.state         = KdaShadowState::RESERVED;
        auto [inserted, _]   = records_.emplace(command.key, std::move(record));
        return result(inserted->second, true);
    }

    if (it == records_.end()) {
        return {false, false, KdaShadowState::ABSENT, {}, {}, "KDA shadow record does not exist"};
    }

    auto& record = it->second;
    switch (command.type) {
        case KdaShadowCommandType::ADOPT:
            break;
        case KdaShadowCommandType::LOAD:
            if (record.state == KdaShadowState::RESERVED) {
                record.state = KdaShadowState::LOADING;
                return result(record, true);
            }
            if (record.state == KdaShadowState::LOADING || record.state == KdaShadowState::READY) {
                return result(record, true, true);
            }
            break;
        case KdaShadowCommandType::COMMIT:
            if (record.state == KdaShadowState::LOADING) {
                record.state = KdaShadowState::READY;
                return result(record, true);
            }
            if (record.state == KdaShadowState::READY) {
                return result(record, true, true);
            }
            break;
        case KdaShadowCommandType::FAIL:
            if (record.state == KdaShadowState::ERROR) {
                return result(record, true, true);
            }
            if (isLive(record.state) && record.state != KdaShadowState::RELEASING) {
                record.state = KdaShadowState::ERROR;
                record.error = command.error.empty() ? "KDA shadow load failed" : command.error;
                return result(record, true);
            }
            break;
        case KdaShadowCommandType::ROLLBACK:
            if (record.state == KdaShadowState::RELEASED) {
                return result(record, true, true);
            }
            if (record.state == KdaShadowState::RESERVED || record.state == KdaShadowState::LOADING
                || record.state == KdaShadowState::READY || record.state == KdaShadowState::ERROR) {
                record.state = KdaShadowState::RELEASING;
                std::string error;
                if (!releaseBlocks(record, error)) {
                    record.state = KdaShadowState::ERROR;
                    record.error = std::move(error);
                    return result(record, false);
                }
                record.state = KdaShadowState::RELEASED;
                record.error.clear();
                return result(record, true);
            }
            break;
        case KdaShadowCommandType::RELEASE:
            if (record.state == KdaShadowState::RELEASED) {
                return result(record, true, true);
            }
            if (isLive(record.state)) {
                record.state = KdaShadowState::RELEASING;
                std::string error;
                if (!releaseBlocks(record, error)) {
                    record.state = KdaShadowState::ERROR;
                    record.error = std::move(error);
                    return result(record, false);
                }
                record.state = KdaShadowState::RELEASED;
                record.error.clear();
                return result(record, true);
            }
            break;
        case KdaShadowCommandType::RESERVE:
            break;
    }

    record.error = invalidTransition(record.state, command.type);
    return result(record, false);
}

std::vector<BlockIndicesType>
KdaShadowRegistry::buildReadyKernelBlockRows(
    const std::vector<std::optional<KdaShadowKey>>& rank_major_keys) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<BlockIndicesType> rows;
    rows.reserve(rank_major_keys.size());
    for (const auto& key : rank_major_keys) {
        if (!key) {
            rows.emplace_back();
            continue;
        }
        const auto it = records_.find(*key);
        if (it == records_.end() || it->second.state != KdaShadowState::READY) {
            throw std::runtime_error("KDA shadow kernel block table requested before local shard is READY");
        }
        rows.push_back(it->second.kernel_blocks);
    }
    return rows;
}

std::vector<KdaShadowRecord> KdaShadowRegistry::readyRecords() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<KdaShadowRecord> result;
    for (const auto& [_, record] : records_) {
        if (record.state == KdaShadowState::READY) {
            result.push_back(record);
        }
    }
    return result;
}

std::optional<KdaShadowRecord> KdaShadowRegistry::record(const KdaShadowKey& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  it = records_.find(key);
    return it == records_.end() ? std::nullopt : std::optional<KdaShadowRecord>(it->second);
}

std::optional<KdaShadowRecord> KdaShadowRegistry::readyRecord(const KdaShadowKey& key) const {
    auto value = record(key);
    if (!value || value->state != KdaShadowState::READY) {
        return std::nullopt;
    }
    return value;
}

std::vector<BlockIndicesType>
KdaShadowRegistry::buildReadyBlockRows(const std::vector<std::optional<KdaShadowKey>>& rank_major_keys) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<BlockIndicesType> rows;
    rows.reserve(rank_major_keys.size());
    for (const auto& key : rank_major_keys) {
        if (!key) {
            rows.emplace_back();
            continue;
        }
        const auto it = records_.find(*key);
        if (it == records_.end() || it->second.state != KdaShadowState::READY) {
            throw std::runtime_error("KDA shadow block table requested before local shard is READY");
        }
        rows.push_back(it->second.blocks);
    }
    return rows;
}

size_t KdaShadowRegistry::liveRecordCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t                      count = 0;
    for (const auto& [_, record] : records_) {
        count += isLive(record.state) ? 1 : 0;
    }
    return count;
}

size_t KdaShadowRegistry::liveBlockCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t                      count = 0;
    for (const auto& [_, record] : records_) {
        count += isLive(record.state) ? record.blocks.size() : 0;
    }
    return count;
}

}  // namespace rtp_llm
