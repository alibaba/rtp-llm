#include "rtp_llm/cpp/disaggregate/cache_store/remote_mock/RemoteMockState.h"
#include "rtp_llm/cpp/disaggregate/cache_store/remote_mock/RemoteMockFormat.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>

namespace rtp_llm {
namespace {
uint64_t newEpoch() {
    std::random_device random;
    const uint64_t     value =
        (uint64_t(random()) << 32) ^ random() ^ uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
    return value ? value : 1;
}
bool add(uint64_t a, uint64_t b, uint64_t& result) {
    if (b > UINT64_MAX - a)
        return false;
    result = a + b;
    return true;
}
bool multiply(uint64_t a, uint64_t b, uint64_t& result) {
    if (a && b > UINT64_MAX / a)
        return false;
    result = a * b;
    return true;
}
absl::Status invalid(const char* message) {
    return absl::InvalidArgumentError(message);
}
absl::Status exhausted(const char* message) {
    return absl::ResourceExhaustedError(message);
}
bool rangeFits(uint64_t offset, uint64_t length, uint64_t size) {
    return length && offset <= size && length <= size - offset;
}
}  // namespace
struct RemoteMockState::Data {
    struct Group {
        RemoteMockLayout      layout;
        std::vector<uint8_t>  bytes;
        std::vector<uint32_t> checksums;
    };
    struct Rank {
        uint64_t                  layout_hash = 0;
        std::map<uint32_t, Group> groups;
        uint64_t                  high_sequence = 0;
        std::vector<uint64_t>     seen;
    };
    RemoteMockSession        session;
    uint32_t                 expected_ranks = 0;
    uint32_t                 read_delay_ms  = 0;
    uint32_t                 write_delay_ms = 0;
    uint64_t                 source_bytes   = 0;
    uint64_t                 write_bytes    = 0;
    uint64_t                 metadata_bytes = 0;
    std::map<uint32_t, Rank> ranks;
    std::vector<uint8_t>     writes;
    std::vector<bool>        busy_slots;
    std::mutex               quota_mutex;
    uint64_t                 read_charge  = 0;
    uint64_t                 write_charge = 0;
    uint32_t                 reads        = 0;
};
bool RemoteMockSession::operator==(const RemoteMockSession& rhs) const {
    return server_epoch == rhs.server_epoch && session_id == rhs.session_id && generation == rhs.generation;
}
RemoteMockReadLease::~RemoteMockReadLease() {
    if (release_)
        release_();
}
RemoteMockWriteLease::~RemoteMockWriteLease() {
    if (release_)
        release_();
}
RemoteMockState::RemoteMockState(RemoteMockLimits limits, uint64_t epoch):
    limits_(limits), server_epoch_(epoch ? epoch : newEpoch()) {}
RemoteMockState::~RemoteMockState() = default;
uint64_t RemoteMockState::lastGeneration() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_generation_;
}
uint64_t RemoteMockState::allocatedBytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  data = active_ ? active_ : retiring_.lock();
    return data ? data->source_bytes + data->write_bytes + data->metadata_bytes : 0;
}
absl::Status RemoteMockState::checkReady(const RemoteMockSession& session) const {
    if (!active_ || !(active_->session == session))
        return absl::FailedPreconditionError("inactive or stale session");
    if (active_->ranks.size() != active_->expected_ranks)
        return absl::FailedPreconditionError("not all ranks registered");
    return absl::OkStatus();
}
absl::Status RemoteMockState::ready(const RemoteMockSession& session) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return checkReady(session);
}
absl::Status RemoteMockState::close(const RemoteMockSession& session) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_ || !(active_->session == session))
        return absl::FailedPreconditionError("inactive or stale session");
    retiring_ = active_;
    active_.reset();
    return absl::OkStatus();
}
absl::StatusOr<RemoteMockSession> RemoteMockState::registerRank(const RemoteMockManifest& m) {
    std::lock_guard<std::mutex> lock(mutex_);
    // A stale/foreign caller must never tear down the current session.
    if (active_ && !(active_->session == m.session))
        return absl::FailedPreconditionError("another session is active");
    if (active_ && active_->ranks.size() == active_->expected_ranks)
        return absl::AlreadyExistsError("session already initialized");
    if (!active_ && !retiring_.expired())
        return exhausted("closed session leases have not drained");
    if (m.session.server_epoch != server_epoch_ || !m.session.session_id || !m.session.generation
        || (!active_ && m.session.generation <= last_generation_))
        return invalid("stale or invalid session identity");
    // Every failed rank registration belonging to the initializing session rolls it back.
    const auto fail = [&](absl::Status status) -> absl::StatusOr<RemoteMockSession> {
        last_generation_ = std::max(last_generation_, m.session.generation);
        active_.reset();
        return status;
    };
    const auto& l = limits_;
    if (!l.write_slots || !l.write_slot_bytes || !l.read_staging_bytes || !l.write_staging_bytes || !l.max_rpc_bytes
        || !l.max_blocks_per_rpc || !l.max_read_operations || !l.replay_window || l.write_slots > 65536
        || l.replay_window > (1U << 20) || l.max_ranks > 4096 || l.max_groups_per_rank > 65536
        || l.max_members_per_group > 4096 || l.max_sources_per_group > 65536)
        return fail(invalid("invalid server limits"));
    const auto minimum_charge = remoteMockStagingCharge(1, 1, l);
    if (!minimum_charge.ok() || l.read_staging_bytes < *minimum_charge || l.write_staging_bytes < *minimum_charge)
        return fail(invalid("read and write must each reserve a minimum progress budget"));
    if (!m.expected_ranks || m.expected_ranks > l.max_ranks || m.rank >= m.expected_ranks || !m.layout_hash
        || m.groups.empty() || m.groups.size() > l.max_groups_per_rank || m.read_delay_ms > l.max_delay_ms
        || m.write_delay_ms > l.max_delay_ms)
        return fail(invalid("invalid rank manifest"));
    if (active_
        && (active_->expected_ranks != m.expected_ranks || active_->read_delay_ms != m.read_delay_ms
            || active_->write_delay_ms != m.write_delay_ms || active_->ranks.count(m.rank)))
        return fail(invalid("inconsistent or duplicate rank registration"));
    // Conservative accounting for structs, map nodes, vector backing, and strings.
    // The process runtime/allocator arena itself is not claimed as bounded by this payload budget.
    uint64_t metadata      = active_ ? active_->metadata_bytes : sizeof(Data) + 256 + l.write_slots;
    uint64_t rank_metadata = sizeof(Data::Rank) + 256 + uint64_t(l.replay_window) * sizeof(uint64_t);
    if (!add(metadata, rank_metadata, metadata))
        return fail(invalid("metadata capacity overflow"));
    uint64_t           source_bytes = 0;
    std::set<uint32_t> groups;
    std::set<uint32_t> pools;
    for (const auto& group : m.groups) {
        auto valid = validateRemoteMockLayout(group, l);
        if (!valid.ok())
            return fail(valid);
        if (!groups.insert(group.group).second || !pools.insert(group.pool).second)
            return fail(invalid("duplicate group or pool"));
        const uint64_t group_metadata =
            sizeof(Data::Group) + 256 + 32 + uint64_t(group.member_offsets.size()) * sizeof(uint64_t);
        if (!add(metadata, group_metadata, metadata))
            return fail(invalid("metadata capacity overflow"));
        uint64_t stride_and_checksum, bytes;
        if (!add(group.stride_bytes, sizeof(uint32_t), stride_and_checksum)
            || !multiply(stride_and_checksum, group.source_count, bytes) || !add(source_bytes, bytes, source_bytes))
            return fail(invalid("source capacity overflow"));
    }
    uint64_t total_sources = 0, writes = 0, total = 0;
    if (!add(active_ ? active_->source_bytes : 0, source_bytes, total_sources)
        || !multiply(l.write_slots, l.write_slot_bytes, writes) || !add(total_sources, writes, total)
        || !add(total, l.read_staging_bytes, total) || !add(total, l.write_staging_bytes, total)
        || !add(total, metadata, total) || source_bytes > SIZE_MAX || writes > SIZE_MAX)
        return fail(invalid("manifest capacity overflow"));
    // Output before allocation, including already registered ranks and both reserved network budgets.
    std::clog << "remote_mock capacity session=" << m.session.session_id << " generation=" << m.session.generation
              << " rank=" << m.rank << " source_and_checksums=" << total_sources << " write_pool=" << writes
              << " read_staging=" << l.read_staging_bytes << " write_staging=" << l.write_staging_bytes
              << " metadata_reservation=" << metadata << " total=" << total << " limit=" << l.total_bytes << '\n';
    if (total_sources > l.source_bytes || total > l.total_bytes)
        return fail(exhausted("manifest exceeds memory budget"));
    try {
        auto data = active_;
        if (!data) {
            data                 = std::make_shared<Data>();
            data->session        = m.session;
            data->expected_ranks = m.expected_ranks;
            data->read_delay_ms  = m.read_delay_ms;
            data->write_delay_ms = m.write_delay_ms;
            data->writes.resize(writes, 0);
            data->write_bytes = writes;
            data->busy_slots.resize(l.write_slots, false);
        }
        Data::Rank rank;
        rank.layout_hash = m.layout_hash;
        rank.seen.resize(l.replay_window, 0);
        for (const auto& layout : m.groups) {
            Data::Group group;
            group.layout = layout;
            group.bytes.resize(layout.stride_bytes * layout.source_count);
            group.checksums.resize(layout.source_count);
            for (uint32_t index = 0; index < layout.source_count; ++index) {
                auto* record = group.bytes.data() + index * layout.stride_bytes;
                auto  status = initializeRemoteMockRecord(layout, record, layout.stride_bytes);
                if (!status.ok())
                    return fail(status);
                group.checksums[index] = remoteMockCrc32c(record, layout.storage_bytes);
            }
            rank.groups.emplace(layout.group, std::move(group));
        }
        data->ranks.emplace(m.rank, std::move(rank));
        data->source_bytes   = total_sources;
        data->metadata_bytes = metadata;
        active_              = std::move(data);
        last_generation_     = m.session.generation;
        return m.session;
    } catch (const std::exception& e) {
        return fail(exhausted(e.what()));
    }
}
absl::StatusOr<RemoteMockSessionInfo> RemoteMockState::sessionInfo() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_)
        return absl::NotFoundError("no active session");
    RemoteMockSessionInfo info;
    info.session          = active_->session;
    info.expected_ranks   = active_->expected_ranks;
    info.registered_ranks = active_->ranks.size();
    info.read_delay_ms    = active_->read_delay_ms;
    info.write_delay_ms   = active_->write_delay_ms;
    info.ready            = info.registered_ranks == info.expected_ranks;
    return info;
}
absl::StatusOr<std::vector<RemoteMockSourceDescriptor>> RemoteMockState::describeRank(const RemoteMockSession& session,
                                                                                      uint32_t rank) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_ || !(active_->session == session))
        return absl::FailedPreconditionError("inactive or stale session");
    auto r = active_->ranks.find(rank);
    if (r == active_->ranks.end())
        return absl::NotFoundError("rank has not registered");
    std::vector<RemoteMockSourceDescriptor> result;
    result.reserve(r->second.groups.size());
    for (const auto& entry : r->second.groups) {
        const auto& g = entry.second;
        result.push_back(
            {rank, g.layout.group, g.layout.pool, g.layout.source_count, r->second.layout_hash, g.checksums.front()});
    }
    return result;
}
absl::StatusOr<std::string> RemoteMockState::sourceKey(const RemoteMockSession& session,
                                                       uint32_t                 rank,
                                                       uint32_t                 group,
                                                       uint32_t                 index,
                                                       uint64_t                 offset,
                                                       uint64_t                 length) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        status = checkReady(session);
    if (!status.ok())
        return status;
    auto r = active_->ranks.find(rank);
    if (r == active_->ranks.end())
        return invalid("unknown rank");
    auto g = r->second.groups.find(group);
    if (g == r->second.groups.end())
        return invalid("unknown group");
    const auto& source = g->second;
    if (index >= source.layout.source_count || !rangeFits(offset, length, source.layout.storage_bytes)
        || length > limits_.max_rpc_bytes || length > UINT32_MAX)
        return invalid("invalid source range");
    RemoteMockKey key;
    key.session     = session;
    key.layout_hash = r->second.layout_hash;
    key.rank        = rank;
    key.group       = group;
    key.pool        = source.layout.pool;
    key.index       = index;
    key.offset      = offset;
    key.length      = length;
    key.checksum    = source.checksums[index];
    return encodeRemoteMockKey(key);
}
absl::StatusOr<std::shared_ptr<const RemoteMockSourceLease>> RemoteMockState::resolve(const std::string& encoded,
                                                                                      uint64_t           length) const {
    auto parsed = decodeRemoteMockKey(encoded);
    if (!parsed.ok())
        return parsed.status();
    const auto&                 key = *parsed;
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        status = checkReady(key.session);
    if (!status.ok())
        return status;
    if (key.write || key.length != length || length > limits_.max_rpc_bytes)
        return invalid("invalid read key/length");
    auto r = active_->ranks.find(key.rank);
    if (r == active_->ranks.end() || r->second.layout_hash != key.layout_hash)
        return invalid("unknown rank/layout");
    auto g = r->second.groups.find(key.group);
    if (g == r->second.groups.end())
        return invalid("unknown group");
    const auto& source = g->second;
    if (key.pool != source.layout.pool || key.index >= source.layout.source_count
        || !rangeFits(key.offset, length, source.layout.storage_bytes) || key.checksum != source.checksums[key.index])
        return invalid("invalid source identity/range/checksum");
    auto lease       = std::make_shared<RemoteMockSourceLease>();
    lease->owner_    = active_;
    lease->data_     = source.bytes.data() + key.index * source.layout.stride_bytes + key.offset;
    lease->size_     = length;
    lease->checksum_ = key.checksum;
    return std::shared_ptr<const RemoteMockSourceLease>(std::move(lease));
}
absl::StatusOr<std::shared_ptr<RemoteMockReadLease>> RemoteMockState::reserveRead(const RemoteMockSession& session,
                                                                                  uint64_t                 bytes,
                                                                                  uint32_t                 blocks,
                                                                                  uint64_t additional_metadata_bytes) {
    auto charge = remoteMockStagingCharge(bytes, blocks, limits_, additional_metadata_bytes);
    if (!charge.ok())
        return charge.status();
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        status = checkReady(session);
    if (!status.ok())
        return status;
    auto                        data = active_;
    std::lock_guard<std::mutex> quota(data->quota_mutex);
    if (data->reads >= limits_.max_read_operations || *charge > limits_.read_staging_bytes - data->read_charge)
        return exhausted("read progress quota exhausted");
    auto lease      = std::shared_ptr<RemoteMockReadLease>(new RemoteMockReadLease());
    lease->release_ = [data, charge = *charge] {
        std::lock_guard<std::mutex> guard(data->quota_mutex);
        --data->reads;
        data->read_charge -= charge;
    };
    ++data->reads;
    data->read_charge += *charge;
    return lease;
}
absl::StatusOr<std::shared_ptr<RemoteMockWriteLease>> RemoteMockState::reserveWrite(
    const RemoteMockWriteOperation& op, uint64_t bytes, uint32_t blocks, uint64_t additional_metadata_bytes) {
    auto charge = remoteMockStagingCharge(bytes, blocks, limits_, additional_metadata_bytes);
    if (!charge.ok())
        return charge.status();
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        status = checkReady(op.session);
    if (!status.ok())
        return status;
    auto data = active_;
    auto r    = data->ranks.find(op.rank);
    if (r == data->ranks.end() || r->second.layout_hash != op.layout_hash || !op.sequence
        || op.slot >= limits_.write_slots)
        return invalid("invalid write operation");
    std::lock_guard<std::mutex> quota(data->quota_mutex);
    auto&                       rank = r->second;
    if ((rank.high_sequence >= limits_.replay_window && op.sequence <= rank.high_sequence - limits_.replay_window)
        || rank.seen[op.sequence % limits_.replay_window] == op.sequence)
        return absl::AlreadyExistsError("stale or replayed write operation");
    if (bytes > limits_.write_slot_bytes || data->busy_slots[op.slot]
        || *charge > limits_.write_staging_bytes - data->write_charge)
        return exhausted("write slot or staging quota exhausted");
    auto lease      = std::shared_ptr<RemoteMockWriteLease>(new RemoteMockWriteLease());
    lease->data_    = data->writes.data() + op.slot * limits_.write_slot_bytes;
    lease->size_    = bytes;
    lease->release_ = [data, slot = op.slot, charge = *charge] {
        std::lock_guard<std::mutex> guard(data->quota_mutex);
        data->busy_slots[slot] = false;
        data->write_charge -= charge;
    };
    data->busy_slots[op.slot] = true;
    data->write_charge += *charge;
    rank.high_sequence                             = std::max(rank.high_sequence, op.sequence);
    rank.seen[op.sequence % limits_.replay_window] = op.sequence;
    return lease;
}
absl::Status RemoteMockState::validateWriteKey(const RemoteMockWriteOperation& op,
                                               const std::string&              encoded,
                                               uint64_t                        length) const {
    auto parsed = decodeRemoteMockKey(encoded);
    if (!parsed.ok())
        return parsed.status();
    const auto&                 key = *parsed;
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        status = checkReady(op.session);
    if (!status.ok())
        return status;
    if (!key.write || !(key.session == op.session) || key.layout_hash != op.layout_hash || key.rank != op.rank
        || key.index != op.slot || key.operation != op.sequence || key.length != length
        || length > limits_.max_rpc_bytes)
        return invalid("write key does not match operation");
    auto r = active_->ranks.find(key.rank);
    if (r == active_->ranks.end() || r->second.layout_hash != key.layout_hash)
        return invalid("unknown rank/layout");
    auto g = r->second.groups.find(key.group);
    if (g == r->second.groups.end() || key.pool != g->second.layout.pool
        || !rangeFits(key.offset, length, g->second.layout.storage_bytes))
        return invalid("invalid write group/range");
    return absl::OkStatus();
}
}  // namespace rtp_llm
