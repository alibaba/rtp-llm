#include "rtp_llm/cpp/disaggregate/cache_store/remote_mock/RemoteMockFormat.h"
#include "rtp_llm/cpp/utils/Crc32c.h"
#include <array>
#include <charconv>
#include <cstring>
#include <locale>
#include <sstream>

namespace rtp_llm {
namespace {
absl::Status invalid(const char* message) {
    return absl::InvalidArgumentError(message);
}
}  // namespace
uint32_t remoteMockCrc32c(const uint8_t* bytes, size_t length) {
    return cpuCrc32c(bytes, length);
}
std::string encodeRemoteMockKey(const RemoteMockKey& k) {
    std::ostringstream s;
    s.imbue(std::locale::classic());
    s << "rm1:" << (k.write ? 'w' : 'r') << ':' << k.session.server_epoch << ':' << k.session.session_id << ':'
      << k.session.generation << ':' << k.layout_hash << ':' << k.rank << ':' << k.group << ':' << k.pool << ':'
      << k.index << ':' << k.operation << ':' << k.record_index << ':' << k.offset << ':' << k.length << ':'
      << k.checksum;
    return s.str();
}
absl::StatusOr<RemoteMockKey> decodeRemoteMockKey(const std::string& key) {
    if (key.size() > kRemoteMockMaxKeyBytes || key.size() < 6
        || (key.compare(0, 6, "rm1:r:") != 0 && key.compare(0, 6, "rm1:w:") != 0)) {
        return invalid("invalid mock key prefix or size");
    }
    std::array<uint64_t, 13> values{};
    size_t                   pos = 6;
    for (size_t i = 0; i < values.size(); ++i) {
        const size_t end = key.find(':', pos);
        if ((i + 1 == values.size()) != (end == std::string::npos))
            return invalid("invalid mock key field count");
        const size_t length = (end == std::string::npos ? key.size() : end) - pos;
        if (!length || (length > 1 && key[pos] == '0') || key[pos] < '0' || key[pos] > '9')
            return invalid("noncanonical mock key number");
        auto parsed = std::from_chars(key.data() + pos, key.data() + pos + length, values[i]);
        if (parsed.ec != std::errc() || parsed.ptr != key.data() + pos + length)
            return invalid("invalid mock key number");
        pos += length + 1;
    }
    for (size_t i : {4, 5, 6, 7, 12}) {
        if (values[i] > UINT32_MAX)
            return invalid("mock key field exceeds uint32");
    }
    RemoteMockKey k;
    k.write        = key[4] == 'w';
    k.session      = {values[0], values[1], values[2]};
    k.layout_hash  = values[3];
    k.rank         = values[4];
    k.group        = values[5];
    k.pool         = values[6];
    k.index        = values[7];
    k.operation    = values[8];
    k.record_index = values[9];
    k.offset       = values[10];
    k.length       = values[11];
    k.checksum     = values[12];
    if (!k.session.server_epoch || !k.session.session_id || !k.session.generation || !k.layout_hash || !k.length
        || k.length > UINT32_MAX || (k.write ? !k.operation : (k.operation != 0 || k.record_index != 0)))
        return invalid("invalid mock key identity or length");
    return k;
}
absl::Status validateRemoteMockLayout(const RemoteMockLayout& l, const RemoteMockLimits& limits) {
    if (!l.payload_bytes || !l.source_count || l.source_count > limits.max_sources_per_group || !l.alignment
        || (l.alignment & (l.alignment - 1)) || l.stride_bytes % l.alignment || l.stride_bytes < l.storage_bytes
        || l.stride_bytes > SIZE_MAX || !l.tokens_per_block || !l.cp_world_size || l.cp_rank >= l.cp_world_size
        || l.dtype.empty() || l.dtype.size() > 32 || l.crc_format_version != 1
        || l.member_offsets.size() > limits.max_members_per_group)
        return invalid("invalid layout geometry");
    uint64_t expected = l.payload_bytes;
    if (l.crc_enabled) {
        if (expected > UINT64_MAX - 19)
            return invalid("record size overflow");
        expected = (expected + 19) & ~uint64_t(15);
    }
    if (l.storage_bytes != expected || (l.crc_enabled ? l.crc_footer_offset != expected - 4 : l.crc_footer_offset != 0))
        return invalid("incompatible CRC storage/footer layout");
    for (size_t i = 0; i < l.member_offsets.size(); ++i) {
        if (l.member_offsets[i] >= l.payload_bytes || (i && l.member_offsets[i] <= l.member_offsets[i - 1]))
            return invalid("invalid member offsets");
    }
    return absl::OkStatus();
}
absl::Status initializeRemoteMockRecord(const RemoteMockLayout& l, uint8_t* bytes, size_t capacity, uint8_t fill) {
    auto valid = validateRemoteMockLayout(l, RemoteMockLimits{});
    if (!valid.ok())
        return valid;
    if (!bytes || capacity < l.stride_bytes)
        return invalid("short record allocation");
    std::memset(bytes, 0, l.stride_bytes);
    std::memset(bytes, fill, l.payload_bytes);
    if (l.crc_enabled) {
        const uint32_t crc = cpuCrc32c(bytes, l.payload_bytes);
        // Existing CrcBlockCopy footer is a native uint32 (supported platforms are little endian).
        std::memcpy(bytes + l.crc_footer_offset, &crc, sizeof(crc));
    }
    return absl::OkStatus();
}
absl::Status validateRemoteMockRecord(const RemoteMockLayout& l, const uint8_t* bytes, size_t length) {
    auto valid = validateRemoteMockLayout(l, RemoteMockLimits{});
    if (!valid.ok())
        return valid;
    if (!bytes || length != l.storage_bytes)
        return invalid("incomplete record");
    if (l.crc_enabled) {
        uint32_t expected;
        std::memcpy(&expected, bytes + l.crc_footer_offset, sizeof(expected));
        if (expected != cpuCrc32c(bytes, l.payload_bytes))
            return absl::DataLossError("record CRC32c mismatch");
    }
    return absl::OkStatus();
}
absl::StatusOr<uint64_t> remoteMockStagingCharge(uint64_t                bytes,
                                                 uint32_t                blocks,
                                                 const RemoteMockLimits& limits,
                                                 uint64_t                additional_metadata_bytes) {
    if (!bytes || bytes > limits.max_rpc_bytes || bytes > UINT32_MAX || !blocks || blocks > limits.max_blocks_per_rpc)
        return invalid("RPC payload or block count exceeds limits");
    const uint64_t envelope = uint64_t(blocks) * 512 + 4096;
    if (bytes + envelope > INT32_MAX)
        return invalid("complete RPC envelope exceeds protobuf limit");
    // Two payload copies, per-block key/protobuf allowance, and message envelope.
    const uint64_t base = 2 * bytes + envelope;
    if (additional_metadata_bytes > UINT64_MAX - base)
        return invalid("RPC staging metadata charge overflows");
    return base + additional_metadata_bytes;
}
}  // namespace rtp_llm
