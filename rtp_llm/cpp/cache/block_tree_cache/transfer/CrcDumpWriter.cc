#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcDumpWriter.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

namespace rtp_llm {
namespace {
namespace fs = std::filesystem;

std::atomic<uint64_t> dump_sequence{0};

uint32_t diagnosticCrc32c(const uint8_t* data, size_t bytes) {
    static const auto table = [] {
        std::array<uint32_t, 256> values{};
        for (uint32_t i = 0; i < values.size(); ++i) {
            uint32_t value = i;
            for (int bit = 0; bit < 8; ++bit)
                value = (value >> 1) ^ ((value & 1) ? 0x82f63b78U : 0);
            values[i] = value;
        }
        return values;
    }();
    uint32_t crc = ~uint32_t(0);
    for (size_t i = 0; i < bytes; ++i)
        crc = table[(crc ^ data[i]) & 255] ^ (crc >> 8);
    return ~crc;
}

std::string jsonString(const std::string& value) {
    std::string out = "\"";
    for (const unsigned char c : value) {
        if (c == '\"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (c < 32) {
            const char hex[] = "0123456789abcdef";
            out += "\\u00";
            out += hex[c >> 4];
            out += hex[c & 15];
        } else {
            out += c;
        }
    }
    return out + "\"";
}

void writeBytes(const fs::path& path, const void* bytes, size_t size) {
    std::ofstream file;
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file.open(path, std::ios::binary);
    file.write(static_cast<const char*>(bytes), static_cast<std::streamsize>(size));
    file.close();
}

void removeDumpDirectory(const fs::path& path) {
    // Evidence directories are flat. Avoid libtorch's interposed remove_all,
    // and never traverse a symlink or a directory we did not create.
    if (!fs::is_directory(fs::symlink_status(path)))
        throw std::runtime_error("CRC dump cleanup requires a directory");
    for (const auto& entry : fs::directory_iterator(path)) {
        const auto status = entry.symlink_status();
        if (!fs::is_regular_file(status) && !fs::is_symlink(status))
            throw std::runtime_error("unexpected entry in CRC dump directory");
        if (::unlink(entry.path().c_str()) != 0)
            throw std::system_error(errno, std::generic_category(), "CRC dump unlink failed");
    }
    if (::rmdir(path.c_str()) != 0)
        throw std::system_error(errno, std::generic_category(), "CRC dump rmdir failed");
}
}  // namespace

CrcDumpWriter::CrcDumpWriter(int64_t world_rank, std::string root, size_t backup_count):
    world_rank_(world_rank),
    rank_dir_((fs::path(root) / ("rank_" + std::to_string(world_rank))).string()),
    backup_count_(std::max<size_t>(backup_count, 1)) {}

std::shared_ptr<CrcDumpWriter> CrcDumpWriter::forRank(int64_t world_rank) {
    static std::mutex                                        mutex;
    static std::map<int64_t, std::shared_ptr<CrcDumpWriter>> writers;
    std::lock_guard<std::mutex>                              lock(mutex);
    auto&                                                    writer = writers[world_rank];
    if (!writer)
        writer = std::make_shared<CrcDumpWriter>(world_rank);
    return writer;
}

void CrcDumpWriter::rotate() {
    struct Entry {
        fs::path           path;
        fs::file_time_type time;
    };
    std::vector<Entry> entries;
    for (const auto& entry : fs::directory_iterator(rank_dir_)) {
        if (!fs::is_directory(entry.symlink_status()) || entry.path().filename().string().find("crc_") != 0)
            continue;
        const auto manifest = entry.path() / "manifest.json";
        if (fs::is_regular_file(fs::symlink_status(manifest)))
            entries.push_back({entry.path(), fs::last_write_time(manifest)});
    }
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        return a.time != b.time ? a.time < b.time : a.path < b.path;
    });
    for (size_t i = backup_count_; i < entries.size(); ++i)
        removeDumpDirectory(entries[i - backup_count_].path);
}

std::string CrcDumpWriter::write(const CrcCopyItem&        item,
                                 const CrcCopyFailure&     verdict,
                                 const CrcSnapshotCapture& capture_snapshot) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    fs::path                    directory;
    std::string                 dump_path;
    Metadata                    metadata;
    try {
        fs::create_directories(rank_dir_);
        if (item.dump_metadata)
            metadata = item.dump_metadata();
        const auto   failure                 = capture_snapshot();
        const size_t encoded                 = CrcBlockCopyBatch::encodedBytes(item.payload_bytes);
        const size_t footer_offset           = encoded - sizeof(uint32_t);
        const bool   cpu_captured            = failure.cpu_snapshot.size() == encoded;
        const bool   gpu_captured            = failure.staging_snapshot.size() == encoded;
        metadata["snapshot_format_version"]  = "4";
        metadata["rank"]                     = std::to_string(world_rank_);
        metadata["copy_item_index"]          = std::to_string(failure.item_index);
        metadata["payload_bytes"]            = std::to_string(item.payload_bytes);
        metadata["storage_bytes"]            = std::to_string(encoded);
        metadata["footer_offset"]            = std::to_string(footer_offset);
        metadata["captured_bytes"]           = std::to_string(cpu_captured ? encoded : 0);
        metadata["truncated"]                = "false";
        metadata["crc_scope"]                = "payload only; excludes alignment padding and footer";
        metadata["checked_footer_available"] = failure.checked_footer_available ? "true" : "false";
        metadata["expected_crc32c"] =
            failure.checked_footer_available ? std::to_string(failure.expected) : "unavailable";
        metadata["actual_crc32c"] =
            failure.status == CrcCopyStatus::CRC_COMPUTE_ERROR ? "unavailable" : std::to_string(failure.actual);
        metadata["gpu_crc_status"] = std::to_string(failure.nvcomp_status);
        metadata["failure_stage"]  = failure.status == CrcCopyStatus::CRC_COMPUTE_ERROR ? "crc_compute" : "source_crc";
        metadata["cpu_address"]    = std::to_string(reinterpret_cast<uintptr_t>(item.host));
        metadata["cpu_capacity_bytes"] = std::to_string(item.capacity_bytes);
        metadata["cpu_captured"]       = cpu_captured ? "true" : "false";
        metadata["staging_captured"]   = gpu_captured ? "true" : "false";
        metadata["capture_error"]      = failure.capture_error;
        metadata["temporal_limit"] =
            "CPU snapshot is copied after failure, not atomically with transfer or GPU validation. "
            "Concurrent mutation, stored-footer corruption and transfer faults cannot be uniquely "
            "distinguished. Matching CRC alone does not establish matching bytes.";
        metadata["tile_count"] = std::to_string(item.tiles.size());
        for (size_t i = 0; i < item.tiles.size(); ++i) {
            const auto key            = "tile_" + std::to_string(i) + "_";
            metadata[key + "address"] = std::to_string(reinterpret_cast<uintptr_t>(item.tiles[i].device));
            metadata[key + "offset"]  = std::to_string(item.tiles[i].offset);
            metadata[key + "bytes"]   = std::to_string(item.tiles[i].bytes);
            metadata[key + "is_cuda"] = "true";
        }
        if (cpu_captured) {
            uint32_t footer;
            std::memcpy(&footer, failure.cpu_snapshot.data() + footer_offset, sizeof(footer));
            metadata["cpu_footer_crc32c"] = std::to_string(footer);
            metadata["cpu_crc32c"] = std::to_string(diagnosticCrc32c(failure.cpu_snapshot.data(), item.payload_bytes));
            if (failure.checked_footer_available)
                metadata["cpu_footer_differs_from_gpu_checked_footer"] = footer != failure.expected ? "true" : "false";
        }
        if (gpu_captured) {
            uint32_t footer;
            std::memcpy(&footer, failure.staging_snapshot.data() + footer_offset, sizeof(footer));
            metadata["gpu_footer_crc32c"] = std::to_string(footer);
            metadata["gpu_snapshot_cpu_crc32c"] =
                std::to_string(diagnosticCrc32c(failure.staging_snapshot.data(), item.payload_bytes));
        }
        if (cpu_captured && gpu_captured) {
            size_t differing = 0;
            size_t first     = item.payload_bytes;
            for (size_t i = 0; i < item.payload_bytes; ++i) {
                if (failure.cpu_snapshot[i] != failure.staging_snapshot[i]) {
                    first = std::min(first, i);
                    ++differing;
                }
            }
            metadata["cpu_gpu_payload_differing_bytes"]  = std::to_string(differing);
            metadata["cpu_gpu_payload_first_difference"] = first == item.payload_bytes ? "none" : std::to_string(first);
            metadata["diagnosis"] =
                differing ? "CPU-after-failure differs from checked GPU staging; see temporal_limit" :
                            "CPU-after-failure and checked GPU payload bytes agree; inspect footers and CRC status";
        }
        std::ostringstream json;
        json << "{\n";
        bool first = true;
        for (const auto& [key, value] : metadata) {
            if (!first)
                json << ",\n";
            first = false;
            json << "  " << jsonString(key) << ": " << jsonString(value);
        }
        json << "\n}\n";
        const auto manifest = json.str();
        // Preserve write order even when consecutive manifests share a filesystem timestamp.
        std::ostringstream name;
        name << "crc_" << std::setw(20) << std::setfill('0') << dump_sequence.fetch_add(1, std::memory_order_relaxed)
             << "_XXXXXX";
        std::string pattern = (fs::path(rank_dir_) / name.str()).string();
        if (!mkdtemp(pattern.data()))
            throw std::runtime_error("cannot create CRC dump directory");
        directory = pattern;
        if (cpu_captured) {
            writeBytes(directory / "cpu.bin", failure.cpu_snapshot.data(), encoded);
            writeBytes(directory / "cpu_footer.bin", failure.cpu_snapshot.data() + footer_offset, sizeof(uint32_t));
        }
        if (gpu_captured) {
            writeBytes(directory / "gpu_staging.bin", failure.staging_snapshot.data(), item.payload_bytes);
            writeBytes(directory / "gpu_staging_footer.bin",
                       failure.staging_snapshot.data() + footer_offset,
                       sizeof(uint32_t));
        }
        if (failure.checked_footer_available)
            writeBytes(directory / "checked_footer.bin", &failure.expected, sizeof(failure.expected));
        // Manifest is the completion marker; never leave a partial diagnostic set.
        writeBytes(directory / "manifest.json", manifest.data(), manifest.size());
        dump_path = directory.string();
        rotate();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "CRC dump write failed: %s\n", error.what());
    } catch (...) {
        std::fprintf(stderr, "CRC dump write failed\n");
    }
    if (dump_path.empty() && !directory.empty()) {
        try {
            removeDumpDirectory(directory);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "CRC dump cleanup failed: %s\n", error.what());
        } catch (...) {
            std::fprintf(stderr, "CRC dump cleanup failed\n");
        }
    }
    const auto descriptor = metadata.find("copy_item");
    RTP_LLM_LOG_WARNING("CRC backing failure: rank=%ld item=%zu %s status=%d dump=%s payload=%zu expected=%08x "
                        "actual=%08x nvcomp_status=%d",
                        world_rank_,
                        verdict.item_index,
                        descriptor == metadata.end() ? "" : descriptor->second.c_str(),
                        static_cast<int>(verdict.status),
                        dump_path.c_str(),
                        item.payload_bytes,
                        verdict.expected,
                        verdict.actual,
                        verdict.nvcomp_status);
    return dump_path;
}
}  // namespace rtp_llm
