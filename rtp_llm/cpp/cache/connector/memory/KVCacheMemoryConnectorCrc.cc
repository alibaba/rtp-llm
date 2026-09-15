#include "rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unistd.h>

#include "autil/legacy/json.h"
#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

namespace {
constexpr uint64_t kCrcDumpQuotaBytes = 2ULL * 1024 * 1024 * 1024;

bool fullCrcDumpFits(size_t bytes) {
    // At most three full blocks (source, candidate output, staging), plus metadata.
    // Check before either the GPU capture or the CPU allocation; never truncate.
    return bytes <= (kCrcDumpQuotaBytes - 1024 * 1024) / 3 - 80;
}

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

void describeFooter(std::map<std::string, std::string>& manifest,
                    const std::string&                  prefix,
                    const CrcBlockFooter&               footer) {
    manifest[prefix + "crc32c"] = std::to_string(footer.crc32c);
}
}  // namespace

void KVCacheMemoryConnector::initCrcCopy() {
    const auto          slots = layerRegionSlots();
    std::vector<size_t> sizes;
    if (usePrefixTreeMemoryCache())
        sizes = {prefixKindBlockSize(CacheBlockKind::COMPRESSED_KV, slots),
                 prefixKindBlockSize(CacheBlockKind::STATE_SWA_KV, slots)};
    else if (isDualPool()) {
        sizes.push_back(complete_block_size_);
        if (incomplete_pool_)
            sizes.push_back(incomplete_block_size_);
    } else
        sizes.push_back(memoryCacheBlockSizeBytes());
    crc_layout_ = 14695981039346656037ULL;
    auto mix    = [this](uint64_t value) {
        for (int byte = 0; byte < 8; ++byte) {
            crc_layout_ ^= (value >> (byte * 8)) & 255;
            crc_layout_ *= 1099511628211ULL;
        }
    };
    mix(cache_config_.seq_size_per_block);
    mix(cache_config_.kernel_seq_size_per_block);
    for (const auto& slot : slots) {
        mix(slot.layer_id);
        mix(static_cast<uint64_t>(slot.region_name));
        mix(slot.group_id);
        mix(slot.stride_bytes);
    }
    crc_copy_slots_.reserve(kCopyThreadCount);
    for (size_t i = 0; i < kCopyThreadCount; ++i) {
        CrcCopySlot slot;
        slot.copy = std::make_unique<CrcBlockCopy>(sizes, slots.size() * 2);
        crc_copy_slots_.push_back(std::move(slot));
    }
    RTP_LLM_LOG_INFO("memory cache block CRC enabled: workspaces=%zu max_payload=%zu layout=%lu dump_path=%s",
                     kCopyThreadCount,
                     *std::max_element(sizes.begin(), sizes.end()),
                     crc_layout_,
                     crc_dump_path_.c_str());
}

bool KVCacheMemoryConnector::reserveCrcDump() {
    std::lock_guard<std::mutex> lock(crc_mutex_);
    const int64_t               now = currentTimeUs();
    if (now - crc_dump_window_us_ >= 60000000) {
        crc_dump_window_us_ = now;
        crc_dump_count_     = 0;
    }
    if (crc_dump_count_ >= 2)
        return false;
    ++crc_dump_count_;
    return true;
}

MemoryOperationResponsePB::ErrorCode
KVCacheMemoryConnector::copyCacheWithCrc(const MemoryOperationRequestPB&     request,
                                         const std::vector<LayerRegionSlot>& slots) {
    using Response       = MemoryOperationResponsePB;
    const bool to_device = request.copy_direction() == MemoryOperationRequestPB::H2D;
    auto       reject    = [&](const char* message, Response::ErrorCode error = Response::INVALID_REQUEST) {
        RTP_LLM_LOG_ERROR("memory cache copy rejected: rank=%ld direction=%d items=%d error=%s detail=%s",
                          parallelism_config_.world_rank,
                          static_cast<int>(request.copy_direction()),
                          request.copy_items_size(),
                          Response::ErrorCode_Name(error).c_str(),
                          message);
        return error;
    };

    if (!request.copy_items_size()
        || (request.copy_direction() != MemoryOperationRequestPB::H2D
            && request.copy_direction() != MemoryOperationRequestPB::D2H)) {
        return reject("invalid CRC copy request");
    }
    size_t slot_index = 0;
    {
        std::unique_lock<std::mutex> lock(crc_mutex_);
        crc_cv_.wait(lock, [&] {
            return stop_.load() || crc_copy_slots_.empty()
                   || std::any_of(
                       crc_copy_slots_.begin(), crc_copy_slots_.end(), [](const auto& slot) { return !slot.busy; });
        });
        if (stop_.load()) {
            return reject("CRC connector stopping", Response::COPY_FAILED);
        }
        if (crc_copy_slots_.empty())
            return reject("CRC workspace not initialized", Response::COPY_FAILED);
        while (crc_copy_slots_[slot_index].busy)
            ++slot_index;
        crc_copy_slots_[slot_index].busy = true;
    }
    auto release = [this, slot_index](CrcBlockCopy*) {
        {
            std::lock_guard<std::mutex> lock(crc_mutex_);
            crc_copy_slots_[slot_index].busy = false;
        }
        crc_cv_.notify_one();
    };
    std::unique_ptr<CrcBlockCopy, decltype(release)> workspace(crc_copy_slots_[slot_index].copy.get(), release);

    for (int item_index = 0; item_index < request.copy_items_size(); ++item_index) {
        const auto& item = request.copy_items(item_index);
        if (!validateCopyItemBacking(item) || item.gpu_blocks_size() != slots.size()) {
            return reject("invalid CRC backing");
        }
        const bool prefix = item.cache_block_kind() == MemoryOperationRequestPB::COMPRESSED_KV
                            || item.cache_block_kind() == MemoryOperationRequestPB::STATE_SWA_KV;
        const auto kind = prefix ? (item.cache_block_kind() == MemoryOperationRequestPB::COMPRESSED_KV ?
                                        CacheBlockKind::COMPRESSED_KV :
                                        CacheBlockKind::STATE_SWA_KV) :
                                   blockKindFromComplete(item.is_complete());
        std::vector<CrcBlockCopyTile> tiles;
        tiles.reserve(slots.size() * 2);
        size_t bytes = 0;
        for (size_t index = 0; index < slots.size(); ++index) {
            const auto& slot = slots[index];
            if (prefix ? kindForSlot(slot) != kind : (!item.is_complete() && !isFullOnlySlot(slot)))
                continue;
            const auto block = static_cast<BlockIdxType>(item.gpu_blocks(index));
            if (block > 0 && !isNullBlockIdx(block)) {
                size_t within = 0;
                for (const auto& buffer : allocator_->convertIndexToBuffer(slot.layer_id, slot.region_name, block)) {
                    if (!buffer.addr || !buffer.size_bytes || within + buffer.size_bytes > slot.stride_bytes) {
                        return reject("invalid CRC layer buffer");
                    }
                    tiles.push_back({buffer.addr, bytes + within, buffer.size_bytes, buffer.is_cuda});
                    within += buffer.size_bytes;
                }
            }
            bytes += slot.stride_bytes;
        }
        auto                                        memory_pool = memoryPoolFor(kind);
        auto                                        disk_pool   = diskPoolFor(kind);
        std::unique_ptr<void, decltype(&std::free)> disk_target(nullptr, &std::free), disk_source(nullptr, &std::free);
        auto                                        backing_error = Response::COPY_FAILED;
        auto                                        backing       = [&](bool source) -> void* {
            const bool disk =
                (source ? item.src_backing_type() : item.backing_type()) == MemoryOperationRequestPB::DISK;
            if (disk) {
                if (!disk_pool || disk_pool->blockSizeBytes() != CrcBlockCopy::storageBytes(bytes))
                    return nullptr;
                auto& storage = source ? disk_source : disk_target;
                void* pointer = nullptr;
                if (posix_memalign(&pointer, 4096, disk_pool->slotStrideBytes()) != 0)
                    return nullptr;
                storage.reset(pointer);
                std::memset(pointer, 0, disk_pool->slotStrideBytes());
                const int disk_slot = source ? item.src_disk_slot() : item.disk_slot();
                if (!disk_pool->validSlot(disk_slot))
                    return nullptr;
                if ((source || to_device) && !disk_pool->read(disk_slot, pointer, disk_pool->slotStrideBytes())) {
                    backing_error = Response::IO_FAILED;
                    return nullptr;
                }
                return pointer;
            }
            if (!memory_pool)
                return nullptr;
            const auto buffers = memory_pool->convertIndexToBuffer(0, source ? item.src_mem_block() : item.mem_block());
            if (buffers.size() != 1 || !buffers[0].addr || buffers[0].is_cuda
                || buffers[0].size_bytes < CrcBlockCopy::storageBytes(bytes))
                return nullptr;
            return buffers[0].addr;
        };
        void*      host = backing(false);
        const bool has_source =
            !to_device
            && (item.src_mem_block_presence_case() == MemoryOperationRequestPB::CopyItem::kSrcMemBlock
                || item.src_disk_slot_presence_case() == MemoryOperationRequestPB::CopyItem::kSrcDiskSlot);
        void* inherited = has_source ? backing(true) : nullptr;
        if (!host || (has_source && !inherited)) {
            return reject("CRC backing read/allocation failed", backing_error);
        }
        const bool host_pinned = item.backing_type() != MemoryOperationRequestPB::DISK && memory_pool
                                 && memory_pool->where() == MemoryType::MEMORY_CPU_PINNED;
        const bool source_pinned = item.src_backing_type() != MemoryOperationRequestPB::DISK && memory_pool
                                   && memory_pool->where() == MemoryType::MEMORY_CPU_PINNED;
        bool                        dump_reserved   = false;
        const std::function<bool()> capture_failure = [&] {
            dump_reserved = fullCrcDumpFits(bytes) && reserveCrcDump();
            return dump_reserved;
        };
        CrcBlockCopyResult result;
        try {
            if (to_device) {
                result = workspace->loadAndValidate(host, bytes, host_pinned, capture_failure);
                if (result.success)
                    workspace->scatter(bytes, tiles);
            } else {
                if (inherited)
                    result = workspace->loadAndValidate(inherited, bytes, source_pinned, capture_failure);
                if (!inherited || result.success) {
                    workspace->gather(bytes, tiles, inherited != nullptr);
                    result = workspace->store(host, bytes, host_pinned, capture_failure);
                }
            }
        } catch (const std::invalid_argument& error) {
            return reject(error.what());
        }
        if (!result.success) {
            const auto error = result.failure_stage == CrcBlockCopyResult::FailureStage::SOURCE_CRC ?
                                   Response::CRC_MISMATCH :
                                   Response::CRC_COMPUTE_FAILED;
            RTP_LLM_LOG_ERROR("memory cache CRC rejected: item=%d mem_block=%d kind=%s rank=%ld "
                              "expected=%u actual=%u gpu_status=%u",
                              item_index,
                              item.mem_block(),
                              cacheBlockKindName(kind),
                              parallelism_config_.world_rank,
                              result.expected_crc,
                              result.actual_crc,
                              result.gpu_crc_status);
            if (dump_reserved) {
                try {
                    dumpCrcFailure(request, item_index, kind, std::move(result), host, bytes, inherited);
                } catch (const std::exception& error) {
                    RTP_LLM_LOG_ERROR(
                        "CRC dump failed: rank=%ld error=%s", parallelism_config_.world_rank, error.what());
                }
            } else {
                RTP_LLM_LOG_WARNING("CRC dump dropped: rank=%ld bytes=%zu reason=%s",
                                    parallelism_config_.world_rank,
                                    bytes,
                                    fullCrcDumpFits(bytes) ? "rate_limit" : "full_block_exceeds_quota");
            }
            return error;
        }
        if (!to_device && item.backing_type() == MemoryOperationRequestPB::DISK
            && !disk_pool->write(item.disk_slot(), host, disk_pool->slotStrideBytes())) {
            return reject("CRC disk write failed", Response::IO_FAILED);
        }
    }
    return Response::NONE;
}

void KVCacheMemoryConnector::dumpCrcFailure(const MemoryOperationRequestPB& request,
                                            int                             item_index,
                                            CacheBlockKind                  kind,
                                            CrcBlockCopyResult              result,
                                            const void*                     host,
                                            size_t                          bytes,
                                            const void*                     inherited) {
    // Serialize rotation and file writes for this rank's dump directory.
    std::lock_guard<std::mutex> lock(crc_dump_mutex_);
    if (!fullCrcDumpFits(bytes))
        throw std::length_error("full CRC dump exceeds quota");
    const auto&  item      = request.copy_items(item_index);
    const bool   to_device = request.copy_direction() == MemoryOperationRequestPB::H2D;
    const void*  source    = inherited ? inherited : host;
    const size_t captured  = CrcBlockCopy::storageBytes(bytes);
    // Finish the dump before replying to rank 0, while CopyPlan owns the source
    // reference and copyCacheWithCrc owns any disk buffers and the GPU workspace.
    // References prevent recycling, not unsynchronized mutation. Only the owned
    // snapshot is stable; it is not an atomic observation of the live block.
    std::vector<uint8_t> cpu(captured), cpu_output;
    std::memcpy(cpu.data(), source, cpu.size());
    if (inherited && result.output_written) {
        cpu_output.resize(captured);
        std::memcpy(cpu_output.data(), host, cpu_output.size());
    }
    CrcBlockFooter footer;
    std::memcpy(&footer, cpu.data() + CrcBlockCopy::footerOffset(bytes), sizeof(footer));
    using Stage                                   = CrcBlockCopyResult::FailureStage;
    const bool                         source_gpu = !result.output_written && (to_device || inherited);
    const char*                        stage      = result.failure_stage == Stage::SOURCE_CRC  ? "source_crc" :
                                                    result.failure_stage == Stage::CRC_COMPUTE ? "crc_compute" :
                                                                                                 "input";
    std::map<std::string, std::string> manifest{
        {"snapshot_format_version", "3"},
        {"copy_item_index", std::to_string(item_index)},
        {"mem_block", std::to_string(inherited ? item.src_mem_block() : item.mem_block())},
        {"rank", std::to_string(parallelism_config_.world_rank)},
        {"kind", std::to_string(static_cast<uint32_t>(kind))},
        {"layout", std::to_string(crc_layout_)},
        {"payload_bytes", std::to_string(bytes)},
        {"storage_bytes", std::to_string(captured)},
        {"footer_offset", std::to_string(CrcBlockCopy::footerOffset(bytes))},
        {"captured_bytes", std::to_string(captured)},
        {"truncated", "false"},
        {"crc_scope", "payload only; excludes alignment padding and footer"},
        {"expected_crc32c", result.checked_footer ? std::to_string(result.checked_footer->crc32c) : "unavailable"},
        {"actual_crc32c", result.gpu_crc_observed ? std::to_string(result.actual_crc) : "unavailable"},
        {"gpu_crc_status",
         result.failure_stage != Stage::INPUT ? std::to_string(result.gpu_crc_status) : "unavailable"},
        {"gpu_crc_role",
         result.failure_stage == Stage::INPUT ? "unavailable" :
         source_gpu                           ? "source" :
                                                "output"},
        {"output_written", result.output_written ? "true" : "false"},
        {"direction", to_device ? "H2D" : "D2H"},
        {"failure_stage", stage},
        {"failure_source",
         result.output_written ? "output" :
         inherited             ? "merge_source" :
         to_device             ? "restore" :
                                 "input"},
        {"cpu_snapshot_role",
         inherited             ? "inherited_source_after_failure" :
         to_device             ? "restore_source_after_failure" :
         result.output_written ? "candidate_output_after_failure" :
                                 "unwritten_destination_after_rejection"},
        {"gpu_snapshot_role",
         result.staging_snapshot.empty() ? "unavailable" :
         source_gpu                      ? "checked_source_before_overlay" :
                                           "candidate_output_after_overlay"},
        {"gpu_footer_role", source_gpu ? "source" : "candidate_output"},
        {"original_gpu_source_available", source_gpu && !result.staging_snapshot.empty() ? "true" : "false"},
        {"temporal_limit",
         "CPU snapshot is copied after failure, not atomically with transfer or GPU validation. "
         "Concurrent mutation, stored-footer corruption and transfer faults cannot be uniquely "
         "distinguished. Matching CRC alone does not establish matching bytes."},
        {"cpu_address", std::to_string(reinterpret_cast<uintptr_t>(source))},
        {"cpu_output_address", std::to_string(reinterpret_cast<uintptr_t>(host))},
        {"cpu_backing",
         (inherited ? item.src_backing_type() : item.backing_type()) == MemoryOperationRequestPB::DISK ?
             "disk_read_buffer" :
             "memory_pool"},
        {"disk_slot", std::to_string(inherited ? item.src_disk_slot() : item.disk_slot())},
        {"staging_captured", result.staging_snapshot.empty() ? "false" : "true"},
        {"copy_item", item.DebugString()},
    };
    describeFooter(manifest, "cpu_footer_", footer);
    if (result.checked_footer) {
        describeFooter(manifest, "checked_footer_", *result.checked_footer);
        manifest["cpu_footer_differs_from_gpu_checked_footer"] =
            std::memcmp(&footer, &*result.checked_footer, sizeof(footer)) == 0 ? "false" : "true";
    }
    if (result.staging_footer)
        describeFooter(manifest, "gpu_footer_", *result.staging_footer);
    if (!cpu_output.empty()) {
        CrcBlockFooter output_footer;
        std::memcpy(&output_footer, cpu_output.data() + CrcBlockCopy::footerOffset(bytes), sizeof(output_footer));
        describeFooter(manifest, "cpu_output_footer_", output_footer);
    }
    // Record the trusted packing layout independently of possibly corrupt footer metadata.
    size_t     offset      = 0;
    const auto slots       = layerRegionSlots();
    const bool prefix_kind = item.cache_block_kind() == MemoryOperationRequestPB::COMPRESSED_KV
                             || item.cache_block_kind() == MemoryOperationRequestPB::STATE_SWA_KV;
    for (size_t i = 0; i < slots.size(); ++i) {
        const auto& slot         = slots[i];
        const bool  included     = prefix_kind ? kindForSlot(slot) == kind : item.is_complete() || isFullOnlySlot(slot);
        const auto  key          = "slot_" + std::to_string(i) + "_";
        manifest[key + "layer"]  = std::to_string(slot.layer_id);
        manifest[key + "region"] = std::to_string(static_cast<uint32_t>(slot.region_name));
        manifest[key + "group"]  = std::to_string(slot.group_id);
        manifest[key + "stride_bytes"]   = std::to_string(slot.stride_bytes);
        manifest[key + "payload_offset"] = included ? std::to_string(offset) : "excluded";
        if (included)
            offset += slot.stride_bytes;
    }
    const auto  gpu_blocks = item.gpu_blocks();
    std::string block_ids;
    for (const auto id : gpu_blocks)
        block_ids += std::to_string(id) + ",";
    manifest["gpu_blocks"] = block_ids;
    const auto sequence    = crc_dump_sequence_++;
    const auto path        = crc_dump_path_;
    const auto prefix      = "crc_rank" + std::to_string(parallelism_config_.world_rank) + "_";
    const auto name        = prefix + std::to_string(currentTimeUs()) + "_" + std::to_string(sequence);
    // CPU reference hashing only runs on the rate-limited failure path,
    // using owned snapshots of the source and staging payloads.
    const auto cpu_crc                          = diagnosticCrc32c(cpu.data(), bytes);
    manifest["cpu_crc32c"]                      = std::to_string(cpu_crc);
    manifest["cpu_matches_snapshot_footer_crc"] = cpu_crc == footer.crc32c ? "true" : "false";
    if (!cpu_output.empty())
        manifest["cpu_output_crc32c"] = std::to_string(diagnosticCrc32c(cpu_output.data(), bytes));
    const auto& gpu       = result.staging_snapshot;
    std::string diagnosis = manifest["failure_stage"] + "_rejected";
    if (result.gpu_crc_status != 0)
        diagnosis = "gpu_crc_status_failed";
    if (gpu.size() == bytes) {
        const auto gpu_snapshot_crc         = diagnosticCrc32c(gpu.data(), bytes);
        manifest["gpu_snapshot_cpu_crc32c"] = std::to_string(gpu_snapshot_crc);
        const auto& compared_cpu            = source_gpu || cpu_output.empty() ? cpu : cpu_output;
        size_t      different = 0, first = bytes;
        for (size_t i = 0; i < bytes; ++i) {
            if (compared_cpu[i] != gpu[i]) {
                if (different == 0)
                    first = i;
                ++different;
            }
        }
        manifest["compared_cpu_role"]        = source_gpu ? "source" : "candidate_output";
        manifest["payload_different_bytes"]  = std::to_string(different);
        manifest["payload_first_difference"] = different ? std::to_string(first) : "none";
        if (source_gpu && result.checked_footer && result.gpu_crc_observed) {
            const bool cpu_matches_checked      = cpu_crc == result.checked_footer->crc32c;
            manifest["cpu_matches_checked_crc"] = cpu_matches_checked ? "true" : "false";
            if (result.gpu_crc_status != 0)
                diagnosis = "gpu_crc_status_failed";
            else if (gpu_snapshot_crc != result.actual_crc)
                diagnosis = "gpu_crc_observation_disagrees_with_snapshot";
            else if (different == 0)
                diagnosis = "cpu_gpu_payload_agree_checked_crc_mismatch";
            else if (cpu_matches_checked)
                diagnosis = "cpu_snapshot_matches_checked_crc_gpu_differs";
            else
                diagnosis = "cpu_gpu_payload_disagree_checked_crc_mismatch";
        }
    } else {
        manifest["payload_different_bytes"] = "unavailable";
    }
    manifest["diagnosis"] = diagnosis;
    RTP_LLM_LOG_ERROR("CRC diagnostic: rank=%s direction=%s mem_block=%s "
                      "stage=%s diagnosis=%s checked_crc=%s cpu_crc=%u gpu_crc=%s gpu_status=%s "
                      "different_bytes=%s cpu_role=%s gpu_role=%s "
                      "temporal_cause=undetermined dump=%s/%s",
                      manifest["rank"].c_str(),
                      manifest["direction"].c_str(),
                      manifest["mem_block"].c_str(),
                      manifest["failure_stage"].c_str(),
                      diagnosis.c_str(),
                      manifest["expected_crc32c"].c_str(),
                      cpu_crc,
                      manifest["actual_crc32c"].c_str(),
                      manifest["gpu_crc_status"].c_str(),
                      manifest["payload_different_bytes"].c_str(),
                      manifest["cpu_snapshot_role"].c_str(),
                      manifest["gpu_snapshot_role"].c_str(),
                      path.c_str(),
                      name.c_str());
    const auto     json         = autil::legacy::ToJsonString(manifest);
    const uint64_t record_bytes = cpu.size() + cpu_output.size() + gpu.size() + sizeof(footer)
                                  + (result.checked_footer ? sizeof(CrcBlockFooter) : 0)
                                  + (result.staging_footer ? sizeof(CrcBlockFooter) : 0) + json.size();
    if (record_bytes > kCrcDumpQuotaBytes)
        throw std::length_error("full CRC dump and metadata exceed quota");
    namespace fs = std::filesystem;
    fs::create_directories(path);
    std::vector<std::pair<fs::file_time_type, fs::path>> old;
    uintmax_t                                            total = 0;
    for (const auto& entry : fs::directory_iterator(path)) {
        if (entry.is_symlink() || !entry.is_directory() || entry.path().filename().string().rfind(prefix, 0) != 0)
            continue;
        old.emplace_back(entry.last_write_time(), entry.path());
        for (const auto& file : fs::directory_iterator(entry.path()))
            if (file.is_regular_file())
                total += file.file_size();
    }
    std::sort(old.begin(), old.end());
    for (const auto& [time, directory] : old) {
        if (total <= kCrcDumpQuotaBytes - record_bytes)
            break;
        // Dump directories contain only our flat evidence files. Use POSIX
        // deletion to avoid libtorch's interposed filesystem::remove_all.
        for (const auto& file : fs::directory_iterator(directory)) {
            if (file.is_regular_file()) {
                total -= file.file_size();
            }
            if (::unlink(file.path().c_str()) != 0) {
                throw std::runtime_error("CRC dump rotation unlink failed");
            }
        }
        if (::rmdir(directory.c_str()) != 0) {
            throw std::runtime_error("CRC dump rotation rmdir failed");
        }
    }
    const auto directory = fs::path(path) / name;
    fs::create_directory(directory);
    auto write = [&](const char* filename, const void* data, size_t size) {
        std::ofstream file(directory / filename, std::ios::binary);
        file.write(static_cast<const char*>(data), size);
        file.close();
        if (!file.good())
            throw std::runtime_error("CRC dump write failed");
    };
    write("cpu.bin", cpu.data(), cpu.size());
    if (!cpu_output.empty())
        write("cpu_output.bin", cpu_output.data(), cpu_output.size());
    if (!gpu.empty())
        write("gpu_staging.bin", gpu.data(), gpu.size());
    write("footer.bin", &footer, sizeof(footer));
    if (result.checked_footer)
        write("checked_footer.bin", &*result.checked_footer, sizeof(CrcBlockFooter));
    if (result.staging_footer)
        write("gpu_staging_footer.bin", &*result.staging_footer, sizeof(CrcBlockFooter));
    write("manifest.json", json.data(), json.size());
    RTP_LLM_LOG_INFO("CRC dump written: rank=%ld path=%s", parallelism_config_.world_rank, directory.c_str());
}

}  // namespace rtp_llm
