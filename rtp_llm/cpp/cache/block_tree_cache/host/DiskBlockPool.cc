#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskBlockPool.h"

#include <sstream>
#include <utility>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

namespace {

std::string buildFilePath(const std::string& dir, const DiskBlockPoolConfig& cfg) {
    const std::string pool_name = cfg.pool_name.empty() ? std::string("unnamed") : cfg.pool_name;
    std::ostringstream oss;
    oss << dir << "/disk_block_pool_" << pool_name << "_r" << cfg.world_rank << "_l" << cfg.local_rank << ".bin";
    return oss.str();
}

}  // namespace

std::shared_ptr<const DiskBlockPoolConfig>
DiskBlockPool::normalizeConfig(const std::shared_ptr<const DiskBlockPoolConfig>& config) {
    RTP_LLM_CHECK(config != nullptr);
    RTP_LLM_CHECK(config->pool_type == BlockPoolType::DISK);
    RTP_LLM_CHECK_WITH_INFO(
        config->stride_bytes > 0, "disk block pool [%s] stride_bytes must be > 0", config->pool_name.c_str());

    const size_t computed_physical_block_count = config->disk_size_bytes / config->stride_bytes;
    if (config->physical_block_count != 0) {
        RTP_LLM_CHECK_WITH_INFO(config->physical_block_count == computed_physical_block_count,
                                 "disk block pool [%s] physical_block_count [%zu] does not match "
                                 "disk_size_bytes [%zu] / stride_bytes [%zu] = [%zu]",
                                 config->pool_name.c_str(),
                                 config->physical_block_count,
                                 config->disk_size_bytes,
                                 config->stride_bytes,
                                 computed_physical_block_count);
    }
    RTP_LLM_CHECK_WITH_INFO(computed_physical_block_count > 1,
                             "disk block pool [%s] physical_block_count [%zu] (disk_size_bytes=%zu / "
                             "stride_bytes=%zu) must be > 1",
                             config->pool_name.c_str(),
                             computed_physical_block_count,
                             config->disk_size_bytes,
                             config->stride_bytes);

    auto normalized                  = std::make_shared<DiskBlockPoolConfig>(*config);
    normalized->physical_block_count = computed_physical_block_count;
    return normalized;
}

DiskBlockPool::DiskBlockPool(std::shared_ptr<const DiskBlockPoolConfig> config, std::unique_ptr<DiskBlockIO> io):
    IBlockPool(normalizeConfig(config)), io_(std::move(io)) {}

DiskBlockPool::~DiskBlockPool() = default;

const DiskBlockPoolConfig& DiskBlockPool::config() const {
    return configAs<DiskBlockPoolConfig>(BlockPoolType::DISK);
}

BlockIOStatus DiskBlockPool::mapStatus(DiskBlockIOStatus status) {
    switch (status) {
        case DiskBlockIOStatus::OK:
            return BlockIOStatus::OK;
        case DiskBlockIOStatus::INVALID_SIZE:
            return BlockIOStatus::INVALID_SIZE;
        case DiskBlockIOStatus::ALIGNMENT_ERROR:
            return BlockIOStatus::ALIGNMENT_ERROR;
        case DiskBlockIOStatus::IO_ERROR:
            return BlockIOStatus::IO_ERROR;
        case DiskBlockIOStatus::PARTIAL_FAILURE:
            return BlockIOStatus::PARTIAL_FAILURE;
    }
    return BlockIOStatus::IO_ERROR;
}

bool DiskBlockPool::init() {
    const auto& cfg = config();
    RTP_LLM_CHECK_WITH_INFO(
        cfg.payload_bytes > 0, "disk block pool [%s] payload_bytes must be > 0", cfg.pool_name.c_str());
    RTP_LLM_CHECK_WITH_INFO(cfg.stride_bytes >= cfg.payload_bytes,
                             "disk block pool [%s] stride_bytes [%zu] must be >= payload_bytes [%zu]",
                             cfg.pool_name.c_str(),
                             cfg.stride_bytes,
                             cfg.payload_bytes);
    RTP_LLM_CHECK_WITH_INFO(
        !cfg.work_dir.empty(), "disk block pool [%s] work_dir must not be empty", cfg.pool_name.c_str());

    std::string effective_dir = cfg.work_dir;
    if (cfg.mount_guard != nullptr) {
        effective_dir = cfg.mount_guard->workDir();
    }

    file_path_ = buildFilePath(effective_dir, cfg);

    if (io_ == nullptr) {
        io_ = std::make_unique<PosixDiskBlockIO>();
    }

    // physical_block_count already includes block 0's reserved slot (see
    // normalizeConfig), so the file must be preallocated over the full range.
    const size_t total_bytes = cfg.physical_block_count * cfg.stride_bytes;
    const auto   status      = io_->openAndPreallocate(file_path_, total_bytes, cfg.buffered_io);
    RTP_LLM_CHECK_WITH_INFO(status == DiskBlockIOStatus::OK,
                             "disk block pool [%s] failed to preallocate backing file [%s], bytes=[%zu]",
                             cfg.pool_name.c_str(),
                             file_path_.c_str(),
                             total_bytes);

    markInitialized();
    return true;
}

BlockIOStatus DiskBlockPool::read(BlockIdxType block, void* dst, size_t bytes) {
    RTP_LLM_CHECK(initialized());
    if (!validBlock(block)) {
        return BlockIOStatus::INVALID_BLOCK;
    }
    if (bytes == 0 || bytes > strideBytes()) {
        return BlockIOStatus::INVALID_SIZE;
    }
    const auto status = mapStatus(io_->read(blockOffset(block), dst, bytes));
    if (status == BlockIOStatus::OK) {
        read_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    }
    return status;
}

BlockIOStatus DiskBlockPool::write(BlockIdxType block, const void* src, size_t bytes) {
    RTP_LLM_CHECK(initialized());
    if (!validBlock(block)) {
        return BlockIOStatus::INVALID_BLOCK;
    }
    if (bytes == 0 || bytes > strideBytes()) {
        return BlockIOStatus::INVALID_SIZE;
    }
    const auto status = mapStatus(io_->write(blockOffset(block), src, bytes));
    if (status == BlockIOStatus::OK) {
        write_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    }
    return status;
}

BlockIOStatus DiskBlockPool::read(const BlockIdList& blocks, const std::vector<void*>& dsts, size_t bytes_per_block) {
    RTP_LLM_CHECK(initialized());
    RTP_LLM_CHECK_WITH_INFO(blocks.size() == dsts.size(),
                             "disk block pool [%s] batch read blocks/dsts size mismatch, blocks=[%zu] dsts=[%zu]",
                             poolName().c_str(),
                             blocks.size(),
                             dsts.size());
    if (bytes_per_block == 0 || bytes_per_block > strideBytes()) {
        return BlockIOStatus::INVALID_SIZE;
    }
    for (const auto block : blocks) {
        if (!validBlock(block)) {
            return BlockIOStatus::INVALID_BLOCK;
        }
    }

    std::vector<DiskRead> reads;
    reads.reserve(blocks.size());
    for (size_t i = 0; i < blocks.size(); ++i) {
        reads.push_back(DiskRead{blockOffset(blocks[i]), dsts[i], bytes_per_block});
    }
    const auto status = mapStatus(io_->read(reads));
    if (status == BlockIOStatus::OK) {
        read_bytes_.fetch_add(bytes_per_block * blocks.size(), std::memory_order_relaxed);
    }
    return status;
}

BlockIOStatus
DiskBlockPool::write(const BlockIdList& blocks, const std::vector<const void*>& srcs, size_t bytes_per_block) {
    RTP_LLM_CHECK(initialized());
    RTP_LLM_CHECK_WITH_INFO(blocks.size() == srcs.size(),
                             "disk block pool [%s] batch write blocks/srcs size mismatch, blocks=[%zu] srcs=[%zu]",
                             poolName().c_str(),
                             blocks.size(),
                             srcs.size());
    if (bytes_per_block == 0 || bytes_per_block > strideBytes()) {
        return BlockIOStatus::INVALID_SIZE;
    }
    for (const auto block : blocks) {
        if (!validBlock(block)) {
            return BlockIOStatus::INVALID_BLOCK;
        }
    }

    std::vector<DiskWrite> writes;
    writes.reserve(blocks.size());
    for (size_t i = 0; i < blocks.size(); ++i) {
        writes.push_back(DiskWrite{blockOffset(blocks[i]), srcs[i], bytes_per_block});
    }
    const auto status = mapStatus(io_->write(writes));
    if (status == BlockIOStatus::OK) {
        write_bytes_.fetch_add(bytes_per_block * blocks.size(), std::memory_order_relaxed);
    }
    return status;
}

size_t DiskBlockPool::payloadBytes() const {
    return config().payload_bytes;
}

size_t DiskBlockPool::strideBytes() const {
    return config().stride_bytes;
}

size_t DiskBlockPool::readBytes() const {
    return read_bytes_.load(std::memory_order_relaxed);
}

size_t DiskBlockPool::writeBytes() const {
    return write_bytes_.load(std::memory_order_relaxed);
}

uint64_t DiskBlockPool::blockOffset(BlockIdxType block) const {
    return static_cast<uint64_t>(block) * static_cast<uint64_t>(strideBytes());
}

const std::string& DiskBlockPool::filePath() const {
    return file_path_;
}

std::string DiskBlockPool::debugString() const {
    std::ostringstream oss;
    oss << "DiskBlockPool{" << IBlockPool::debugString() << ", payload_bytes=" << payloadBytes()
        << ", stride_bytes=" << strideBytes() << ", file_path=" << file_path_
        << ", io=" << (io_ ? io_->debugString() : "none") << "}";
    return oss.str();
}

}  // namespace rtp_llm
