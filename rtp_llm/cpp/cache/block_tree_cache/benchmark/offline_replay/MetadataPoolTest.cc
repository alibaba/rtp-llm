#include <filesystem>
#include <iostream>
#include <unistd.h>
#include <stdexcept>
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/config/StaticConfig.h"
using namespace rtp_llm;
namespace {
void require(bool value) {
    if (!value)
        throw std::runtime_error("metadata pool assertion failed");
}
template<class F>
void rejects(F action) {
    bool rejected = false;
    try {
        action();
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected);
}
void checkTreeOwnership(IBlockPool& pool) {
    require(pool.totalBlocksNum() == 16 && pool.freeBlocksNum() == 16);
    auto ids = pool.malloc(16);
    require(ids.has_value());
    pool.incTreeRef(*ids, BlockTreeRefType::CACHE);
    pool.incTreeRef(*ids, BlockTreeRefType::LOAD);
    require(!pool.malloc().has_value());
    require(!pool.canReclaimCacheBlock(ids->front()));
    pool.decTreeRef(*ids, BlockTreeRefType::LOAD);
    require(pool.canReclaimCacheBlock(ids->front()));
    pool.decTreeRef(*ids, BlockTreeRefType::CACHE);
    require(pool.freeBlocksNum() == 16 && pool.usedBlocksNum() == 0);
}
}  // namespace
int main(int argc, char** argv) {
    StaticConfig::user_ft_core_dump_on_exception = false;
    require(argc == 1 || argc == 2);
    const auto disk_path = argc == 2 ?
                               std::filesystem::path(argv[1]) :
                               std::filesystem::temp_directory_path() / ("btc-no-payload-" + std::to_string(getpid()));
    require(!std::filesystem::exists(disk_path));
    constexpr size_t stride  = size_t{1} << 36;
    auto             dc      = std::make_shared<DeviceBlockPoolConfig>();
    dc->pool_type            = BlockPoolType::DEVICE;
    dc->pool_name            = "metadata-device";
    dc->metadata_only        = true;
    dc->physical_block_count = 17;
    dc->total_size_bytes     = 17 * stride;
    MemoryLayoutConfig layout;
    layout.layer_num                = 1;
    layout.block_num                = 17;
    layout.dtype                    = TYPE_INT8;
    layout.kv_block_stride_bytes    = stride;
    layout.kv_block_pool_size_bytes = 17 * stride;
    layout.block_stride_bytes       = stride;
    layout.total_size_bytes         = 17 * stride;
    layout.seq_size_per_block       = 256;
    dc->memory_layouts.push_back(layout);
    DeviceBlockPool device(dc);
    require(device.init());
    require(device.getAllocationSizeBytes() == 0);
    rejects([&] { device.getBaseAddress(); });
    auto id = device.malloc();
    require(id.has_value());
    device.incRef(*id);
    device.incTreeRef(*id, BlockTreeRefType::CACHE);
    require(!device.canReclaimCacheBlock(*id));
    device.decRef(*id);
    require(device.canReclaimCacheBlock(*id));
    device.decTreeRef(*id, BlockTreeRefType::CACHE);
    checkTreeOwnership(device);
    auto hc                  = std::make_shared<HostBlockPoolConfig>();
    hc->pool_type            = BlockPoolType::HOST;
    hc->pool_name            = "metadata-host";
    hc->metadata_only        = true;
    hc->physical_block_count = 17;
    hc->payload_bytes        = stride;
    hc->stride_bytes         = stride;
    HostBlockPool host(hc);
    require(host.init());
    checkTreeOwnership(host);
    rejects([&] { host.blockBuffer(1); });
    auto kc                  = std::make_shared<DiskBlockPoolConfig>();
    kc->pool_type            = BlockPoolType::DISK;
    kc->pool_name            = "metadata-disk";
    kc->metadata_only        = true;
    kc->physical_block_count = 17;
    kc->payload_bytes        = stride;
    kc->stride_bytes         = stride;
    kc->disk_size_bytes      = 17 * stride;
    kc->work_dir             = disk_path.string();
    DiskBlockPool disk(kc);
    require(disk.init());
    checkTreeOwnership(disk);
    char data = 0;
    rejects([&] { disk.read(1, &data, 1); });
    rejects([&] { disk.write(1, &data, 1); });
    require(disk.readBytes() == 0 && disk.writeBytes() == 0 && !std::filesystem::exists(disk_path));
    std::cout << "PASS: metadata allocation/refcounts, no KV backing, forbidden payload access, no disk file\n";
}
