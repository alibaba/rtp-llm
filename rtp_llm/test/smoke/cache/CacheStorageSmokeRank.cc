#include "rtp_llm/test/smoke/cache/CacheSmokeSupport.h"

#include <atomic>
#include <cerrno>
#include <filesystem>
#include <fcntl.h>
#include <mutex>
#include <unistd.h>

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/DeviceBlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

// This test-only linker seam affects only the explicitly named probe file.
// ENOSPC is a syscall-interface injection, never a claim that shared disk is full.
namespace {
std::string           enospc_path;
std::atomic<unsigned> enospc_calls{0};
}  // namespace
extern "C" int __real_posix_fallocate(int fd, off_t offset, off_t bytes);
extern "C" int __wrap_posix_fallocate(int fd, off_t offset, off_t bytes) {
    if (!enospc_path.empty()) {
        char       buffer[4096];
        const auto count = ::readlink(("/proc/self/fd/" + std::to_string(fd)).c_str(), buffer, sizeof(buffer));
        if (count > 0 && std::string(buffer, count) == enospc_path) {
            enospc_calls.fetch_add(1);
            return ENOSPC;
        }
    }
    return __real_posix_fallocate(fd, offset, bytes);
}

namespace rtp_llm::cache_smoke {
namespace {
using Bytes = std::vector<uint8_t>;
std::string boolean(bool value) {
    return value ? "true" : "false";
}
std::string ioName(DiskBlockIOStatus status) {
    switch (status) {
        case DiskBlockIOStatus::OK:
            return "OK";
        case DiskBlockIOStatus::INVALID_SIZE:
            return "INVALID_SIZE";
        case DiskBlockIOStatus::ALIGNMENT_ERROR:
            return "ALIGNMENT_ERROR";
        case DiskBlockIOStatus::IO_ERROR:
            return "IO_ERROR";
        case DiskBlockIOStatus::PARTIAL_FAILURE:
            return "PARTIAL_FAILURE";
    }
    return "UNKNOWN";
}

// All normal I/O delegates unchanged to production PosixDiskBlockIO. The only
// fault switch is a labelled test interface write failure; reads remain real.
class ObservedIO final: public DiskBlockIO {
public:
    bool                      fail_writes = false;
    std::vector<std::string>  events;
    std::mutex                mutex;
    BlockTreePosixDiskBlockIO real;

    void record(const std::string& event) {
        std::lock_guard<std::mutex> lock(mutex);
        events.push_back(event);
    }
    std::string take() {
        std::lock_guard<std::mutex> lock(mutex);
        std::string                 out = "[";
        for (const auto& event : events) {
            if (out.size() > 1)
                out += ',';
            out += event;
        }
        events.clear();
        return out + ']';
    }
    DiskBlockIOStatus openAndPreallocate(const std::string& path, size_t bytes, bool buffered) override {
        auto status = real.openAndPreallocate(path, bytes, buffered);
        record("{\"operation\":\"open\",\"path\":" + quote(path) + ",\"bytes\":" + std::to_string(bytes)
               + ",\"status\":" + quote(ioName(status)) + '}');
        return status;
    }
    DiskBlockIOStatus read(uint64_t offset, void* dst, size_t bytes) override {
        auto status = real.read(offset, dst, bytes);
        record("{\"operation\":\"read\",\"offset\":" + std::to_string(offset) + ",\"bytes\":" + std::to_string(bytes)
               + ",\"status\":" + quote(ioName(status)) + '}');
        return status;
    }
    DiskBlockIOStatus write(uint64_t offset, const void* src, size_t bytes) override {
        auto status = fail_writes ? DiskBlockIOStatus::IO_ERROR : real.write(offset, src, bytes);
        record("{\"operation\":\"write\",\"offset\":" + std::to_string(offset) + ",\"bytes\":" + std::to_string(bytes)
               + ",\"injected\":" + boolean(fail_writes) + ",\"status\":" + quote(ioName(status)) + '}');
        return status;
    }
    DiskBlockIOStatus read(const std::vector<DiskRead>& items) override {
        const auto status = real.read(items);
        record("{\"operation\":\"read_batch\",\"count\":" + std::to_string(items.size())
               + ",\"status\":" + quote(ioName(status)) + '}');
        return status;
    }
    DiskBlockIOStatus write(const std::vector<DiskWrite>& items) override {
        const auto status = fail_writes ? DiskBlockIOStatus::IO_ERROR : real.write(items);
        record("{\"operation\":\"write_batch\",\"count\":" + std::to_string(items.size())
               + ",\"injected\":" + boolean(fail_writes) + ",\"status\":" + quote(ioName(status)) + '}');
        return status;
    }
    void close() override {
        real.close();
    }
    std::string debugString() const override {
        return "ObservedIO{" + real.debugString() + '}';
    }
};

struct Fixture {
    std::shared_ptr<DeviceBlockPool>              device;
    std::shared_ptr<HostBlockPool>                host;
    std::shared_ptr<BlockTreeDiskBlockPool>       disk;
    std::shared_ptr<BlockTreeDiskBlockPoolConfig> disk_config;
    std::shared_ptr<PerRankBlockTransferEngine>   engine;
    ObservedIO*                                   io = nullptr;
    PayloadOracle                                 oracle;
    int                                           layers, count;
    size_t                                        payload, stride;
    size_t                                        kv_bytes, scale_bytes, scale_offset;
    BlockIdList                                   device_blocks, host_blocks, disk_blocks;
    std::string                                   config_json;

    Fixture(const std::string& directory,
            uint64_t           seed,
            int                layer_count,
            int                heads,
            int                dim,
            int                tokens,
            int                block_tokens,
            int                staging,
            int                batch):
        oracle{seed, heads, dim, block_tokens, 1, true}, layers(layer_count), count(tokens / block_tokens) {
        ModelConfig model;
        model.num_layers                   = layers;
        model.hidden_size                  = heads * dim;
        model.data_type                    = TYPE_INT8;
        model.attn_config.head_num         = heads;
        model.attn_config.kv_head_num      = heads;
        model.attn_config.size_per_head    = dim;
        model.attn_config.tokens_per_block = block_tokens;
        model.kv_cache_spec_descs.resize(layers);
        for (auto& descs : model.kv_cache_spec_descs) {
            KVCacheSpecDesc desc;
            desc.tag        = "default";
            desc.dtype      = TYPE_INT8;
            desc.cache_type = KVCacheSpecType::MultiHeadAttention;
            descs.push_back(desc);
        }
        ParallelismConfig parallel;
        KVCacheConfig     options;
        options.test_block_num      = count + 3;
        options.reserve_block_ratio = 0;
        options.reuse_cache         = false;
        auto config                 = CacheConfigCreator::createConfig(model, parallel, options);
        config.finalizeBlockNums(
            CacheConfigCreator::computeLocalBlockNum(config, model, RuntimeConfig{}, options, parallel),
            RuntimeConfig{});
        require(config.group("default").kvBlockStrideBytes() == oracle.payload(0, 0, false).size(),
                "real KV geometry differs");
        require(config.group("default").kvScaleStrideBytes() == oracle.payload(0, 0, true).size(),
                "real scale geometry differs");
        auto device_config = std::make_shared<DeviceBlockPoolConfig>(
            DeviceBlockPoolConfigHelper::createConfigForGroup(config, config.group("default")));
        // Independent geometry for this fixture's single-layout INT8 MHA pool.
        // Do not derive checker addresses from the production block converter.
        kv_bytes     = size_t(2) * heads * block_tokens * dim;
        scale_bytes  = size_t(2) * heads * block_tokens * sizeof(float);
        scale_offset = size_t(layers) * (count + 3) * kv_bytes;
        require(device_config->memory_layouts.size() == 1, "storage oracle requires one MHA layout");
        const auto& layout = device_config->memory_layouts.front();
        require(layout.layer_num == static_cast<size_t>(layers) && layout.block_num == static_cast<size_t>(count + 3)
                    && layout.kv_cache_offset_bytes == 0 && layout.kv_scale_offset_bytes == scale_offset
                    && layout.kv_block_stride_bytes == kv_bytes && layout.kv_scale_stride_bytes == scale_bytes
                    && layout.kv_block_pool_size_bytes == scale_offset
                    && layout.kv_scale_pool_size_bytes == size_t(layers) * (count + 3) * scale_bytes
                    && device_config->total_size_bytes == size_t(layers) * (count + 3) * (kv_bytes + scale_bytes),
                "independent MHA pool geometry differs from production allocation");
        device_config->use_device_malloc_backing = true;
        device                                   = std::make_shared<DeviceBlockPool>(device_config);
        require(device->init(), "device init failed");
        require(device->where() == MEMORY_GPU, "storage requires real GPU backing");
        payload                = layers * (oracle.payload(0, 0, false).size() + oracle.payload(0, 0, true).size());
        stride                 = ((payload + 4095) / 4096) * 4096;
        auto host_config       = std::make_shared<HostBlockPoolConfig>();
        host_config->pool_type = BlockPoolType::HOST;
        host_config->pool_name = "storage_host";
        host_config->physical_block_count = count + 3;
        host_config->payload_bytes        = payload;
        host_config->stride_bytes         = stride;
        host_config->alignment            = 4096;
        host                              = std::make_shared<HostBlockPool>(host_config);
        require(host->init(), "host init failed");
        disk_config                  = std::make_shared<BlockTreeDiskBlockPoolConfig>();
        disk_config->pool_type       = BlockPoolType::DISK;
        disk_config->pool_name       = "storage";
        disk_config->work_dir        = directory;
        disk_config->disk_size_bytes = (count + 3) * stride;
        disk_config->payload_bytes   = payload;
        disk_config->stride_bytes    = stride;
        disk_config->buffered_io     = true;
        auto observed                = std::make_unique<ObservedIO>();
        io                           = observed.get();
        disk                         = std::make_shared<BlockTreeDiskBlockPool>(disk_config, std::move(observed));
        require(disk->init(), "disk init failed");
        auto group = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{device}, host, disk);
        group->initialize(0, config.topologyPtr(), {"default"});
        require(group->payloadBytes() == payload, "resolved group payload differs");
        engine = std::make_shared<PerRankBlockTransferEngine>(
            std::vector<GroupSetPtr>{group}, true, DeviceHostCopyOptions{}, staging, batch, 2);
        for (auto* pool : std::vector<IBlockPool*>{device.get(), host.get(), disk.get()}) {
            auto blocks = pool->malloc(count + 2);
            require(blocks.has_value(), "fixture allocate failed");
            pool->incTreeRef(*blocks, BlockTreeRefType::STORE);
            if (pool == device.get())
                device_blocks = *blocks;
            else if (pool == host.get())
                host_blocks = *blocks;
            else
                disk_blocks = *blocks;
        }
        for (int i = 0; i < count + 2; ++i) {
            put(Tier::DEVICE, i, Bytes(payload, 0xA5));
            put(Tier::HOST, i, Bytes(payload, 0xA5));
            put(Tier::DISK, i, Bytes(payload, 0xA5));
        }
        config_json =
            "{\"tokens\":" + std::to_string(tokens) + ",\"block_tokens\":" + std::to_string(block_tokens)
            + ",\"pages\":" + std::to_string(count) + ",\"layers\":" + std::to_string(layers)
            + ",\"kv_heads\":" + std::to_string(heads) + ",\"head_dim\":" + std::to_string(dim)
            + ",\"dtype\":\"INT8\",\"scale_dtype\":\"FP32\",\"payload_bytes_per_block\":" + std::to_string(payload)
            + ",\"disk_stride\":" + std::to_string(stride) + ",\"physical_blocks\":" + std::to_string(count + 3)
            + ",\"group_type\":" + quote("FULL") + ",\"staging_blocks\":" + std::to_string(staging)
            + ",\"actual_lane_capacity\":" + std::to_string(staging / 2) + ",\"max_descriptors_per_batch\":"
            + std::to_string(batch) + ",\"device_bytes\":" + std::to_string(device->getTotalSizeBytes())
            + ",\"host_pinned_bytes\":" + std::to_string((count + 3) * stride) + ",\"disk_preallocated_bytes\":"
            + std::to_string(disk_config->disk_size_bytes) + ",\"device_address_oracle\":\"raw_base_mha_int8\""
            + ",\"production_spec\":" + quote(config.group("default").spec->debugString()) + '}';
    }
    Bytes expected(int logical, uint64_t seed) const {
        auto reference = oracle;
        reference.seed = seed;
        Bytes bytes;
        for (int layer = 0; layer < layers; ++layer)
            for (bool scale : {false, true}) {
                auto part = reference.payload(layer, logical, scale);
                bytes.insert(bytes.end(), part.begin(), part.end());
            }
        return bytes;
    }
    BlockIdxType block(Tier tier, int index) const {
        return (tier == Tier::DEVICE ? device_blocks : tier == Tier::HOST ? host_blocks : disk_blocks).at(index);
    }
    void* deviceAddress(int layer, BlockIdxType physical_block, bool scale) const {
        require(layer >= 0 && layer < layers && physical_block >= 0 && physical_block < count + 3,
                "independent DEVICE coordinate out of range");
        const size_t bytes  = scale ? scale_bytes : kv_bytes;
        const size_t offset = (scale ? scale_offset : 0) + (size_t(layer) * (count + 3) + physical_block) * bytes;
        require(offset + bytes <= device->getTotalSizeBytes() && (scale || offset + bytes <= scale_offset),
                "independent DEVICE span overlaps or exceeds backing");
        return static_cast<uint8_t*>(device->getBaseAddress()) + offset;
    }
    void put(Tier tier, int index, const Bytes& bytes) {
        require(bytes.size() == payload, "write payload differs");
        if (tier == Tier::DEVICE) {
            size_t offset = 0;
            for (int layer = 0; layer < layers; ++layer)
                for (bool scale : {false, true}) {
                    const size_t size = scale ? scale_bytes : kv_bytes;
                    require(offset + size <= bytes.size(), "device component overflow");
                    require(cudaMemcpy(deviceAddress(layer, block(tier, index), scale),
                                       bytes.data() + offset,
                                       size,
                                       cudaMemcpyHostToDevice)
                                == cudaSuccess,
                            "GPU write failed");
                    offset += size;
                }
            require(offset == payload, "device payload geometry differs");
        } else if (tier == Tier::HOST) {
            auto buffer = host->blockBuffer(block(tier, index));
            std::memset(buffer.addr, 0xEE, stride);
            std::memcpy(buffer.addr, bytes.data(), payload);
        } else {
            Bytes padded(stride, 0xEE);
            std::copy(bytes.begin(), bytes.end(), padded.begin());
            require(disk->write(block(tier, index), padded.data(), stride) == BlockIOStatus::OK, "disk write failed");
        }
    }
    Bytes get(Tier tier, int index) {
        Bytes bytes;
        if (tier == Tier::DEVICE) {
            for (int layer = 0; layer < layers; ++layer)
                for (bool scale : {false, true}) {
                    auto part =
                        readBytes(deviceAddress(layer, block(tier, index), scale), scale ? scale_bytes : kv_bytes);
                    bytes.insert(bytes.end(), part.begin(), part.end());
                }
        } else if (tier == Tier::HOST) {
            auto buffer = host->blockBuffer(block(tier, index));
            auto begin  = static_cast<uint8_t*>(buffer.addr);
            bytes.assign(begin, begin + payload);
        } else {
            bytes.resize(stride);
            require(disk->read(block(tier, index), bytes.data(), stride) == BlockIOStatus::OK, "disk snapshot failed");
            bytes.resize(payload);
        }
        require(bytes.size() == payload, "read geometry differs");
        return bytes;
    }
    std::string checkAddressPermutations(uint64_t seed) {
        const auto first = get(Tier::DEVICE, 1), second = get(Tier::DEVICE, 2);
        require(first == expected(0, seed) && second == expected(1, seed), "checker baseline differs");
        put(Tier::DEVICE, 1, second);
        put(Tier::DEVICE, 2, first);
        auto block_swap = snapshot("checker_block_swap", Tier::DEVICE, seed);
        require(get(Tier::DEVICE, 1) != first && get(Tier::DEVICE, 2) != second, "block swap not detected");
        put(Tier::DEVICE, 1, first);
        put(Tier::DEVICE, 2, second);
        require(get(Tier::DEVICE, 1) == first && get(Tier::DEVICE, 2) == second, "block swap restore differs");
        std::string layer_swap = "null";
        if (layers >= 2) {
            auto         swapped     = first;
            const size_t layer_bytes = kv_bytes + scale_bytes;
            std::swap_ranges(swapped.begin(), swapped.begin() + layer_bytes, swapped.begin() + layer_bytes);
            put(Tier::DEVICE, 1, swapped);
            layer_swap = snapshot("checker_layer_swap", Tier::DEVICE, seed);
            require(get(Tier::DEVICE, 1) != first, "layer swap not detected");
            put(Tier::DEVICE, 1, first);
            require(get(Tier::DEVICE, 1) == first, "layer swap restore differs");
        }
        return "{\"block_swap\":" + block_swap + ",\"layer_swap\":" + layer_swap + ",\"restored\":true}";
    }
    std::string snapshot(const std::string& name, Tier tier, uint64_t seed, bool guard = false) {
        std::string units      = "[";
        size_t      mismatches = 0;
        for (int i = guard ? 0 : 1; i < (guard ? count + 2 : count + 1); ++i) {
            if (guard && i != 0 && i != count + 1)
                continue;
            auto   actual    = get(tier, i);
            auto   reference = guard ? Bytes(payload, 0xA5) : expected(i - 1, seed);
            size_t offset    = 0;
            for (int layer = 0; layer < layers; ++layer)
                for (bool scale : {false, true}) {
                    size_t size = oracle.payload(layer, 0, scale).size();
                    size_t bad  = 0;
                    for (size_t j = 0; j < size; ++j)
                        bad += actual[offset + j] != reference[offset + j];
                    mismatches += bad;
                    if (units.size() > 1)
                        units += ',';
                    units += "{\"logical_block\":" + std::to_string(i - 1) + ",\"physical_block\":"
                             + std::to_string(block(tier, i)) + ",\"layer\":" + std::to_string(layer)
                             + ",\"component\":" + quote(scale ? "KV_scale" : "KV")
                             + ",\"bytes\":" + std::to_string(size)
                             + ",\"expected_hash\":" + quote(hex(signature(reference.data() + offset, size)))
                             + ",\"actual_hash\":" + quote(hex(signature(actual.data() + offset, size)))
                             + ",\"mismatch_bytes\":" + std::to_string(bad) + ",\"equal\":" + boolean(bad == 0) + '}';
                    offset += size;
                }
        }
        return "{\"phase\":" + quote(name) + ",\"tier\":"
               + quote(tier == Tier::DEVICE ? "DEVICE" :
                       tier == Tier::HOST   ? "HOST" :
                                              "DISK")
               + ",\"mismatch_bytes\":" + std::to_string(mismatches) + ",\"equal\":" + boolean(mismatches == 0)
               + ",\"units\":" + units + "]}";
    }
    std::string transfer(const std::string& phase, Tier from, Tier to, bool expect_success = true) {
        std::vector<TransferDescriptor> descriptors;
        for (int i = 1; i <= count; ++i) {
            TransferDescriptor desc;
            desc.group_set_id  = 0;
            desc.source_tier   = from;
            desc.target_tier   = to;
            desc.source_blocks = {block(from, i)};
            desc.target_blocks = {block(to, i)};
            descriptors.push_back(desc);
        }
        io->take();
        std::vector<ErrorInfo> errors;
        // Production DEVICE->DISK entry requires exactly one descriptor; it is
        // issued sequentially. DISK->DEVICE receives the complete batch.
        if (from == Tier::DEVICE && to == Tier::DISK) {
            for (auto desc : descriptors) {
                auto context = engine->execute(TransferTask({desc}, std::chrono::seconds(30)));
                context->waitDone();
                errors.push_back(context->errorInfo());
                if (!context->success())
                    break;
            }
        } else {
            auto context = engine->execute(TransferTask(descriptors, std::chrono::seconds(30)));
            context->waitDone();
            errors.push_back(context->errorInfo());
        }
        bool        success = std::all_of(errors.begin(), errors.end(), [](const auto& e) { return e.ok(); });
        std::string result  = "{\"phase\":" + quote(phase) + ",\"success\":" + boolean(success) + ",\"api\":[";
        for (size_t i = 0; i < errors.size(); ++i) {
            if (i)
                result += ',';
            result += "{\"code\":" + std::to_string(static_cast<int>(errors[i].code()))
                      + ",\"message\":" + quote(errors[i].ToString()) + '}';
        }
        result += "],\"io_events\":" + io->take() + ",\"expected_success\":" + boolean(expect_success) + '}';
        // Return the raw production behavior. The supervisor independently checks
        // the contract; this function never maps or changes error codes.
        return result;
    }
    std::string release() {
        engine.reset();  // Real task-pool idle barrier before any pool reuse.
        std::string result = "[";
        for (auto* pool : std::vector<IBlockPool*>{device.get(), host.get(), disk.get()}) {
            auto blocks = pool == device.get() ? device_blocks : pool == host.get() ? host_blocks : disk_blocks;
            pool->decTreeRef(blocks, BlockTreeRefType::STORE);
            size_t free = pool->freeBlocksNum();
            auto   all  = pool->malloc(count + 2);
            bool   unique =
                all && std::set<BlockIdxType>(all->begin(), all->end()).size() == static_cast<size_t>(count + 2);
            bool exhausted = !pool->malloc().has_value();
            if (all) {
                pool->incTreeRef(*all, BlockTreeRefType::STORE);
                pool->decTreeRef(*all, BlockTreeRefType::STORE);
            }
            if (result.size() > 1)
                result += ',';
            result += "{\"tier\":"
                      + quote(pool == device.get() ? "DEVICE" :
                              pool == host.get()   ? "HOST" :
                                                     "DISK")
                      + ",\"free_after_engine_destroy\":" + std::to_string(free) + ",\"full_capacity_allocated\":"
                      + boolean(unique) + ",\"extra_allocation_rejected\":" + boolean(exhausted)
                      + ",\"free_after_probe\":" + std::to_string(pool->freeBlocksNum()) + '}';
        }
        return result + ']';
    }
};

void save(const std::string& path, const std::string& json) {
    std::ofstream out(path + ".tmp");
    out << json << '\n';
    out.close();
    require(static_cast<bool>(out), "report write failed");
    std::filesystem::rename(path + ".tmp", path);
}

int run(int argc, char** argv) {
    require(argc == 12, "storage smoke expects 11 arguments");
    const std::string scene = argv[1], directory = argv[2], output = argv[3];
    uint64_t          seed   = std::stoull(argv[4]);
    int               layers = std::stoi(argv[5]), heads = std::stoi(argv[6]), dim = std::stoi(argv[7]);
    int               tokens = std::stoi(argv[8]), block_tokens = std::stoi(argv[9]), staging = std::stoi(argv[10]),
        batch = std::stoi(argv[11]);
    initLogger();
    initRuntime(0, false, false, MlaOpsType::AUTO);
    Fixture                  env(directory, seed, layers, heads, dim, tokens, block_tokens, staging, batch);
    std::vector<std::string> observations;
    auto                     snap = [&](const std::string& phase, Tier tier, uint64_t value) {
        observations.push_back(env.snapshot(phase, tier, value));
    };
    auto move = [&](const std::string& phase, Tier from, Tier to, bool success = true) {
        observations.push_back(env.transfer(phase, from, to, success));
    };
    auto fill = [&](Tier tier, uint64_t value) {
        for (int i = 1; i <= env.count; ++i)
            env.put(tier, i, env.expected(i - 1, value));
    };
    auto poison = [&](Tier tier) {
        for (int i = 1; i <= env.count; ++i)
            env.put(tier, i, Bytes(env.payload, 0xA5));
    };
    fill(Tier::DEVICE, seed);
    const auto address_checks = env.checkAddressPermutations(seed);
    snap("source_before", Tier::DEVICE, seed);
    move("baseline_device_host", Tier::DEVICE, Tier::HOST);
    snap("baseline_device_host", Tier::HOST, seed);
    move("baseline_host_disk", Tier::HOST, Tier::DISK);
    snap("baseline_host_disk", Tier::DISK, seed);
    poison(Tier::HOST);
    move("baseline_disk_host", Tier::DISK, Tier::HOST);
    snap("baseline_disk_host", Tier::HOST, seed);
    poison(Tier::DEVICE);
    move("baseline_host_device", Tier::HOST, Tier::DEVICE);
    snap("baseline_host_device", Tier::DEVICE, seed);
    move("baseline_device_disk", Tier::DEVICE, Tier::DISK);
    poison(Tier::DEVICE);
    move("baseline_disk_device", Tier::DISK, Tier::DEVICE);
    snap("baseline_disk_device", Tier::DEVICE, seed);
    std::string injection = "{\"kind\":\"none\"}";
    if (scene == "corruption") {
        int fd = ::open(env.disk->filePath().c_str(), O_RDWR | O_CLOEXEC);
        require(fd >= 0, "corruption open failed");
        uint64_t offset = env.disk->blockOffset(env.disk_blocks.at(1 + env.count / 2)) + env.payload / 2;
        uint8_t  old    = 0;
        require(::pread(fd, &old, 1, offset) == 1, "corruption read failed");
        uint8_t changed = old ^ 0x5A;
        require(::pwrite(fd, &changed, 1, offset) == 1, "corruption write failed");
        require(::fsync(fd) == 0, "corruption fsync failed");
        ::close(fd);
        injection = "{\"kind\":\"actual_pwrite_equal_length_corruption\",\"offset\":" + std::to_string(offset)
                    + ",\"original_byte\":" + std::to_string(old) + ",\"changed_byte\":" + std::to_string(changed)
                    + '}';
        snap("fault_disk_source", Tier::DISK, seed);
        poison(Tier::DEVICE);
        move("fault_disk_device", Tier::DISK, Tier::DEVICE);
        snap("fault_target", Tier::DEVICE, seed);
        fill(Tier::DISK, seed);  // Explicit fixture repair, not production repair.
    } else if (scene == "truncate") {
        uint64_t offset = env.disk->blockOffset(env.disk_blocks.at(1 + env.count / 2)) + env.payload / 2;
        require(::truncate(env.disk->filePath().c_str(), offset) == 0, "truncate failed");
        injection = "{\"kind\":\"actual_truncate_mid_payload\",\"file_bytes\":" + std::to_string(offset) + '}';
        poison(Tier::DEVICE);
        move("fault_disk_device", Tier::DISK, Tier::DEVICE, false);
        snap("fault_target", Tier::DEVICE, seed);
        int fd = ::open(env.disk->filePath().c_str(), O_RDWR | O_CLOEXEC);
        require(fd >= 0, "repair open failed");
        require(::ftruncate(fd, env.disk_config->disk_size_bytes) == 0, "repair ftruncate failed");
        ::close(fd);
        for (int i = 0; i < env.count + 2; ++i)
            env.put(Tier::DISK, i, i == 0 || i == env.count + 1 ? Bytes(env.payload, 0xA5) : env.expected(i - 1, seed));
    } else if (scene == "unlink") {
        require(::unlink(env.disk->filePath().c_str()) == 0, "unlink failed");
        require(!std::filesystem::exists(env.disk->filePath()), "unlinked path remains");
        injection = "{\"kind\":\"actual_unlink_open_backing_file\",\"path_absent\":true,\"open_fd_remains\":true}";
        poison(Tier::DEVICE);
        move("fault_disk_device", Tier::DISK, Tier::DEVICE);
        snap("fault_target", Tier::DEVICE, seed);
    } else if (scene == "write_io_error") {
        fill(Tier::DEVICE, seed + 1);
        env.io->fail_writes = true;
        injection           = "{\"kind\":\"test_DiskBlockIO_write_interface_IO_ERROR\",\"physical_disk_full\":false}";
        move("fault_device_disk", Tier::DEVICE, Tier::DISK, false);
        env.io->fail_writes = false;
        snap("fault_disk_unchanged", Tier::DISK, seed);
        snap("fault_source_retained", Tier::DEVICE, seed + 1);
    } else if (scene == "init_existing" || scene == "init_enospc") {
        auto config       = std::make_shared<BlockTreeDiskBlockPoolConfig>(*env.disk_config);
        config->pool_name = "probe";
        std::string path  = directory + "/disk_block_pool_probe_r0_l0.bin";
        if (scene == "init_existing") {
            int fd = ::open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
            require(fd >= 0, "probe create failed");
            ::close(fd);
        } else
            enospc_path = path;
        auto        observer = std::make_unique<ObservedIO>();
        auto*       raw      = observer.get();
        auto        probe    = std::make_shared<BlockTreeDiskBlockPool>(config, std::move(observer));
        bool        rejected = false;
        std::string error;
        try {
            probe->init();
        } catch (const std::exception& e) {
            rejected = true;
            error    = e.what();
        }
        enospc_path.clear();
        injection =
            "{\"kind\":"
            + quote(scene == "init_existing" ? "actual_existing_file_O_EXCL" : "test_posix_fallocate_ENOSPC_syscall")
            + ",\"init_rejected\":" + boolean(rejected) + ",\"exception\":" + quote(error)
            + ",\"io_events\":" + raw->take() + ",\"enospc_syscall_count\":" + std::to_string(enospc_calls.load())
            + ",\"physical_disk_full\":false}";
        probe.reset();
        require(::unlink(path.c_str()) == 0, "probe file remove failed");
        auto repaired = std::make_shared<BlockTreeDiskBlockPool>(config);
        require(repaired->init(), "fresh probe init failed");
        auto block = repaired->malloc();
        require(block.has_value(), "fresh probe allocate failed");
        repaired->incTreeRef(*block, BlockTreeRefType::STORE);
        Bytes bytes(env.stride, 0x69), readback(env.stride);
        require(repaired->write(*block, bytes.data(), bytes.size()) == BlockIOStatus::OK, "probe repair write failed");
        require(repaired->read(*block, readback.data(), readback.size()) == BlockIOStatus::OK && bytes == readback,
                "probe repair read failed");
        repaired->decTreeRef(*block, BlockTreeRefType::STORE);
        repaired.reset();
    } else if (scene == "pool_exhaustion") {
        std::string pools = "[";
        for (auto* pool : std::vector<IBlockPool*>{env.device.get(), env.host.get(), env.disk.get()}) {
            size_t before   = pool->freeBlocksNum();
            bool   rejected = !pool->malloc().has_value();
            if (pools.size() > 1)
                pools += ',';
            pools += "{\"tier\":"
                     + quote(pool == env.device.get() ? "DEVICE" :
                             pool == env.host.get()   ? "HOST" :
                                                        "DISK")
                     + ",\"free_before\":" + std::to_string(before) + ",\"extra_allocation_rejected\":"
                     + boolean(rejected) + ",\"free_after\":" + std::to_string(pool->freeBlocksNum()) + '}';
        }
        injection =
            "{\"kind\":\"configured_pool_capacity_exhaustion\",\"disk_space_error\":false,\"pools\":" + pools + "]}";
    } else
        require(scene == "staging_full", "unknown storage scene");
    if (scene == "corruption" || scene == "truncate" || scene == "unlink") {
        poison(Tier::DEVICE);
        move("after_explicit_fixture_repair", Tier::DISK, Tier::DEVICE);
        snap("after_explicit_fixture_repair", Tier::DEVICE, seed);
    }
    fill(Tier::DEVICE, seed + 2);
    snap("recovery_source", Tier::DEVICE, seed + 2);
    move("recovery_device_disk", Tier::DEVICE, Tier::DISK);
    snap("recovery_disk", Tier::DISK, seed + 2);
    poison(Tier::DEVICE);
    move("recovery_disk_device", Tier::DISK, Tier::DEVICE);
    snap("recovery_target", Tier::DEVICE, seed + 2);
    for (auto tier : {Tier::DEVICE, Tier::HOST, Tier::DISK})
        observations.push_back(env.snapshot("guards", tier, seed, true));
    std::string report = "{\"pid\":" + std::to_string(::getpid()) + ",\"scene\":" + quote(scene) + ",\"seed\":"
                         + std::to_string(seed) + ",\"resolved\":" + env.config_json + ",\"address_self_checks\":"
                         + address_checks + ",\"injection\":" + injection + ",\"observations\":[";
    for (size_t i = 0; i < observations.size(); ++i) {
        if (i)
            report += ',';
        report += observations[i];
    }
    report += "],\"capacity\":" + env.release() + ",\"phase\":\"complete\"}";
    save(output, report);
    return 0;
}
}  // namespace
}  // namespace rtp_llm::cache_smoke
int main(int argc, char** argv) {
    try {
        return rtp_llm::cache_smoke::run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "storage smoke: " << e.what() << std::endl;
        return 1;
    }
}
