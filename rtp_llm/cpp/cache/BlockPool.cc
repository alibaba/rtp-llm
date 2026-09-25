#include "rtp_llm/cpp/cache/BlockPool.h"

// ---- dcu leak-probe (combined): ledger + block-level account ----
#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <unordered_map>
namespace rtp_llm {
std::atomic<size_t> g_probe_malloc_total{0};
std::atomic<size_t> g_probe_reqfree_total{0};
thread_local int64_t g_tl_probe_request_id = 0;
thread_local void* g_tl_probe_owner_ids   = nullptr;
std::atomic<int>   g_probe_pool_seq{0};
struct ProbeBlockInfo {
    int64_t request_id;
    int64_t alloc_ms;
    void*   frames[8];
    void*   owner_ids;   // address of the BlockIds object this block was appended to
    int     pool_seq;
};
std::mutex                                              g_probe_block_mu;
std::unordered_map<int, ProbeBlockInfo> g_probe_blocks;
#include <execinfo.h>
#include <array>
#include <climits>
#include <vector>
#include <algorithm>
void probeRecordBlocks(const BlockIndicesType& block_ids) {
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
    void* frames[11] = {};
    const int n = backtrace(frames, 11);
    ProbeBlockInfo info;
    info.request_id = g_tl_probe_request_id;
    info.alloc_ms   = now_ms;
    info.owner_ids  = g_tl_probe_owner_ids;
    info.pool_seq   = g_probe_pool_seq.fetch_add(1) + 1;
    for (int i = 0; i < 8; ++i) {
        info.frames[i] = (2 + i < n) ? frames[2 + i] : nullptr;
    }
    std::lock_guard<std::mutex> g(g_probe_block_mu);
    for (int idx : block_ids) {
        g_probe_blocks[idx] = info;
    }
}
void probeEraseBlocks(const BlockIndicesType& block_ids) {
    std::lock_guard<std::mutex> g(g_probe_block_mu);
    for (int idx : block_ids) {
        g_probe_blocks.erase(idx);
    }
}
void probeDumpHanging(const char* pool) {
    std::lock_guard<std::mutex> g(g_probe_block_mu);
    if (g_probe_blocks.empty()) {
        return;
    }
    std::unordered_map<int64_t, size_t> by_req;
    std::map<std::array<void*, 8>, size_t> by_stack;
    for (auto& kv : g_probe_blocks) {
        by_req[kv.second.request_id] += 1;
        std::array<void*, 8> key;
        for (int i = 0; i < 8; ++i) {
            key[static_cast<size_t>(i)] = kv.second.frames[static_cast<size_t>(i)];
        }
        by_stack[key] += 1;
    }
    std::string req_hist;
    size_t shown = 0;
    for (auto it = by_req.begin(); it != by_req.end() && shown < 8; ++it, ++shown) {
        req_hist += " req=" + std::to_string(it->first) + ":" + std::to_string(it->second);
    }
    std::map<void*, size_t> by_owner;
    for (auto& kv : g_probe_blocks) {
        by_owner[kv.second.owner_ids] += 1;
    }
    std::string owner_hist;
    size_t os = 0;
    char obuf[128];
    for (auto it = by_owner.rbegin(); it != by_owner.rend() && os < 12; ++it, ++os) {
        snprintf(obuf, sizeof(obuf), " %p:%zu", it->first, it->second);
        owner_hist += obuf;
    }
    std::string idx_hist;
    {
        // a few concrete hanging block indices from the biggest owner
        char ibuf[160];
        size_t taken = 0;
        void*  best_owner = nullptr;
        size_t best_cnt = 0;
        for (auto& kv : by_owner) {
            if (kv.second > best_cnt) { best_cnt = kv.second; best_owner = kv.first; }
        }
        if (best_owner != nullptr) {
            std::vector<int> sample;
            for (auto& kv : g_probe_blocks) {
                if (kv.second.owner_ids == best_owner && taken < 8) {
                    sample.push_back(kv.first);
                    ++taken;
                }
            }
            std::string s;
            for (size_t i = 0; i < sample.size(); ++i) {
                s += std::to_string(sample[i]);
                if (i + 1 < sample.size()) s += ",";
            }
            snprintf(ibuf, sizeof(ibuf), " best_owner=%p idx=[%s]", best_owner, s.c_str());
            idx_hist = ibuf;
        }
    }
    RTP_LLM_LOG_WARNING("[dcu-owner] pool=%s owners%s%s",
                        pool,
                        owner_hist.c_str(),
                        idx_hist.c_str());
    std::string stack_hist;
    shown = 0;
    char buf[640];
    const int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now().time_since_epoch())
                               .count();
    for (auto it = by_stack.begin(); it != by_stack.end() && shown < 4; ++it, ++shown) {
        int64_t tmin = INT64_MAX, tmax = 0;
        for (auto& kv : g_probe_blocks) {
            std::array<void*, 8> key;
            for (int i = 0; i < 8; ++i) {
                key[static_cast<size_t>(i)] = kv.second.frames[static_cast<size_t>(i)];
            }
            if (key == it->first) {
                tmin = std::min(tmin, kv.second.alloc_ms);
                tmax = std::max(tmax, kv.second.alloc_ms);
            }
        }
        snprintf(buf, sizeof(buf), " ST%d:[%p|%p]x%zu age(-%lld..-%lldms)",
                 shown,
                 it->first[0], it->first[1],
                 it->second,
                 (long long)(now_ms - tmax), (long long)(now_ms - tmin));
        stack_hist += buf;
    }
    RTP_LLM_LOG_WARNING("[dcu-hang] pool=%s hanging=%zu distinct_req=%zu%s stacks:%s",
                        pool,
                        g_probe_blocks.size(),
                        by_req.size(),
                        req_hist.c_str(),
                        stack_hist.c_str());
}
void probeCheckLeakClear(const BlockIndicesType& blocks, void* owner_ids) {
    std::lock_guard<std::mutex> g(g_probe_block_mu);
    int unreturned = 0;
    for (int idx : blocks) {
        if (g_probe_blocks.count(idx)) {
            ++unreturned;
        }
    }
    if (unreturned > 0) {
        void* frames[8] = {};
        const int n = backtrace(frames, 8);
        RTP_LLM_LOG_WARNING("[dcu-clear-leak] unreturned=%d total=%zu owner=%p bt=[%p|%p|%p|%p]",
                            unreturned,
                            blocks.size(),
                            owner_ids,
                            (2 < n) ? frames[2] : nullptr,
                            (3 < n) ? frames[3] : nullptr,
                            (4 < n) ? frames[4] : nullptr,
                            (5 < n) ? frames[5] : nullptr);
    }
}
void probeOnBlockIdsDtor(const BlockIndicesType& blocks, void* self) {
    std::lock_guard<std::mutex> g(g_probe_block_mu);
    int unreturned = 0;
    for (int idx : blocks) {
        if (g_probe_blocks.count(idx)) {
            ++unreturned;
        }
    }
    if (unreturned > 0) {
        void* frames[8] = {};
        const int n = backtrace(frames, 8);
        RTP_LLM_LOG_WARNING("[dcu-dtor-leak] unreturned=%d total=%zu ids=%p bt=[%p|%p|%p|%p]",
                            unreturned,
                            blocks.size(),
                            self,
                            (2 < n) ? frames[2] : nullptr,
                            (3 < n) ? frames[3] : nullptr,
                            (4 < n) ? frames[4] : nullptr,
                            (5 < n) ? frames[5] : nullptr);
    }
}
void probeLedgerDump(const char* pool, const char* where) {
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
    static std::mutex              s_mu;
    static std::map<std::string, int64_t> s_last;
    {
        std::lock_guard<std::mutex> g(s_mu);
        auto it = s_last.find(pool);
        if (it != s_last.end() && now_ms - it->second < 30000) {
            return;
        }
        s_last[pool] = now_ms;
    }
    RTP_LLM_LOG_WARNING("[dcu-ledger] pool=%s at=%s malloc=%zu reqfree=%zu",
                        pool,
                        where,
                        g_probe_malloc_total.load(),
                        g_probe_reqfree_total.load());
    probeDumpHanging(pool);
}
}  // namespace rtp_llm
// ---- end dcu leak-probe ----
#include "rtp_llm/cpp/cache/MemoryLayoutStrategy.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

#include <cstdlib>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <utility>

#include <sys/mman.h>
#include <unistd.h>

#if USING_CUDA
#include <cuda_runtime.h>
#elif USING_ROCM
#include <hip/hip_runtime.h>
#endif

namespace rtp_llm {

namespace {

bool shouldPinHostBlockPool();

const char* allocationTypeName(AllocationType allocation_type) {
    switch (allocation_type) {
        case AllocationType::HOST:
            return "HOST";
        case AllocationType::DEVICE:
            return "DEVICE";
    }
    return "UNKNOWN";
}

const char* memoryTypeName(MemoryType memory_type) {
    switch (memory_type) {
        case MemoryType::MEMORY_CPU:
            return "CPU";
        case MemoryType::MEMORY_CPU_PINNED:
            return "CPU_PINNED";
        case MemoryType::MEMORY_GPU:
            return "GPU";
    }
    return "UNKNOWN";
}

const char*
requestedBackingName(AllocationType allocation_type, bool use_pinned_cpu_backing, bool use_device_malloc_backing) {
    if (allocation_type == AllocationType::HOST) {
        return shouldPinHostBlockPool() ? "CPU_PINNED_OR_CPU_FALLBACK" : "CPU";
    }
    if (use_device_malloc_backing) {
#if USING_CUDA
        return "GPU_CUDA_MALLOC";
#elif USING_ROCM
        return "GPU_HIP_MALLOC";
#else
        return "GPU_DEVICE_MALLOC";
#endif
    }
    return use_pinned_cpu_backing ? "CPU_PINNED" : "GPU";
}

bool shouldPinHostBlockPool() {
    static const bool cached = [] {
        const char* value = std::getenv("RTP_LLM_PIN_HOST_BLOCK_POOL");
        if (value == nullptr) {
            return true;
        }
        const std::string flag(value);
        return flag != "0" && flag != "false" && flag != "FALSE" && flag != "off" && flag != "OFF";
    }();
    return cached;
}

void markHostBlockPoolDontDump(const char* pool_name, void* ptr, size_t size) {
#ifdef MADV_DONTDUMP
    if (ptr == nullptr || size == 0) {
        return;
    }

    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        page_size = 4096;
    }

    const auto begin         = reinterpret_cast<uintptr_t>(ptr);
    const auto page_mask     = static_cast<uintptr_t>(page_size - 1);
    const auto aligned_begin = begin & ~page_mask;
    const auto aligned_end   = (begin + size + page_mask) & ~page_mask;
    const auto aligned_size  = static_cast<size_t>(aligned_end - aligned_begin);

    if (madvise(reinterpret_cast<void*>(aligned_begin), aligned_size, MADV_DONTDUMP) != 0) {
        RTP_LLM_LOG_WARNING("madvise MADV_DONTDUMP failed for host block pool, pool_name=%s ptr=%p, size=%zu, "
                            "error=%s",
                            pool_name,
                            ptr,
                            size,
                            std::strerror(errno));
    } else {
        RTP_LLM_LOG_INFO("madvise MADV_DONTDUMP success for host block pool, pool_name=%s ptr=%p, size=%zu, "
                         "aligned_ptr=%p, aligned_size=%zu",
                         pool_name,
                         ptr,
                         size,
                         reinterpret_cast<void*>(aligned_begin),
                         aligned_size);
    }
#else
    RTP_LLM_LOG_WARNING(
        "MADV_DONTDUMP is not defined, host block pool may be included in coredump, pool_name=%s ptr=%p, size=%zu",
        pool_name,
        ptr,
        size);
#endif
}

}  // namespace

BlockPool::BlockPool(const BlockPoolConfig& config,
                     AllocationType         allocation_type,
                     bool                   use_pinned_cpu_backing,
                     bool                   use_device_malloc_backing):
    config_(config),
    allocation_type_(allocation_type),
    use_pinned_cpu_backing_(use_pinned_cpu_backing),
    use_device_malloc_backing_(use_device_malloc_backing) {}

BlockPool::~BlockPool() {
    cache_aligned_buffer_ = torch::Tensor();
}

void BlockPool::validateConfig() const {
    RTP_LLM_CHECK_WITH_INFO(!(use_pinned_cpu_backing_ && use_device_malloc_backing_),
                            "BlockPool cannot use both pinned CPU backing and raw device malloc backing");
    RTP_LLM_CHECK_WITH_INFO(!config_.memory_layouts.empty(), "BlockPoolConfig.memory_layouts must not be empty");
    RTP_LLM_CHECK_WITH_INFO(config_.block_num > 0, "BlockPoolConfig.block_num must be > 0");

    for (size_t layout_idx = 0; layout_idx < config_.memory_layouts.size(); ++layout_idx) {
        const auto& layout_cfg = config_.memory_layouts[layout_idx];

        RTP_LLM_CHECK_WITH_INFO(layout_cfg.block_num == config_.block_num,
                                "MemoryLayoutConfig.block_num mismatch: layout[%zu].block_num=%u, pool.block_num=%u",
                                layout_idx,
                                layout_cfg.block_num,
                                config_.block_num);
        RTP_LLM_CHECK_WITH_INFO(
            layout_cfg.layer_num > 0, "MemoryLayoutConfig.layer_num must be > 0 (layout=%zu)", layout_idx);
        RTP_LLM_CHECK_WITH_INFO(layout_cfg.kv_block_pool_size_bytes > 0,
                                "MemoryLayoutConfig.kv_block_pool_size_bytes must be > 0 (layout=%zu)",
                                layout_idx);
    }
}

void BlockPool::initializeCacheBuffer() {
    if (allocation_type_ == AllocationType::HOST) {
        auto cpu_buffer = torch::empty({static_cast<int64_t>(config_.total_size_bytes)},
                                       torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
        if (shouldPinHostBlockPool()) {
            try {
                cache_aligned_buffer_ = cpu_buffer.pin_memory();
            } catch (const std::exception& e) {
                RTP_LLM_LOG_WARNING("pin host block pool failed, fallback to pageable CPU memory, pool_name=%s "
                                    "total_size=%zu bytes, error=%s",
                                    config_.pool_name.c_str(),
                                    config_.total_size_bytes,
                                    e.what());
                cache_aligned_buffer_ = std::move(cpu_buffer);
            }
        } else {
            RTP_LLM_LOG_INFO("host block pool uses pageable CPU memory, pool_name=%s total_size=%zu bytes",
                             config_.pool_name.c_str(),
                             config_.total_size_bytes);
            cache_aligned_buffer_ = std::move(cpu_buffer);
        }
        RTP_LLM_LOG_INFO("mark host block pool dont dump, pool_name=%s ptr=%p, size=%zu",
                         config_.pool_name.c_str(),
                         cache_aligned_buffer_.data_ptr(),
                         config_.total_size_bytes);
        markHostBlockPoolDontDump(
            config_.pool_name.c_str(), cache_aligned_buffer_.data_ptr(), config_.total_size_bytes);
    } else if (use_pinned_cpu_backing_) {
        initializePinnedCpuBuffer("device block pool pinned CPU backing");
    } else if (use_device_malloc_backing_) {
        initializeDeviceMallocBuffer();
    } else {
        cache_aligned_buffer_ = torch::empty({static_cast<int64_t>(config_.total_size_bytes)},
                                             torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    }
    cache_base_ptr_ = cache_aligned_buffer_.data_ptr();
    RTP_LLM_CHECK_WITH_INFO(cache_base_ptr_ != nullptr, "block pool allocate cache aligned buffer is null");
    const bool              is_cuda     = cache_aligned_buffer_.is_cuda();
    const bool              is_pinned   = !is_cuda && cache_aligned_buffer_.is_pinned();
    static constexpr double kBytesPerMB = 1024.0 * 1024.0;
    RTP_LLM_LOG_INFO("BlockPool backing selected: pool_name=%s allocation_type=%s requested_backing=%s "
                     "actual_backing=%s is_cuda=%d is_pinned=%d ptr=%p total_size=%zu bytes total_size_mb=%.2f "
                     "block_num=%u memory_layouts=%zu",
                     config_.pool_name.c_str(),
                     allocationTypeName(allocation_type_),
                     requestedBackingName(allocation_type_, use_pinned_cpu_backing_, use_device_malloc_backing_),
                     memoryTypeName(where()),
                     is_cuda,
                     is_pinned,
                     cache_base_ptr_,
                     config_.total_size_bytes,
                     static_cast<double>(config_.total_size_bytes) / kBytesPerMB,
                     config_.block_num,
                     config_.memory_layouts.size());
}

void BlockPool::initializePinnedCpuBuffer(const char* log_context) {
    RTP_LLM_LOG_WARNING(
        "%s, pool_name=%s, total_size=%zu bytes", log_context, config_.pool_name.c_str(), config_.total_size_bytes);
    auto cpu_buffer = torch::empty({static_cast<int64_t>(config_.total_size_bytes)},
                                   torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    try {
        cache_aligned_buffer_ = cpu_buffer.pin_memory();
    } catch (const std::exception& e) {
        RTP_LLM_FAIL("%s pin failed, pool_name=%s total_size=%zu bytes, error=%s",
                     log_context,
                     config_.pool_name.c_str(),
                     config_.total_size_bytes,
                     e.what());
    }
}

void BlockPool::initializeDeviceMallocBuffer() {
#if USING_CUDA
    RTP_LLM_CHECK_WITH_INFO(allocation_type_ == AllocationType::DEVICE,
                            "raw device malloc block pool backing requires DEVICE allocation");
    RTP_LLM_CHECK_WITH_INFO(config_.total_size_bytes > 0,
                            "raw device malloc block pool total_size_bytes must be > 0");

    int  device_id  = -1;
    auto device_err = cudaGetDevice(&device_id);
    RTP_LLM_CHECK_WITH_INFO(device_err == cudaSuccess,
                            "cudaGetDevice failed before cudaMalloc block pool allocation, error=%s",
                            cudaGetErrorString(device_err));

    void*      ptr = nullptr;
    const auto err = cudaMalloc(&ptr, config_.total_size_bytes);
    RTP_LLM_CHECK_WITH_INFO(err == cudaSuccess,
                            "cudaMalloc block pool failed, pool_name=%s, total_size=%zu bytes, error=%s",
                            config_.pool_name.c_str(),
                            config_.total_size_bytes,
                            cudaGetErrorString(err));

    auto deleter = [device_id](void* p) {
        if (p == nullptr) {
            return;
        }
        int current_device = -1;
        if (cudaGetDevice(&current_device) == cudaSuccess && current_device != device_id) {
            (void)cudaSetDevice(device_id);
            (void)cudaFree(p);
            (void)cudaSetDevice(current_device);
            return;
        }
        (void)cudaFree(p);
    };
    cache_aligned_buffer_ =
        torch::from_blob(ptr,
                         {static_cast<int64_t>(config_.total_size_bytes)},
                         std::move(deleter),
                         torch::TensorOptions().dtype(torch::kUInt8).device(torch::Device(torch::kCUDA, device_id)));
    RTP_LLM_LOG_INFO("cudaMalloc block pool backing allocated, pool_name=%s, ptr=%p, total_size=%zu bytes, device=%d",
                     config_.pool_name.c_str(),
                     ptr,
                     config_.total_size_bytes,
                     device_id);
#elif USING_ROCM
    RTP_LLM_CHECK_WITH_INFO(allocation_type_ == AllocationType::DEVICE,
                            "raw device malloc block pool backing requires DEVICE allocation");
    RTP_LLM_CHECK_WITH_INFO(config_.total_size_bytes > 0,
                            "raw device malloc block pool total_size_bytes must be > 0");

    int  device_id  = -1;
    auto device_err = hipGetDevice(&device_id);
    RTP_LLM_CHECK_WITH_INFO(device_err == hipSuccess,
                            "hipGetDevice failed before hipMalloc block pool allocation, error=%s",
                            hipGetErrorString(device_err));

    void*      ptr = nullptr;
    const auto err = hipMalloc(&ptr, config_.total_size_bytes);
    RTP_LLM_CHECK_WITH_INFO(err == hipSuccess,
                            "hipMalloc block pool failed, pool_name=%s, total_size=%zu bytes, error=%s",
                            config_.pool_name.c_str(),
                            config_.total_size_bytes,
                            hipGetErrorString(err));

    auto deleter = [device_id](void* p) {
        if (p == nullptr) {
            return;
        }
        int current_device = -1;
        if (hipGetDevice(&current_device) == hipSuccess && current_device != device_id) {
            (void)hipSetDevice(device_id);
            (void)hipFree(p);
            (void)hipSetDevice(current_device);
            return;
        }
        (void)hipFree(p);
    };
    cache_aligned_buffer_ =
        torch::from_blob(ptr,
                         {static_cast<int64_t>(config_.total_size_bytes)},
                         std::move(deleter),
                         torch::TensorOptions().dtype(torch::kUInt8).device(torch::Device(torch::kCUDA, device_id)));
    RTP_LLM_LOG_INFO("hipMalloc block pool backing allocated, pool_name=%s, ptr=%p, total_size=%zu bytes, device=%d",
                     config_.pool_name.c_str(),
                     ptr,
                     config_.total_size_bytes,
                     device_id);
#else
    RTP_LLM_FAIL("raw device malloc block pool backing requires a CUDA or ROCm build, pool_name=%s",
                 config_.pool_name.c_str());
#endif
}

void BlockPool::initializeLayerMappings() {
    torch::Tensor full_tensor = cache_aligned_buffer_;

    size_t total_layers = 0;
    for (const auto& layout_cfg : config_.memory_layouts) {
        total_layers += static_cast<size_t>(layout_cfg.layer_num);
    }
    global_layer_to_local_.assign(total_layers, {-1, -1});
    global_layer_kv_tensors_.assign(total_layers, torch::Tensor());
    global_layer_kv_scale_tensors_.assign(total_layers, torch::Tensor());
}

void BlockPool::initializeLayoutStrategies() {
    layout_strategies_.resize(config_.memory_layouts.size());
    torch::Tensor full_tensor = cache_aligned_buffer_;

    size_t global_layer_begin = 0;
    for (size_t layout_idx = 0; layout_idx < config_.memory_layouts.size(); ++layout_idx) {
        processMemoryLayout(layout_idx, full_tensor, global_layer_begin);
        global_layer_begin += static_cast<size_t>(config_.memory_layouts[layout_idx].layer_num);
    }
}

void BlockPool::processMemoryLayout(size_t layout_idx, const torch::Tensor& full_tensor, size_t& global_layer_begin) {
    const auto& layout_cfg = config_.memory_layouts[layout_idx];

    // 创建 KV 缓存张量
    torch::Tensor kv_cache_tensor = createTensor(full_tensor,
                                                 static_cast<int64_t>(layout_cfg.kv_cache_offset_bytes),
                                                 static_cast<int64_t>(layout_cfg.kv_block_pool_size_bytes),
                                                 layout_idx,
                                                 "kv");
    // 创建缩放张量（如果需要）
    torch::Tensor kv_scale_tensor;
    if (layout_cfg.hasScale()) {
        kv_scale_tensor = createTensor(full_tensor,
                                       static_cast<int64_t>(layout_cfg.kv_scale_offset_bytes),
                                       static_cast<int64_t>(layout_cfg.kv_scale_pool_size_bytes),
                                       layout_idx,
                                       "kv_scale");
    }

    // 初始化内存布局策略
    initializeLayoutStrategy(layout_idx, layout_cfg, kv_cache_tensor, kv_scale_tensor);

    // 处理层张量映射
    processLayerTensors(layout_idx, layout_cfg, global_layer_begin);

    // 记录初始化信息
    RTP_LLM_LOG_INFO("MemoryLayout[%zu] initialized: pool_name=%s layer_num=%u block_num=%u kv_off=%zu kv_bytes=%zu "
                     "scale_off=%zu scale_bytes=%zu",
                     layout_idx,
                     config_.pool_name.c_str(),
                     layout_cfg.layer_num,
                     layout_cfg.block_num,
                     layout_cfg.kv_cache_offset_bytes,
                     layout_cfg.kv_block_pool_size_bytes,
                     layout_cfg.kv_scale_offset_bytes,
                     layout_cfg.kv_scale_pool_size_bytes);
}

torch::Tensor BlockPool::createTensor(
    const torch::Tensor& full_tensor, int64_t offset, int64_t size, size_t layout_idx, const std::string& tensor_type) {
    RTP_LLM_CHECK_WITH_INFO(offset >= 0 && size >= 0 && offset + size <= full_tensor.numel(),
                            "layout[%zu] %s tensor out of range: off=%ld bytes=%ld full=%ld",
                            layout_idx,
                            tensor_type.c_str(),
                            offset,
                            size,
                            full_tensor.numel());
    return full_tensor.narrow(0, offset, size);
}

void BlockPool::initializeLayoutStrategy(size_t                    layout_idx,
                                         const MemoryLayoutConfig& layout_cfg,
                                         torch::Tensor&            kv_cache_tensor,
                                         torch::Tensor&            kv_scale_tensor) {
    void* layout_cache_base_ptr =
        static_cast<void*>(static_cast<char*>(cache_base_ptr_) + layout_cfg.kv_cache_offset_bytes);

    layout_strategies_[layout_idx] = std::make_unique<MemoryLayoutStrategy>();
    RTP_LLM_CHECK_WITH_INFO(layout_strategies_[layout_idx] != nullptr,
                            "Failed to create memory layout strategy for layout[%zu]",
                            layout_idx);

    RTP_LLM_CHECK_WITH_INFO(
        layout_strategies_[layout_idx]->init(layout_cfg, kv_cache_tensor, kv_scale_tensor, layout_cache_base_ptr),
        "Failed to initialize memory layout strategy for layout[%zu]",
        layout_idx);
}

void BlockPool::processLayerTensors(size_t                    layout_idx,
                                    const MemoryLayoutConfig& layout_cfg,
                                    size_t&                   global_layer_begin) {
    // 获取层张量
    auto layer_tensors = layout_strategies_[layout_idx]->getLayerCacheTensors();
    RTP_LLM_CHECK_WITH_INFO(layer_tensors.size() == static_cast<size_t>(layout_cfg.layer_num),
                            "layout[%zu] layer tensors size mismatch: got=%zu expect=%u",
                            layout_idx,
                            layer_tensors.size(),
                            layout_cfg.layer_num);

    // 映射全局层到局部层，并设置KV张量
    for (size_t local_layer = 0; local_layer < static_cast<size_t>(layout_cfg.layer_num); ++local_layer) {
        const size_t global_layer = global_layer_begin + local_layer;
        RTP_LLM_CHECK_WITH_INFO(global_layer < global_layer_to_local_.size(), "global layer index out of range");
        global_layer_to_local_[global_layer]   = {static_cast<int>(layout_idx), static_cast<int>(local_layer)};
        global_layer_kv_tensors_[global_layer] = layer_tensors[local_layer];
    }

    // 处理缩放张量（如果存在）
    auto scale_tensors = layout_strategies_[layout_idx]->getLayerScaleCacheTensors();
    if (!scale_tensors.empty()) {
        RTP_LLM_CHECK_WITH_INFO(scale_tensors.size() == static_cast<size_t>(layout_cfg.layer_num),
                                "layout[%zu] scale tensors size mismatch: got=%zu expect=%u",
                                layout_idx,
                                scale_tensors.size(),
                                layout_cfg.layer_num);
        for (size_t local_layer = 0; local_layer < static_cast<size_t>(layout_cfg.layer_num); ++local_layer) {
            const size_t global_layer                    = global_layer_begin + local_layer;
            global_layer_kv_scale_tensors_[global_layer] = scale_tensors[local_layer];
        }
    }
}

bool BlockPool::init() {
    validateConfig();
    initializeCacheBuffer();
    initializeLayerMappings();
    initializeLayoutStrategies();
    initFreeBlocks();

    RTP_LLM_LOG_INFO("BlockPool init success: pool_name=%s memory_layouts=%zu, total_layers=%zu, total_size=%zu bytes",
                     config_.pool_name.c_str(),
                     config_.memory_layouts.size(),
                     global_layer_to_local_.size(),
                     config_.total_size_bytes);
    return true;
}

void BlockPool::initFreeBlocks() {
    // block 0 is reserved
    for (BlockIdxType i = 1; i < static_cast<BlockIdxType>(config_.block_num); ++i) {
        free_block_ids_.insert(i);
    }
    request_ref_counter_.init(config_.block_num);
    connector_ref_counter_.init(config_.block_num);
    req_con_ref_counter_.init(config_.block_num);
    block_cache_ref_counter_.init(config_.block_num);
    req_cache_ref_counter_.init(config_.block_num);
}

std::vector<torch::Tensor> BlockPool::allLayerCacheBase() const {
    return global_layer_kv_tensors_;
}

std::vector<torch::Tensor> BlockPool::allLayerScaleCacheBase() const {
    return global_layer_kv_scale_tensors_;
}

BlockIndicesType BlockPool::malloc(int num_blocks) {
    RTP_LLM_PROFILE_FUNCTION();
    if (num_blocks <= 0) {
        return {};
    }
    BlockIndicesType block_ids;
    block_ids.reserve(num_blocks);

    {
        std::scoped_lock lock(ref_mu_, free_mu_);
        if (free_block_ids_.size() < static_cast<size_t>(num_blocks)) {
            RTP_LLM_LOG_WARNING("Block pool only has %zu free blocks, cannot allocate %d blocks, pool_name=%s",
                                free_block_ids_.size(),
                                num_blocks,
                                config_.pool_name.c_str());
            return {};
        }
        auto first = free_block_ids_.begin();
        auto last  = std::next(first, num_blocks);
        block_ids.assign(first, last);
        free_block_ids_.erase(first, last);
        request_ref_counter_.incrementRefCounter(block_ids);
        req_con_ref_counter_.incrementRefCounter(block_ids);
        req_cache_ref_counter_.incrementRefCounter(block_ids);
    }
    g_probe_malloc_total.fetch_add(block_ids.size());
    probeRecordBlocks(block_ids);
    probeLedgerDump(config_.pool_name.c_str(), "malloc");

    return block_ids;
}

void BlockPool::requestFree(BlockIdxType block_idx) {
    auto block_ids = {block_idx};
    requestFree(block_ids);
}

void BlockPool::requestFree(const BlockIndicesType& block_ids) {
    RTP_LLM_PROFILE_FUNCTION();
    std::scoped_lock lock(ref_mu_, free_mu_);
    request_ref_counter_.decrementRefCounter(block_ids);
    req_con_ref_counter_.decrementRefCounter(block_ids);
    req_cache_ref_counter_.decrementRefCounter(block_ids);
    tryFreeBlocks(block_ids);
    g_probe_reqfree_total.fetch_add(block_ids.size());
    probeEraseBlocks(block_ids);
    probeLedgerDump(config_.pool_name.c_str(), "requestFree");
}

void BlockPool::connectorFree(BlockIdxType block_idx) {
    auto block_ids = {block_idx};
    connectorFree(block_ids);
}

void BlockPool::connectorFree(const BlockIndicesType& block_indices) {
    RTP_LLM_PROFILE_FUNCTION();
    std::scoped_lock lock(ref_mu_, free_mu_);
    connector_ref_counter_.decrementRefCounter(block_indices);
    req_con_ref_counter_.decrementRefCounter(block_indices);
    tryFreeBlocks(block_indices);
}

void BlockPool::blockCacheFree(BlockIdxType block_idx) {
    auto block_ids = {block_idx};
    blockCacheFree(block_ids);
}

void BlockPool::blockCacheFree(const BlockIndicesType& block_ids) {
    RTP_LLM_PROFILE_FUNCTION();
    std::scoped_lock lock(ref_mu_, free_mu_);
    block_cache_ref_counter_.decrementRefCounter(block_ids);
    req_cache_ref_counter_.decrementRefCounter(block_ids);
    tryFreeBlocks(block_ids);
}

// Must be called with ref_mu_ and free_mu_ held.
void BlockPool::tryFreeBlocks(const BlockIndicesType& block_ids) {
    RTP_LLM_PROFILE_FUNCTION();
    for (const auto& block_id : block_ids) {
        if (req_con_ref_counter_.getRefCounter(block_id) == 0
            && block_cache_ref_counter_.getRefCounter(block_id) == 0) {
            free_block_ids_.insert(block_id);
        }
    }
}

void BlockPool::requestReference(BlockIdxType block_idx) {
    BlockIndicesType block_ids = {block_idx};
    requestReference(block_ids);
}

void BlockPool::requestReference(const BlockIndicesType& block_ids) {
    RTP_LLM_PROFILE_FUNCTION();
    std::scoped_lock lock(ref_mu_, free_mu_);
    request_ref_counter_.incrementRefCounter(block_ids);
    req_con_ref_counter_.incrementRefCounter(block_ids);
    req_cache_ref_counter_.incrementRefCounter(block_ids);
    for (const auto& block_id : block_ids) {
        free_block_ids_.erase(block_id);
    }
}

void BlockPool::connectorReference(BlockIdxType block_idx) {
    BlockIndicesType block_ids = {block_idx};
    connectorReference(block_ids);
}

void BlockPool::connectorReference(const BlockIndicesType& block_indices) {
    RTP_LLM_PROFILE_FUNCTION();
    std::scoped_lock lock(ref_mu_, free_mu_);
    connector_ref_counter_.incrementRefCounter(block_indices);
    req_con_ref_counter_.incrementRefCounter(block_indices);
    for (const auto& block_id : block_indices) {
        free_block_ids_.erase(block_id);
    }
}

void BlockPool::blockCacheReference(BlockIdxType block_idx) {
    BlockIndicesType block_ids = {block_idx};
    blockCacheReference(block_ids);
}

void BlockPool::blockCacheReference(const BlockIndicesType& block_ids) {
    RTP_LLM_PROFILE_FUNCTION();
    std::scoped_lock lock(ref_mu_, free_mu_);
    block_cache_ref_counter_.incrementRefCounter(block_ids);
    req_cache_ref_counter_.incrementRefCounter(block_ids);
    for (const auto& block_id : block_ids) {
        free_block_ids_.erase(block_id);
    }
}

void BlockPool::regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store) {
    if (cache_store) {
        cache_store_ = std::move(cache_store);
    }
    if (cache_store_ && !kvcache_reg_mr_) {
        RTP_LLM_LOG_INFO("start to register user mr, pool_name=%s", config_.pool_name.c_str());
        auto       memory_util = cache_store_->getMemoryUtil();
        const bool gpu         = where() == MemoryType::MEMORY_GPU;

        for (size_t layout_idx = 0; layout_idx < config_.memory_layouts.size(); ++layout_idx) {
            const auto& layout_cfg = config_.memory_layouts[layout_idx];

            // Register KV buffer
            registerUserMrForBuffer(memory_util,
                                    layout_idx,
                                    layout_cfg.kv_cache_offset_bytes,
                                    layout_cfg.kv_block_pool_size_bytes,
                                    layout_cfg.kv_block_stride_bytes,
                                    gpu,
                                    "kv");

            // Register scale buffer if present
            if (layout_cfg.hasScale()) {
                registerUserMrForBuffer(memory_util,
                                        layout_idx,
                                        layout_cfg.kv_scale_offset_bytes,
                                        layout_cfg.kv_scale_pool_size_bytes,
                                        layout_cfg.kv_scale_stride_bytes,
                                        gpu,
                                        "scale");
            }
        }

        kvcache_reg_mr_ = true;
    }
}

void BlockPool::deregUserMr() {
    if (kvcache_reg_mr_ && cache_store_) {
        RTP_LLM_LOG_INFO("start to deregister user mr, pool_name=%s", config_.pool_name.c_str());
        auto       memory_util = cache_store_->getMemoryUtil();
        const bool gpu         = where() == MemoryType::MEMORY_GPU;

        for (size_t layout_idx = 0; layout_idx < config_.memory_layouts.size(); ++layout_idx) {
            const auto& layout_cfg = config_.memory_layouts[layout_idx];

            // Deregister KV buffer
            deregisterUserMrForBuffer(memory_util, layout_idx, layout_cfg.kv_cache_offset_bytes, gpu, "kv");

            // Deregister scale buffer if present
            if (layout_cfg.hasScale()) {
                deregisterUserMrForBuffer(memory_util, layout_idx, layout_cfg.kv_scale_offset_bytes, gpu, "scale");
            }
        }

        RTP_LLM_LOG_INFO("deregister user mr for block pool success, pool_name=%s", config_.pool_name.c_str());
        kvcache_reg_mr_ = false;
    }
}

void BlockPool::registerUserMrForBuffer(std::shared_ptr<rtp_llm::MemoryUtil> memory_util,
                                        size_t                               layout_idx,
                                        size_t                               offset_bytes,
                                        size_t                               bytes,
                                        size_t                               stride_bytes,
                                        bool                                 gpu,
                                        const std::string&                   buffer_type) {
    void* base_ptr = static_cast<void*>(static_cast<char*>(cache_base_ptr_) + static_cast<ptrdiff_t>(offset_bytes));
    auto  start_us = currentTimeUs();

    if (!memory_util->regUserMr(base_ptr, bytes, gpu, stride_bytes)) {
        RTP_LLM_FAIL("register user mr for block pool layout[%zu] %s buffer failed, pool_name=%s",
                     layout_idx,
                     buffer_type.c_str(),
                     config_.pool_name.c_str());
    }

    auto cost_ms = (currentTimeUs() - start_us) / 1000;
    mr_cost_time_ms_ += cost_ms;

    RTP_LLM_LOG_INFO("register user mr success: pool_name=%s layout[%zu] %s base=%p len=%zu aligned=%zu cost=%ld ms",
                     config_.pool_name.c_str(),
                     layout_idx,
                     buffer_type.c_str(),
                     base_ptr,
                     bytes,
                     stride_bytes,
                     cost_ms);
}

void BlockPool::deregisterUserMrForBuffer(std::shared_ptr<rtp_llm::MemoryUtil> memory_util,
                                          size_t                               layout_idx,
                                          size_t                               offset_bytes,
                                          bool                                 gpu,
                                          const std::string&                   buffer_type) {
    void* base_ptr = static_cast<void*>(static_cast<char*>(cache_base_ptr_) + static_cast<ptrdiff_t>(offset_bytes));

    if (!memory_util->deregUserMr(base_ptr, gpu)) {
        RTP_LLM_FAIL("deregister user mr for block pool layout[%zu] %s buffer failed, pool_name=%s",
                     layout_idx,
                     buffer_type.c_str(),
                     config_.pool_name.c_str());
    }
}

size_t BlockPool::freeBlocksNum() const {
    std::lock_guard<std::mutex> free_lock(free_mu_);
    return free_block_ids_.size();
}

size_t BlockPool::totalBlocksNum() const {
    // reserve block 0 for internal use
    return config_.block_num - 1;
}

// Available blocks need to satisfy two conditions:
// 1. not referenced by a request
// 2. not referenced by connector(read or write)
size_t BlockPool::availableBlocksNum() const {
    std::lock_guard<std::mutex> lock(ref_mu_);
    return req_con_ref_counter_.freeBlockNum();
}

size_t BlockPool::requestRefBlocksNum() const {
    std::lock_guard<std::mutex> lock(ref_mu_);
    return request_ref_counter_.busyBlockNum();
}

size_t BlockPool::connectorRefBlocksNum() const {
    std::lock_guard<std::mutex> lock(ref_mu_);
    return connector_ref_counter_.busyBlockNum();
}

size_t BlockPool::blockCacheRefBlocksNum() const {
    std::lock_guard<std::mutex> lock(ref_mu_);
    return block_cache_ref_counter_.busyBlockNum();
}

size_t BlockPool::notInUseBlocksNum() const {
    std::lock_guard<std::mutex> lock(ref_mu_);
    return req_cache_ref_counter_.freeBlockNum();
}

// MTP support: Map global_layer_id to (model_index, local_layer_id).
// Returns {layout_index, local_layer_id}. layout_index is the index in BlockPoolConfig.memory_layouts.
std::pair<int, int> BlockPool::mapGlobalLayerIdToLocal(int global_layer_id) const {
    if (global_layer_id < 0 || static_cast<size_t>(global_layer_id) >= global_layer_to_local_.size()) {
        RTP_LLM_LOG_ERROR("Global layer_id %d out of range (total layers: %zu), pool_name=%s",
                          global_layer_id,
                          global_layer_to_local_.size(),
                          config_.pool_name.c_str());
        return {-1, -1};
    }

    return global_layer_to_local_[static_cast<size_t>(global_layer_id)];
}

BlockAddrInfo BlockPool::convertIndexToAddr(int layer_id, int block_id) const {
    auto [layout_index, local_layer_id] = mapGlobalLayerIdToLocal(layer_id);
    checkLayoutValidity(layout_index);
    return layout_strategies_[static_cast<size_t>(layout_index)]->convertIndexToAddr(local_layer_id, block_id);
}

std::vector<BlockInfo> BlockPool::convertIndexToBuffer(int layer_id, int block_id) const {
    auto [layout_index, local_layer_id] = mapGlobalLayerIdToLocal(layer_id);
    checkLayoutValidity(layout_index);
    return layout_strategies_[static_cast<size_t>(layout_index)]->convertIndexToBuffer(local_layer_id, block_id);
}

std::vector<BlockInfo>
BlockPool::convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const {
    auto [layout_index, local_layer_id] = mapGlobalLayerIdToLocal(layer_id);
    checkLayoutValidity(layout_index);

    return layout_strategies_[static_cast<size_t>(layout_index)]->convertIndexToBuffer(
        local_layer_id, block_id, partition_count, partition_id);
}

MemoryType BlockPool::where() const {
    if (cache_aligned_buffer_.is_cuda()) {
        return MemoryType::MEMORY_GPU;
    }
    return cache_aligned_buffer_.is_pinned() ? MemoryType::MEMORY_CPU_PINNED : MemoryType::MEMORY_CPU;
}

void BlockPool::checkLayoutValidity(int layout_id) const {
    RTP_LLM_CHECK_WITH_INFO(layout_id >= 0 && static_cast<size_t>(layout_id) < layout_strategies_.size(),
                            "Memory layout ID %d out of range (max: %zu)",
                            layout_id,
                            layout_strategies_.size());
}

}  // namespace rtp_llm
