#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include "rtp_llm/cpp/devices/CommonDefines.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/kernels/unfused_attention_kernels.h"
#include "rtp_llm/cpp/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention.h"
#include "rtp_llm/cpp/kernels/kv_cache_kernels.h"
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#include "rtp_llm/cpp/rocm/hip_host_utils.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/rocm_impl/aiterPA.h"
#include "rtp_llm/cpp/config/StaticConfig.h"
#include "rtp_llm/cpp/utils/RopeCosSin.h"
#include <filesystem>

#include "ck_tile/host.hpp"
#include "torch/mha_fwd.h"
// ===== helper: 形状/stride 打印（纯 std::cout） =====
#include <iostream>
#include <sstream>
#include <vector>
#include <torch/torch.h>  // 或 #include <ATen/ATen.h>

#include <cstdint>
#include <iostream>
#include <sstream>

namespace {
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <torch/torch.h>
#include <hip/hip_runtime.h>
#include <torch/torch.h>
#include <utility>

// per-tensor 量化：BF16 → FP8(e4m3fnuz)
inline std::pair<torch::Tensor, torch::Tensor>
quantize_bf16_to_fp8(const torch::Tensor& x_bf16,
                     c10::optional<torch::Tensor> scale_opt = c10::nullopt)
{
    auto x32 = x_bf16.to(torch::kFloat32);
    torch::Tensor scale;

    constexpr double kFp8Max = 448.0;

    if (scale_opt.has_value()) {
        scale = scale_opt.value().to(torch::kFloat32).view({1});
    } else {
        auto amax = x32.abs().amax();
        scale = (amax / kFp8Max).clamp_min(1e-12).view({1});
    }

    auto y = x32 / scale;
    auto q = y.to(at::ScalarType::Float8_e4m3fnuz);
    return {q, scale};
}

// 反量化：FP8 → FP32
inline torch::Tensor
dequantize_fp8_to_fp32(const torch::Tensor& q_fp8,
                       const torch::Tensor& scale)
{
    return q_fp8.to(torch::kFloat32) * scale.to(torch::kFloat32);
}


// 把 pickled 的 CPU Tensor（一般是 uint8）加载出来
torch::Tensor loadTensorFromFile(const std::string& fileName) {
    std::ifstream fin(fileName, std::ios::in | std::ios::binary);
    if (!fin) throw std::runtime_error("Failed to open file: " + fileName);
    fin.seekg(0, std::ios::end); size_t fileSize = fin.tellg(); fin.seekg(0, std::ios::beg);
    std::vector<char> buf(fileSize);
    fin.read(buf.data(), fileSize);
    return torch::pickle_load(buf).toTensor();
}

// 将 CPU uint8/FP8 字节装入 CUDA 上的 FP8(E4M3FNUZ) Tensor（按给定形状）
torch::Tensor toCudaFp8FromCpuBytes(const torch::Tensor& cpu_u8_like,
                                    c10::IntArrayRef sizes,
                                    c10::Device dev = c10::Device(torch::kCUDA, 0)) {
    TORCH_CHECK(cpu_u8_like.device().is_cpu(), "expect CPU tensor");
    TORCH_CHECK(cpu_u8_like.scalar_type() == torch::kByte ||
                cpu_u8_like.scalar_type() == torch::kFloat8_e4m3fnuz ||
                cpu_u8_like.scalar_type() == torch::kFloat8_e4m3fn,
                "expect uint8 or fp8 tensor in file");

    auto contiguous = cpu_u8_like.contiguous();
    const size_t bytes = contiguous.nbytes();

    auto opts = torch::TensorOptions()
                    .dtype(torch::kFloat8_e4m3fnuz)
                    .device(dev);
    auto t_cuda = torch::empty(sizes, opts);                      // 目标 dtype=fp8_e4m3fnuz
    TORCH_CHECK((size_t)t_cuda.nbytes() == bytes, "size mismatch");

    // 直接按字节搬运（不做数值变换）
    hipError_t e = hipMemcpy(t_cuda.data_ptr(), contiguous.data_ptr(), bytes, hipMemcpyHostToDevice);
    TORCH_CHECK(e == hipSuccess, "hipMemcpy H2D failed");

    return t_cuda;
}


// 小工具：局部保存/恢复 cout 的格式状态，防止污染外界
struct IosFmtGuard {
    explicit IosFmtGuard(std::ostream& os) : os_(os) { saved_.copyfmt(os_); }
    ~IosFmtGuard() { os_.copyfmt(saved_); }
private:
    std::ostream& os_;
    std::ios saved_{nullptr};
};
// 可选：简单 dtype 名称（修正：去掉 .c_str()，直接返回 c10::toString(t)）
inline const char* dtype_name(c10::ScalarType t) {
    using T = c10::ScalarType;
    switch (t) {
        case T::Float:     return "float32";
        case T::Half:      return "float16";
        case T::BFloat16:  return "bfloat16";
#if TORCH_VERSION_MAJOR >= 2
#  ifdef C10_SCALAR_TYPE_FLOAT8_E4M3FN
        case T::Float8_e4m3fn:     return "float8_e4m3fn";
#  endif
#  ifdef C10_SCALAR_TYPE_FLOAT8_E4M3FNUZ
        case T::Float8_e4m3fnuz:   return "float8_e4m3fnuz";
#  endif
#  ifdef C10_SCALAR_TYPE_FLOAT8_E5M2
        case T::Float8_e5m2:       return "float8_e5m2";
#  endif
#  ifdef C10_SCALAR_TYPE_FLOAT8_E5M2FNUZ
        case T::Float8_e5m2fnuz:   return "float8_e5m2fnuz";
#  endif
#endif
        case T::Byte:      return "uint8";
        default:           return c10::toString(t);  // 注意：不要再 .c_str()
    }
}


// 打印张量元信息 + 前 n 个值（若能安全转 float32 则打印为 f32），并打印 raw u8 头部
inline void debugTensor(const torch::Tensor& t, const std::string& tag = "T", int64_t n = 20) {
    try {
        IosFmtGuard guard(std::cout);                // 保护外部 cout 状态
        std::cout << std::defaultfloat << std::setprecision(7);  // 局部打印风格

        // 元信息
        std::cout << "[" << tag << "] shape=" << t.sizes()
                  << " stride=" << t.strides()
                  << " dtype=" << dtype_name(t.scalar_type())
                  << " device=" << t.device()
                  << " contig=" << (t.is_contiguous() ? 1 : 0)
                  << " storage_off=" << t.storage_offset()
                  << " data=" << t.data_ptr()
                  << "\n";

        if (t.numel() == 0) {
            std::cout << "[" << tag << " head 0] (empty tensor)\n";
            return;
        }

        // 统一搬到 CPU 并 contiguous
        auto cpu = t.detach();
        if (!cpu.is_cpu()) cpu = cpu.to(torch::kCPU, /*non_blocking=*/true, /*copy=*/true);
        if (!cpu.is_contiguous()) cpu = cpu.contiguous();

        // 打印原始字节（便于与 Python 对齐）
        {
            auto bytes = cpu.view(torch::kUInt8);
            const auto* bptr = bytes.data_ptr<uint8_t>();
            const int64_t bk = std::min<int64_t>(n, bytes.numel());
            std::cout << "[" << tag << " raw u8 head " << bk << "]: ";
            for (int64_t i = 0; i < bk; ++i) {
                std::cout << int(bptr[i]) << (i + 1 == bk ? "" : " ");
            }
            std::cout << "\n";
        }

        // 尝试转成 float32 打印（支持 fp32/fp16/bf16/fp8，包括 e4m3fnuz）
        bool printed_as_float = false;
        try {
            auto as_f32 = cpu.to(torch::kFloat32);
            const float* ptr = as_f32.data_ptr<float>();
            const int64_t total = as_f32.numel();
            const int64_t k = std::min<int64_t>(n, total);

            float mn = std::numeric_limits<float>::infinity();
            float mx = -std::numeric_limits<float>::infinity();
            double sum = 0.0;

            std::cout << "[" << tag << " first " << k << " as f32]: ";
            for (int64_t i = 0; i < k; ++i) {
                std::cout << ptr[i] << (i + 1 == k ? "" : " ");
                if (std::isfinite(ptr[i])) {
                    mn = std::min(mn, ptr[i]);
                    mx = std::max(mx, ptr[i]);
                    sum += ptr[i];
                }
            }
            std::cout << "\n";
            std::cout << "[" << tag << " stats(f32 on head)]: "
                      << "min=" << mn << " max=" << mx
                      << " mean=" << (k > 0 ? float(sum / double(k)) : 0.0f)
                      << " (printed=" << k << "/" << total << ")\n";

            printed_as_float = true;
        } catch (...) {
            printed_as_float = false;
        }

        // 若无法安全转 float，则退回打印原始字节段（已打印过 u8 头部，这里不用重复）
        if (!printed_as_float) {
            const size_t bytes = cpu.nbytes();
            const size_t k = static_cast<size_t>(std::min<int64_t>(n * cpu.element_size(), bytes));
            const uint8_t* p = reinterpret_cast<const uint8_t*>(cpu.data_ptr());
            std::cout << "[" << tag << " raw bytes head " << k << "B]: ";
            for (size_t i = 0; i < k; ++i) {
                // 用十六进制显示原始字节
                std::cout << std::hex << std::setw(2) << std::setfill('0') << (unsigned)(p[i])
                          << (i + 1 == k ? "" : " ");
            }
            std::cout << std::dec << "\n";
        }
    } catch (const std::exception& e) {
        // 不要抛到外层，避免影响调用方
        IosFmtGuard guard(std::cout);
        std::cout << "[debugTensor] exception: " << e.what() << "\n";
    }
}

inline const char* to_cstr(rtp_llm::MemoryType m) {
    using rtp_llm::MemoryType;
    switch (m) {
        case MemoryType::MEMORY_CPU: return "CPU";
        case MemoryType::MEMORY_GPU: return "GPU";
        default: return "Unknown";
    }
}

inline const char* to_cstr(rtp_llm::DataType t) {
    using rtp_llm::DataType;
    switch (t) {
        case DataType::TYPE_FP32:       return "fp32";
        case DataType::TYPE_FP16:       return "fp16";
        case DataType::TYPE_BF16:       return "bf16";
        case DataType::TYPE_FP64:       return "fp64";
        case DataType::TYPE_INT8:       return "int8";
        case DataType::TYPE_INT32:      return "int32";
        case DataType::TYPE_QINT8:      return "qint8";
        case DataType::TYPE_QINT4X2:    return "qint4x2";
        case DataType::TYPE_QFP8_E4M3:  return "qfp8_e4m3";
        // 需要的话补齐你项目里的其它枚举
        default: return "other";
    }
}

inline std::string vec_to_str(const std::vector<size_t>& v) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) oss << ", ";
        oss << v[i];
    }
    oss << "]";
    return oss.str();
}

inline std::string vec_to_str(const std::vector<int64_t>& v) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) oss << ", ";
        oss << v[i];
    }
    oss << "]";
    return oss.str();
}

inline void printBufferMeta(const char* tag, const rtp_llm::Buffer& buf) {
    std::cout << "[BUF] " << tag
              << " where="    << to_cstr(buf.where())
              << " dtype="    << to_cstr(buf.type())
              << " shape="    << vec_to_str(buf.shape())
              << " strides="  << vec_to_str(buf.strides())
              << " elems="    << buf.size()
              << " bytes="    << buf.sizeBytes()
              << " dim="      << buf.dim()
              << " data="     << buf.data()
              << std::endl;
}

// 适配 shared_ptr
inline void printBufferMeta(const char* tag, const rtp_llm::BufferPtr& p) {
    if (p) printBufferMeta(tag, *p);
    else   std::cout << "[BUF] " << tag << " = nullptr\n";
}

inline void printBufferMeta(const char* tag, const rtp_llm::ConstBufferPtr& p) {
    if (p) printBufferMeta(tag, *p);
    else   std::cout << "[BUF] " << tag << " = nullptr\n";
}

} // anonymous namespace

// 把任意“1 字节/元素”的张量（fp8/uint8/int8）按 uint8 视图导出；
// 返回的是 clone 后的独立存储，安全可保存。
static inline torch::Tensor as_uint8_bytes(const torch::Tensor& t) {
    TORCH_CHECK(t.defined(), "as_uint8_bytes: tensor is undefined");
    TORCH_CHECK(
        t.element_size() == 1, "as_uint8_bytes expects 1-byte-per-element tensor, got element_size=", t.element_size());

    auto tc = t.contiguous();

    // 注意：这里返回的是 const void*
    const void*  base_ptr    = tc.storage().unsafeGetStorageImpl()->data();
    const size_t byte_offset = static_cast<size_t>(tc.storage_offset()) * tc.element_size();

    // 计算实际首地址，并用 const_cast<void*> 交给 from_blob（我们随后会 clone()）
    void* raw_ptr = const_cast<void*>(static_cast<const void*>(static_cast<const uint8_t*>(base_ptr) + byte_offset));

    auto alias_u8 = torch::from_blob(
        raw_ptr,
        tc.sizes(),
        tc.strides(),
        [](void*) {},  // 不接管内存
        tc.options().dtype(torch::kUInt8));

    return alias_u8.clone().contiguous();  // 独立存储，便于保存/跨进程读取
}

// 把 uint8 字节张量按 FP8 视图“读回”；返回 clone 后的独立存储。
// fp8_dtype 请选择你工程实际使用的 FP8 类型（kFloat8_e4m3fnuz 或 kFloat8_e4m3fn）
static inline torch::Tensor bytes_to_fp8_e4m3(const torch::Tensor& bytes_u8,
                                                    c10::ScalarType      fp8_dtype = torch::kFloat8_e4m3fnuz) {
    TORCH_CHECK(bytes_u8.dtype() == torch::kUInt8, "bytes_to_fp8_view_clone expects uint8 input");
    auto tc = bytes_u8.contiguous();

    const void*  base_ptr    = tc.storage().unsafeGetStorageImpl()->data();
    const size_t byte_offset = static_cast<size_t>(tc.storage_offset()) * tc.element_size();
    void* raw_ptr = const_cast<void*>(static_cast<const void*>(static_cast<const uint8_t*>(base_ptr) + byte_offset));

    auto alias_fp8 = torch::from_blob(
        raw_ptr,
        tc.sizes(),
        tc.strides(),
        [](void*) {},  // 不接管内存
        tc.options().dtype(fp8_dtype));

    return alias_fp8.clone().contiguous();  // 独立存储，后续使用更安全
}

using namespace std;
namespace fs = std::filesystem;
namespace rtp_llm {
// ---- debug helpers (header/top of file) ----
static inline const char* dtype_name(at::ScalarType t){
    switch(t){
        case at::kByte: return "uint8";
        case at::kChar: return "int8";
        case at::kHalf: return "fp16";
        case at::kBFloat16: return "bf16";
        case at::kFloat: return "fp32";
#ifdef TORCH_CHECKTYPE_SUPPORTS_FLOAT8
        case at::kFloat8_e4m3fn:   return "fp8_e4m3fn";
        case at::kFloat8_e4m3fnuz: return "fp8_e4m3fnuz";
#endif
        default: return "other";
    }
}

static inline void dump_u8_head(const torch::Tensor& t, int n = 32, const char* tag = ""){
    try{
        auto a = t.contiguous().flatten().to(torch::kUInt8).cpu();
        auto p = a.data_ptr<uint8_t>();
        const int m = std::min<int64_t>(n, a.numel());
        std::ostringstream oss;
        oss << "[" << tag << " u8 head " << m << "B] ";
        for(int i=0;i<m;++i){
            oss << std::hex << std::setw(2) << std::setfill('0') << int(p[i]) << " ";
        }
        std::cout << oss.str() << std::dec << std::endl;
    }catch(...){
        std::cout << "[" << tag << "] dump_u8_head failed" << std::endl;
    }
}

static inline void log_tensor_meta(const char* name, const torch::Tensor& t){
    std::cout << "[T] " << name
              << " shape=" << t.sizes()
              << " stride=" << t.strides()
              << " dtype=" << dtype_name(t.scalar_type())
              << " device=" << t.device()
              << " contig=" << (t.is_contiguous() ? 1 : 0)
              << " storage_off=" << t.storage_offset()
              << " data=" << (const void*)t.data_ptr()
              << std::endl;
}

static inline void log_tensor_stats_f32(const char* name, const torch::Tensor& t){
    try{
        auto f = t.detach().to(torch::kFloat32).flatten().cpu();
        if(f.numel()==0){ std::cout << "[S] " << name << " empty\n"; return; }
        auto p = f.data_ptr<float>();
        float mn = std::numeric_limits<float>::infinity();
        float mx = -std::numeric_limits<float>::infinity();
        double sum = 0.0; size_t n_nan = 0, n_inf = 0;
        const int64_t N = f.numel();
        const int64_t M = std::min<int64_t>(N, 1'000'000); // cap for speed
        for(int64_t i=0;i<M;++i){
            float v = p[i];
            if(std::isnan(v)) { n_nan++; continue; }
            if(std::isinf(v)) { n_inf++; continue; }
            mn = std::min(mn, v); mx = std::max(mx, v); sum += v;
        }
        std::cout << "[S] " << name << " N=" << N
                  << " sample=" << M
                  << " min=" << mn << " max=" << mx
                  << " mean=" << (sum / std::max<int64_t>(1, (M - n_nan - n_inf)))
                  << " nan=" << n_nan << " inf=" << n_inf << std::endl;
    }catch(...){
        std::cout << "[S] " << name << " stats failed\n";
    }
}

// 统计 FP8(e4m3) 字节里 exponent==0xF 的数量（NaN/Inf 候选）
auto count_fp8_expF = [](const torch::Tensor& t, const char* tag){
    auto u8 = t.view({-1}).to(torch::kUInt8);
    auto cpu = u8.cpu();
    const uint8_t* p = cpu.data_ptr<uint8_t>();
    int64_t N = cpu.numel();
    int64_t expF = 0, expF_mantNonZero = 0;
    for (int64_t i=0;i<N;++i){
        uint8_t b = p[i];
        uint8_t exp = (b >> 3) & 0x1F; // e4m3: 高 5 位是 exponent；若实现是 e4m3(4bit exp)，这里按具体格式改一下
        uint8_t mant = b & 0x07;
        if (exp == 0x0F) { expF++; if (mant) expF_mantNonZero++; }
    }
    std::cout << "[FP8 expF] " << tag << " total="<<N
              << " expF="<<expF<<" expF&mant!=0="<<expF_mantNonZero << std::endl;
};

// #define DEBUG_PRINT_PARAMS(...) printParams(__VA_ARGS__)
#define DEBUG_PRINT_PARAMS(...)                                                                                        \
    do {                                                                                                               \
    } while (0)

void printParams(const AttentionModuleParams& params,
                 ROCmDevice*                  device,
                 const std::string&           prefix,
                 BufferPtr                    sliceQ = nullptr) {
    if (params.common.kv_cache && params.common.kv_cache->kv_cache_block_id) {
        auto kv_cache_block_id_host = device->clone({*params.common.kv_cache->kv_cache_block_id, AllocationType::HOST});

        auto getUniqueDumpDir = [](const std::string& root, const std::string& prefix) -> std::string {
            int count = 0;
            while (true) {
                std::ostringstream oss;
                oss << root << "/" << prefix << "_" << count;
                std::string path = oss.str();
                if (!fs::exists(path)) {
                    fs::create_directories(path);
                    return path;
                }
                ++count;
            }
        };

        std::string dump_dir = getUniqueDumpDir("attn", prefix);

        auto saveOneKVBlock =
            [](const BufferPtr& buffer, const std::string& dump_dir, const std::string& tag, int32_t block_id) {
                std::ostringstream oss;
                oss << dump_dir << "/" << tag << "_block_" << block_id << ".pt";
                std::string file_path = oss.str();

                printf("📦 Saving %s block_id [%d] → %s\n", tag.c_str(), block_id, file_path.c_str());
                saveBufferDataToTorch(*buffer, nullptr, file_path.c_str());
                printf("✅ Done saving %s block_id [%d]\n", tag.c_str(), block_id);
            };

        auto saveKVCacheToFile = [&]() {
            auto kv_cache = params.common.kv_cache;

            for (int i = 0; i < kv_cache_block_id_host->size(); ++i) {
                int32_t   block_id = *kv_cache_block_id_host->dataWithOffset<int32_t>(i);
                BufferPtr k_block  = kv_cache->k_cache_buffer->index(block_id);
                BufferPtr v_block  = kv_cache->v_cache_buffer->index(block_id);
                saveOneKVBlock(k_block, dump_dir, "k", block_id);
                saveOneKVBlock(v_block, dump_dir, "v", block_id);
            }

            saveBufferDataToTorch(*params.common.sequence_lengths, nullptr, dump_dir + "/sequence_lengths.pt");
            saveBufferDataToTorch(params.input, nullptr, dump_dir + "/qkv.pt");

            if (sliceQ) {
                saveBufferDataToTorch(*sliceQ, nullptr, dump_dir + "/q.pt");
            }
        };

        saveKVCacheToFile();

        printf("🧾 kv_cache_block_id:\n%s\n", kv_cache_block_id_host->debugStringWithData<int32_t>().c_str());
    } else {
        printf("❌ params.common.kv_cache->kv_cache_block_id is nullptr\n");
    }

    // k_cache_buffer
    if (params.common.kv_cache && params.common.kv_cache->k_cache_buffer) {
        printf("params.common.k_cache_buffer\n%s\n", params.common.kv_cache->k_cache_buffer->debugString().c_str());
    } else {
        printf("params.common.k_cache_buffer is nullptr\n");
    }

    // input_lengths
    if (params.common.input_lengths) {
        auto input_lengths = device->clone({*params.common.input_lengths, AllocationType::HOST});
        printf("params.common.input_lengths\n%s\n", input_lengths->debugStringWithData<int32_t>().c_str());
    } else {
        printf("params.common.input_lengths is nullptr\n");
    }

    // sequence_lengths
    if (params.common.sequence_lengths) {
        auto sequence_lengths = device->clone({*params.common.sequence_lengths, AllocationType::HOST});
        printf("params.common.sequence_lengths\n%s\n", sequence_lengths->debugStringWithData<int32_t>().c_str());
    } else {
        printf("params.common.sequence_lengths is nullptr\n");
    }

    // cu_seqlens
    if (params.common.cu_seqlens) {
        auto cu_seqlens = device->clone({*params.common.cu_seqlens, AllocationType::HOST});
        printf("params.common.cu_seqlens\n%s\n", cu_seqlens->debugStringWithData<int32_t>().c_str());
    } else {
        printf("params.common.cu_seqlens is nullptr\n");
    }

    // cu_kv_seqlens
    if (params.common.cu_kv_seqlens) {
        auto cu_kv_seqlens = device->clone({*params.common.cu_kv_seqlens, AllocationType::HOST});
        printf("params.common.cu_kv_seqlens\n%s\n", cu_kv_seqlens->debugStringWithData<int32_t>().c_str());
    } else {
        printf("params.common.cu_kv_seqlens is nullptr\n");
    }

    // padding_offset
    if (params.common.padding_offset) {
        auto padding_offset = device->clone({*params.common.padding_offset, AllocationType::HOST});
        printf("params.common.padding_offset\n%s\n", padding_offset->debugStringWithData<int32_t>().c_str());
    } else {
        printf("params.common.padding_offset is nullptr\n");
    }

    // input
    if (params.input.data()) {
        printf("params.input\n%s\n", params.input.debugString().c_str());
    } else {
        printf("params.input is nullptr\n");
    }

    // prefix_prompt_lengths
    if (params.common.prefix_prompt_lengths) {
        auto prefix_prompt_lengths = device->clone({*params.common.prefix_prompt_lengths, AllocationType::HOST});
        printf("params.common.prefix_prompt_lengths\n%s\n",
               prefix_prompt_lengths->debugStringWithData<int32_t>().c_str());
    } else {
        printf("params.common.prefix_prompt_lengths is nullptr\n");
    }

    printf("Context Batch Size       : %d\n", params.common.context_batch_size);
    printf("Decoder Batch Size       : %d\n", params.common.decoder_batch_size);
    printf("Context Max Seq Length   : %d\n", params.common.context_max_seq_len);
    printf("Decode Max Seq Length   : %d\n", params.common.decoder_max_seq_len);
    printf("Prefix Length (Max)      : %d\n", params.common.max_prefix_length);
    printf("==================================\n");
}

void flashInferAttnParamsDeleter(void* p) {
    delete (FlashInferAttnParams*)p;
}

void aiterAttnParamsDeleter(void* p) {
    delete (AiterAttnParams*)p;
}

void prepareDecodeFlashInferAttnParamsImpl(FlashInferAttnParams*            params,
                                           rtp_llm::DeviceBase*             device,
                                           const rtp_llm::AttentionConfigs& attn_configs,
                                           const BufferPtr&                 sequence_lengths_host,
                                           const BufferPtr&                 kv_cache_block_id_host,
                                           const uint64_t                   batch_size,
                                           const uint64_t                   tokens_per_block,
                                           const uint64_t                   max_batch_blocks) {
    RTP_LLM_CHECK_WITH_INFO(max_batch_blocks > 0 && kv_cache_block_id_host,
                            "max_batch_blocks and kv_cache_block_id_host must be set for decode");
    params->float_workspace =
        device->allocateBuffer({DataType::TYPE_INT8, {128 * 1024 * 1024}, AllocationType::DEVICE}, {"float_workspace"});
    params->int_workspace =
        device->allocateBuffer({DataType::TYPE_INT8, {8 * 1024 * 1024}, AllocationType::DEVICE}, {"int_workspace"});
    params->int_host_workspace =
        device->allocateBuffer({DataType::TYPE_INT8, {8 * 1024 * 1024}, AllocationType::HOST}, {"int_host_workspace"});
    params->page_indptr_host =
        device->allocateBuffer({DataType::TYPE_INT32, {batch_size + 1}, AllocationType::HOST}, {"page_indptr_host"});
    params->qo_indptr_host =
        device->allocateBuffer({DataType::TYPE_INT32, {batch_size + 1}, AllocationType::HOST}, {"qo_indptr_host"});

    params->batch_indice_host =
        device->allocateBuffer({DataType::TYPE_INT32, {batch_size}, AllocationType::HOST}, {"batch_indice_host"});
    params->positions_host =
        device->allocateBuffer({DataType::TYPE_INT32, {batch_size}, AllocationType::HOST}, {"positions_host"});
    params->kvlen_host =
        device->allocateBuffer({DataType::TYPE_INT32, {batch_size}, AllocationType::HOST}, {"kvlen_host"});
    params->paged_kv_last_page_len_host = device->allocateBuffer(
        {DataType::TYPE_INT32, {batch_size}, AllocationType::HOST}, {"paged_kv_last_page_len_host"});
    params->paged_kv_last_page_len_1_host = device->allocateBuffer(
        {DataType::TYPE_INT32, {batch_size}, AllocationType::HOST}, {"paged_kv_last_page_len_1_host"});

    vector<int> page_indice_vec;
    params->qo_indptr_host->data<int>()[0]   = 0;
    params->page_indptr_host->data<int>()[0] = 0;
    for (int i = 0; i < int(batch_size); i++) {
        params->batch_indice_host->data<int>()[i]           = i;
        params->paged_kv_last_page_len_host->data<int>()[i] = sequence_lengths_host->data<int>()[i] % tokens_per_block;
        params->paged_kv_last_page_len_1_host->data<int>()[i] = params->paged_kv_last_page_len_host->data<int>()[i] + 1;
        params->positions_host->data<int>()[i]                = sequence_lengths_host->data<int>()[i];
        params->kvlen_host->data<int>()[i]                    = sequence_lengths_host->data<int>()[i] + 1;
        // sequence_length_host here is the index of the last token in the sequence, equals to length - 1
        int page_nums = (sequence_lengths_host->data<int>()[i] + tokens_per_block) / tokens_per_block;
        for (int j = 0; j < page_nums - 1; j++) {
            auto page_idx = kv_cache_block_id_host->data<int>()[i * max_batch_blocks + j];
            for (int k = page_idx * tokens_per_block; k < (page_idx + 1) * tokens_per_block; k++) {
                page_indice_vec.push_back(k);
            }
        }
        auto page_idx = kv_cache_block_id_host->data<int>()[i * max_batch_blocks + page_nums - 1];
        for (int k = page_idx * tokens_per_block;
             k < page_idx * tokens_per_block + params->paged_kv_last_page_len_1_host->data<int>()[i];
             k++) {
            page_indice_vec.push_back(k);
        }
        params->page_indptr_host->data<int>()[i + 1] = int(page_indice_vec.size());
        params->qo_indptr_host->data<int>()[i + 1]   = i + 1;
    }

    params->page_indice_host = device->allocateBuffer(
        {DataType::TYPE_INT32, {size_t(page_indice_vec.size())}, AllocationType::HOST}, {"page_indice_host"});
    std::copy(page_indice_vec.begin(), page_indice_vec.end(), params->page_indice_host->data<int>());

    params->kv_cache_block_id        = device->clone({*kv_cache_block_id_host, AllocationType::DEVICE});
    params->batch_indice             = device->clone({*params->batch_indice_host, AllocationType::DEVICE});
    params->positions                = device->clone({*params->positions_host, AllocationType::DEVICE});
    params->paged_kv_last_page_len   = device->clone({*params->paged_kv_last_page_len_host, AllocationType::DEVICE});
    params->paged_kv_last_page_len_1 = device->clone({*params->paged_kv_last_page_len_1_host, AllocationType::DEVICE});
    params->page_indptr              = device->clone({*params->page_indptr_host, AllocationType::DEVICE});
    params->qo_indptr                = device->clone({*params->qo_indptr_host, AllocationType::DEVICE});
    params->page_indice              = device->clone({*params->page_indice_host, AllocationType::DEVICE});
    params->kvlen                    = device->clone({*params->kvlen_host, AllocationType::DEVICE});

    params->float_workspace_t    = Buffer2torchTensor(params->float_workspace, false);
    params->int_workspace_t      = Buffer2torchTensor(params->int_workspace, false);
    params->int_host_workspace_t = Buffer2torchTensor(params->int_host_workspace, false);

    params->batch_indice_t             = Buffer2torchTensor(params->batch_indice, false);
    params->positions_t                = Buffer2torchTensor(params->positions, false);
    params->paged_kv_last_page_len_t   = Buffer2torchTensor(params->paged_kv_last_page_len, false);
    params->paged_kv_last_page_len_1_t = Buffer2torchTensor(params->paged_kv_last_page_len_1, false);

    params->qo_indptr_t         = Buffer2torchTensor(params->qo_indptr, false);
    params->qo_indptr_host_t    = Buffer2torchTensor(params->qo_indptr_host, false);
    params->page_indptr_t       = Buffer2torchTensor(params->page_indptr, false);
    params->page_indptr_host_t  = Buffer2torchTensor(params->page_indptr_host, false);
    params->page_indice_t       = Buffer2torchTensor(params->page_indice, false);
    params->kvlen_host_t        = Buffer2torchTensor(params->kvlen_host, false);
    params->kvlen_t             = Buffer2torchTensor(params->kvlen, false);
    params->kv_cache_block_id_t = Buffer2torchTensor(params->kv_cache_block_id, false);
}

// for mla, we need to prepare additional params for write kvcache and de rotary embedding
void prepareContextMLAFlashInferAttnParamsImpl(FlashInferAttnParams*            params,
                                               rtp_llm::DeviceBase*             device,
                                               const rtp_llm::AttentionConfigs& attn_configs,
                                               const BufferPtr&                 sequence_lengths_host,
                                               const BufferPtr&                 input_lengths_host,
                                               const BufferPtr&                 kv_cache_block_id_host,
                                               const uint64_t                   prefill_token_num,
                                               const uint64_t                   context_batch_size,
                                               const uint64_t                   tokens_per_block,
                                               const uint64_t                   max_batch_blocks,
                                               const uint64_t                   batch_size) {
    params->batch_indice_host = device->allocateBuffer(
        {DataType::TYPE_INT32, {prefill_token_num}, AllocationType::HOST}, {"prefill_batch_indices_host"});
    params->positions_host = device->allocateBuffer({DataType::TYPE_INT32, {prefill_token_num}, AllocationType::HOST},
                                                    {"prefill_positions_host"});
    params->paged_kv_last_page_len_1_host = device->allocateBuffer(
        {DataType::TYPE_INT32, {context_batch_size}, AllocationType::HOST}, {"prefill_kv_last_page_len_1_host"});
    params->page_indptr_host = device->allocateBuffer(
        {DataType::TYPE_INT32, {context_batch_size + 1}, AllocationType::HOST}, {"prefill_page_indptr_host"});
    params->page_indptr_host->data<int>()[0] = 0;
    std::vector<int> prefill_page_indices_vec;

    int offset = 0;
    for (int i = 0; i < context_batch_size; i++) {
        auto input_length = input_lengths_host->data<int>()[i + batch_size];
        for (int j = 0; j < input_length; j++) {
            params->batch_indice_host->data<int>()[offset] = i;
            params->positions_host->data<int>()[offset]    = j;
            offset += 1;
        }
        if (kv_cache_block_id_host) {
            int page_nums = (input_length + tokens_per_block - 1) / tokens_per_block;
            for (int j = 0; j < page_nums; j++) {
                auto page_idx = kv_cache_block_id_host->data<int>()[(i + batch_size) * max_batch_blocks + j];
                prefill_page_indices_vec.push_back(page_idx);
            }
            params->paged_kv_last_page_len_1_host->data<int>()[i] = (input_length - 1) % tokens_per_block + 1;
            params->page_indptr_host->data<int>()[i + 1]          = prefill_page_indices_vec.size();
        }
    }
    if (kv_cache_block_id_host) {
        params->page_indice_host = device->allocateBuffer(
            {DataType::TYPE_INT32, {size_t(prefill_page_indices_vec.size())}, AllocationType::HOST},
            {"prefill_page_indices_host"});
        std::copy(
            prefill_page_indices_vec.begin(), prefill_page_indices_vec.end(), params->page_indice_host->data<int>());
        params->page_indice   = device->clone({*params->page_indice_host, AllocationType::DEVICE});
        params->page_indice_t = Buffer2torchTensor(params->page_indice, false);
    }

    params->batch_indice             = device->clone({*params->batch_indice_host, AllocationType::DEVICE});
    params->positions                = device->clone({*params->positions_host, AllocationType::DEVICE});
    params->paged_kv_last_page_len_1 = device->clone({*params->paged_kv_last_page_len_1_host, AllocationType::DEVICE});
    params->page_indptr              = device->clone({*params->page_indptr_host, AllocationType::DEVICE});

    params->batch_indice_t             = Buffer2torchTensor(params->batch_indice, false);
    params->positions_t                = Buffer2torchTensor(params->positions, false);
    params->paged_kv_last_page_len_1_t = Buffer2torchTensor(params->paged_kv_last_page_len_1, false);
    params->page_indptr_t              = Buffer2torchTensor(params->page_indptr, false);
}

void prepareDecodeAiterAttnParamsImpl(AiterAttnParams*     params,
                                      rtp_llm::DeviceBase* device,
                                      const BufferPtr&     sequence_lengths_host,
                                      const uint64_t       batch_size) {
    if (device->nativeGraphCapturing()) {
        params->sequence_lengths_host = nullptr;
        params->sequence_lengths      = device->clone({*sequence_lengths_host, AllocationType::DEVICE});
        params->sequence_lengths_t    = Buffer2torchTensor(params->sequence_lengths, false);
        params->sequence_lengths_t += 1;
        return;
    }
    params->sequence_lengths_host =
        device->allocateBuffer({DataType::TYPE_INT32, {batch_size}, AllocationType::HOST}, {"sequence_lengths_host"});

    for (int i = 0; i < int(batch_size); i++) {
        params->sequence_lengths_host->data<int>()[i] = sequence_lengths_host->data<int>()[i] + 1;
    }

    params->sequence_lengths = device->clone({*params->sequence_lengths_host, AllocationType::DEVICE});

    params->sequence_lengths_t = Buffer2torchTensor(params->sequence_lengths, false);
}

ParamsPtr FlashInferAttnParams::preparePrefillFlashInferAttnParams(rtp_llm::DeviceBase*             device,
                                                                   const rtp_llm::AttentionConfigs& attn_configs,
                                                                   const BufferPtr&                 prefix_lengths_host,
                                                                   const BufferPtr&  sequence_lengths_host,
                                                                   const BufferPtr&  input_lengths_host,
                                                                   const BufferPtr&  kv_cache_block_id_host,
                                                                   rtp_llm::DataType dtype) {
    const size_t batch_size         = sequence_lengths_host->shape()[0];
    const size_t context_batch_size = input_lengths_host->shape()[0] - batch_size;
    if (context_batch_size == 0) {
        return nullptr;
    }

    const int tokens_per_block = attn_configs.tokens_per_block;

    const int    max_batch_blocks  = kv_cache_block_id_host ? kv_cache_block_id_host->shape()[1] : -1;
    const size_t prefill_token_num = std::accumulate(input_lengths_host->data<int>() + batch_size,
                                                     input_lengths_host->data<int>() + context_batch_size + batch_size,
                                                     0);
    auto         ret               = ParamsPtr(new FlashInferAttnParams, flashInferAttnParamsDeleter);
    auto         params            = (FlashInferAttnParams*)ret.get();
    prepareContextMLAFlashInferAttnParamsImpl(params,
                                              device,
                                              attn_configs,
                                              sequence_lengths_host,
                                              input_lengths_host,
                                              kv_cache_block_id_host,
                                              prefill_token_num,
                                              context_batch_size,
                                              tokens_per_block,
                                              max_batch_blocks,
                                              batch_size);
    return ret;
}

ParamsPtr FlashInferAttnParams::prepareDecodeFlashInferAttnParams(rtp_llm::DeviceBase*             device,
                                                                  const rtp_llm::AttentionConfigs& attn_configs,
                                                                  const BufferPtr&  sequence_lengths_host,
                                                                  const BufferPtr&  input_lengths_host,
                                                                  const BufferPtr&  kv_cache_block_id_host,
                                                                  rtp_llm::DataType dtype) {
    const char* disable_flash_infer_env = getenv("DISABLE_FLASH_INFER");
    if (rtp_llm::rocm::get_sm() < 80 || (disable_flash_infer_env && strcmp(disable_flash_infer_env, "1") == 0)) {
        return nullptr;
    }

    const size_t batch_size = sequence_lengths_host->shape()[0];
    if (batch_size == 0) {
        return nullptr;
    }

    auto      cuda_device       = dynamic_cast<ROCmDevice*>(device);
    const int local_head_num    = attn_configs.head_num;
    const int local_head_num_kv = attn_configs.kv_head_num;
    const int size_per_head     = attn_configs.size_per_head;
    const int group_size        = local_head_num / local_head_num_kv;
    const int tokens_per_block  = attn_configs.tokens_per_block;

    if (!cuda_device || (dtype != DataType::TYPE_FP16 && dtype != DataType::TYPE_BF16)
        || attn_configs.kv_cache_dtype != KvCacheDataType::BASE
        || (attn_configs.rope_config.style != RopeStyle::Base && attn_configs.rope_config.style != RopeStyle::No)
        || attn_configs.mask_type != causalMask || attn_configs.q_scaling != 1.0f || attn_configs.use_logn_attn
        || (size_per_head != 64 && size_per_head != 128 && size_per_head != 192)
        || (group_size > 10 && group_size != 16)) {
        return nullptr;
    }

    const int max_batch_blocks = kv_cache_block_id_host ? kv_cache_block_id_host->shape()[1] : -1;
    auto      ret              = ParamsPtr(new FlashInferAttnParams, flashInferAttnParamsDeleter);
    auto      params           = (FlashInferAttnParams*)ret.get();
    if (group_size > 5) {
        params->decode = false;
    } else {
        params->decode = true;
    }

    // prepare flashinfer params for decode
    prepareDecodeFlashInferAttnParamsImpl(params,
                                          device,
                                          attn_configs,
                                          sequence_lengths_host,
                                          kv_cache_block_id_host,
                                          batch_size,
                                          tokens_per_block,
                                          max_batch_blocks);
    return ret;
}

ParamsPtr AiterAttnParams::prepareDecodeAiterAttnParams(rtp_llm::DeviceBase* device,
                                                        const BufferPtr&     sequence_lengths_host) {

    if (!device->initParams().use_aiter_pa) {
        return nullptr;
    }

    const size_t batch_size = sequence_lengths_host->shape()[0];
    if (batch_size == 0) {
        return nullptr;
    }

    auto ret    = ParamsPtr(new AiterAttnParams, aiterAttnParamsDeleter);
    auto params = (AiterAttnParams*)ret.get();

    prepareDecodeAiterAttnParamsImpl(params, device, sequence_lengths_host, batch_size);
    return ret;
}

KVBlockArray ROCmDevice::getKVBlockArray(const AttentionModuleParams& params,
                                         const Buffer&                kv_cache_offset_pointers,
                                         int                          batch_size,
                                         bool                         use_fp8_fmha,
                                         bool                         use_offset_array) {
    const auto& kv_cache         = params.common.kv_cache;
    const auto& kv_blocks_offset = *(kv_cache->kv_cache_block_id);
    const auto& kv_block_offset  = (kv_cache->k_cache_buffer)->shape()[0] * kv_cache->layer_num;
    RUNTIME_ASSERT_OP_ARG(kv_blocks_offset.shape()[0] == batch_size,
                          "context attention kv blocks batch size expected [%d] but buffer[%s]",
                          (int)batch_size,
                          kv_blocks_offset.debugString().c_str());
    const auto  max_blocks_per_batch = kv_blocks_offset.shape()[1];
    const auto& k_cache              = *(kv_cache->k_cache_buffer);
    const auto& v_cache              = *(kv_cache->v_cache_buffer);
    auto const  elemSize = kv_cache->k_scale_buffer || use_fp8_fmha ? sizeof(int8_t) : 2;  // 2 for kv cache fp16
    // RTP_LLM_LOG_INFO("kv_cache[0].typeSize():%d", kv_cache[0].typeSize());
    RTP_LLM_LOG_DEBUG("kv_blocks_offset size:%d, k_cache:%p, v_cache:%p, "
                      "k_cache[0].sizeBytes():%d, params.configs.tokens_per_block:%d, "
                      "kv_block_offset:%d, k_cache (int): %lu, v_cache (int): %lu, "
                      "max_blocks_per_batch:%d",
                      kv_blocks_offset.size(),
                      static_cast<void*>(k_cache.data()),  // for %p
                      static_cast<void*>(v_cache.data()),  // for %p
                      k_cache[0].sizeBytes(),
                      params.configs.tokens_per_block,
                      kv_block_offset,
                      static_cast<unsigned long>(reinterpret_cast<uintptr_t>(k_cache.data())),  // for %lu
                      static_cast<unsigned long>(reinterpret_cast<uintptr_t>(v_cache.data())),
                      max_blocks_per_batch);
    auto const   sizePerToken = params.configs.kv_head_num * params.configs.size_per_head * elemSize;
    KVBlockArray kv_cache_buffer =
        KVBlockArray(batch_size,
                     max_blocks_per_batch,
                     params.configs.tokens_per_block,
                     sizePerToken,
                     0,
                     0,
                     (uint64_t*)k_cache.data(),
                     nullptr,
                     (rtp_llm::KVBlockArrayForContextFMHA::DataType*)kv_cache_offset_pointers.data());

    if (!use_offset_array) {
        invokeConvertOffsetToBlockArrayData((int32_t*)kv_cache_offset_pointers.data(),
                                            (int*)kv_blocks_offset.data(),
                                            batch_size,
                                            max_blocks_per_batch,
                                            kv_block_offset,
                                            stream_);
    }
    check_cuda_error();
    if (kv_cache->k_scale_buffer) {
        RUNTIME_ASSERT_OP_ARG(kv_cache->v_scale_buffer,
                              "v scale buffer should has value when use k scale buffer has value");
        const auto& k_scale                 = *(kv_cache->k_scale_buffer);
        kv_cache_buffer.scale               = k_scale.data();
        kv_cache_buffer.mScaleBytesPerBlock = k_scale[0].sizeBytes();
    }
    KvCacheDataType cache_type = KvCacheDataType::BASE;
#if defined(ENABLE_FP8)
    if (use_fp8_fmha_) {
        cache_type = KvCacheDataType::FP8;
    } else
#endif
    if (use_fp8_fmha) {
        cache_type = KvCacheDataType::FP8;
    } else if (kv_cache->k_scale_buffer && params.configs.kv_cache_dtype == KvCacheDataType::INT8) {
        RTP_LLM_LOG_DEBUG("now use kv_cache int8");
        cache_type = KvCacheDataType::INT8;
    }
    kv_cache_buffer.cache_type = cache_type;
    check_cuda_error();
    return kv_cache_buffer;
}

ParamsPtr ROCmDevice::PrepareCKAttn(const AttentionConfigs& configs,
                                    int                     kv_block_offset,
                                    const BufferPtr&        kv_cache_block_id,
                                    int                     batch_size) {
    RTP_LLM_LOG_DEBUG("PrepareCKAttn: kv_block_offset: %d, batch_size: %d, kv_cache_block_id: %s",
                      kv_block_offset,
                      batch_size,
                      kv_cache_block_id ? kv_cache_block_id->debugString().c_str() : "nullptr");
    if (kv_block_offset <= 0 || batch_size <= 0 || !kv_cache_block_id) {
        return nullptr;
    }
    auto            ck_attn    = std::make_shared<CKAttn>();
    KvCacheDataType cache_type = KvCacheDataType::BASE;
#ifdef ENABLE_FP8
    if (use_fp8_fmha_) {
        cache_type = KvCacheDataType::FP8;
    } else
#endif
        if (configs.kv_cache_dtype == KvCacheDataType::INT8) {
        RTP_LLM_LOG_DEBUG("now use kv_cache int8");
        cache_type = KvCacheDataType::INT8;
    }
    const auto max_blocks_per_batch = kv_cache_block_id->shape()[1];
    auto const elemSize             = 2;  // 2 for kv cache fp16

    ck_attn->kv_cache_offset =
        allocateBuffer({DataType::TYPE_INT32, {size_t(batch_size), 1, 2, max_blocks_per_batch}, AllocationType::DEVICE},
                       {"kv_cache_offset"});
    ck_attn->kv_block_array                     = KVBlockArray(batch_size,
                                           max_blocks_per_batch,
                                           configs.tokens_per_block,
                                           configs.kv_head_num * configs.size_per_head * elemSize,
                                           0,
                                           0,
                                           nullptr,  // (uint64_t*)k_cache.data(),
                                           nullptr,
                                           (rtp_llm::KVCacheIndex*)ck_attn->kv_cache_offset->data<int>());
    ck_attn->kv_block_array.cache_type          = cache_type;
    ck_attn->kv_block_array.mScaleBytesPerBlock = configs.tokens_per_block * configs.kv_head_num * sizeof(float);
    invokeConvertOffsetToBlockArrayData(ck_attn->kv_cache_offset->data<int>(),
                                        kv_cache_block_id->data<int>(),
                                        batch_size,
                                        max_blocks_per_batch,
                                        kv_block_offset,
                                        stream_);
    check_cuda_error();
    return ck_attn;
}

AttentionModuleOutput ROCmDevice::contextAttention(const AttentionModuleParams& params) {
    auto datatype            = params.input.type();
    auto token_num           = params.input.shape()[0];
    auto batch_size          = params.common.context_batch_size;
    auto decoder_batch_size  = params.common.decoder_batch_size;
    auto seq_len             = params.common.context_max_seq_len;
    auto seq_len_with_prefix = seq_len + params.common.max_prefix_length;
    // auto context_token_num   = params.common.context_token_num;
    auto head_num      = params.configs.head_num;
    auto kv_head_num   = params.configs.kv_head_num;
    auto size_per_head = params.configs.size_per_head;

    auto q_output = allocateBuffer(
        {params.input.type(), {batch_size, head_num, seq_len, size_per_head}, AllocationType::DEVICE}, {"q_output"});

    auto k_output = allocateBuffer(
        {params.input.type(), {batch_size, kv_head_num, seq_len_with_prefix, size_per_head}, AllocationType::DEVICE},
        {"k_output"});

    auto v_output = allocateBuffer(
        {params.input.type(), {batch_size, kv_head_num, seq_len_with_prefix, size_per_head}, AllocationType::DEVICE},
        {"v_output"});

    BufferPtr kv_cache_block_id = nullptr;

    KVBlockArray                  kv_block_array;
    PrefixPromptBatchWeightsParam prefix_prompt_param;

    bool use_fp8_fmha = false;
    if (params.common.kv_cache) {
        const auto max_blocks_per_batch = params.common.kv_cache->kv_cache_block_id->shape()[1];
        kv_cache_block_id =
            allocateBuffer({DataType::TYPE_INT32, {batch_size, 1, 2, max_blocks_per_batch}, AllocationType::DEVICE},
                           {"kv_cache_block_id"});
        kv_block_array = getKVBlockArray(params, *kv_cache_block_id, batch_size, params.common.kv_cache->k_cache_buffer->type() == DataType::TYPE_FP8_E4M3);
        prefix_prompt_param.kv_block_array = kv_block_array;

        if (params.common.prefix_prompt_lengths) {
            prefix_prompt_param.d_prefix_prompt_lengths  = params.common.prefix_prompt_lengths->data<int>();
            prefix_prompt_param.max_prefix_prompt_length = params.common.max_prefix_length;
            prefix_prompt_param.count_length             = 1;
        }
        use_fp8_fmha = kv_block_array.cache_type == KvCacheDataType::FP8;
    }
    printBufferData(*params.common.input_lengths, "input_lengths");
    if (params.common.cu_seqlens) {
        printBufferData(*params.common.cu_seqlens, "cu_seqlens");
        printBufferData(*params.common.cu_kv_seqlens, "cu_kv_seqlens");
    }

    BufferPtr qkv_buf_fp8 = nullptr;
    if (use_fp8_fmha) {
        qkv_buf_fp8 = allocateBuffer({DataType::TYPE_FP8_E4M3,
                                      {batch_size, (head_num + kv_head_num * 2), seq_len_with_prefix, size_per_head},
                                      AllocationType::DEVICE},
                                     {"qkv_fp8_output"});}


    // printBufferMeta("params.input", params.input);                 // Buffer
    // printBufferMeta("qkv_buf_fp8", qkv_buf_fp8);                   // BufferPtr/ConstBufferPtr
    // printBufferMeta("output", params.output);                   // Buffer

    // int8
    float* scale_out_ptr = nullptr;
    int    int8_mode     = 0;

    if (prefix_prompt_param.max_prefix_prompt_length > 0) {
        if (init_params_.use_aiter_pa) {
            if (init_params_.use_asm_pa) {
                DISPATCH_CUDA_FUNCTION_DATA_TYPE(datatype,
                                             invokeLoadPrefixKVCacheAiter,
                                             q_output->data(),
                                             k_output->data(),
                                             v_output->data(),
                                             &prefix_prompt_param,
                                             batch_size,
                                             seq_len,
                                             head_num,
                                             kv_head_num,
                                             size_per_head,
                                             scale_out_ptr,
                                             int8_mode,
                                             stream_);
            } else {
                RUNTIME_ASSERT_OP_ARG(init_params_.use_asm_pa, "Should use asm_pa");
            }
        } else {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(datatype,
                                             invokeLoadPrefixKVCache,
                                             q_output->data(),
                                             k_output->data(),
                                             v_output->data(),
                                             &prefix_prompt_param,
                                             batch_size,
                                             seq_len,
                                             head_num,
                                             kv_head_num,
                                             size_per_head,
                                             scale_out_ptr,
                                             int8_mode,
                                             stream_);
        }
    }

    // printBufferMeta("params.input", params.input);                 // Buffer
    // printBufferMeta("qkv_buf_fp8", qkv_buf_fp8);                   // BufferPtr/ConstBufferPtr
    // printBufferMeta("output", params.output);                   // Buffer

    bool store_qkv   = true;
    bool store_q     = true;
    bool store_kv    = true;
    bool store_cache = params.common.kv_cache.has_value();

    // if all condition satisfy, no need to do invokeAddFusedQKVBiasTranspose
    bool skip_add_bias_transpose = (params.configs.rope_config.style == RopeStyle::No && !params.common.kv_cache
                                    && !params.configs.fuse_qkv_add_bias);
    RTP_LLM_LOG_DEBUG("skip_add_bias_transpose: %d", skip_add_bias_transpose);
    if (!skip_add_bias_transpose) {
        static torch::Tensor cos_sin_cache = getRopeCosSin(params.configs.rope_config.style,
                                                           params.configs.rope_config.dim,
                                                           params.configs.rope_config.base,
                                                           params.configs.rope_config.scale,
                                                           init_params_.max_seq_len);
        if (init_params_.use_aiter_pa) {
            if (init_params_.use_asm_pa) {
                // std::string dir = "/mnt/raid0/zhaoan12/cache/tmp_ck/";
                // if (token_num != 1){
                //     torch::Tensor input0 = Buffer2torchTensor(params.input, false);
                //     saveTorchDataTofile(input0, dir +"input.pt");
                // }
                DISPATCH_CUDA_FUNCTION_DATA_TYPE(datatype,
                                             invokeAddFusedQKVBiasTransposePrefill,
                                             q_output->data(),
                                             k_output->data(),
                                             v_output->data(),
                                             &prefix_prompt_param,
                                             params.input.data(),
                                             qkv_buf_fp8? qkv_buf_fp8->data(): nullptr,
                                             params.common.position_ids ?
                                                 params.common.position_ids->dataWithOffset<int>(
                                                     decoder_batch_size * params.configs.rope_config.index_factor) :
                                                 nullptr,
                                             params.configs.fuse_qkv_add_bias && params.weights.qkv_weight->bias ?
                                                 params.weights.qkv_weight->bias->data() :
                                                 nullptr,
                                             params.common.padding_offset->data<int>(),
                                             params.common.cu_seqlens->data<int>(),
                                             batch_size,
                                             seq_len,
                                             token_num,
                                             head_num,
                                             kv_head_num,
                                             size_per_head,
                                             params.configs.rope_config,
                                             params.configs.use_logn_attn,
                                             scale_out_ptr,
                                             int8_mode,
                                             false,
                                             store_qkv,
                                             store_q,
                                             store_kv,
                                             store_cache,
                                             cos_sin_cache.defined() ? static_cast<float2*>(cos_sin_cache.data_ptr()) : nullptr,
                                             stream_);
            // dir = "/mnt/raid0/zhaoan12/cache/tmp_aiter/";
            // if (token_num != 1){
            //     torch::Tensor input0 = Buffer2torchTensor(params.input, false);
            //     saveTorchDataTofile(input0, dir +"input.pt");
            // }                                             
            } else {
                RUNTIME_ASSERT_OP_ARG(init_params_.use_asm_pa, "Should use asm_pa");
            }
            check_cuda_error();
        } else {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(datatype,
                                             invokeAddFusedQKVBiasTranspose,
                                             nullptr,
                                             q_output->data(),
                                             k_output->data(),
                                             v_output->data(),
                                             &prefix_prompt_param,
                                             params.input.data(),
                                             nullptr,
                                             params.common.position_ids ?
                                                 params.common.position_ids->dataWithOffset<int>(
                                                     decoder_batch_size * params.configs.rope_config.index_factor) :
                                                 nullptr,
                                             params.configs.fuse_qkv_add_bias && params.weights.qkv_weight->bias ?
                                                 params.weights.qkv_weight->bias->data() :
                                                 nullptr,
                                             params.common.padding_offset->data<int>(),
                                             params.common.cu_seqlens->data<int>(),
                                             batch_size,
                                             seq_len,
                                             token_num,
                                             head_num,
                                             kv_head_num,
                                             size_per_head,
                                             params.configs.rope_config,
                                             params.configs.use_logn_attn,
                                             scale_out_ptr,
                                             int8_mode,
                                             false,
                                             store_qkv,
                                             false,
                                             store_q,
                                             store_kv,
                                             store_cache,
                                             stream_);
            check_cuda_error();
        }
        writeCacheStore(params);
    }
    if (use_fp8_fmha){
        fmha_runner_->setup(
            DataType::TYPE_FP8_E4M3, params.configs.mask_type, head_num, kv_head_num, size_per_head, params.configs.q_scaling);
    } else {
        fmha_runner_->setup(
            datatype, params.configs.mask_type, head_num, kv_head_num, size_per_head, params.configs.q_scaling);
    }
    
    printBufferData(*q_output, "q_output");

    const size_t hidden_units    = head_num * size_per_head;
    const size_t hidden_units_kv = kv_head_num * size_per_head;

    auto lse_acc_buf = allocateBuffer({DataType::TYPE_FP32, {1, 1, 1, 1}, AllocationType::DEVICE}, {"lse_acc_buf"});

    printBufferData(*q_output, "run_ck_q_output");
    printBufferData(*k_output, "run_ck_k_output");
    printBufferData(*v_output, "run_ck_v_output");
    printBufferData(params.input, "run_ck_input");

    // std::string dir = "/mnt/raid0/zhaoan12/cache/tmp_golden_bf16";
    // if (token_num != 1){
    //     torch::Tensor input0 = Buffer2torchTensor(params.input, false);
    //     saveTorchDataTofile(input0, dir +"input.pt");
    // }     
    // std::cout << "skip_add_bias_transpose = " << skip_add_bias_transpose << std::endl;
    if (skip_add_bias_transpose || prefix_prompt_param.max_prefix_prompt_length <= 0) {
        // not implemented reuse cache for this branch
        if (use_fp8_fmha){
            std::string dir = "/mnt/raid0/zhaoan12/cache/tmp_ck/";
            // if ((int64_t)seq_len != 1){ // dump data
                // saveTorchDataTofile(as_uint8_bytes(A_quant_tensor),  dir + "CK_A_kernel.pt");
            //     torch::Tensor qkv_buf_fp8_tensor1       = Buffer2torchTensor(qkv_buf_fp8, false);
            //     saveTorchDataTofile(as_uint8_bytes(qkv_buf_fp8_tensor1), dir +"qkv_buf_fp8.pt");
            // }
            // Bufferptr q_load, k_load, v_load;

            // if ((int64_t)seq_len != 1){ // dump data
            //     // 假设路径如下
            //     std::string q_path = "/mnt/raid0/zhaoan12/cache/tmp_ck/q_tensor.pt";
            //     std::string k_path = "/mnt/raid0/zhaoan12/cache/tmp_ck/k_tensor.pt";
            //     std::string v_path = "/mnt/raid0/zhaoan12/cache/tmp_ck/v_tensor.pt";

            //     // 加载为 torch::Tensor
            //     torch::Tensor q_tensor = bytes_to_fp8_e4m3(loadTensorFromFile(q_path));
            //     torch::Tensor k_tensor = bytes_to_fp8_e4m3(loadTensorFromFile(k_path));
            //     torch::Tensor v_tensor = bytes_to_fp8_e4m3(loadTensorFromFile(v_path));
                
            //     std::cout << "[DEBUG_contextAttention] start debugTensor"  << std::endl;
            //     // 假设你已经有 torch::Tensor q_tensor
            //     debugTensor(q_tensor, "Q", 20);
            //     debugTensor(k_tensor, "K", 20);
            //     debugTensor(v_tensor, "V", 20);
            // }
// if ((int64_t)seq_len != 1){ // dump data
//                 // 假设路径如下
//                 std::string q_path = "/mnt/raid0/zhaoan12/cache/tmp_ck/q_tensor.pt";
//                 std::string k_path = "/mnt/raid0/zhaoan12/cache/tmp_ck/k_tensor.pt";
//                 std::string v_path = "/mnt/raid0/zhaoan12/cache/tmp_ck/v_tensor.pt";

//                 // 加载为 torch::Tensor
//                 // 例：B=1, S_q=7, S_k=7, H_q=32, H_k=8, D=128
//                 auto q_cpu = loadTensorFromFile(q_path);
//                 auto k_cpu = loadTensorFromFile(k_path);
//                 auto v_cpu = loadTensorFromFile(v_path);

//                 debugTensor(q_cpu, "Q_CPU", 20);
//                 debugTensor(k_cpu, "K_CPU", 20);
//                 debugTensor(v_cpu, "V_CPU", 20);
//                 const int64_t B   = (int64_t)batch_size;
//                 const int64_t S_q = (int64_t)seq_len;
//                 const int64_t S_k = (int64_t)seq_len_with_prefix;
//                 const int64_t H_q = (int64_t)head_num;
//                 const int64_t H_k = (int64_t)(kv_head_num > 0 ? kv_head_num : head_num);
//                 const int64_t D   = (int64_t)size_per_head;
//                 auto q_tensor_1 = toCudaFp8FromCpuBytes(q_cpu, {B, S_q, H_q, D});
//                 auto k_tensor_1 = toCudaFp8FromCpuBytes(k_cpu, {B, S_k, H_k, D});
//                 auto v_tensor_1 = toCudaFp8FromCpuBytes(v_cpu, {B, S_k, H_k, D});

//                 std::cout << "[DEBUG_contextAttention] start debugTensor"  << std::endl;
//                 // 假设你已经有 torch::Tensor q_tensor
//                 debugTensor(q_tensor_1, "Q", 20);
//                 debugTensor(k_tensor_1, "K", 20);
//                 debugTensor(v_tensor_1, "V", 20);
//                     std::cout << "[DEBUG_contextAttention] start debugTensor 2 "  << std::endl;
//                 auto q_buffer =  torchTensor2Buffer(q_tensor_1);
//                 auto k_buffer =  torchTensor2Buffer(k_tensor_1);
//                 auto v_buffer =  torchTensor2Buffer(v_tensor_1);

//             //     at::Tensor q_tensor_2 = q_tensor_1.device(torch::kCUDA);
//             //     at::Tensor k_tensor_2 = k_tensor_1.device(torch::kCUDA);
//             //     at::Tensor v_tensor_2 = v_tensor_1.device(torch::kCUDA);

//             // debugTensor(q_tensor_2, "Q", 20);
//             // debugTensor(k_tensor_2, "K", 20);
//             // debugTensor(v_tensor_2, "V", 20);

            // const int64_t B   = (int64_t)batch_size;
            // const int64_t S_q = (int64_t)seq_len;
            // const int64_t S_k = (int64_t)seq_len_with_prefix;
            // const int64_t H_q = (int64_t)head_num;
            // const int64_t H_k = (int64_t)(kv_head_num > 0 ? kv_head_num : head_num);
            // const int64_t D   = (int64_t)size_per_head;

            // const int64_t q_elems = B * S_q * H_q * D;
            // const int64_t k_elems = B * S_k * H_k * D;
            // const int64_t v_elems = B * S_k * H_k * D;

            // // auto input_reshape = params.input.reshape({batch_size * seq_len_with_prefix, head_num + kv_head_num*2, size_per_head});

            // // torch::Tensor scales_tensor = torch::full({1}, 1.0f, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            // // torch::Tensor input_tensor  = Buffer2torchTensor(params.input, false);   
            // // auto [q_fp8, scale] = quantize_bf16_to_fp8(input_tensor, scales_tensor);
            // // torch::Tensor qkv_buf_fp8_tensor = q_fp8;

            // // auto input_reshape = params.input.reshape({batch_size * seq_len_with_prefix, head_num + kv_head_num*2, size_per_head});

            // QBufferPtr q_hidden = std::dynamic_pointer_cast<QBuffer>(
            //     quantize(QuantizeParams(params.input, DataType::TYPE_QFP8_E4M3, 1, QScheme::Qfp8PerTensor, 0, 0)));
            // torch::Tensor qkv_buf_fp8_tensor  = Buffer2torchTensor(q_hidden->kernelPtr(), false);
            
            // int64_t q_size = H_q * D;
            // int64_t k_size = H_k * D;
            // int64_t v_size = H_k * D;

            // auto splits = torch::split(qkv_buf_fp8_tensor, {q_size, k_size, v_size}, /*dim=*/1);

            // auto q_flat = splits[0]; // [7, q_size]
            // auto k_flat = splits[1]; // [7, k_size]
            // auto v_flat = splits[2]; // [7, v_size]

            // torch::Tensor q_tensor = q_flat.view({B, S_q, H_q, D}).contiguous();
            // torch::Tensor k_tensor = k_flat.view({B, S_k, H_k, D}).contiguous();
            // torch::Tensor v_tensor = v_flat.view({B, S_k, H_k, D}).contiguous();

            // auto q_bhs = q_tensor.permute({0, 2, 1, 3}).contiguous();   // [B, Hq, S, D], stride=(B*H*S*D, S*D, D, 1)
            // auto k_bhs = k_tensor.permute({0, 2, 1, 3}).contiguous();   // [B, Hk, S, D]
            // auto v_bhs = v_tensor.permute({0, 2, 1, 3}).contiguous();   // [B, Hk, S, D]


            // // dir = "/mnt/raid0/zhaoan12/cache/tmp_ck_qkvo/";
            // // if ((int64_t)seq_len != 1){ // dump data
            // //     // saveTorchDataTofile(as_uint8_bytes(qkv_buf_fp8_tensor), dir +"qkv_buf_fp8.pt");
            // //     // saveTorchDataTofile(as_uint8_bytes(q_tensor), dir + "q_tensor.pt");
            // //     // saveTorchDataTofile(as_uint8_bytes(k_tensor), dir + "k_tensor.pt");
            // //     // saveTorchDataTofile(as_uint8_bytes(v_tensor), dir + "v_tensor.pt");
            // //     auto qkv_u8 = qkv_buf_fp8_tensor.view(torch::kUInt8);
            // //     auto q_u8   = q_tensor.view(torch::kUInt8);
            // //     auto k_u8   = k_tensor.view(torch::kUInt8);
            // //     auto v_u8   = v_tensor.view(torch::kUInt8);

            // //     saveTorchDataTofile(qkv_u8, dir + "qkv_buf_fp8.pt");
            // //     saveTorchDataTofile(q_u8,   dir + "q_tensor.pt");
            // //     saveTorchDataTofile(k_u8,   dir + "k_tensor.pt");
            // //     saveTorchDataTofile(v_u8,   dir + "v_tensor.pt");                
            // // }
            // // debugTensor(q_tensor, "Q", 100);
            // // debugTensor(k_tensor, "K", 100);
            // // debugTensor(v_tensor, "V", 100);



            // // std::cout << "[DEBUG_contextAttention] enter ck"  << std::endl;

            // fmha_runner_->runCKFmha(q_bhs.data_ptr()  ,
            //                     k_bhs.data_ptr()  ,
            //                     v_bhs.data_ptr()  ,
            //                     params.output.data(),
            //                     nullptr,  // buffer for store out softmax_lse, looks like not used by RTP
            //                     batch_size,
            //                     seq_len,
            //                     prefix_prompt_param.max_prefix_prompt_length,
            //                     // context_token_num,
            //                     params.common.cu_seqlens->data(),
            //                     params.common.cu_kv_seqlens->data(),
            //                     lse_acc_buf->data(),
            //                     params.common.linear_bias_slopes ? params.common.linear_bias_slopes->data() : nullptr,
            //                     nullptr,
            //                     true,
            //                     false);

                // auto out_tensor  = Buffer2torchTensor(params.output ,false).view({B,S_q, H_q,D});
                


                // // dir = "/mnt/raid0/zhaoan12/cache/tmp_ck/";
                // if ((int64_t)seq_len != 1){ // dump data
                //     saveTorchDataTofile(out_tensor, dir +"output.pt");
                //     debugTensor(out_tensor, "OUT", 200);
                //     std::exit(1);
                // }
// //             }
// //             fmha_runner_->runCKFmha(qkv_buf_fp8->data(),
// //                                 qkv_buf_fp8->dataWithOffset(hidden_units),
// //                                 qkv_buf_fp8->dataWithOffset(hidden_units + hidden_units_kv),
// //                                 params.output.data(),
// //                                 nullptr,  // buffer for store out softmax_lse, looks like not used by RTP
// //                                 batch_size,
// //                                 seq_len,
// //                                 prefix_prompt_param.max_prefix_prompt_length,
// //                                 // context_token_num,
// //                                 params.common.cu_seqlens->data(),
// //                                 params.common.cu_kv_seqlens->data(),
// //                                 lse_acc_buf->data(),
// //                                 params.common.linear_bias_slopes ? params.common.linear_bias_slopes->data() : nullptr,
// //                                 nullptr,
// //                                 false,
// //                                 false);

            
            // std::cout << "[DEBUG_contextAttention] start 1"  << std::endl;
            // torch::Tensor qkv_buf_fp8_tensor   = Buffer2torchTensor(qkv_buf_fp8, false);
            const int64_t B   = (int64_t)batch_size;
            const int64_t S_q = (int64_t)seq_len;
            const int64_t S_k = (int64_t)seq_len_with_prefix;
            const int64_t H_q = (int64_t)head_num;
            const int64_t H_k = (int64_t)(kv_head_num > 0 ? kv_head_num : head_num);
            const int64_t D   = (int64_t)size_per_head;

            const int64_t q_elems = B * S_q * H_q * D;
            const int64_t k_elems = B * S_k * H_k * D;
            const int64_t v_elems = B * S_k * H_k * D;

            // auto input_reshape = params.input.reshape({batch_size * seq_len_with_prefix, head_num + kv_head_num*2, size_per_head});
            QBufferPtr q_hidden = std::dynamic_pointer_cast<QBuffer>(
                quantize(QuantizeParams(params.input, DataType::TYPE_QFP8_E4M3, 1, QScheme::Qfp8PerTensor, 0, 0)));
            torch::Tensor qkv_buf_fp8_tensor  = Buffer2torchTensor(q_hidden->kernelPtr(), false);
            
            // torch::Tensor scales_tensor = torch::full({1}, 1.0f, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            // torch::Tensor input_tensor  = Buffer2torchTensor(params.input, false);   
            // auto [qkv_buf_fp8_tensor, scale] = quantize_bf16_to_fp8(input_tensor, scales_tensor);
            // torch::Tensor qkv_buf_fp8_tensor = q_fp8;


            int64_t q_size = H_q * D;
            int64_t k_size = H_k * D;
            int64_t v_size = H_k * D;

            auto splits = torch::split(qkv_buf_fp8_tensor, {q_size, k_size, v_size}, /*dim=*/1);

            auto q_flat = splits[0]; // [bs, q_size]
            auto k_flat = splits[1]; // [bs, k_size]
            auto v_flat = splits[2]; // [bs, v_size]

            
            torch::Tensor q_tensor = q_flat.view({B, S_q, H_q, D}).contiguous();
            torch::Tensor k_tensor = k_flat.view({B, S_k, H_k, D}).contiguous();
            torch::Tensor v_tensor = v_flat.view({B, S_k, H_k, D}).contiguous();
            // torch::Tensor qkv_buf_fp8_tensor  = input_tensor.to(torch::kFP8E4M3Nuz) fuck

            // debugTensor(input_tensor, "input_tensor", 32);
            // input_tensor: [0.0140380859375, 0.150390625, 0.0908203125, -0.007659912109375, 0.294921875, 0.26953125, -0.36328125, 
            // -0.154296875, 0.4765625, 0.173828125, 0.1171875, 0.271484375, 2.296875, 0.232421875, -1.6953125, 0.0947265625, 
            // -0.373046875, -1.78125, 0.197265625, -0.74609375, 1.53125, -1.1328125, -1.359375, 1.234375, 1.2734375, 1.1796875,
            //  0.416015625, -1.8515625, -1.3046875, -1.578125, -0.07666015625, -0.2333984375]
            // debugTensor(qkv_buf_fp8_tensor, "qkv_buf_fp8_tensor", 129);

//    qkv_buf_fp8_tensor:  [0.013671875, 0.15625, 0.09375, -0.0078125, 0.28125, 0.28125, -0.375, -0.15625, 0.46875, 
//                          0.171875, 0.1171875, 0.28125, 2.25, 0.234375, -1.75, 0.09375, -0.375, -1.75, 0.203125, 
//                          -0.75, 1.5, -1.125, -1.375, 1.25, 1.25, 1.125, 0.40625, -1.875, -1.25, -1.625, -0.078125,
//                       -0.234375, 2.0, -0.013671875, 0.02734375, -1.5, -0.05078125, 0.1875, 0.5625, 1.375, 
//                       -1.125, 0.3125, 0.009765625, -1.875, -0.34375, -0.5625, -0.75, 0.3125, 0.234375, -0.9375,
//                       -1.125, -0.005859375, 1.75, 1.125, 3.75, 5.5, 0.6875, 0.40625, -2.5, 0.8125, 4.5, 0.5, 1.125,
//                        1.875, -0.025390625, -0.15625, -0.1875, -0.25, 0.140625, 0.21875, 0.234375, -0.34375, -0.125, 
//                        -2.25, 0.28125, -0.5, 0.234375, -0.625, -1.75, -0.1875, -1.125, 2.25, 0.8125, -0.005859375, -0.4375,
//                       0.4375, -0.25, -0.75, -1.0, -0.8125, 1.0, 0.875, 0.140625, 0.25, -0.6875, 0.375, -0.875, 0.34375, -0.8125,
//                        0.5625, -1.625, 0.21875, -0.375, 1.25, 0.046875, -0.75, 0.375, 1.25, -0.0703125, -0.5625, 0.6875, -0.5625, 
//                       -5.0, -0.01953125, -1.875, -2.0, -1.625, 0.375, 0.875, 0.375, -0.3125, -0.9375, -0.28125, 1.125, 2.25, -5.0,
//                        1.25, 1.125, 2.75]
            
 
 // log_tensor_meta("QKV_BUF", qkv_buf_fp8_tensor);
            // dump_u8_head(qkv_buf_fp8_tensor, 32, "QKV_BUF");            
            // std::cout << "[DEBUG_contextAttention] start 2" << std::endl;

            // 安全拉平（优先 reshape；如果你强制零拷贝，可用 as_strided 替代）
            // auto flat = qkv_buf_fp8_tensor.reshape({-1});
            // std::cout << "[DEBUG_contextAttention] start 3" << std::endl;
            // 若你想绝对零拷贝：
            // auto flat = qkv_buf_fp8_tensor.as_strided({qkv_buf_fp8_tensor.numel()}, {1});



            //（可选）健壮性检查：flat.numel() == q_elems + k_elems + v_elems
            // TORCH_CHECK(flat.numel() == q_elems + k_elems + v_elems, "QKV total numel mismatch");

            // std::cout << "[DEBUG_contextAttention] shape scalars"
            //         << " B=" << B << " S_q=" << S_q << " S_k=" << S_k
            //         << " H_q=" << H_q << " H_k=" << H_k << " D=" << D << " Dv=" << D
            //         << " hidden_units=" << hidden_units << " hidden_units_kv=" << hidden_units_kv
            //         << std::endl;

            // std::cout << "[DEBUG_contextAttention] elems"
            //         << " q=" << q_elems << " k=" << k_elems << " v=" << v_elems
            //         << " flat=" << flat.numel()
            //         << " (q+k+v)=" << (q_elems + k_elems + v_elems)
            //         << std::endl;

            // 只提示，不中断
            // if(flat.numel() != (q_elems + k_elems + v_elems)){
            //     std::cout << "[WARN] flat.numel != q+k+v. 当前切片假设为Monolithic(Q|K|V整块)。"
            //                 " 如果缓冲实际是“每token交错Q|K|V”，请改用 per-token 切法做对比验证（仅调试）。"
            //             << std::endl;
            // }
            // Q  [b, h, s, d] -> [b, s, h, d]
            // torch::Tensor q_tensor = flat.narrow(0, 0, q_elems)
            //     .view({B,  S_q,H_q, D})
            //     // .permute({0, 2, 1, 3})
            //     .contiguous();

            // // K [b, h, s, d] -> [b, s, h, d]
            // torch::Tensor k_tensor = flat.narrow(0, q_elems, k_elems)
            //     .view({B,  S_k, H_k,D})
            //     // .permute({0, 2, 1, 3})
            //     .contiguous();

            // // V [b, h, s, d] -> [b, s, h, d]
            // torch::Tensor v_tensor = flat.narrow(0, q_elems + k_elems, v_elems)
            //     .view({B,  S_k, H_k,D})
            //     // .permute({0, 2, 1, 3})
            //     .contiguous();

            // int64_t q_size = H_q * D;
            // int64_t k_size = H_k * D;
            // int64_t v_size = H_k * D;

            // auto splits = torch::split(qkv_buf_fp8_tensor, {q_size, k_size, v_size}, /*dim=*/1);

            // auto q_flat = splits[0]; // [7, q_size]
            // auto k_flat = splits[1]; // [7, k_size]
            // auto v_flat = splits[2]; // [7, v_size]

            // torch::Tensor q_tensor = q_flat.view({B, S_q, H_q, D}).contiguous();
            // torch::Tensor k_tensor = k_flat.view({B, S_k, H_k, D}).contiguous();
            // torch::Tensor v_tensor = v_flat.view({B, S_k, H_k, D}).contiguous();

            // dir = "/mnt/raid0/zhaoan12/cache/tmp_aiter/";
            // if ((int64_t)seq_len != 1){ // dump data
            //     // saveTorchDataTofile(as_uint8_bytes(qkv_buf_fp8_tensor), dir +"qkv_buf_fp8.pt");
            //     // saveTorchDataTofile(as_uint8_bytes(q_tensor), dir + "q_tensor.pt");
            //     // saveTorchDataTofile(as_uint8_bytes(k_tensor), dir + "k_tensor.pt");
            //     // saveTorchDataTofile(as_uint8_bytes(v_tensor), dir + "v_tensor.pt");
            //     auto qkv_u8 = qkv_buf_fp8_tensor.view(torch::kUInt8);
            //     auto q_u8   = q_tensor.view(torch::kUInt8);
            //     auto k_u8   = k_tensor.view(torch::kUInt8);
            //     auto v_u8   = v_tensor.view(torch::kUInt8);

            //     saveTorchDataTofile(qkv_u8, dir + "qkv_buf_fp8.pt");
            //     saveTorchDataTofile(q_u8,   dir + "q_tensor.pt");
            //     saveTorchDataTofile(k_u8,   dir + "k_tensor.pt");
            //     saveTorchDataTofile(v_u8,   dir + "v_tensor.pt");                
            // }

            // std::cout << "[DType] QKV_BUF.scalar_type=" << static_cast<int>(qkv_buf_fp8_tensor.scalar_type()) << std::endl;
            // std::cout << "[DType] Q.scalar_type=" << static_cast<int>(q_tensor.scalar_type()) << std::endl;

            // log_tensor_stats_f32("Q(fp32)", q_tensor);
            // log_tensor_stats_f32("K(fp32)", k_tensor);
            // log_tensor_stats_f32("V(fp32)", v_tensor);

            // count_fp8_expF(q_tensor, "Q");
            // count_fp8_expF(k_tensor, "K");
            // count_fp8_expF(v_tensor, "V");

            // log_tensor_meta("Q", q_tensor);
            // log_tensor_meta("K", k_tensor);
            // log_tensor_meta("V", v_tensor);  

            // dump_u8_head(q_tensor, 32, "Q");
            // dump_u8_head(k_tensor, 32, "K");
            // dump_u8_head(v_tensor, 32, "V");       

            // debugTensor(q_tensor, "Q", 100);
            // debugTensor(k_tensor, "K", 100);
            // debugTensor(v_tensor, "V", 100);
            // std::cout << "[DEBUG_contextAttention] start 4" << std::endl;

            torch::Tensor out_tensor = Buffer2torchTensor(params.output, /*copyData=*/false).view({B,S_q, H_q,D}); 

            // debugTensor(q_tensor, "Q", 100);
            // debugTensor(k_tensor, "K", 100);
            // debugTensor(v_tensor, "V", 100);                
            // log_tensor_meta("OUT(init)", out_tensor);

            // std::cout << "[DEBUG_contextAttention] start 5"  << std::endl;
            const float p_dropout     = 0.f;

            float softmax_scale = 0.f;

            if (softmax_scale == .0f)
                softmax_scale = 1.0 / ck_tile::sqrt(static_cast<float>(size_per_head));

                
            const bool  is_causal     = true; // mask=2
            const int   win_left      = -1;     // 全窗口
            const int   win_right     = is_causal ? 0 : -1;


            // std::cout << "[DEBUG_contextAttention] call_params"
            //         << " p_drop=0"
            //         << " scale_s=" << softmax_scale
            //         << " is_causal=" << is_causal
            //         << " win=(" << win_left << "," << win_right << ")"
            //         << " return_lse=0 return_rand=0"
            //         << std::endl;


            // std::cout << "[DEBUG_contextAttention] start aiter::torch_itfs::mha_fwd"  << std::endl;
            // if ((int64_t)seq_len != 1){ // dump data
            //     // 假设路径如下
            //     std::string q_path = "/mnt/raid0/zhaoan12/cache/tmp_ck/q_tensor.pt";
            //     std::string k_path = "/mnt/raid0/zhaoan12/cache/tmp_ck/k_tensor.pt";
            //     std::string v_path = "/mnt/raid0/zhaoan12/cache/tmp_ck/v_tensor.pt";

            //     // 加载为 torch::Tensor
            //     // 例：B=1, S_q=7, S_k=7, H_q=32, H_k=8, D=128
            //     auto q_cpu = loadTensorFromFile(q_path);
            //     auto k_cpu = loadTensorFromFile(k_path);
            //     auto v_cpu = loadTensorFromFile(v_path);

            //     debugTensor(q_cpu, "Q_CPU", 20);
            //     debugTensor(k_cpu, "K_CPU", 20);
            //     debugTensor(v_cpu, "V_CPU", 20);
            //     auto q_tensor_1 = toCudaFp8FromCpuBytes(q_cpu, {B, S_q, H_q, D});
            //     auto k_tensor_1 = toCudaFp8FromCpuBytes(k_cpu, {B, S_k, H_k, D});
            //     auto v_tensor_1 = toCudaFp8FromCpuBytes(v_cpu, {B, S_k, H_k, D});

            //     std::cout << "[DEBUG_contextAttention] start debugTensor"  << std::endl;
            //     // 假设你已经有 torch::Tensor q_tensor
            //     debugTensor(q_tensor_1, "Q", 20);
            //     debugTensor(k_tensor_1, "K", 20);
            //     debugTensor(v_tensor_1, "V", 20);
            //                     std::cout << "[DEBUG_contextAttention] start debugTensor 2 "  << std::endl;
                                
            // //     at::Tensor q_tensor_2 = q_tensor_1.device(torch::kCUDA);
            // //     at::Tensor k_tensor_2 = k_tensor_1.device(torch::kCUDA);
            // //     at::Tensor v_tensor_2 = v_tensor_1.device(torch::kCUDA);

            // // debugTensor(q_tensor_2, "Q", 20);
            // // debugTensor(k_tensor_2, "K", 20);
            // // debugTensor(v_tensor_2, "V", 20);

            // aiter::torch_itfs::mha_fwd(/*q*/ q_tensor_1, // b s h d
            //         /*k*/ k_tensor_1,
            //         /*v*/ v_tensor_1,
            //         /*p_dropout*/ p_dropout,
            //         /*softmax_scale*/ softmax_scale,
            //         /*is_causal*/ is_causal,
            //         /*window_size_left*/  win_left, // -1
            //         /*window_size_right*/ win_right, // 0
            //         /*return_softmax_lse*/ false,
            //         /*return_dropout_randval*/ false,
            //         /*cu_seqlens_q*/  std::nullopt,
            //         /*cu_seqlens_kv*/ std::nullopt,
            //         /*out_*/ out_tensor,         // b s h d        
            //         /*bias_*/ std::nullopt,
            //         /*alibi_slopes_*/ std::nullopt,
            //         /*gen_*/ std::nullopt);
            // std::cout << "[DEBUG_contextAttention] end aiter::torch_itfs::mha_fwd"  << std::endl;

            // debugTensor(out_tensor, "OUT", 20);

            // dir = "/mnt/raid0/zhaoan12/cache/tmp_aiter/";
            // if ((int64_t)seq_len != 1){ // dump data

            //     saveTorchDataTofile(out_tensor, dir +"output.pt");
            //     std::exit(1);
            // }            
            // }
// std::cout << "[DEBUG_contextAttention] start debugTensor 3 "  << std::endl;
//             debugTensor(q_tensor, "Q", 20);
//             debugTensor(k_tensor, "K", 20);
//             debugTensor(v_tensor, "V", 20);

//             std::cout << "[DEBUG_contextAttention] start debugTensor 4 "  << std::endl;
//             debugTensor(q_tensor, "Q", 20);
//             debugTensor(k_tensor, "K", 20);
//             debugTensor(v_tensor, "V", 20);
            aiter::torch_itfs::mha_fwd(/*q*/ q_tensor, // b s h d
                    /*k*/ k_tensor,
                    /*v*/ v_tensor,
                    /*p_dropout*/ p_dropout,
                    /*softmax_scale*/ softmax_scale,
                    /*is_causal*/ is_causal,
                    /*window_size_left*/  win_left, // -1
                    /*window_size_right*/ win_right, // 0
                    /*return_softmax_lse*/ false,
                    /*return_dropout_randval*/ false,
                    /*cu_seqlens_q*/  std::nullopt,
                    /*cu_seqlens_kv*/ std::nullopt,
                    /*out_*/ out_tensor,         // b s h d        
                    /*bias_*/ std::nullopt,
                    /*alibi_slopes_*/ std::nullopt,
                    /*gen_*/ std::nullopt);
            // std::cout << "[DEBUG_contextAttention] end aiter::torch_itfs::mha_fwd"  << std::endl;

            // debugTensor(out_tensor, "OUT", 20);

            // dir = "/mnt/raid0/zhaoan12/cache/tmp_aiter/";
            // if ((int64_t)seq_len != 1){ // dump data

            //     saveTorchDataTofile(out_tensor, dir +"output.pt");
            //     saveBufferDataToTorch(params.output, nullptr , "/mnt/raid0/zhaoan12/cache/tmp_flat/output.pt");
            //     std::exit(1);
            // }
            // torch::Tensor input1 = Buffer2torchTensor(params.input, false);
            // saveTorchDataTofile(input1, dir +"input.pt");
            // // 输出检查（不改变逻辑）
            // log_tensor_meta("OUT(final)", out_tensor);
            // log_tensor_stats_f32("OUT(final)", out_tensor);                  

        } else {
            fmha_runner_->runCKFmha(params.input.data(),
                                params.input.dataWithOffset(hidden_units),
                                params.input.dataWithOffset(hidden_units + hidden_units_kv),
                                params.output.data(),
                                nullptr,  // buffer for store out softmax_lse, looks like not used by RTP
                                batch_size,
                                seq_len,
                                prefix_prompt_param.max_prefix_prompt_length,
                                // context_token_num,
                                params.common.cu_seqlens->data(),
                                params.common.cu_kv_seqlens->data(),
                                lse_acc_buf->data(),
                                params.common.linear_bias_slopes ? params.common.linear_bias_slopes->data() : nullptr,
                                nullptr,
                                false,
                                false);
        }
        // if ((int64_t)seq_len != 1){ // dump data
        //     printBufferMeta("params.output", params.output);       
        //     std::exit(1);
        // } 
    } else {
        // Processing continuous/variable-length sequences
        torch::Tensor q_output_tensor, k_output_tensor, v_output_tensor;
        auto          q_contiguous = allocateBuffer(
            {params.input.type(), {head_num, seq_len * batch_size, size_per_head}, AllocationType::DEVICE},
            {"q_contiguous"});
        bufMemset(*q_contiguous, 0);
        auto k_contiguous = allocateBuffer({params.input.type(),
                                            {kv_head_num, seq_len_with_prefix * batch_size, size_per_head},
                                            AllocationType::DEVICE},
                                           {"k_contiguous"});
        bufMemset(*k_contiguous, 0);
        auto v_contiguous = allocateBuffer({params.input.type(),
                                            {kv_head_num, seq_len_with_prefix * batch_size, size_per_head},
                                            AllocationType::DEVICE},
                                           {"v_contiguous"});
        bufMemset(*v_contiguous, 0);
        const int hidden_size_q  = head_num * size_per_head;
        const int hidden_size_kv = kv_head_num * size_per_head;
        DISPATCH_CUDA_FUNCTION_DATA_TYPE(datatype,
                                         invokeGatherSequencesCombined,
                                         q_contiguous->data(),
                                         k_contiguous->data(),
                                         v_contiguous->data(),
                                         q_output->data(),
                                         k_output->data(),
                                         v_output->data(),
                                         params.common.cu_seqlens->data<int>(),
                                         params.common.cu_kv_seqlens->data<int>(),
                                         batch_size,
                                         seq_len,
                                         seq_len_with_prefix,
                                         head_num,
                                         kv_head_num,
                                         size_per_head,
                                         stream_);
        printBufferData(*q_contiguous, "q_contiguous");
        printBufferData(*k_contiguous, "k_contiguous");
        printBufferData(*v_contiguous, "v_contiguous");

        fmha_runner_->setup(
            datatype, params.configs.mask_type, head_num, kv_head_num, size_per_head, params.configs.q_scaling);

        auto lse_acc_buf = allocateBuffer({DataType::TYPE_FP32, {1, 1, 1, 1}, AllocationType::DEVICE}, {"lse_acc_buf"});
        if (fmha_runner_->runCKFmhaV2(q_contiguous->data(),
                                      k_contiguous->data(),
                                      v_contiguous->data(),
                                      params.output.data(),
                                      nullptr,
                                      batch_size,
                                      seq_len,
                                      params.common.max_prefix_length,
                                      params.common.cu_seqlens->data(),
                                      params.common.cu_kv_seqlens->data(),
                                      lse_acc_buf->data(),
                                      params.common.linear_bias_slopes ? params.common.linear_bias_slopes->data() :
                                                                         nullptr,
                                      nullptr,
                                      token_num,
                                      true,
                                      false)) {
            printBufferData(params.output, "run_ck_data_output");
            return;
        } else {
            RTP_LLM_CHECK_WITH_INFO(
                q_output && k_output && v_output,
                "q_output/k_output/v_output must be provided for default context attention implementation");
            q_output->updateShape({batch_size, kv_head_num, (head_num / kv_head_num) * seq_len, size_per_head});
            auto qk_output = gemm({*q_output,
                                   *k_output,
                                   std::nullopt,
                                   nullptr,
                                   DataType::TYPE_FP32,
                                   DataType::TYPE_FP32,
                                   TransposeOperation::NONE,
                                   TransposeOperation::TRANSPOSE});
            qk_output->updateShape({batch_size, head_num, seq_len, seq_len_with_prefix});
            printBufferData(*qk_output, "qk_output: ");
            float scale = (1.0f / sqrtf(size_per_head * 1.0f));  // * params.configs.softmax_extra_scale;
            auto  lengths_host =
                clone({params.common.input_lengths->view(decoder_batch_size, batch_size), AllocationType::HOST});
            auto prefix_lengths_host =
                params.common.prefix_prompt_lengths ?
                    clone({*params.common.prefix_prompt_lengths, AllocationType::HOST}) :
                    BufferPtr(new Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_INVALID, {0}, nullptr));
            auto attention_mask    = attentionMask({*lengths_host,
                                                    *prefix_lengths_host,
                                                    q_output->type(),
                                                    params.configs.mask_type == AttentionMaskType::causalMask});
            auto softmax_qk_output = softmax({std::move(qk_output), *attention_mask, nullopt, scale, datatype});
            softmax_qk_output->updateShape(
                {batch_size, kv_head_num, (head_num / kv_head_num) * seq_len, seq_len_with_prefix});
            printBufferData(*softmax_qk_output, "softmax_qk_output: ");

            auto qkv_output = gemm(
                {*softmax_qk_output, *v_output, std::nullopt, nullptr, DataType::TYPE_INVALID, params.compute_type});
            qkv_output->updateShape({batch_size, head_num, seq_len, size_per_head});
            printBufferData(*qkv_output, "qkv_output");
            auto& qkv_transpose_output = params.output;
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(datatype,
                                             invokeTransposeAttentionOutRemovePadding,
                                             qkv_output->data(),
                                             qkv_transpose_output.data(),
                                             token_num,
                                             batch_size,
                                             seq_len,
                                             head_num,
                                             size_per_head,
                                             params.common.padding_offset->data<int>(),
                                             nullptr,
                                             0,
                                             stream_);
            printBufferData(params.output, "run_ck_data_output");
            return;
        }
    }
}

template<typename T>
void selfAttentionwrapper(const AttentionModuleParams params,
                          bool                        use_multi_block_mode,
                          size_t                      max_seq_len_tile,
                          void*                       partial_out,
                          float*                      partial_sum,
                          float*                      partial_max,
                          int*                        block_counter,
                          KVBlockArray                kv_block_array,
                          cudaStream_t                stream) {
    size_t      batch_size        = params.common.decoder_batch_size;
    size_t      step              = params.common.decoder_max_seq_len + 1;
    size_t      local_head_num    = params.configs.head_num;
    size_t      local_head_num_kv = params.configs.kv_head_num;
    size_t      size_per_head     = params.configs.size_per_head;
    const auto& output            = params.output;

    const T* qkv_buf_ptr  = params.input.data<T>();
    void*    attn_out_ptr = nullptr;
    attn_out_ptr          = output.data();

    const T* bias_ptr = (params.weights.qkv_weight->bias == nullptr || !params.configs.fuse_qkv_add_bias) ?
                            nullptr :
                            params.weights.qkv_weight->bias->data<T>();

    const auto* input_lengths    = params.common.input_lengths->data<int>();
    const auto* sequence_lengths = params.common.sequence_lengths->data<int>();

    float        q_scaling = params.configs.q_scaling;
    const float* linear_bias_slopes =
        params.common.linear_bias_slopes ? params.common.linear_bias_slopes->data<float>() : nullptr;

    tensorrt_llm::common::QuantMode kv_cache_quant_mode =
        trt_common::QuantMode::fromDescription(false, false, false, false, false, false, false, false);
    if (params.configs.kv_cache_dtype == KvCacheDataType::INT8) {
        kv_cache_quant_mode =
            trt_common::QuantMode::fromDescription(true, true, false, false, false, true, false, true);
    }

    const float* attention_output_orig_quant_scale = nullptr;
    if (params.weights.static_scale_reciprocal_weight) {
        attention_output_orig_quant_scale = params.weights.static_scale_reciprocal_weight->kernel->data<float>();
    }

    fusedQKV_masked_attention_dispatch<T, KVBlockArray>(
        qkv_buf_ptr,
        bias_ptr,
        nullptr,  // relative_attention_bias
        nullptr,  // cache_indir
        reinterpret_cast<T*>(attn_out_ptr),
        nullptr,  // finished
        sequence_lengths,
        batch_size,
        1,  // beam_width
        local_head_num,
        local_head_num_kv,
        size_per_head,
        params.configs.rope_config,
        params.configs.use_logn_attn,
        params.common.position_ids ? params.common.position_ids->data<int>() : nullptr,
        step,
        nullptr,  // prefix_prompt_lengths
        0,        // max_prefix_prompt_length
        true,     // count_prefix_length
        input_lengths,
        step,
        q_scaling,
        0,  // relative_attention_bias_stride,
        linear_bias_slopes,
        nullptr,  // masked_tokens,
        nullptr,  // query_weight_scale_out
        attention_output_orig_quant_scale,
        0,  // int8_mode,
        kv_cache_quant_mode,
        use_multi_block_mode,
        (int)max_seq_len_tile,
        reinterpret_cast<T*>(partial_out),
        partial_sum,
        partial_max,
        block_counter,
        params.configs.softmax_extra_scale,
        kv_block_array,
        stream);
    check_cuda_error();
}

AttentionModuleOutput ROCmDevice::decoderSelfAttention(const AttentionModuleParams& params) {
    auto      datatype         = params.input.type();
    size_t    max_seq_len_tile = 0;
    BufferPtr partial_out      = nullptr;
    BufferPtr partial_sum      = nullptr;
    BufferPtr partial_max      = nullptr;
    BufferPtr block_counter    = nullptr;

    size_t batch_size     = params.common.decoder_batch_size;
    size_t local_head_num = params.configs.head_num;
    size_t size_per_head  = params.configs.size_per_head;

    if (use_multi_block_mode) {
        const int threads_per_value = pow2roundup(size_per_head) * getTypeSize(datatype) / 16;
        // for allocate partial output results memory. Regardless to THDS_PER_BLOCK
        max_seq_len_tile = 256 / threads_per_value;
        partial_out      = allocateBuffer(
            {datatype, {batch_size, max_seq_len_tile, local_head_num, size_per_head}, AllocationType::DEVICE},
            {"partial_out"});
        partial_sum = allocateBuffer(
            {DataType::TYPE_FP32, {batch_size, max_seq_len_tile, local_head_num}, AllocationType::DEVICE},
            {"partial_sum"});
        partial_max = allocateBuffer(
            {DataType::TYPE_FP32, {batch_size, max_seq_len_tile, local_head_num}, AllocationType::DEVICE},
            {"partial_max"});
        block_counter = allocateBuffer({DataType::TYPE_INT32, {batch_size, local_head_num}, AllocationType::DEVICE},
                                       {"block_counter"});
        // TODO(lidongjin) use fill op to set zeros.
        cudaMemsetAsync(block_counter->data(), 0, sizeof(int) * batch_size * local_head_num, stream_);
    }
    void*  partial_out_data   = (partial_out == nullptr) ? nullptr : partial_out->data();
    float* partial_sum_data   = (partial_sum == nullptr) ? nullptr : partial_sum->data<float>();
    float* partial_max_data   = (partial_max == nullptr) ? nullptr : partial_max->data<float>();
    int*   block_counter_data = (block_counter == nullptr) ? nullptr : block_counter->data<int>();

    RUNTIME_ASSERT_OP_ARG(params.common.kv_cache, "kv cache can not be null for decoder self-attention");
    const auto max_blocks_per_batch = params.common.kv_cache->kv_cache_block_id->shape()[1];
    auto       kv_cache_offset      = allocateBuffer(
        {DataType::TYPE_INT32, {batch_size, 1, 2, max_blocks_per_batch}, AllocationType::DEVICE}, {"kv_cache_offset"});

    if (init_params_.use_aiter_pa) {
        PrefixPromptBatchWeightsParam prefix_prompt_param;
        if (init_params_.use_asm_pa) {
            KVBlockArray kv_block_array = getKVBlockArray(params, *kv_cache_offset, batch_size, params.common.kv_cache->k_cache_buffer->type() == DataType::TYPE_FP8_E4M3, false);
            prefix_prompt_param.kv_block_array = kv_block_array;
        }
        else {
            KVBlockArray kv_block_array = getKVBlockArray(params, *kv_cache_offset, batch_size, params.common.kv_cache->k_cache_buffer->type() == DataType::TYPE_FP8_E4M3, true);
            auto offset_kv_block_array = OffsetIndexedKVBlockArray(
                                            kv_block_array,
                                            (rtp_llm::KVBlockArrayForContextFMHA::DataType*)params.common.kv_cache->kv_cache_block_id->data(),
                                            params.common.kv_cache->k_cache_buffer->shape()[0] * params.common.kv_cache->layer_num);
            prefix_prompt_param.kv_block_array = kv_block_array;
            prefix_prompt_param.offset_kv_block_array = offset_kv_block_array;
        }

        auto   token_num          = params.input.shape()[0];
        auto   decoder_batch_size = params.common.decoder_batch_size;
        auto   head_num           = params.configs.head_num;
        auto   kv_head_num        = params.configs.kv_head_num;
        size_t seq_len            = 1;

        auto q_output = allocateBuffer(
            {params.input.type(), {batch_size, head_num, size_per_head}, AllocationType::DEVICE}, {"q_output"});

        bool        store_qkv        = false;
        bool        store_q          = true;
        bool        store_kv         = false;
        bool        store_cache      = params.common.kv_cache.has_value();
        const auto* sequence_lengths = params.common.sequence_lengths->data<int>();
        const auto* input_lengths    = params.common.input_lengths->data<int>();

        bool skip_add_bias_transpose = (params.configs.rope_config.style == RopeStyle::No && !params.common.kv_cache
                                        && !params.configs.fuse_qkv_add_bias);
        printBufferData(*params.common.input_lengths, "input_lengths");
        if (!skip_add_bias_transpose) {
            static torch::Tensor cos_sin_cache = getRopeCosSin(params.configs.rope_config.style,
                                                               params.configs.rope_config.dim,
                                                               params.configs.rope_config.base,
                                                               params.configs.rope_config.scale,
                                                               init_params_.max_seq_len);
            if (init_params_.use_asm_pa) {
                DISPATCH_CUDA_FUNCTION_DATA_TYPE(datatype,
                                             invokeAddFusedQKVBiasTransposeDecode,
                                             q_output->data(),
                                             nullptr,
                                             nullptr,
                                             &prefix_prompt_param,
                                             input_lengths,
                                             params.input.data(),
                                             nullptr,
                                             params.common.position_ids ? params.common.position_ids->data<int>() :
                                                                          nullptr,
                                             params.configs.fuse_qkv_add_bias && params.weights.qkv_weight->bias ?
                                                 params.weights.qkv_weight->bias->data() :
                                                 nullptr,
                                             /*params.common.padding_offset->data<int>(),*/ nullptr,
                                             /*params.common.cu_seqlens->data<int>(),*/ nullptr,
                                             params.common.sequence_lengths->data<int>(),
                                             batch_size,
                                             seq_len,
                                             token_num,
                                             head_num,
                                             kv_head_num,
                                             size_per_head,
                                             params.configs.rope_config,
                                             params.configs.use_logn_attn,
                                             nullptr,
                                             0,
                                             false,
                                             store_qkv,
                                             store_q,
                                             store_kv,
                                             store_cache,
                                             cos_sin_cache.defined() ? static_cast<float2*>(cos_sin_cache.data_ptr()) : nullptr,
                                             stream_);
            } else {
                RUNTIME_ASSERT_OP_ARG(init_params_.use_asm_pa, "Should use asm_pa");
            }
            check_cuda_error();
            DEBUG_PRINT_PARAMS(params, this, "decode_writeKVCache", q_output);
            if (init_params_.use_asm_pa) {
                runAiterAsmPA(params, this, *q_output);
            }
            else {
                runAiterPA(params, this, *q_output);
            }
            check_cuda_error();
        }
    } else {
        KVBlockArray kv_block_array = getKVBlockArray(params, *kv_cache_offset, batch_size, params.common.kv_cache->k_cache_buffer->type() == DataType::TYPE_FP8_E4M3);

        DISPATCH_CUDA_FUNCTION_DATA_TYPE(datatype,
                                         selfAttentionwrapper,
                                         params,
                                         use_multi_block_mode,
                                         max_seq_len_tile,
                                         partial_out_data,
                                         partial_sum_data,
                                         partial_max_data,
                                         block_counter_data,
                                         kv_block_array,
                                         stream_);
        check_cuda_error();
        DEBUG_PRINT_PARAMS(params, this, "decode_attn");
    }
}

}  // namespace rtp_llm
