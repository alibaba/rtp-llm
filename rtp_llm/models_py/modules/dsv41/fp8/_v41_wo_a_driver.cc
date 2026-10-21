#include <cuda.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

struct EpilogueArgs {
    uint32_t* sfd;
    uint32_t  stride, m, n;
    float     legacy;
};

struct V41Module {
    CUmodule   module       = nullptr;
    CUfunction functions[2] = {};
    CUcontext  context      = nullptr;
    unsigned   metadata[5]  = {};
};

#define V41_CHECK(expr)                                                                                                \
    do {                                                                                                               \
        CUresult code = (expr);                                                                                        \
        if (code != CUDA_SUCCESS)                                                                                      \
            return code;                                                                                               \
    } while (0)

extern "C" int v41_wo_a_init(const char* cubin, int device, void** output) {
    auto     module = std::make_unique<V41Module>();
    CUdevice current;
    V41_CHECK(cuCtxGetDevice(&current));
    if (current != device)
        return CUDA_ERROR_INVALID_CONTEXT;
    V41_CHECK(cuCtxGetCurrent(&module->context));
    V41_CHECK(cuModuleLoad(&module->module, cubin));
    // All validation and function loading happen during startup, outside capture.
    auto validate = [&]() -> CUresult {
        CUdeviceptr metadata;
        size_t      bytes;
        V41_CHECK(cuModuleGetGlobal(&metadata, &bytes, module->module, "v41_wo_a_metadata"));
        if (bytes != sizeof(module->metadata))
            return CUDA_ERROR_INVALID_VALUE;
        V41_CHECK(cuMemcpyDtoH(module->metadata, metadata, bytes));
        if (module->metadata[3] != sizeof(EpilogueArgs) || module->metadata[4] != offsetof(EpilogueArgs, legacy))
            return CUDA_ERROR_INVALID_VALUE;
        unsigned count;
        V41_CHECK(cuModuleGetFunctionCount(&count, module->module));
        if (count != 2)
            return CUDA_ERROR_INVALID_VALUE;
        CUfunction functions[2];
        V41_CHECK(cuModuleEnumerateFunctions(functions, 2, module->module));
        for (auto function : functions) {
            const char* name;
            V41_CHECK(cuFuncGetName(&name, function));
            // Identify the two fixed instantiations, independent of dynamic M/SMS.
            int index = std::strstr(name, "ELj1024ELj4096ELj32ELj128ELj128ELj8")  ? 0 :
                        std::strstr(name, "ELj1024ELj4096ELj128ELj256ELj128ELj8") ? 1 :
                                                                                    -1;
            if (index < 0 || module->functions[index])
                return CUDA_ERROR_INVALID_VALUE;
            V41_CHECK(
                cuFuncSetAttribute(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, module->metadata[index]));
            V41_CHECK(cuFuncLoad(function));
            module->functions[index] = function;
        }
        return CUDA_SUCCESS;
    };
    const auto status = validate();
    if (status != CUDA_SUCCESS) {
        cuModuleUnload(module->module);
        return status;
    }
    *output = module.release();
    return CUDA_SUCCESS;
}

extern "C" int v41_wo_a_is_current(void* handle) {
    CUcontext context = nullptr;
    return handle && cuCtxGetCurrent(&context) == CUDA_SUCCESS && context != nullptr
           && context == static_cast<V41Module*>(handle)->context;
}

static CUresult
make3(CUtensorMap& map, void* ptr, unsigned m, unsigned n, uint64_t stride_m, uint64_t stride_g, unsigned box_m) {
    const cuuint64_t dims[] = {n, m, 8}, strides[] = {stride_m, stride_g};
    const cuuint32_t box[] = {128, box_m, 1}, elem[] = {1, 1, 1};
    return cuTensorMapEncodeTiled(&map,
                                  CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                  3,
                                  ptr,
                                  dims,
                                  strides,
                                  box,
                                  elem,
                                  CU_TENSOR_MAP_INTERLEAVE_NONE,
                                  CU_TENSOR_MAP_SWIZZLE_128B,
                                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

static CUresult make_sf(CUtensorMap& map, void* ptr, unsigned m, unsigned box_m) {
    const cuuint64_t dims[] = {m, 256}, strides[] = {m * 4ull};
    const cuuint32_t box[] = {box_m, 1}, elem[] = {1, 1};
    return cuTensorMapEncodeTiled(&map,
                                  CU_TENSOR_MAP_DATA_TYPE_INT32,
                                  2,
                                  ptr,
                                  dims,
                                  strides,
                                  box,
                                  elem,
                                  CU_TENSOR_MAP_INTERLEAVE_NONE,
                                  CU_TENSOR_MAP_SWIZZLE_NONE,
                                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

extern "C" int v41_wo_a_launch(void*    handle,
                               void*    a,
                               void*    sa,
                               void*    b,
                               void*    sb,
                               void*    q,
                               void*    sq,
                               unsigned m,
                               uint64_t stride_m,
                               uint64_t stride_g,
                               unsigned legacy,
                               void*    stream) {
    auto&     module = *static_cast<V41Module*>(handle);
    CUcontext context;
    V41_CHECK(cuCtxGetCurrent(&context));
    if (context != module.context)
        return CUDA_ERROR_INVALID_CONTEXT;
    // Two-CTA M pairing requires an even number of M tiles.
    const bool  small = ((m + 127) / 128) % 2 != 0;
    const int   index = small ? 0 : 1;
    CUtensorMap ma, mb, msa, msb, md;
    V41_CHECK(make3(ma, a, m, 4096, stride_m, stride_g, small ? 16 : 128));
    V41_CHECK(make3(mb, b, 1024, 4096, 4096, 1024ull * 4096, 128));
    V41_CHECK(make3(md, q, m, 1024, 8192, 1024, small ? 16 : 128));
    V41_CHECK(make_sf(msa, sa, (m + 3) / 4 * 4, 128));
    V41_CHECK(make_sf(msb, sb, 1024, small ? 128 : 256));
    EpilogueArgs      epilogue{static_cast<uint32_t*>(sq), (m + 3) / 4 * 4, m, 1024, legacy ? 1.0f : 0.0f};
    CUlaunchAttribute attr{};
    attr.id               = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    attr.value.clusterDim = {2, 1, 1};
    CUlaunchConfig config{};
    config.gridDimX = module.metadata[2];
    config.gridDimY = config.gridDimZ = 1;
    config.blockDimX                  = 256;
    config.blockDimY = config.blockDimZ = 1;
    config.sharedMemBytes               = module.metadata[index];
    config.hStream                      = static_cast<CUstream>(stream);
    config.attrs                        = &attr;
    config.numAttrs                     = 1;
    void*    layout                     = nullptr;
    unsigned n = 1024, k = 4096;
    void*    args[] = {&layout, &m, &n, &k, &epilogue, &ma, &mb, &msa, &msb, &md};
    return cuLaunchKernelEx(&config, module.functions[index], args, nullptr);
}

#undef V41_CHECK
