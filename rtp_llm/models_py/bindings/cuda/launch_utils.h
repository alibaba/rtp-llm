#pragma once

namespace rtp_llm {

// Programmatic Dependent Launch is unconditionally enabled on SM>=90:
// the retired disable_pdl switch was never set to enable PDL anywhere,
// and the launched kernels carry griddepcontrol.wait/launch_dependents.
// Pre-Hopper devices keep the plain launch.
#if USING_CUDA
#define LAUNCH_KERNEL_WITH_PDL(kernel, grid_dim, block_dim, smem_size, stream, ...)                                     \
    do {                                                                                                                \
        int cc_major__ = getComputeCapabilityMajor();                                                                   \
        if (cc_major__ >= 9) {                                                                                          \
            cudaLaunchConfig_t config__;                                                                                \
            config__.gridDim          = (grid_dim);                                                                     \
            config__.blockDim         = (block_dim);                                                                    \
            config__.dynamicSmemBytes = (smem_size);                                                                    \
            config__.stream           = (stream);                                                                       \
            cudaLaunchAttribute attrs__[1];                                                                             \
            attrs__[0].id                                         = cudaLaunchAttributeProgrammaticStreamSerialization; \
            attrs__[0].val.programmaticStreamSerializationAllowed = 1;                                                  \
            config__.numAttrs                                     = 1;                                                  \
            config__.attrs                                        = attrs__;                                            \
            cudaLaunchKernelEx(&config__, &(kernel), __VA_ARGS__);                                                      \
        } else {                                                                                                        \
            (kernel)<<<(grid_dim), (block_dim), (smem_size), (stream)>>>(__VA_ARGS__);                                  \
        }                                                                                                               \
    } while (0)

#else
#define LAUNCH_KERNEL_WITH_PDL(kernel, grid_dim, block_dim, smem_size, stream, ...)                                    \
    (kernel)<<<(grid_dim), (block_dim), (smem_size), (stream)>>>(__VA_ARGS__);

#endif

}  // namespace rtp_llm
