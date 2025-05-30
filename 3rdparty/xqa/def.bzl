load("//:def.bzl", "cuda_copts")

def compile_xqa_libs():
    xqa_libs = []
    for head_dim in [64, 128, 256]:
        for page_size in [16, 32, 64, 128]:
            for group_size in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
                xqa_name = 'xqa_sm90' + '_hd' + str(head_dim) + '_ps' + str(page_size) + '_gs' + str(group_size)
                xqa_kernel_name = 'xqa_kernel_sm90' + '_hd' + str(head_dim) + '_ps' + str(page_size) + '_gs' + str(group_size)
                native.cc_library(
                    name = xqa_name,
                    srcs = ["mha_sm90.cu"],
                    hdrs = native.glob([
                        "*.h",
                        "*.cuh",
                    ]),
                    deps = [
                        "@local_config_cuda//cuda:cuda_headers",
                        "@local_config_cuda//cuda:cudart",
                        "@local_config_cuda//cuda:cuda_driver",
                    ],
                    copts = cuda_copts() + [
                        '--cuda-include-ptx=sm_90a',
                        '--cuda-gpu-arch=sm_90a',
                        "--expt-relaxed-constexpr",
                        '-DXQA_FUNC_SM90=' + xqa_name,
                        '-DXQA_KERNEL_SM90=' + xqa_kernel_name,
                        '-DHEAD_ELEMS=' + str(head_dim),
                        '-DTOKENS_PER_PAGE=' + str(page_size),
                        '-DHEAD_GRP_SIZE=' + str(group_size)
                    ],
                    visibility = ["//visibility:public"],
                )
                xqa_libs.append(xqa_name)
    return xqa_libs
