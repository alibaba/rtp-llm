load("//:def.bzl", "cuda_copts")

def compile_xqa_libs():
    xqa_libs = []
    for page_size in [16, 32, 64, 128]:
        for group_size in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
            xqa_name = 'xqa_sm90' + '_ps' + str(page_size) + '_gs' + str(group_size)
            native.cc_library(
                name = xqa_name,
                srcs = native.glob([
                    "*.cu",
                ]),
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
                    '-DTOKENS_PER_PAGE=' + str(page_size),
                    '-DHEAD_GRP_SIZE=' + str(group_size)
                ],
                visibility = ["//visibility:public"],
            )
            xqa_libs.append(xqa_name)
    return xqa_libs
