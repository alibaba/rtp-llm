load("//:def.bzl", "cuda_copts")


def build_lib(head_dim, page_size, group_size, xqa_name, xqa_kernel_name, input_type_id, kv_cache_type_id, output_type_id, spec_dec):
    addition_cuda_copts = [
        '--cuda-include-ptx=sm_90a',
        '--cuda-gpu-arch=sm_90a',
        "-nvcc_options=expt-relaxed-constexpr",
        '-nvcc_options=use_fast_math',
        "-nvcc_options=diag-suppress=177,550,186",
        '-t 0',
        '-res-usage',
        '-DUSE_INPUT_KV=0',
        '-DXQA_FUNC_SM90=' + xqa_name,
        '-DXQA_KERNEL_SM90=' + xqa_kernel_name,
        '-DHEAD_ELEMS=' + str(head_dim),
        '-DTOKENS_PER_PAGE=' + str(page_size),
        '-DHEAD_GRP_SIZE=' + str(group_size),
        '-DINPUT_FP16=' + str(input_type_id),
        '-DCACHE_ELEM_ENUM=' + str(kv_cache_type_id),
        '-DLOW_PREC_OUTPUT=' + str(output_type_id)
    ]
    if spec_dec:
        addition_cuda_copts += [
            '-DSPEC_DEC=1'
        ]
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
            "//3rdparty:cuda_driver"
        ],
        copts = cuda_copts() + addition_cuda_copts,
        visibility = ["//visibility:public"],
    )


def compile_xqa_libs():
    xqa_libs = []
    for head_dim in [64, 128, 256]:
        for page_size in [16, 32, 64, 128]:
            for group_size in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
                for input_type in ["__nv_bfloat16", "half"]:
                    for kv_cache_type in [input_type, "__nv_fp8_e4m3"]:
                        for output_type in [input_type, "__nv_fp8_e4m3"]:
                            input_type_id = 0 if input_type == "__nv_bfloat16" else 1
                            kv_cache_type_id = 2 if kv_cache_type == "__nv_fp8_e4m3" else 0
                            output_type_id = 1 if output_type == "__nv_fp8_e4m3" else 0
                            xqa_name = 'xqa_sm90' + '_hd' + str(head_dim) + '_ps' + str(page_size) + '_gs' + str(group_size) + '_input_' + input_type + '_kv_cache_' + kv_cache_type + '_output_' + output_type
                            xqa_kernel_name = 'xqa_kernel_sm90' + '_hd' + str(head_dim) + '_ps' + str(page_size) + '_gs' + str(group_size) + '_input_' + input_type + '_kv_cache_' + kv_cache_type + '_output_' + output_type
                            if output_type != "__nv_fp8_e4m3":
                                build_lib(head_dim, page_size, group_size, xqa_name, xqa_kernel_name, input_type_id, kv_cache_type_id, output_type_id, False)
                                xqa_libs.append(xqa_name)

                                if kv_cache_type == "__nv_fp8_e4m3":
                                    xqa_name += '_spec_dec'
                                    xqa_kernel_name += '_spec_dec'
                                    build_lib(head_dim, page_size, group_size, xqa_name, xqa_kernel_name, input_type_id, kv_cache_type_id, output_type_id, True)
                                    xqa_libs.append(xqa_name)
                            else:
                                if kv_cache_type == "__nv_fp8_e4m3":
                                    build_lib(head_dim, page_size, group_size, xqa_name, xqa_kernel_name, input_type_id, kv_cache_type_id, output_type_id, False)
                                    xqa_libs.append(xqa_name)

                                    xqa_name += '_spec_dec'
                                    xqa_kernel_name += '_spec_dec'
                                    build_lib(head_dim, page_size, group_size, xqa_name, xqa_kernel_name, input_type_id, kv_cache_type_id, output_type_id, True)
                                    xqa_libs.append(xqa_name)

    return xqa_libs
