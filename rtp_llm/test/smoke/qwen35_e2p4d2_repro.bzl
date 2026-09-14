load('//rtp_llm/test/smoke:defs.bzl', 'SMOKE_FRAMEWORK_DEPS')

def qwen35_e2p4d2_repro_suite():
    native.py_test(
        name = 'qwen35_e2p4d2_repro',
        main = 'qwen35_e2p4d2_repro.py',
        srcs = ['qwen35_e2p4d2_repro.py'],
        args = ['--e-batch=8'],
        deps = SMOKE_FRAMEWORK_DEPS,
        data = native.glob(['qwen35_e2p4d2_data/**']),
        env = {'GPU_COUNT': '8', 'WORLD_SIZE': '8', 'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7', 'PYTHONNOUSERSITE': '1'},
        exec_properties = {'gpu': 'L20D', 'gpu_count': '8'},
        timeout = 'eternal',
        tags = ['manual', 'L20D'],
        legacy_create_init = 0,
    )
