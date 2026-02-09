py_test(
    name = "vip_server_test",
    srcs = [
        "vip_server_test.py",
    ],
    deps = [
        "//rtp_llm:ops",
        "//rtp_llm:testlib",
    ],
    exec_properties = {'gpu': 'A10'},
    tags = ["frontend"],
)

# py_test(
#     name = "frontend_start_test",
#     srcs = [
#         "frontend_app_test.py",
#     ],
#     main = "frontend_app_test.py",
#     deps = [
#         "//rtp_llm:ops",
#         "//rtp_llm:testlib",
#     ],
#     env = {
#         "CHECKPOINT_PATH": "/mnt/nas1/hf/Qwen2-1.5B-Instruct/",
#         "MODEL_TYPE": "qwen_2",
#         "ACT_TYPE": "bf16",
#         "CONCURRENCY_LIMIT": "128",
#         "GPU_COUNT": "0",
#     },
#     exec_properties = {'gpu': 'NO_GPU', "gpu_count": "0"},
#     tags = ["frontend"],
# )