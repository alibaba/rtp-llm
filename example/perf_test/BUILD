load("//example/perf_test:defs.bzl", "latency_test", "throughput_test", "vit_test")

test_input_default = [
    [1, 16],
    [1, 128],
    [1, 512],
    [1, 1024],
    [4, 16],
    [4, 128],
    [4, 512],
    [8, 16],
    [8, 128],
    [8, 512],
    [4, 1024], # use last case seqlen as max seqlen
]

test_input_2048 = [
    [1, 2048],
    [4, 2048],
    [6, 2048],
]

test_input_long_seq_7b_1 = [
    [1, 4096],
    [4, 4096],
    [8, 4096],
    [1, 8192],
    [4, 8192],
    [8, 8192],
]

test_input_long_seq_7b_2 = [
    [1, 16384],
    [2, 16384],
    [1, 24576],
    [2, 24576],
    [1, 32768],
    [2, 32768],
]

test_input_long_seq_13b_1 = [
    [1, 4096],
    [4, 4096],
    [1, 8192],
    [4, 8192],
]

test_input_long_seq_13b_2 = [
    [1, 16384],
    [2, 16384],
    [1, 24576],
    [2, 24576],
    [1, 32768],
]

test_input_batch_7b = [
    [8, 1024],
    [16, 1024],
    [32, 1024],
    [8, 2048],
    [16, 2048],
    [32, 2048],
    [8, 4096],
    [16, 4096],
]

test_input_qwen72B = [
    [1, 16], 
    [1, 128],
    [1, 512],
    [1, 1024],     
    [4, 16],
    [4, 128],
    [4, 512],
    [8, 16],
    [8, 128],
    [1, 2048],
]

latency_test('qwen_1b8', '1.8', 'int8', 'Qwen/Qwen-1_8B', test_input_default + test_input_2048)
latency_test('qwen_1b8', '1.8', 'fp16', 'Qwen/Qwen-1_8B', test_input_default + test_input_2048)
throughput_test('qwen_1b8', '1.8', 'int8', 'Qwen/Qwen-1_8B')
throughput_test('qwen_1b8', '1.8', 'fp16', 'Qwen/Qwen-1_8B')

latency_test('llama', '7', 'int8', 'OpenBuddy/openbuddy-openllama-7b-v12-bf16', test_input_default)
latency_test('llama', '7', 'int8', 'OpenBuddy/openbuddy-openllama-7b-v12-bf16', test_input_default + test_input_2048 + test_input_long_seq_7b_1 + test_input_batch_7b + test_input_long_seq_7b_2, name_suffix='_80G', reserve_mem=4096)
latency_test('llama', '7', 'fp16', 'OpenBuddy/openbuddy-openllama-7b-v12-bf16', test_input_default)
latency_test('llama', '7', 'fp16', 'OpenBuddy/openbuddy-openllama-7b-v12-bf16', test_input_default + test_input_2048 + test_input_long_seq_7b_1 + test_input_batch_7b + test_input_long_seq_7b_2, name_suffix='_80G', reserve_mem=8192)

throughput_test('llama', '7', 'int8', 'OpenBuddy/openbuddy-openllama-7b-v12-bf16')
throughput_test('llama', '7', 'fp16', 'OpenBuddy/openbuddy-openllama-7b-v12-bf16')
throughput_test('llama', '7', 'int8', 'OpenBuddy/openbuddy-openllama-7b-v12-bf16', name_suffix='_80G', reserve_mem=4096)
throughput_test('llama', '7', 'fp16', 'OpenBuddy/openbuddy-openllama-7b-v12-bf16', name_suffix='_80G', reserve_mem=4096)

latency_test('qwen_2', '7', 'int4', 'Qwen/Qwen1.5-7B-Chat-GPTQ-Int4', test_input_default)
latency_test('qwen_2', '7', 'int4', 'Qwen/Qwen1.5-7B-Chat-GPTQ-Int4', test_input_default + test_input_2048 + test_input_long_seq_13b_1 + test_input_long_seq_13b_2, name_suffix='_80G', reserve_mem=4096)
latency_test('qwen_2', '72', 'fp16', 'Qwen/Qwen2-72B-Instruct', test_input_qwen72B)

test_suite(
    name = "all_perf_test",
    tests = [
        ':qwen_1b8_1.8_int8',
        ':qwen_1b8_1.8_fp16',
        ':llama_7_int8',
        ':llama_7_fp16',
        ':qwen_2_7_int4',
        ':qwen_1b8_1.8_int8_throughput',
        ':qwen_1b8_1.8_fp16_throughput',
        ':llama_7_int8_throughput',
        ':llama_7_fp16_throughput',
    ]
)

test_suite(
    name = "80G_perf_test",
    tests = [
        ':llama_7_int8_80G',
        ':llama_7_fp16_80G',
        ':qwen_2_7_int4_80G',
        ':llama_7_int8_80G_throughput',
        ':llama_7_fp16_80G_throughput',
    ]
)

test_image_input = [
    300, 500, 1000, 2000, 4000
]

vit_test('qwen2_vl', '/mnt/nas1/hf/Qwen2-VL-7B-Instruct/', test_image_input)

test_suite(
    name = "vit_test",
    tests = ['qwen2_vl_vit_test']
)
