load("//maga_transformer/test/perf_test:defs.bzl", "perf_test")

test_input_default = [
    [1, 16],
    [1, 128],
    [1, 512],
    [1, 1024],
    [4, 16],
    [4, 128],
    [4, 512],
    [4, 1024],
    [8, 16],
    [8, 128],
    [8, 512],
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

test_suite(
    name = "all_perf_test",
    tests = [
        perf_test('qwen_1b8', '1.8', 'int8', '/mnt/nas1/perf_test/qwen-1b8', test_input_default, max_seq_len=1024),
        perf_test('turing_006_3b', '3', 'int8', '/mnt/nas1/perf_test/turing-3b', test_input_default, max_seq_len=1024),
        perf_test('llama', '7', 'int8', '/mnt/nas1/perf_test/llama-7b', test_input_default, max_seq_len=1024),
        perf_test('llama', '13', 'int8', '/mnt/nas1/perf_test/llama-13b', test_input_default, max_seq_len=1024),
        perf_test('gpt_bigcode', '15', 'int8', '/mnt/nas1/perf_test/wizardcoder-15b', test_input_default, max_seq_len=1024),
        perf_test('qwen_1b8', '1.8', 'int8', '/mnt/nas1/perf_test/qwen-1b8', test_input_2048, name_suffix='_2048'),
        perf_test('turing_006_3b', '3', 'int8', '/mnt/nas1/perf_test/turing-3b', test_input_2048, name_suffix='_2048'),
        perf_test('llama', '7', 'int8', '/mnt/nas1/perf_test/llama-7b', test_input_2048, name_suffix='_2048'),
    ]
)

test_suite(
    name = "80G_perf_test",
    tests = [
        perf_test('llama', '7', 'int8', '/mnt/nas1/perf_test/llama-7b', test_input_long_seq_7b_1, name_suffix='_long_seq_1'),
        perf_test('llama', '7', 'int8', '/mnt/nas1/perf_test/llama-7b', test_input_long_seq_7b_2, name_suffix='_long_seq_2'),
        perf_test('llama', '13', 'int8', '/mnt/nas1/perf_test/llama-13b', test_input_long_seq_13b_1, name_suffix='_long_seq_1'),
        perf_test('llama', '13', 'int8', '/mnt/nas1/perf_test/llama-13b', test_input_long_seq_13b_2, name_suffix='_long_seq_2'),
    ]
)
