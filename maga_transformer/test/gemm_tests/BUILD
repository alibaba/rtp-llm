load("//maga_transformer/test/gemm_tests:defs.bzl", "gemm_test")

# tp 1, 2
test_7b = [
    [4096, 12288],
    [4096, 4096],
    [4096, 11008],
    [11008, 4096],
    [4096, 6144],
    [2048, 4096],
    [4096, 5504],
    [5504, 4096],
]

test_3b = [
    [3072, 9216],
    [3072, 3072],
    [3072, 8192],
    [8192, 3072],
]

test_1b8 = [
    [2048, 6144],
    [2048, 2048],
    [2048, 5504],
    [5504, 2048],
]

# tp 1, 2
test_13b = [
    [5120, 15360],
    [5120, 5120],
    [5120, 13696],
    [13696, 5120],
    [5120, 7680],
    [2560, 5120],
    [5120, 6848],
    [6848, 5120],
]

# tp 1, 2
test_15b = [
    [6144, 6400],
    [6144, 6144],
    [6144, 24576],
    [24576, 6144],
    [6144, 3200],
    [3072, 6144],
    [6144, 12288],
    [12288, 6144],
]

# tp 1, 2, 4
test_72b = [
    [8192, 24576],
    [8192, 8192],
    [4096, 8192],
    [2048, 8192],
    [24576, 8192],
    [8192, 12288],
    [12288, 8192],
    [8192, 6144],
    [6144, 8192],
]

# tp 1, 2, 4
test_70b = [
    [8192, 9216],
    [8192, 28672],
    [28672, 8192],
    [8192, 14336],
    [14336, 8192],
    [8192, 7168],
    [7168, 8192],
]

test_7b_2 = [
    [3584, 18944],
    [18944, 3584],
    [3584, 4096],
    [3584, 3584],
    [3584, 2048],
    [1792, 3584],
    [3584, 9472],
    [9472, 3584],
]

test_72b_2 = [
    [8192, 29568],
    [29568, 8192],
    [8192, 14784],
    [14784, 8192],
    [8192, 7424],
    [7424, 8192],
    [8192, 10240],
    [8192, 5120],
    [8192, 2560],
]

test_1b5 = [
    [1536, 1536],
    [1536, 8960],
    [8960, 1536],
    [1536, 4480],
    [4480, 1536],
    [1536, 1024],
    [1536, 2048],
    [768, 1536],
]

gemm_test('4096', '1_8', test_1b8, 'int8')
gemm_test('4096', '3', test_3b, 'int8')
gemm_test('4096', '7', test_7b, 'int8')
gemm_test('4096', '13', test_13b, 'int8')
gemm_test('4096', '15', test_15b, 'int8')
gemm_test('4096', '70', test_70b, 'int8')
gemm_test('4096', '72', test_72b, 'int8')

gemm_test('4096', '1_8', test_1b8, 'int4')
gemm_test('4096', '3', test_3b, 'int4')
gemm_test('4096', '7', test_7b, 'int4')
gemm_test('4096', '13', test_13b, 'int4')
gemm_test('4096', '15', test_15b, 'int4')
gemm_test('4096', '70', test_70b, 'int4')
gemm_test('4096', '72', test_72b, 'int4')

gemm_test('4096', '1_5', test_1b5, 'int8')
gemm_test('4096', '7_2', test_7b_2,  'int8')
gemm_test('4096', '72_2', test_72b_2, 'int8')
gemm_test('4096', '1_5', test_1b5, 'int4')
gemm_test('4096', '7_2', test_7b_2,  'int4')
gemm_test('4096', '72_2', test_72b_2, 'int4')

gemm_test('4096', '1_8', test_1b8, 'w8a8')
gemm_test('4096', '3', test_3b, 'w8a8')
gemm_test('4096', '7', test_7b, 'w8a8')
gemm_test('4096', '13', test_13b, 'w8a8')
gemm_test('4096', '15', test_15b, 'w8a8')
gemm_test('4096', '70', test_70b, 'w8a8')
gemm_test('4096', '72', test_72b, 'w8a8')

gemm_test('4096', '1_5', test_1b5, 'w8a8')
gemm_test('4096', '7_2', test_7b_2,  'w8a8')
gemm_test('4096', '72_2', test_72b_2, 'w8a8')

test_suite(
    name = "int8_gemm_test",
    tests = [
        ':gemm_test_4096_1_8_int8',
        ':gemm_test_4096_3_int8',
        ':gemm_test_4096_7_int8',
        ':gemm_test_4096_13_int8',
        ':gemm_test_4096_15_int8',
        ':gemm_test_4096_70_int8',
        ':gemm_test_4096_72_int8',
    ]
)

test_suite(
    name = "int4_gemm_test",
    tests = [
        ':gemm_test_4096_1_8_int4',
        ':gemm_test_4096_3_int4',
        ':gemm_test_4096_7_int4',
        ':gemm_test_4096_13_int4',
        ':gemm_test_4096_15_int4',
        ':gemm_test_4096_70_int4',
        ':gemm_test_4096_72_int4',
    ]
)

test_suite(
    name = "new_gemm_test",
    tests = [
        ':gemm_test_4096_1_5_int4',
        ':gemm_test_4096_7_2_int4',
        ':gemm_test_4096_72_2_int4',
        ':gemm_test_4096_1_5_int8',
        ':gemm_test_4096_7_2_int8',
        ':gemm_test_4096_72_2_int8',
    ]
)

test_suite(
    name = "w8a8_gemm_test",
    tests = [
        ':gemm_test_4096_1_8_w8a8',
        ':gemm_test_4096_3_w8a8',
        ':gemm_test_4096_7_w8a8',
        ':gemm_test_4096_13_w8a8',
        ':gemm_test_4096_15_w8a8',
        ':gemm_test_4096_70_w8a8',
        ':gemm_test_4096_72_w8a8',
        ':gemm_test_4096_1_5_w8a8',
        ':gemm_test_4096_7_2_w8a8',
        ':gemm_test_4096_72_2_w8a8',
    ]
)
