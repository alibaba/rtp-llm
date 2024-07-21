def gemm_test(max_m,
              model_size,
              kn_list,
              prec,
              upload_to_oss='True'):
    test_name = 'gemm_test_' + max_m + '_' + model_size + '_' + prec
    k = ','.join([str(x[0]) for x in kn_list])
    n = ','.join([str(x[1]) for x in kn_list])
    native.py_test(
        name = test_name,
        main =  "run.py",
        srcs = [
            "gemm_test.py",
            "run.py"
        ],
        timeout = "eternal",
        imports = [],
        deps = [
            "//maga_transformer:testlib",
            "//maga_transformer/test/utils:device_resource"
        ],
        data = [
            "//tests:test_ops"
        ],
        args = [
            "--max_m", max_m,
            "--prec", prec,
            "--k", k,
            "--n", n,
            "--model_size", model_size,
            "--upload_to_oss", upload_to_oss
        ],
        env = {
        },
        tags = ["manual"],
    )
    return test_name

