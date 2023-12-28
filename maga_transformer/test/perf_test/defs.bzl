def perf_test(model_type,
              model_size,
              prec,
              ckpt_path,
              test_input,
              tokenizer_path=None,
              name_suffix=None,
              max_batch_size=None,
              max_seq_len=None):
    if tokenizer_path == None:
        tokenizer_path = ckpt_path
    test_name = model_type + '_' + model_size
    if name_suffix != None:
        test_name = test_name + name_suffix
    if max_seq_len == None:
        max_seq_len = test_input[-1][1]
    if max_batch_size == None:
        max_batch_size = test_input[-1][0]
    test_batch_size = ','.join([str(x[0]) for x in test_input])
    test_input_len = ','.join([str(x[1]) for x in test_input])
    native.py_test(
        name = test_name,
        main =  "perf_test.py",
        srcs = [
            "perf_test.py",
        ],
        timeout = "eternal",
        imports = [],
        deps = [
            "//maga_transformer:pyodps",
            "//maga_transformer:testlib",
        ],
        data = [
            "//maga_transformer:sdk"
        ],
        args = [
            "--model_type", model_type,
            "--model_size", model_size,
            "--ckpt_path", ckpt_path,
            "--tokenizer_path", tokenizer_path,
            "--batch_size", test_batch_size,
            "--input_len", test_input_len,
            "--prec", prec,
        ],
        env = {
            "LOAD_CKPT_NUM_PROCESS" : "3",
            "MAX_CONTEXT_BATCH_SIZE": str(max_batch_size + 1),
            "MAX_SEQ_LEN": str(max_seq_len + 16),
            "RESERVER_RUNTIME_MEM_MB": "2048",
	    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256",
        },
        tags = ["manual"],
    )
    return test_name
