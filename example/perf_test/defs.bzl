def latency_test(model_type,
              model_size,
              prec,
              ckpt_path,
              test_input,
              tokenizer_path=None,
              name_suffix=None,
              max_batch_size=None,
              max_seq_len=None,
              lora_infos=None,
              reserve_mem=2048):
    if tokenizer_path == None:
        tokenizer_path = ckpt_path
    test_name = model_type + '_' + model_size + '_' + prec
    if name_suffix != None:
        test_name = test_name + name_suffix
    if max_seq_len == None:
        max_seq_len = test_input[-1][1]
    if lora_infos == None:
        lora_infos = "{}"
    else:
        lora_infos = ','.join([str(x) for x in lora_infos])
    test_batch_size = ','.join([str(x[0]) for x in test_input])
    test_input_len = ','.join([str(x[1]) for x in test_input])
    native.py_test(
        name = test_name,
        main =  "latency_test.py",
        srcs = [
            "latency_test.py",
            "test_util.py",
        ],
        timeout = "eternal",
        imports = [],
        deps = [
            "//maga_transformer:pyodps",
            "//maga_transformer:testlib",
        ],
        data = [
            "//maga_transformer:sdk",
            "ShareGPT_V3_test_data_lens.json"
        ],
        args = [
            "--model_type", model_type,
            "--model_size", model_size,
            "--ckpt_path", ckpt_path,
            "--tokenizer_path", tokenizer_path,
            "--batch_size", test_batch_size,
            "--input_len", test_input_len,
            "--prec", prec,
            "--lora_infos", lora_infos
        ],
        env = {
            "PERF_TEST": "1",
            "MAX_SEQ_LEN": str(max_seq_len + 16),
            "DEVICE_RESERVE_MEMORY_BYTES": str(-512 * 1024 * 1024),
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256",
        },
        tags = ["manual"],
    )
    return test_name

def throughput_test(model_type,
                    model_size,
                    prec,
                    ckpt_path,
                    tokenizer_path=None,
                    name_suffix=None,
                    lora_infos=None,
                    reserve_mem=2048):
    max_seq_len = 2048
    if tokenizer_path == None:
        tokenizer_path = ckpt_path
    test_name = model_type + '_' + model_size + '_' + prec
    if name_suffix != None:
        test_name = test_name + name_suffix
    test_name = test_name + '_throughput'
    if lora_infos == None:
        lora_infos = "{}"
    else:
        lora_infos = ','.join([str(x) for x in lora_infos])
    native.py_test(
        name = test_name,
        main =  "throughput_test.py",
        srcs = [
            "throughput_test.py",
            "test_util.py",
        ],
        timeout = "eternal",
        imports = [],
        deps = [
            "//maga_transformer:pyodps",
            "//maga_transformer:testlib",
        ],
        data = [
            "//maga_transformer:sdk",
            "ShareGPT_V3_test_data_lens.json"
        ],
        args = [
            "--model_size", model_size,
            "--prec", prec,
            "--lora_infos", lora_infos
        ],
        env = {
            "MODEL_TYPE": model_type,
            "TOKENIZER_PATH": ckpt_path,
            "CHECKPOINT_PATH": tokenizer_path,
            "CONCURRENCY_LIMIT": "128",
            "START_PORT": "25888",
            "WEIGHT_TYPE": prec,
            "MAX_SEQ_LEN": str(max_seq_len + 16),
            "DEVICE_RESERVE_MEMORY_BYTES": str(-512 * 1024 * 1024),
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256",
        },
        tags = ["manual"],
    )
    return test_name

def vit_test(model_type, ckpt_path, image_size):
    image_size_str = ','.join([str(x) for x in image_size])
    test_name = model_type + "_vit_test"
    native.py_test(
        name = test_name,
        main =  "vit_test.py",
        srcs = [
            "vit_test.py"
        ],
        timeout = "eternal",
        imports = [],
        deps = [
            "//maga_transformer:testlib"
        ],
        data = [
            "//maga_transformer:sdk"
        ],
        args = [
            "--model_type", model_type,
            "--ckpt_path", ckpt_path,
            "--image_size", image_size_str,
        ],
        tags = ["manual"],
    )
    return test_name