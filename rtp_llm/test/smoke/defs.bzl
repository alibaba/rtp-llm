
def extract_data(envs):
    data = []
    for env in envs:
        if env.startswith('MULTI_TASK_PROMPT='):
            data.append(env.split('=')[1][len('internal_source/rtp_llm/test/smoke/'):])
    return data

def extract_multi_task_prompt_data(smoke_args):
    """Extract --multi_task_prompt paths from smoke_args for Bazel data dependencies"""
    data = []
    args_to_check = []

    # Collect all argument strings to check
    if type(smoke_args) == 'string':
        args_to_check = [smoke_args]
    elif type(smoke_args) == 'dict':
        args_to_check = smoke_args.values()

    # Extract --multi_task_prompt paths
    for args_str in args_to_check:
        if "--multi_task_prompt" in args_str:
            tokens = args_str.split(" ")
            for i in range(len(tokens)):
                if tokens[i] == '--multi_task_prompt' and i + 1 < len(tokens):
                    path = tokens[i + 1]
                    if path.startswith('internal_source/rtp_llm/test/smoke/'):
                        relative_path = path[len('internal_source/rtp_llm/test/smoke/'):]
                        if relative_path not in data:
                            data.append(relative_path)
                    break
    return data

def get_world_size_from_smoke_args(smoke_args):
    """Parse --tp_size, --dp_size, --pp_size, --world_size from smoke_args to compute gpu count."""
    if not smoke_args:
        return 1
    args_to_check = []
    if type(smoke_args) == "string":
        args_to_check = [smoke_args]
    elif type(smoke_args) == "dict":
        args_to_check = smoke_args.values()
    else:
        return 1
    max_world = 1
    for s in args_to_check:
        parts = s.split(" ")
        world_size = None
        tp = 1
        pp = 1
        dp = 1
        skip_until = -1
        for i in range(len(parts)):
            if i < skip_until:
                continue
            if parts[i] == "--world_size" and i + 1 < len(parts):
                world_size = int(parts[i + 1])
                skip_until = i + 2
                continue
            if parts[i] == "--tp_size" and i + 1 < len(parts):
                tp = int(parts[i + 1])
                skip_until = i + 2
                continue
            if parts[i] == "--dp_size" and i + 1 < len(parts):
                dp = int(parts[i + 1])
                skip_until = i + 2
                continue
            if parts[i] == "--pp_size" and i + 1 < len(parts):
                pp = int(parts[i + 1])
                skip_until = i + 2
                continue
        size = world_size if world_size != None else tp * pp * dp
        if size > max_world:
            max_world = size
    return max_world

def get_aiter_envs(name, envs):
    for env in envs:
        k, _ = env.split('=')
        if 'AITER_ASM_DIR' == k:
            return []
    # relative path to cwd where rtp_llm.start_server is launched in MagaServerManager
    # files in bazel-out/k8-opt/bin
    return ["AITER_ASM_DIR=../../../../../../../bin/internal_source/rtp_llm/test/smoke/" + name + ".runfiles/pip_gpu_rocm_torch_aiter/site-packages/aiter_meta/hsa/"]

def smoke_test(name, task_info, tags=[], envs=[], gpu_type=[], data=[], smoke_args="",
               kvcm_envs=[], sleep_time_qr=0, kill_remote=False, concurrency_test=False):
    path = '/'.join(task_info.split('/')[:-1])
    data = data + native.glob([path + '/*.pt',
                               path + '/*.jpg',
                               path + '/*.jpeg',
                               path + '/*.mp4'])
    multi_task_data = extract_multi_task_prompt_data(smoke_args)
    for item in multi_task_data:
        if item not in data:
            data.append(item)
    gpu_count = 0
    if type(smoke_args) == 'dict':
        part_env_list = []
        for k, role_args in smoke_args.items():
            v = envs.get(k, []) if type(envs) == "dict" else []
            world_size = get_world_size_from_smoke_args(role_args)
            v = v + ['WORLD_SIZE=' + str(world_size)]
            gpu_count += world_size
            part_env_list.append("\"" + k + "\": " + "[" + ",".join(["\"" + x + "\"" for x in v]) +  "]")
            data.extend(extract_data(v))
        env_str = "'{" + ','.join(part_env_list) + "}'"
    else:
        envs_list = envs if type(envs) == "list" else []
        world_size = get_world_size_from_smoke_args(smoke_args)
        envs_list = envs_list + ['WORLD_SIZE=' + str(world_size)]
        gpu_count += world_size
        env_str = "[" + ",".join(["\\\"" + x + "\\\"" for x in envs_list]) +  "]"
        data.extend(extract_data(envs_list))

    if type(smoke_args) == 'string':
        smoke_args_str = "\"" + smoke_args + "\""
    elif type(smoke_args) == 'dict':
        part_args_list = []
        for k, v in smoke_args.items():
            part_args_list.append("\"" + k + "\": " + "\"" + v + "\"")
        smoke_args_str = "'{" + ','.join(part_args_list) + "}'"
    elif type(smoke_args) == 'list':
        smoke_args_str = "\"" + " ".join(smoke_args) + "\""
    else:
        fail("unknown smoke_args type: " + str(type(smoke_args)))

    kvcm_envs_str = "[" + ",".join(["\\\"" + x + "\\\"" for x in kvcm_envs]) + "]"

    local_srcs = native.glob(["*.py", "mainse/*.py"])
    has_entry = bool([f for f in local_srcs if f == "entry.py" or f.endswith("/entry.py")])
    if has_entry:
        all_srcs = local_srcs
        entry_main = "entry.py"
        extra_deps = []
    else:
        all_srcs = local_srcs + ["//rtp_llm/test/smoke:entry.py"]
        entry_main = "//rtp_llm/test/smoke:entry.py"
        extra_deps = []
        data = data + ["//rtp_llm/test/smoke:smoke_framework_srcs"]

    native.py_test(
        name = name,
        main = entry_main,
        srcs = all_srcs,
        timeout = "eternal",
        imports = [".."] if has_entry else ["../../../../rtp_llm/test", ".."],
        deps = [
            "//rtp_llm/test/utils:maga_server_manager",
            "//rtp_llm:uvicorn",
            "//rtp_llm:fastapi",
            "//rtp_llm:psutil",
            "//rtp_llm:tiktoken",
            "//rtp_llm:testlib",
            "//rtp_llm:pydantic",
            "//rtp_llm:json5",
            "//rtp_llm:dashscope",
            "//rtp_llm:jieba",
            "//rtp_llm:partial_json_parser",
            "//rtp_llm:openai",
            "//rtp_llm/test/utils:device_resource",
            "//rtp_llm/test/utils:test_util",
        ] + extra_deps + select({
            "//conditions:default": [],
        }),
        data = data + [
            task_info,
            "data/prompt_candidates.json",
            "//rtp_llm:sdk",
        ],
        tags = tags + ["smoke_case", "manual"] + gpu_type,
        legacy_create_init=0,
        args = [
            "--suite_name", name,
            "--task_info", task_info,
            "--envs", env_str,
            "--gpu_card", gpu_type[0],
            "--smoke_args", smoke_args_str,
            "--kvcm_envs", kvcm_envs_str,
            "--sleep_time_qr", str(sleep_time_qr),
            "--kill_remote", str(kill_remote),
            "--concurrency_test", str(concurrency_test),
        ],
        exec_properties = {
            'gpu':gpu_type[0],
            'gpu_count': str(gpu_count),
        },
        env = {
            "GPU_COUNT": str(gpu_count),
        },
    )
    return name
