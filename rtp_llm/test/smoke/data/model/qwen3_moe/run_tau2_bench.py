#!/usr/bin/env python3
"""用 evalscope 跑 τ²-Bench,打本地服务(仅 IP+port,无 API key)。

设计:
  - agent 和 user simulator 都打同一个本地服务(全部自洽,无外部依赖)
  - 全量 3 domain(airline + retail + telecom)
  - repeats=2(pass@2)
  - 并发尽量高,追求快

用法:
  # 方式 1:改下面的 CONFIG 区直接跑
  python3 scripts/run_tau2_bench.py

  # 方式 2:命令行覆盖
  python3 scripts/run_tau2_bench.py --host 127.0.0.1 --port 30000 \
      --model your-model --concurrency 32

  # 方式 3:环境变量
  SERVER_HOST=10.0.0.1 SERVER_PORT=8000 MODEL_NAME=qwen \
      python3 scripts/run_tau2_bench.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.request

# === 日志压制 ===
# tau2 内部用 loguru 打 user_simulator / orchestrator / domains.* 的 DEBUG+INFO,
# 满屏刷 "Step 3. Sending message ..." 那类。LOGURU_LEVEL 必须在 `from loguru import
# logger` 之前设,所以放在最顶。evalscope/tau2 的 import 都是后面懒加载,安全。
os.environ.setdefault("LOGURU_LEVEL", "INFO")   # 想看细节:LOGURU_LEVEL=DEBUG python ...
# tqdm 进度条每步刷新也很吵,至少 30s 才更新一次
os.environ.setdefault("TQDM_MININTERVAL", "30")

# evalscope 自己走 stdlib logging,压到 WARNING
logging.getLogger("evalscope").setLevel(logging.WARNING)


# ======================================================================
# 默认配置(命令行 / 环境变量可覆盖)
# ======================================================================
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 36000
DEFAULT_MODEL = "your-model-name"   # 必须和服务 /v1/models 返回的 id 一致
DEFAULT_CONCURRENCY = 8             # --eval-batch-size,按本地服务承载力调
DEFAULT_REPEATS = 1                 # pass@k 的 k。贪心的话设 1,采样的话 4 或 8
DEFAULT_TEMPERATURE = 0.0           # smoke 稳定性:贪心采样,消除单次运行方差
DEFAULT_TOP_P = 1.0                 # temp=0 时 top_p 无意义,设 1 避免边界坑
DEFAULT_AGENT_MAX_TOKENS = 1024     # 压住 agent 啰嗦,让多轮对话累积增速变慢
DEFAULT_USER_MAX_TOKENS = 2048
DEFAULT_SUBSETS = ["airline", "retail", "telecom"]
DEFAULT_WORK_DIR = "./tau2_bench_results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host",        default=os.environ.get("SERVER_HOST", DEFAULT_HOST))
    p.add_argument("--port",  type=int, default=int(os.environ.get("SERVER_PORT", DEFAULT_PORT)))
    p.add_argument("--model",       default=os.environ.get("MODEL_NAME", DEFAULT_MODEL))
    p.add_argument("--concurrency", type=int, default=int(os.environ.get("EVAL_BATCH_SIZE", DEFAULT_CONCURRENCY)))
    p.add_argument("--repeats",     type=int, default=DEFAULT_REPEATS)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                   help="采样温度。REPEATS>1 必须 >0 才能让 pass@k 有意义。")
    p.add_argument("--top-p",       type=float, default=DEFAULT_TOP_P, dest="top_p")
    p.add_argument("--subsets",     nargs="+", default=DEFAULT_SUBSETS,
                   help="Domain 子集,可填 airline / retail / telecom")
    p.add_argument("--limit",       type=int, default=None,
                   help="每个 subset 取前 N 个 task(调试时用,不设=全量)")
    p.add_argument("--work-dir",    default=os.environ.get("WORK_DIR", DEFAULT_WORK_DIR))
    p.add_argument("--skip-preflight", action="store_true", help="跳过启动前的连通性检查")
    p.add_argument("--task-ids-file", default=None,
                   help="JSON 文件,格式 {domain: [task_id, ...]};只跑这些 task")
    return p.parse_args()


def preflight(api_base: str, model: str) -> None:
    """连通性 + 模型名 + 依赖 三合一预检,失败尽量早。"""
    print(f"=== 预检 ===")
    print(f"Server:       {api_base}")
    print(f"Model:        {model}")

    # 1) /v1/models 通不通
    try:
        req = urllib.request.Request(f"{api_base}/models", headers={"Authorization": "Bearer EMPTY"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"[ERR] {api_base}/models 不通: {e}")
        print("      检查服务是否启动,以及 IP/port 是否正确。")
        sys.exit(1)

    # 2) 模型名存在
    ids = [m.get("id") for m in data.get("data", [])]
    if model not in ids:
        print(f"[WARN] /v1/models 里没找到 '{model}'。可用模型:")
        for i in ids:
            print(f"        - {i}")
        print(f"[WARN] 如果后续报 model not found,请把 --model 改成上面列出的值。")
    else:
        print(f"[OK]   model '{model}' 存在")

    # 3) 依赖
    try:
        import evalscope  # noqa: F401
        print(f"[OK]   evalscope {getattr(evalscope, '__version__', '?')}")
    except ImportError:
        print("[ERR] 未安装 evalscope。pip install evalscope")
        sys.exit(1)
    try:
        import tau2  # noqa: F401
        print(f"[OK]   tau2 imported")
    except ImportError:
        print("[ERR] 未安装 tau2。")
        print("      pip install 'git+https://github.com/sierra-research/tau2-bench@v0.2.0'")
        sys.exit(1)


def _install_task_filter(task_ids_map: dict) -> None:
    """Monkey-patch Tau2BenchAdapter.load,只保留 task_ids_map 列出的 task。

    task_ids_map: {domain: [task_id_str, ...]}
    """
    from evalscope.benchmarks.tau_bench.tau2_bench.tau2_bench_adapter import Tau2BenchAdapter

    if getattr(Tau2BenchAdapter, "_task_filter_installed", False):
        return
    original_load = Tau2BenchAdapter.load

    # 规范化为 str 集合,便于比较
    normalized = {d: set(str(x) for x in ids) for d, ids in task_ids_map.items()}

    def patched_load(self):
        # 把 adapter 的 subset_list 压到有 filter 的 domain,避免加载无关域
        self.subset_list = [d for d in self.subset_list if normalized.get(d)]

        # 原版 load 的逻辑,外加 task 过滤
        import os as _os
        from collections import defaultdict

        from evalscope.api.dataset.dataset import DatasetDict
        from evalscope.api.dataset.loader import DictDataLoader

        dataset_name_or_path = self.dataset_id
        if _os.path.exists(dataset_name_or_path):
            dataset_path = dataset_name_or_path
        else:
            from modelscope import dataset_snapshot_download
            dataset_path = dataset_snapshot_download(dataset_name_or_path)
        _os.environ["TAU2_DATA_DIR"] = dataset_path

        from tau2.agent.llm_agent import LLMGTAgent
        from tau2.registry import registry

        data_dict = defaultdict(dict)
        for domain_name in self.subset_list:
            allowed = normalized.get(domain_name, set())
            if not allowed:
                continue
            task_loader = registry.get_tasks_loader(domain_name)
            tasks = task_loader()
            tasks = [t for t in tasks if LLMGTAgent.check_valid_task(t)]
            tasks = [t for t in tasks if str(getattr(t, "id", "")) in allowed]
            tasks = [t.model_dump(exclude_unset=True) for t in tasks]
            missing = allowed - {str(t.get("id")) for t in tasks}
            if missing:
                print(f"[WARN] domain={domain_name} 下未找到 task_id: {sorted(missing)}")
            dataset = DictDataLoader(
                dict_list=tasks,
                sample_fields=self.record_to_sample,
                limit=self.limit,
                repeats=self.repeats,
                shuffle=self.shuffle,
            ).load()
            data_dict[domain_name] = dataset
        return DatasetDict(data_dict), None

    Tau2BenchAdapter.load = patched_load
    Tau2BenchAdapter._task_filter_installed = True


def main() -> None:
    args = parse_args()
    api_base = f"http://{args.host}:{args.port}/v1"

    if not args.skip_preflight:
        preflight(api_base, args.model)

    print(f"\n=== 配置 ===")
    print(f"Concurrency:  {args.concurrency}")
    print(f"Repeats:      {args.repeats} (pass@{args.repeats})")
    print(f"Temperature:  {args.temperature}")
    print(f"Top_p:        {args.top_p}")
    print(f"Subsets:      {args.subsets}")
    print(f"Limit:        {args.limit if args.limit is not None else 'none (全量)'}")
    print(f"Work dir:     {args.work_dir}")
    if args.repeats > 1 and args.temperature == 0.0:
        print(f"[WARN] repeats={args.repeats} 但 temperature=0.0,k 次采样完全一样,pass@k 等于 pass@1。")
    print("")

    # 构造 τ²-Bench 的 dataset-args
    #   user simulator 也打同一本地服务 → user_model 填 MODEL_NAME、api_base 指过去
    extra_params = {
        "user_model": args.model,
        "api_key":    "EMPTY",
        "api_base":   api_base,
        "generation_config": {
            "temperature": args.temperature,
            "top_p":       args.top_p,
            "max_tokens":  DEFAULT_USER_MAX_TOKENS,
        },
    }
    dataset_args = {
        "tau2_bench": {
            "subset_list":  args.subsets,
            "extra_params": extra_params,
        },
    }

    # 用 evalscope Python API 跑(不需要拼 CLI 字符串,JSON 类型安全)
    from evalscope import TaskConfig, run_task

    # --- 可选:仅跑指定 task_id(从 passing_tasks.json 读取) ---
    if args.task_ids_file:
        with open(args.task_ids_file) as f:
            task_ids_map = json.load(f)   # {domain: [task_id, ...]}
        # 自动把 subsets 限制到有 task 的 domain
        args.subsets = [d for d in args.subsets if d in task_ids_map and task_ids_map[d]]
        dataset_args["tau2_bench"]["subset_list"] = args.subsets
        _install_task_filter(task_ids_map)
        total = sum(len(v) for v in task_ids_map.values())
        print(f"[INFO] 仅跑 {args.task_ids_file} 指定的 {total} 个 task")

    os.makedirs(args.work_dir, exist_ok=True)

    task_cfg = TaskConfig(
        model=args.model,
        api_url=api_base,
        api_key="EMPTY",
        eval_type="openai_api",
        datasets=["tau2_bench"],
        dataset_args=dataset_args,
        eval_batch_size=args.concurrency,
        repeats=args.repeats,
        limit=args.limit,
        work_dir=args.work_dir,
        ignore_errors=True,       # 单个 task 挂了不拖垮整体
        generation_config={
            "max_tokens":        DEFAULT_AGENT_MAX_TOKENS,
            "temperature":       args.temperature,
            "top_p":             args.top_p,
            # 防止 Qwen3 thinking 时进入 repetition loop
            "presence_penalty":  0.3,
            "frequency_penalty": 0.3,
        },
    )

    print("=== Launch ===")
    result = run_task(task_cfg)
    print("\n=== Done ===")
    print(f"结果目录: {args.work_dir}")

    # 尝试打印个 summary;run_task 返回结构体随 evalscope 版本变,所以兜底
    try:
        print(f"\nSummary: {json.dumps(result, indent=2, default=str, ensure_ascii=False)[:2000]}")
    except Exception:
        print(f"(raw result: {result})")


if __name__ == "__main__":
    main()
