# DSV4 xgrammar JSON 远程运行 runbook

本文用于在远程机器上复现 DeepSeek-V4-Flash PD 1P1D xgrammar JSON smoke，并按需保活服务采集 timeline。

## 固定信息

- 远程宿主机：`luoli.hn@11.163.39.110`
- 容器：`luoli_gpu`
- 父仓库目录：`/home/luoli.hn/work/rtp_llm_3`
- 代码目录：`/home/luoli.hn/work/rtp_llm_3/github-opensource-dsv4-json-format-mtp-clean`
- 代码分支：`codex/dsv4-json-format-mtp-clean-20260524`
- 代码 commit：`9ce7869991 fix: implement async MTP xgrammar logits processor`
- GitHub 已推分支：`origin/codex/dsv4-json-format-mtp-clean-20260524`
- 父仓库 commit：`e08bf9bec fix: update dsv4 xgrammar code submodule`
- GitLab 父仓库已推分支：`origin/dev/dsv4-xgrammar-json-smoke-20260524`
- smoke target：`//internal_source/rtp_llm/test/smoke:v4_flash_pd_cp4_tp1ep1dp1_xgrammar_json_sm100`
- smoke fixture：`internal_source/rtp_llm/test/smoke/data/model/deepseek_v4/q_r_v4_flash_pd_cp4_tp1ep1dp1_xgrammar_json_sm100_arm.json`
- fixture case 数：`query_result` 共 17 条
- bazel cache：`/home/luoli.hn/work/rtp_llm_3/bazel_cache`

注意：跑 bazel 必须在容器里用 `luoli.hn` 用户，不能用 root。root 没有 GitLab key，fetch 依赖时会失败。

## 1. 登录远程并进入容器

交互式方式，适合手动调试：

```bash
ssh -t luoli.hn@11.163.39.110
docker exec -it -u luoli.hn luoli_gpu bash -l
cd /home/luoli.hn/work/rtp_llm_3/github-opensource-dsv4-json-format-mtp-clean
```

本地直接执行容器命令时，统一用这个形态：

```bash
ssh luoli.hn@11.163.39.110 'docker exec -u luoli.hn luoli_gpu bash -lc "cd /home/luoli.hn/work/rtp_llm_3/github-opensource-dsv4-json-format-mtp-clean && whoami && pwd"'
```

预期 `whoami` 是 `luoli.hn`。

## 2. 确认分支、commit 和 fixture

在容器内执行：

```bash
export PARENT=/home/luoli.hn/work/rtp_llm_3
export CODE=$PARENT/github-opensource-dsv4-json-format-mtp-clean
export FIXTURE=$PARENT/internal_source/rtp_llm/test/smoke/data/model/deepseek_v4/q_r_v4_flash_pd_cp4_tp1ep1dp1_xgrammar_json_sm100_arm.json

cd $CODE
git rev-parse --abbrev-ref HEAD
git log -1 --oneline
git branch -a --contains 9ce7869991acbca17a5dbf9426910ba947f40628

cd $PARENT
git rev-parse --abbrev-ref HEAD
git log -1 --oneline
git branch -a --contains e08bf9bec2b2aacb86670c4cb24fd569e04995ef

python3 - <<'PY'
import json
p = "/home/luoli.hn/work/rtp_llm_3/internal_source/rtp_llm/test/smoke/data/model/deepseek_v4/q_r_v4_flash_pd_cp4_tp1ep1dp1_xgrammar_json_sm100_arm.json"
with open(p) as f:
    data = json.load(f)
print("model_type:", data["model_type"])
print("model_path:", data["model_path"])
print("query_result cases:", len(data["query_result"]))
PY
```

预期重点：

- 代码仓库当前分支是 `codex/dsv4-json-format-mtp-clean-20260524`，最新 commit 是 `9ce7869991 fix: implement async MTP xgrammar logits processor`。
- 代码 commit 同时在 `remotes/origin/codex/dsv4-json-format-mtp-clean-20260524` 上。
- 父仓库最新 commit 是 `e08bf9bec fix: update dsv4 xgrammar code submodule`。本地父仓库分支可能显示为 `dev/dsv4-xgrammar-json-smoke`，远端包含 `remotes/origin/dev/dsv4-xgrammar-json-smoke-20260524`。
- fixture 的 `query_result cases` 是 `17`。远端宿主机没有 `jq` 时用上面的 Python 命令即可。

## 3. 跑普通 smoke 验证

推荐在本地直接复制执行：

```bash
ssh luoli.hn@11.163.39.110 'docker exec -u luoli.hn luoli_gpu bash -lc "cd /home/luoli.hn/work/rtp_llm_3/github-opensource-dsv4-json-format-mtp-clean && bazelisk test //internal_source/rtp_llm/test/smoke:v4_flash_pd_cp4_tp1ep1dp1_xgrammar_json_sm100 --config=cuda13 --config=sm10x --disk_cache=/home/luoli.hn/work/rtp_llm_3/bazel_cache --test_output=errors --nocache_test_results"'
```

如果已经在容器 shell 里：

```bash
cd /home/luoli.hn/work/rtp_llm_3/github-opensource-dsv4-json-format-mtp-clean
bazelisk test //internal_source/rtp_llm/test/smoke:v4_flash_pd_cp4_tp1ep1dp1_xgrammar_json_sm100 \
  --config=cuda13 \
  --config=sm10x \
  --disk_cache=/home/luoli.hn/work/rtp_llm_3/bazel_cache \
  --test_output=errors \
  --nocache_test_results
```

## 4. keep-alive 启服务

建议开两个终端。终端 A 启服务并保持不退出，终端 B 查 `live_info`、发请求、停服务。

终端 A，在容器 shell 内执行：

```bash
export PARENT=/home/luoli.hn/work/rtp_llm_3
export CODE=$PARENT/github-opensource-dsv4-json-format-mtp-clean
export CACHE=$PARENT/bazel_cache
export TARGET=//internal_source/rtp_llm/test/smoke:v4_flash_pd_cp4_tp1ep1dp1_xgrammar_json_sm100
export RUN_ROOT=$PARENT/timeline_runs
export BASE=$RUN_ROOT/dsv4_xgrammar_$(date +%Y%m%d_%H%M%S)

mkdir -p $BASE/profiler
echo $BASE > $RUN_ROOT/dsv4_xgrammar_latest_base
cd $CODE

bazelisk test $TARGET \
  --config=cuda13 \
  --config=sm10x \
  --disk_cache=$CACHE \
  --nocache_test_results \
  --test_env=SMOKE_KEEP_SERVER_ALIVE=True \
  --test_env=SMOKE_LIVE_INFO=$BASE/smoke_live_info.json \
  --test_env=SMOKE_STOP_FILE=$BASE/smoke_stop \
  --test_env=TORCH_CUDA_PROFILER_DIR=$BASE/profiler \
  --test_env=GEN_TIMELINE_SYNC=1 \
  --test_env=PYTHONUNBUFFERED=1 \
  --test_output=streamed
```

如果必须从本地一条命令启动，注意所有给容器内 shell 展开的变量都要写成 `\$BASE` 这种形式：

```bash
ssh luoli.hn@11.163.39.110 'docker exec -u luoli.hn luoli_gpu bash -lc "set -euo pipefail
PARENT=/home/luoli.hn/work/rtp_llm_3
CODE=\$PARENT/github-opensource-dsv4-json-format-mtp-clean
CACHE=\$PARENT/bazel_cache
TARGET=//internal_source/rtp_llm/test/smoke:v4_flash_pd_cp4_tp1ep1dp1_xgrammar_json_sm100
RUN_ROOT=\$PARENT/timeline_runs
BASE=\$RUN_ROOT/dsv4_xgrammar_\$(date +%Y%m%d_%H%M%S)
mkdir -p \$BASE/profiler
echo \$BASE > \$RUN_ROOT/dsv4_xgrammar_latest_base
cd \$CODE
bazelisk test \$TARGET --config=cuda13 --config=sm10x --disk_cache=\$CACHE --nocache_test_results --test_env=SMOKE_KEEP_SERVER_ALIVE=True --test_env=SMOKE_LIVE_INFO=\$BASE/smoke_live_info.json --test_env=SMOKE_STOP_FILE=\$BASE/smoke_stop --test_env=TORCH_CUDA_PROFILER_DIR=\$BASE/profiler --test_env=GEN_TIMELINE_SYNC=1 --test_env=PYTHONUNBUFFERED=1 --test_output=streamed"'
```

`BASE` 推荐放在 `/home/luoli.hn/work/rtp_llm_3/timeline_runs/...`，不要放 `/tmp`，可以避开容器/宿主机权限和清理问题。

## 5. 查 live_info 和端口

终端 B，进入同一个容器用户后执行：

```bash
export BASE=$(cat /home/luoli.hn/work/rtp_llm_3/timeline_runs/dsv4_xgrammar_latest_base)
while [ ! -s $BASE/smoke_live_info.json ]; do
  date
  sleep 5
done
python3 -m json.tool $BASE/smoke_live_info.json
```

这个 PD smoke 的 keep-alive 会写多角色信息，通常类似：

```json
{
  "servers": {
    "prefill": {"port": 26000, "server_pid": 12345, "log_file": "..."},
    "decode": {"port": 26001, "server_pid": 12346, "log_file": "..."}
  },
  "stop_file": "/home/luoli.hn/work/rtp_llm_3/timeline_runs/.../smoke_stop"
}
```

PD 请求通常打 prefill server port；如果 `servers.frontend` 存在，则优先打 frontend。用下面命令自动取请求端口：

```bash
export PORT=$(python3 - <<'PY'
import json, os
base = os.environ["BASE"]
with open(f"{base}/smoke_live_info.json") as f:
    info = json.load(f)
servers = info.get("servers", {})
if "frontend" in servers:
    print(servers["frontend"]["port"])
elif "prefill" in servers:
    print(servers["prefill"]["port"])
else:
    print(info["port"])
PY
)
echo $PORT
```

查看各角色 pid 和日志：

```bash
python3 - <<'PY'
import json, os
base = os.environ["BASE"]
with open(f"{base}/smoke_live_info.json") as f:
    info = json.load(f)
for role, item in info.get("servers", {}).items():
    print(role, "port=", item.get("port"), "pid=", item.get("server_pid"), "log=", item.get("log_file"))
if "servers" not in info:
    print("main", "port=", info.get("port"), "pid=", info.get("server_pid"), "log=", info.get("log_file"))
PY
```

## 6. 发普通 JSON 请求

先确认服务可用：

```bash
curl -sS -X POST http://127.0.0.1:$PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Return only a JSON object with key answer and value ok."}],
    "max_tokens": 64,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "enable_thinking": false,
    "response_format": {"type": "json_object"}
  }'
```

如果要复用 fixture 中的某条 case，可以从 `query_result[i].query` 抽出来发到同一个 `/v1/chat/completions` endpoint。

## 7. 发 timeline 请求

代码确认点：OpenAI `/v1/chat/completions` 的 timeline 参数必须放在 `extra_configs` 里，因为 OpenAI endpoint 通过 `request.extra_configs or GenerateConfig()` 构造生成配置；`GenerateConfig` 里定义了 `gen_timeline`、`profile_step`、`profile_trace_name`。Raw `/inference` 路径才使用 `generate_config`。不要把这三个字段直接放 OpenAI payload 顶层来依赖隐式透传。

对 prefill 端口建议 `profile_step=1`，因为 prefill 节点一个请求通常只跑一步；如果打 decode 端口，可以用 `profile_step=3`。示例：

```bash
cat > $BASE/request_timeline.json <<'JSON'
{
  "messages": [
    {
      "role": "user",
      "content": "Return only a JSON object with key answer and value ok."
    }
  ],
  "max_tokens": 64,
  "temperature": 0.0,
  "top_p": 1.0,
  "top_k": 1,
  "enable_thinking": false,
  "response_format": {"type": "json_object"},
  "extra_configs": {
    "gen_timeline": true,
    "profile_step": 1,
    "profile_trace_name": "dsv4_xgrammar_prefill"
  }
}
JSON

curl -sS -X POST http://127.0.0.1:$PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @$BASE/request_timeline.json
```

如果请求返回成功但没有 trace，先等 10-30 秒；仍没有时，再发一条普通请求帮助 profiler flush，或把 prefill 请求的 `profile_step` 改回 `1`。

## 8. 查看 trace JSON

本 runbook 启服务时显式设置了 `TORCH_CUDA_PROFILER_DIR=$BASE/profiler`，trace 会落到这个目录。设置了 `profile_trace_name=dsv4_xgrammar_prefill` 时，文件名通常类似 `dsv4_xgrammar_prefill_wr0_1.json`；没有 trace name 时通常类似 `profiler_ts..._wr0_1.json`。

```bash
sleep 10
find $BASE/profiler -maxdepth 1 -type f -name '*.json' -ls | sort -k7,7nr | head -20

python3 - <<'PY'
import glob, json, os
base = os.environ["BASE"]
files = sorted(glob.glob(f"{base}/profiler/*.json"))
print("trace files:", len(files))
for path in files[:5]:
    with open(path) as f:
        data = json.load(f)
    print(path, "traceEvents=", len(data.get("traceEvents", [])))
PY
```

需要拉到本地看 Perfetto 时，在本地 shell 执行：

```bash
BASE=$(ssh luoli.hn@11.163.39.110 'docker exec -u luoli.hn luoli_gpu bash -lc "cat /home/luoli.hn/work/rtp_llm_3/timeline_runs/dsv4_xgrammar_latest_base"')
mkdir -p ./dsv4_xgrammar_traces
scp luoli.hn@11.163.39.110:$BASE/profiler/*.json ./dsv4_xgrammar_traces/
```

然后打开 https://ui.perfetto.dev/ 导入 JSON。

## 9. 停服务

终端 B 在容器内执行：

```bash
export BASE=$(cat /home/luoli.hn/work/rtp_llm_3/timeline_runs/dsv4_xgrammar_latest_base)
touch $BASE/smoke_stop
```

然后回到终端 A，等待 `bazelisk test` 自己退出。keep-alive 逻辑会轮询 `SMOKE_STOP_FILE` 并停止所有已注册 server。

如果没有退出，先看 live_info 中 pid 和日志：

```bash
python3 - <<'PY'
import json, os
base = os.environ["BASE"]
with open(f"{base}/smoke_live_info.json") as f:
    info = json.load(f)
for role, item in info.get("servers", {}).items():
    print(role, item)
if "servers" not in info:
    print(info)
PY
```

再按日志定位：

```bash
LOG=$(python3 - <<'PY'
import json, os
base = os.environ["BASE"]
with open(f"{base}/smoke_live_info.json") as f:
    info = json.load(f)
servers = info.get("servers", {})
role = "prefill" if "prefill" in servers else next(iter(servers), None)
print((servers.get(role) or info).get("log_file", ""))
PY
)
tail -n 200 $LOG
```

必要时再对 live_info 里的 `server_pid` 发 `TERM`，但优先使用 `touch $BASE/smoke_stop` 让 bazel 测试自然收尾。

## 10. 常见坑

- 不要用 root 跑 `docker exec`。必须是 `docker exec -u luoli.hn luoli_gpu ...`，否则 GitLab 依赖 fetch 会因为没有 key 失败。
- 不要漏 `--disk_cache=/home/luoli.hn/work/rtp_llm_3/bazel_cache`，否则会慢很多，也更容易被远端依赖状态影响。
- keep-alive 必须加 `--nocache_test_results`，否则 bazel 可能复用旧结果，不会真正启动服务。
- `--test_output=streamed` 只用于 keep-alive；普通 smoke 用 `--test_output=errors` 更干净。
- `BASE` 不要放 `/tmp`。用 `/home/luoli.hn/work/rtp_llm_3/timeline_runs/dsv4_xgrammar_$(date +%Y%m%d_%H%M%S)`。
- `TORCH_CUDA_PROFILER_DIR` 指向的目录要提前创建，本 runbook 用 `mkdir -p $BASE/profiler`。
- PD keep-alive 的 `smoke_live_info.json` 多数没有顶层 `port`，要看 `servers.prefill.port`、`servers.decode.port`。打 OpenAI 请求通常用 prefill；如果有 frontend，则打 frontend。
- OpenAI timeline 参数放 `extra_configs`；Raw `/inference` 才放 `generate_config`。顶层 `response_format`、`enable_thinking` 是 OpenAI 请求字段，不等同于 timeline 的 `GenerateConfig` 字段。
- prefill 端口采 timeline 建议 `profile_step=1`。`profile_step>1` 时单个 prefill 请求可能不足以触发保存。
- 远端宿主机可能没有 `rg`/`jq`，用 `git grep` 和 `python3 -m json.tool` 替代。
- 如果 trace 目录没文件，先看 server 日志里是否有 `timeline profiler started`、`profiler trace saved`，再确认请求是否真的带了 `extra_configs.gen_timeline=true`。
