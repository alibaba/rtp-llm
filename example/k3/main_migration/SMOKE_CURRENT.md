# Kimi K3 FP8 双机 PD smoke：当前入口

`text_smoke.py` 只按 checkpoint 的 `num_hidden_layers` 选择两种运行：四层预检或完整 93 层。命令行不接受 `--suite`、阶段筛选或关闭 MTP 的开关。四层先查 PD、复用、MTP 和全部正交路径；93 层再查基础精度、缓存、PageRR、历史 Chunk KV、取消恢复及 Decode Graph/DCP。四层截断模型的回答只作流程诊断，正式答案以 93 层为准。

完整 smoke 使用 Prefill TP8/EP8/DP1、Query CP1，Decode DP2/TP4/EP8。两端启用 FP8 attention、原生 MXFP4 MoE、Native MTP；Prefill 开 Device、Memory 和前缀复用，Decode 关 Memory 和前缀复用，开 CUDA Graph 与 DCP。MLA 的 KV 页按 PageRR 放置，KDA 按普通 TP 切分。Prefill 历史 KV 展开预算为每 rank 6 GiB。启动后核对 `launch.json` 和日志，不能只凭命令行推断配置已生效。

四层预检使用与完整模型相同的 Prefill TP8/EP8 → Decode DP2/TP4/EP8 拓扑，先发现 DP owner、DCP 和 Graph 问题。`--checkpoint` 指向四层权重时，运行器自动执行四层流程和正交预检；换成 93 层权重，同一入口自动执行完整用例。两个运行都传入两个 Decode owner 地址。下面是 93 层命令；四层只需更换权重、端口、目录和 namespace：

```bash
python example/k3/main_migration/text_smoke.py \
  --base-url http://PREFILL_IP:PREFILL_HTTP_PORT \
  --decode-health-url http://DECODE_IP:DECODE_HTTP_PORT/health \
  --decode-role-addr DECODE_IP:OWNER0_HTTP:OWNER0_GRPC \
  --decode-role-addr DECODE_IP:OWNER1_HTTP:OWNER1_GRPC \
  --prefill-event-dir /local/run/logs \
  --prefill-engine-log /local/repo/logs/engine.log \
  --prefill-rpc-runfiles /local/repo/bazel-bin/rtp_llm/rtp_llm_server.runfiles \
  --prefill-grpc-port PREFILL_GRPC_PORT \
  --checkpoint /verified/checkpoint \
  --block-size 4096 --reuse-unit-tokens 32768 --chunk-tokens 65536 \
  --namespace unique-run --output /local/artifacts/result.json
```

93 层的请求预算仍为 368 次：331 条正式答案请求、37 条准备请求。Decode DP 的新检查复用现有 64K 缓存请求：冷请求交给 owner 0，同前缀命中请求交给 owner 1，因此不增加请求数。独立审计要核对两个 owner 的实际加载、MLA 页归属、KDA 头分片，以及 target Verify 和 draft 的 Graph replay。原约定暂缓的六条长输出或已知超时请求会单独记为跳过，不算通过。

每次功能改动后，以服务就绪后的 93 层完整运行和独立审计作为回归。`result.json` 的 HTTP 答案通过不足以证明路径通过；还要保存 Prefill、Decode 全 rank 事件和原始请求，再用 `audit_orthogonal_smoke.py` 核对答案、乱码、重复、截断、缓存层级、Chunk KV 实际执行和 PD/MTP/Graph 链路。每条正式请求限时 300 秒。15 分钟是完整 smoke 的墙钟目标，编译与模型加载单独记录；超过目标时保留正确性结果并分析耗时，不删验收场景。
