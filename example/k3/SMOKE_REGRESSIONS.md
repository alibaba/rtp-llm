# K3 双机 smoke 回归

默认配置为 Prefill TP8/EP8、Decode DP2×TP4/EP8，两端 KTP1、Page-RR；Decode 开启 DCP、CUDA Graph、原生 MTP，设置 NCCL_GRAPH_REGISTER=0。Prefill 开启 CP、chunk prefill 和 chunkwise RDMA。物理 block 默认为1024 tokens，attention 内核页仍为128 tokens，chunk budget为65536。

支持通过 SMOKE_PREFILL_TP_SIZE、SMOKE_DECODE_TP_SIZE、SMOKE_DECODE_DP_SIZE 配置非对称传输和混合并行，例如 P8/EP8→DP2×TP4/EP8 或 P4/EP4→D8/EP8。对称P8→D8可显式设置 `SMOKE_DECODE_TP_SIZE=8 SMOKE_DECODE_DP_SIZE=1`。两端各自最多8卡，attention TP必须相互整除。旧KTP部署、Eagle3或关闭chunkwise RDMA不在该入口的验收范围。

`SMOKE_SUITE=all` 用完整模型验收答案；`flow` 用四层切片验收传输和执行流程，不能代替完整模型语义验证。长前缀默认为110K，可用 SMOKE_LONG_PREFIX_TARGET_TOKENS 显式覆盖。Prefill KV池默认42000MiB、Decode29000MiB，历史展开预算6GiB；更长请求需另行验证容量。110K允许历史前缀落在一个展开块内，多块数值行为由算子测试覆盖。

## 当前用例

| 范围 | 检查 |
|---|---|
| 历史四平方数 | 请求内指定80–83平方，只接受对应完整答案 |
| 滚动补位与并发 | 8请求窗口、16个不同长度请求；完成后补入，检查请求答案和owner |
| batch miss/hit与混合复用 | 冷请求、全部命中及部分命中同批运行 |
| A→B→A与并发A/B | 相同公共前缀、不同尾部；检查实际公共token前缀、reuse和当前答案 |
| 多模态、原生MTP、整chunk单条/批量 | 保留多模态metadata、chunk输入及实际draft接受检查 |
| padding 1/7 | 精确构造chunk budget+1/+7，冷请求和复用；检查实际padding日志 |
| Page-RR边界 | 物理页owner轮转、回绕、两轮复用和一/两chunk边界前中后各1token的cold/repeat |
| Decode跨页 | 从边界前1token起，实际输出覆盖两个页起点，含最后owner回到owner0 |
| 长前缀 | seed后追加检索，检查三条记录与平方数的严格JSON、PD及充分复用 |

默认P8、block1024时，复用单元为8192 tokens。边界组包含66条cold/repeat请求和20条Decode跨页请求，共86条，不因DP2增加请求。cold/repeat轮换Decode DP组，每个边界在两组都有验收；两个整chunk批量请求均要求实际MTP接受。repeat的期望reuse为 `floor((input_len-1)/reuse_unit)*reuse_unit`；不足首个完整checkpoint的repeat仍应miss。Decode跨页按 `[input_len,input_len+output_len-2]` 验收，排除最后一个可能尚未消费的输出token，不把MTP拒绝槽计入覆盖。

长前缀seed固定路由到DP0，追加检索路由到最后一个DP组，响应必须确认目标owner。长前缀命中也按Prefill的checkpoint跨度检查：reuse不得超过实际公共token前缀，必须按复用单元对齐，并达到公共前缀减去少量对话尾部后的完整checkpoint下限。它不能仅凭一次非零命中就通过。

DP1跳过owner轮转、last-owner-only和不均匀DP batch；这些定义及断言全部保留，DP>1时执行。移除重复的identity、single_exact、partial_prefix单独阶段及旧KTP Graph 7→5→6波次；对应语义由batch、A/B分支和并发组覆盖。不把请求并发数量当成实际引擎batch或Graph replay证据。

## 运行时证据与失败处理

每个all-suite必须同时通过答案检查和两端的smoke-runtime-coverage.json。Prefill检查chunk round、padding算术和真实padding1/7；Decode检查各DP组全部rank的DCP A2A communicator、Page-RR物理block与max(P TP,D TP)×block checkpoint跨度、Graph捕获桶及KTP未启用。TP4/MTP3的捕获桶为1/2/4/8，TP8为2/4/8；capture不等于每一步replay。

所有正式请求不重试。输入、原始响应及精确构造的token IDs均保存，失败不会抹去同批已完成记录。RDMA预热只重试连接异常和HTTP408/429/502/503/504；语义、PD metadata、owner、格式、cache错误直接失败。预热前等待各rank的RDMA transport和gRPC listener就绪。

Detached控制器遇到SSH rc255时允许认证/网络恢复，仍受总体超时约束；真实服务失败和总体超时会清理未完成端，不因连续12次SSH查询失败而误杀仍在运行的服务。

未覆盖的分支包括MTP全部接受/部分接受/全部拒绝的可控枚举、拒绝后的cache frontier、逐请求dummy发布证明、长期淘汰和CPU cache恢复。PD取消、迟到回调与block复用由独立引擎测试验证，不冒充端到端故障注入结果。

## 无GPU检查

在Linux、Python3.10+中运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -S -m unittest \
  example.k3.kimi_k3_full_model_two_host_pd_smoke_driver_test \
  example.k3.kimi_k3_full_model_pd_cases_test \
  example.k3.kimi_k3_long_prefix_case_test \
  example.k3.kimi_k3_smoke_regressions_test
```

单测覆盖用例构造、边界判定、DP保留、混合启动配置、失败记录和控制器恢复，不代替合并后双机GPU验收。

## 引入来源

K3_DCP提交925c0a5的block1024版本曾在144/145以P8D8完整模型通过50阶段、153正式请求+4预热，其中86条边界用例全部通过。该次长前缀为600K，不能当作本分支110K、混合DP配置的合并后运行证据。本分支保留110K默认及非对称/混合TPDP逻辑，并按实际Prefill复用单元适配长前缀检查。
