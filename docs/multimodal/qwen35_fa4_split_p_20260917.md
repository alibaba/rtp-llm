# FA4 P 分段交付：96→64 实验

2026-09-17，/home/xieshui.yyx/workspace/RTP-LLM/github-opensource，分支 feat/qwen35_vl_omega，HEAD 2679cfed1b2ae4b3d315c34577e9feb31ef6017f。GPU0，L20D/SM103。采用实验副本，未修改正式代码或已安装 FA4 源码。

## 结论

不采用该候选。保持现有 Q/K80、V72、128×128 tile，只将 split_P_arrive 从 96 提前到 64，未获得性能收益：B1 中位耗时增加 0.64%，B32 增加 2.50%。代表性 B1 Tensor 管线活跃率由 66.06% 降至 64.85%。

## 无 profiler 单层结果

范围：合成 Q/K/V 匹配当前生产形状、dtype、stride 和分段；仅 attention forward。排除分配、padding、RoPE、验证、视频处理、ViT 其他层、RPC/LLM。batch 是视频数量对应的打包张量规模，不是 HTTP 并发。

| 视频 batch | 默认 96 列中位 ms | 候选 64 列中位 ms | 候选耗时增加 |
|---|---:|---:|---:|
| 1 | 10.41490 | 10.48159 | 0.64% |
| 32 | 356.34210 | 365.24872 | 2.50% |

双方分别预热收敛后，六轮正反交替；B1 每轮 20 次，B32 每轮 3 次。GPU0 样本前后均无其他计算进程。未锁 GPU 时钟，不将 NCU 重放耗时混入上述结果。保留全部样本，包括 B32 首轮基线较慢的样本；不报告 P99。

| 视频 batch | 方案 | 平均 ms | 标准差 ms | 六轮原始 ms |
|---|---|---:|---:|---|
| 1 | split96 | 10.40805 | 0.02939 | 10.36942, 10.41773, 10.41306, 10.38020, 10.45114, 10.41674 |
| 1 | split64 | 10.47195 | 0.07890 | 10.48699, 10.34047, 10.55158, 10.47618, 10.54622, 10.43026 |
| 32 | split96 | 357.63805 | 3.90521 | 365.33280, 354.42879, 357.58276, 355.79972, 356.30111, 356.38310 |
| 32 | split64 | 365.03588 | 1.38867 | 366.35278, 362.82857, 366.36308, 365.71757, 364.77987, 364.17338 |

## 正确性与执行路径

- 等长段 129/1025 tokens 的普通随机输入、1025 tokens 的放大随机输入通过完整输出逐位一致、finite、跨段隔离检查。
- 实际形状 B1/B32 的所有正式样本均与当前 production helper 的完整输出逐位一致，max_abs=0。
- 每视频 23 段，每段 10032 tokens；16 heads，Q/K80 的后 8 维为零，V72；BF16；非 causal，scale=72**-0.5。
- Q/K stride=(1280,80,1)，V stride=(3456,72,1)；B1 输出 [230736,16,72]，B32 输出 [7383552,16,72]。
- 候选只改变 P 的首段就绪阈值，采用独立接口编译缓存。NCU 的 NVTX 标签及 kernel module 名确认分别执行原始 flash_fwd_sm100 和 rtp_split_p64_kernel。

## 独立 NCU 归因

仅 B1，每个方案一个代表 kernel，4 次 replay。硬件计数器不是完整 ViT 利用率；未控制 GPU clocks/cache。

| 指标 | 96 列 | 64 列 |
|---|---:|---:|
| Tensor pipe active | 66.0594% | 64.8467% |
| issue active | 53.5583% | 52.2576% |
| eligible warps/scheduler | 0.6675 | 0.6239 |
| achieved occupancy | 23.4355% | 23.4354% |
| barrier warp 周期/issued instruction | 1.5794 | 1.6221 |
| long scoreboard warp 周期/issued instruction | 2.2760 | 2.5302 |

双方实际 Tensor 运算数相同，均为 12.1936805888 TFLOP/次；148 CTA、512 threads/CTA、128 registers/thread、228352 B dynamic shared memory 也相同。不能把 warp stall 指标直接换算成请求耗时占比。

测量支持：提前交付没有带来更好的有效发射/计算重叠，反而出现更多依赖等待。当前源码先调用整行 apply_exp2_convert，再分段写 P 到 TMEM；单改通知阈值没有显式把 exp/转换拆成两段。尚未从 SASS 临界路径精确拆分退化来源，不能断言所有退化都来自某一个 barrier。

## 验证范围

| 层级 | 状态 |
|---|---|
| 当前生产形状单层 correctness / B1、B32 timing | 完成 |
| 代表性 B1 硬件计数器及候选执行核对 | 完成 |
| 真实 397B ViT 完整 embedding 和完整多模态 RT | 未扩展：候选在单层已无收益，未接入正式路径 |
| ViT→LLM 传输、完整 LLM 服务、负载到饱和 | 本次未测 |

本次完成的是对单一优化假设的否定实验，不宣称完成了全模型性能优化。现有 Q/K padding 优化保留。

## 产物

- [原始计时与正确性](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-split-p-20260917-9d8788l_/bench-result.json)
- [实验脚本](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-split-p-20260917-9d8788l_/benchmark.py)
- [精确运行命令](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-split-p-20260917-9d8788l_/command.json)
- [环境与修改前源码哈希](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-split-p-20260917-9d8788l_/contract.json)
- [候选 kernel](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-split-p-20260917-9d8788l_/split64_kernel.py)
- [候选 interface](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-split-p-20260917-9d8788l_/split64_interface.py)
- [NCU 原始 report](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-split-p-20260917-9d8788l_/counters.ncu-rep)
- [NCU CSV](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-split-p-20260917-9d8788l_/counters.ncu-rep.csv)
- [NCU 汇总](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-split-p-20260917-9d8788l_/counters-summary.json)
- [NCU 精确命令](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-split-p-20260917-9d8788l_/ncu-command.json)
