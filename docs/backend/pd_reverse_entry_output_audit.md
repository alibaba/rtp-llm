# PD 反转入口与出口差异

2026-10-08；代码基准 `595845f426`，以下修复尚未提交，旧路径基准 `397a27cfac`。静态核对，未编译或运行测试。FlexLB 生命周期缺口见 [缺口记录](pd_reverse_flexlb_gaps.md)。

## 流程背景

两条路径都支持 Master 选择 P/D、普通 PD 请求和 FlexLB 提前入队。这些能力本身没有因入口反转而消失：旧路径由 Frontend 连接 P，P 返回首包并转发 D 后续输出；新路径由 Frontend 连接 D，P 的首 token 数据经 StartLoad 交给 D 返回。提前入队的结果连接从 P.FetchResponse 改为 D.GenerateStreamCall，仅这一接口变化不构成功能缺口。

## 入口能力差异

| 能力 | 新路径 | 旧路径 |
|---|---|---|
| HTTP /batch_infer | 已恢复混合 FRONTEND 的 PDFUSION batch；实际 PD 目标仍拒绝，批内目标必须一致 | 非反转模式允许进入 batch 流程；反转模式拒绝 |
| 混合 PDFUSION/PD 路由可用性 | 已恢复 FRONTEND 按配置 domain 构造角色列表，不再隐式补齐 P/D；仍沿用旧集合校验，修改待运行验证 | 按配置 domain 构造角色列表，并校验全部配置角色 |
| PD trace 属性 | 已识别 DECODE，并包含 Master 提前入队的实际 PD 判断 | 目标为 PREFILL 时可记录 |

普通入口对 beam、多返回序列、禁 PD、单 token 都可回退 P 单机。Master batch 两条路径均强制 PD，属于共同限制。

## 出口能力差异

| 能力 | 新路径 | 旧路径 |
|---|---|---|
| generation_prefill_cuda_graph_status 准确性 | 已补充 P→D side channel，D 输出 P 实际状态并保留至后续输出；修改待编译和运行验证 | P 首包能携带实际 Prefill 状态 |
| 耗时统计语义 | aux_info 首 token 与累计耗时使用 D stream 基准，覆盖阶段不同 | 首 token 使用 P stream 基准，转发累计耗时使用 P 请求 context 基准 |
| 独立 KV pool 的 reuse 统计 | 已传递 P 的 pool 配置；独立 pool 且 D 复用更长时，顶层取 D | 相同选择规则 |

## 已有能力与共同限制

以下暂未发现能力退化，不能仅凭传输位置变化判定功能不同：

- logits、select_tokens_id、两类 hidden、softmax、cum_log_probs、all_probs：新首 token 通道及公共输出逻辑已存在。hidden 裁剪、归一化沿用 finished 输出条件。
- calculate_loss=1/2：新路径传原始 loss，由 D 按配置转换；测试源码覆盖 2，1 待验证，不能据此认定 1 不支持。
- return_prompt_logits：正常 Python 配置两条路径都强制单 token、非流式、禁复用、禁 PD，由 P 单机返回。新 P2P 缺该字段通道，只在强制 Master 入队时构成待核对风险。
- multimodal_lengths：新路径保留 input 元信息和公共序列化，旧路径在转发时合并；尚无完整 HTTP 对照证据，不能认定能力缺失。
- 多 DP EnqueueBatch：两条路径都限制 dp_size=1。
- HTTP all_hidden_states：PB/Python 有字段，普通 HTTP renderer 未直接输出，是共同限制。

待验证：普通/Master × streaming/nonstreaming × 首 token EOS/继续生成，检查返回配置、数值和多模态信息。
