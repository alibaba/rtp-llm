"use strict";

const state = {
  runs: [],
  activeRunId: "",
  orientation: "request",
  metric: "cost",
  selectedCell: null,
  overviewRequestIndex: null,
  overviewNodeIndex: null,
  overviewNodeIndices: [],
  optimizationRunIndex: 0,
  optimizationStepIndex: 0,
};

const elements = {
  reportTitle: document.querySelector("#report-title"),
  reportMeta: document.querySelector("#report-meta"),
  reportEmpty: document.querySelector("#report-empty"),
  reportContent: document.querySelector("#report-content"),
  reportMetrics: document.querySelector("#report-metrics"),
  reportRealSection: document.querySelector("#report-real-section"),
  reportRealMetrics: document.querySelector("#report-real-metrics"),
  reportRealContext: document.querySelector("#report-real-context"),
  reportRealInsights: document.querySelector("#report-real-insights"),
  reportVllmAudit: document.querySelector("#report-vllm-audit"),
  reportVllmRuns: document.querySelector("#report-vllm-runs"),
  reportVllmCardTime: document.querySelector("#report-vllm-card-time"),
  reportVllmPrediction: document.querySelector("#report-vllm-prediction"),
  reportVllmSimulate: document.querySelector("#report-vllm-simulate"),
  reportVllmConclusion: document.querySelector("#report-vllm-conclusion"),
  reportRealRuns: document.querySelector("#report-real-runs"),
  reportRealDiscreteScene: document.querySelector("#report-real-discrete-scene"),
  reportRealDiscreteRegression: document.querySelector(
    "#report-real-discrete-regression",
  ),
  reportRealDiscreteConclusion: document.querySelector(
    "#report-real-discrete-conclusion",
  ),
  reportRealProductionRange: document.querySelector(
    "#report-real-production-range",
  ),
  reportRealProductionConclusion: document.querySelector(
    "#report-real-production-conclusion",
  ),
  reportRealScaling: document.querySelector("#report-real-scaling"),
  reportRealScalingConclusion: document.querySelector(
    "#report-real-scaling-conclusion",
  ),
  reportRealPowerScan: document.querySelector("#report-real-power-scan"),
  reportRealPowerConclusion: document.querySelector(
    "#report-real-power-conclusion",
  ),
  reportRealSceneContext: document.querySelector("#report-real-scene-context"),
  reportRealSceneRequests: document.querySelector("#report-real-scene-requests"),
  reportRealSceneScheduleRequests: document.querySelector("#report-real-scene-schedule-requests"),
  reportRealSceneNodes: document.querySelector("#report-real-scene-nodes"),
  reportRealSceneDecisions: document.querySelector("#report-real-scene-decisions"),
  reportRealSceneCostDefinitions: document.querySelector("#report-real-scene-cost-definitions"),
  reportRealSceneBatchCosts: document.querySelector("#report-real-scene-batch-costs"),
  reportRealSceneCrossCosts: document.querySelector("#report-real-scene-cross-costs"),
  reportRealSceneObjective: document.querySelector("#report-real-scene-objective"),
  reportRealSceneCandidates: document.querySelector("#report-real-scene-candidates"),
  reportRealSceneTrajectory: document.querySelector("#report-real-scene-trajectory"),
  reportRealSceneConclusion: document.querySelector("#report-real-scene-conclusion"),
  reportRealNormalizationReplay: document.querySelector(
    "#report-real-normalization-replay",
  ),
  reportRealNormalizationAggregate: document.querySelector(
    "#report-real-normalization-aggregate",
  ),
  reportRealSweeps: document.querySelector("#report-real-sweeps"),
  reportRealProfiles: document.querySelector("#report-real-profiles"),
  reportRealPrediction: document.querySelector("#report-real-prediction"),
  reportRealQueueCacheAudit: document.querySelector("#report-real-queue-cache-audit"),
  reportRealQueueCacheMetrics: document.querySelector("#report-real-queue-cache-metrics"),
  reportRealQueueCacheContext: document.querySelector("#report-real-queue-cache-context"),
  reportRealCacheAccuracy: document.querySelector("#report-real-cache-accuracy"),
  reportRealQueueBoundary: document.querySelector("#report-real-queue-boundary"),
  reportRealQueueStages: document.querySelector("#report-real-queue-stages"),
  reportRealQueueRecurrence: document.querySelector("#report-real-queue-recurrence"),
  reportRealQueueCacheConclusion: document.querySelector("#report-real-queue-cache-conclusion"),
  reportRealSimulate: document.querySelector("#report-real-simulate"),
  reportKVFitSummary: document.querySelector("#report-kv-fit-summary"),
  reportKVFitLatencyChart: document.querySelector("#report-kv-fit-latency-chart"),
  reportKVFitChart: document.querySelector("#report-kv-fit-chart"),
  reportKVFitBins: document.querySelector("#report-kv-fit-bins"),
  reportKVFitRequestBins: document.querySelector("#report-kv-fit-request-bins"),
  reportKVFitParameters: document.querySelector("#report-kv-fit-parameters"),
  reportKVFitNotes: document.querySelector("#report-kv-fit-notes"),
  reportRealFitting: document.querySelector("#report-real-fitting"),
  reportRealLedger: document.querySelector("#report-real-ledger"),
  reportRealInvalid: document.querySelector("#report-real-invalid"),
  reportRealConclusion: document.querySelector("#report-real-conclusion"),
  formulaMeta: document.querySelector("#formula-meta"),
  formulaEmpty: document.querySelector("#formula-empty"),
  formulaContent: document.querySelector("#formula-content"),
  formulaMetrics: document.querySelector("#formula-metrics"),
  formulaContext: document.querySelector("#formula-context"),
  formulaConclusions: document.querySelector("#formula-conclusions"),
  formulaCrossSummary: document.querySelector("#formula-cross-summary"),
  formulaTransferGroups: document.querySelector("#formula-transfer-groups"),
  formulaCrossMatrices: document.querySelector("#formula-cross-matrices"),
  formulaCrossWindow: document.querySelector("#formula-cross-window"),
  formulaAblationMeta: document.querySelector("#formula-ablation-meta"),
  formulaAblationEmpty: document.querySelector("#formula-ablation-empty"),
  formulaAblationContent: document.querySelector("#formula-ablation-content"),
  formulaAblationMetrics: document.querySelector("#formula-ablation-metrics"),
  formulaAblationConclusions: document.querySelector("#formula-ablation-conclusions"),
  formulaOfflineAblation: document.querySelector("#formula-offline-ablation"),
  formulaOnlineAblation: document.querySelector("#formula-online-ablation"),
  formulaOnlinePhases: document.querySelector("#formula-online-phases"),
  reportInsightsHeading: document.querySelector("#report-insights-heading"),
  reportInsights: document.querySelector("#report-insights"),
  reportPoolSection: document.querySelector("#report-pool-section"),
  reportPoolInsights: document.querySelector("#report-pool-insights"),
  reportPool: document.querySelector("#report-pool"),
  reportRegressionSection: document.querySelector(
    "#report-regression-section",
  ),
  reportRegression: document.querySelector("#report-regression"),
  reportTable: document.querySelector("#report-table"),
  reportWorkloadSection: document.querySelector(
    "#report-workload-section",
  ),
  reportWorkload: document.querySelector("#report-workload"),
  reportLossSection: document.querySelector("#report-loss-section"),
  reportLoss: document.querySelector("#report-loss"),
  reportNearOneSection: document.querySelector("#report-near-one-section"),
  reportNearOne: document.querySelector("#report-near-one"),
  reportOptimizationSection: document.querySelector(
    "#report-optimization-section",
  ),
  reportOptimization: document.querySelector("#report-optimization"),
  reportScalabilitySection: document.querySelector(
    "#report-scalability-section",
  ),
  reportScalability: document.querySelector("#report-scalability"),
  reportProfileSection: document.querySelector("#report-profile-section"),
  reportProfile: document.querySelector("#report-profile"),
  empty: document.querySelector("#empty-state"),
  workspace: document.querySelector("#workspace"),
  runTable: document.querySelector("#run-table"),
  sourceLine: document.querySelector("#source-line"),
  replaySummary: document.querySelector("#replay-summary"),
  requestTable: document.querySelector("#request-table"),
  nodeTable: document.querySelector("#node-table"),
  cacheTable: document.querySelector("#cache-table"),
  schedulerMatrix: document.querySelector("#scheduler-matrix"),
  selectionDetail: document.querySelector("#selection-detail"),
  optimizationEmpty: document.querySelector("#optimization-empty"),
  optimizationContent: document.querySelector("#optimization-content"),
  optimizationRun: document.querySelector("#optimization-run"),
  optimizationMetrics: document.querySelector("#optimization-metrics"),
  optimizationTiming: document.querySelector("#optimization-timing"),
  optimizationChart: document.querySelector("#optimization-chart"),
  optimizationHistory: document.querySelector("#optimization-history"),
  optimizationStep: document.querySelector("#optimization-step"),
  optimizationStepLabel: document.querySelector("#optimization-step-label"),
  optimizationStepOutput: document.querySelector("#optimization-step-output"),
  optimizationStepMetrics: document.querySelector("#optimization-step-metrics"),
  optimizationMatrix: document.querySelector("#optimization-matrix"),
  rawJson: document.querySelector("#raw-json"),
  files: document.querySelector("#replay-files"),
  refresh: document.querySelector("#refresh-runs"),
};

function minimumRun(runs, field) {
  return runs.reduce((best, run) => (
    !best || field(run) < field(best) ? run : best
  ), null);
}

function maximumRun(runs, field) {
  return runs.reduce((best, run) => (
    !best || field(run) > field(best) ? run : best
  ), null);
}

function lambdaNumber(value) {
  if (value > 0.999 && value < 1) {
    return value.toFixed(5).replace(/0+$/, "");
  }
  return number(value, 3);
}

function insightItem(label, run, value) {
  const item = document.createElement("article");
  item.className = "insight-item";
  const name = document.createElement("span");
  name.textContent = label;
  const output = document.createElement("strong");
  output.textContent = value;
  const detail = document.createElement("p");
  detail.textContent = run.label;
  item.append(name, output, detail);
  return item;
}

const COLUMN_HELP = {
  "scheduler": "本次运行使用的调度策略。",
  "agents": "本次运行中可参与调度的 Agent 数量。",
  "lambda": "执行代价项与排队代价项之间的权衡参数。",
  "loop": "实际优化循环数的平均值和范围，或配置的最大循环数。",
  "alpha": "投影梯度优化的初始更新步长。",
  "window": "触发调度的最大请求数与最长等待时间。",
  "loss avg": "每轮调度最终可行 Loss 的平均值。",
  "loss p99": "每轮调度最终可行 Loss 的 P99。",
  "sched req": "Master 每轮调度处理的平均请求数。",
  "Batch 数": "该调度现场最终形成的物理 Batch 数量。",
  "Batch size 平均 / 范围 / 期望": "物理 Batch 的平均大小、实际范围，以及 Master 下发的期望范围。",
  "Prompt 长度范围": "本轮请求中最短和最长 Prompt 的 token 数。",
  "选中节点 Cache hit": "按最终调度节点统计的 Cache 命中 block 数和 token 比例。",
  "预测 Queue 平均 / 最大": "所有候选 Agent 在决策时的预测 Queue 平均值和最大值。",
  "Inflight / Running": "所有 Agent 上 Inflight 请求数与 Running 请求数之和。",
  "模型 revision": "本轮所有 Agent 上报的 latency model revision 范围。",
  "sched avg ms": "Master 单轮调度耗时的平均值。",
  "sched p99 ms": "Master 单轮调度耗时的 P99。",
  "queue avg ms": "请求在 Prefill Agent 队列中等待时间的平均值。",
  "queue p99 ms": "请求在 Prefill Agent 队列中等待时间的 P99。",
  "e2e avg ms": "从 Client 发起调度到 Prefill 完成的平均端到端延迟。",
  "e2e p99 ms": "从 Client 发起调度到 Prefill 完成的端到端延迟 P99。",
  "e2e max ms": "该流量段端到端延迟的最大值。",
  "qps": "包含排空阶段在内的完成吞吐，每秒完成请求数。",
  "exec ms": "请求实际 Prefill 执行时间的平均值。",
  "other ms": "端到端延迟扣除凑批、排队和执行后的剩余时间。",
  "batch avg": "实际执行 batch 的平均请求数。",
  "utilization avg": "所有 Agent 逻辑忙碌时间占运行时间的平均比例。",
  "busy agents": "所有 Agent 利用率之和，折算为持续忙碌的 Agent 数。",
  "utilization range": "利用率最低与最高 Agent 的区间。",
  "utilization stddev": "Agent 利用率的总体标准差。",
  "cache hit": "实际复用的输入 token 占可缓存输入 token 的比例。",
  "requests / agent": "单个 Agent 完成请求数的最小值与最大值。",
  "drain ms": "停止发压后，系统排空已接收请求所需的逻辑时间。",
  "experiment": "该行所属的实验分组。",
  "run": "该组实验中的运行标签。",
  "pool": "流量使用独立资源池还是统一共享池。",
  "biz / source": "独立 biz，或混合流量中的原始流量来源。",
  "requests": "该行统计包含的请求数量。",
  "share": "该流量段请求数占本次运行总请求数的比例。",
  "prompt avg": "Prompt token 数的平均值。",
  "prompt stddev": "Prompt token 数的总体标准差。",
  "prompt p50": "Prompt token 数的中位数。",
  "prompt p90": "Prompt token 数的 P90。",
  "prompt p99": "Prompt token 数的 P99。",
  "prompt max": "Prompt token 数的最大值。",
  "actual hit tokens": "执行时真实复用的 Cache token 总数。",
  "actual miss tokens": "执行时未命中、需要计算的输入 token 总数。",
  "actual cache hit": "真实命中 token 占可缓存输入 token 的比例。",
  "execution avg ms": "该流量段实际 Prefill 执行时间的平均值。",
  "execution p99 ms": "该流量段实际 Prefill 执行时间的 P99。",
  "pool layout": "该行对应的独立池或共享池布局。",
  "pool cache hit": "整个资源池的实际 Cache token 命中率。",
  "traffic segment": "按原始 biz 或 Prompt 长度划分的流量段。",
  "prompt mean": "该流量段 Prompt token 数的平均值。",
  "status": "与上一版阈值比较后的通过或失败状态。",
  "e2e avg": "平均端到端延迟相对上一版的变化比例。",
  "e2e p99": "端到端延迟 P99 相对上一版的变化比例。",
  "utilization": "平均 Agent 利用率相对上一版的百分点变化。",
  "schedules": "参与 Loss 统计且已按调度 ID 去重的调度轮数。",
  "mean": "该指标所有样本的平均值。",
  "stddev": "该指标所有样本的总体标准差。",
  "min": "该指标的最小值。",
  "p50": "该指标的中位数。",
  "p90": "该指标的 P90。",
  "p99": "该指标的 P99。",
  "max": "该指标的最大值。",
  "loss stddev": "最终可行 Loss 的总体标准差。",
  "loss p50": "最终可行 Loss 的中位数。",
  "loss p90": "最终可行 Loss 的 P90。",
  "loss max": "最终可行 Loss 的最大值。",
  "loops": "该检查点累计执行的优化循环数。",
  "projected loss": "完成对应循环并投影到 simplex 后的可行 Loss。",
  "trace elapsed us": "Replay 记录的累计优化耗时。",
  "loss reduction vs first": "相对第一个检查点的 Loss 降幅。",
  "forward us": "目标函数前向计算的累计耗时。",
  "backward us": "解析梯度计算的累计耗时。",
  "projection us": "simplex 投影的累计耗时。",
  "parallelism": "调度图中参与该阶段计算的并行分片数。",
  "min ms": "重复压测中单次调度耗时的最小值。",
  "median ms": "重复压测中单次调度耗时的中位数。",
  "max ms": "重复压测中单次调度耗时的最大值。",
  "speedup vs serial": "相对单线程执行的中位耗时加速比。",
  "area": "性能热点所属的计算阶段。",
  "measured evidence": "支持热点判断的实测数据。",
  "implementation": "该阶段当前采用的实现或并行方式。",
  "request": "当前 Replay 中请求的顺序编号。",
  "selected node": "调度结果为该请求选择的节点编号。",
  "agent id": "节点在 NaviSched 中的 Agent 标识。",
  "cache hit blocks": "该请求在选中节点预计命中的完整 Cache block 数。",
  "cache hit tokens": "命中 block 数乘 block size 得到的复用 token 数。",
  "cache miss tokens": "Prompt token 数减去命中 token 数。",
  "queue ms": "调度时节点上已预测的排队时间。",
  "cost": "延迟模型计算的该请求在该节点上的执行代价。",
  "weight": "连续优化为该请求和节点分配的最终权重。",
  "decision": "调度成功、失败或限流的原因。",
  "batch id": "Master 为该批调度结果分配的批次标识。",
  "batch size": "决策中预计进入同一执行批的请求数。",
  "request id": "请求的稳定整数标识。",
  "target address": "Client 应转发请求的目标 Agent 地址。",
  "prompt tokens": "请求的输入 token 数。",
  "block size": "每个 Cache block 包含的 token 数。",
  "block hashes": "去掉末尾不完整 block 后的完整 block hash 数。",
  "wait ms": "请求进入本轮调度前已经等待的时间。",
  "timeout left ms": "调度时请求距离超时还剩余的时间。",
  "timed out": "请求是否已经触发调度超时。",
  "node": "当前 Replay 中节点的顺序编号。",
  "server": "Agent 对外提供 Prefill 服务的地址。",
  "blocks": "Agent 初始化时上报的 KV Cache block 总容量。",
  "inflight req": "节点上凑批、排队和运行请求数之和。",
  "batching req": "已经接收、尚未形成执行 batch 的请求数。",
  "queued req": "已经成批、正在等待执行的请求数。",
  "running req": "已经开始 Prefill 执行的请求数。",
  "queue wait ms": "节点剩余排队执行时间的预测总和。",
  "model revision": "节点当前延迟拟合参数的修订编号。",
  "step": "从零开始的投影梯度优化循环编号。",
  "feasible before update": "本轮更新前、满足 simplex 约束的 Loss。",
  "feasible after projection": "更新并投影后的可行 Loss。",
  "elapsed us": "完成该轮优化消耗的微秒数。",
  "phase": "投影梯度优化中的计算阶段。",
  "total us": "该阶段在全部循环中的累计耗时。",
  "average us / step": "该阶段每轮优化的平均耗时。",
  "measured share": "该阶段占已分解优化耗时的比例。",
  "sample": "Replay 文件中的样本名称和序号。",
  "biz": "该轮调度请求所属的目标 biz。",
  "alpha decay": "每轮结束后用于衰减更新步长的乘数。",
  "min alpha": "步长衰减时允许使用的最小值。",
  "max loops": "该样本允许执行的最大优化循环数。",
  "actual loops": "受最大循环数和时间预算限制后实际完成的循环数。",
  "budget ms": "单轮调度允许用于连续优化的墙钟时间预算。",
  "size": "Replay 样本文件的字节大小。",
  "source": "Replay 样本来自服务器文件还是浏览器本地文件。",
  "objective": "第二项优化目标使用按 token 归一延迟或整请求延迟。",
  "artifact": "该轮真实运行使用的 opt/release 二进制指纹组。",
  "concavity": "延迟权重的凹函数参数；0 表示线性。",
  "mean ms": "丢弃前半段请求后的平均端到端延迟。",
  "p95 ms": "丢弃前半段请求后的端到端延迟 P95。",
  "max ms": "丢弃前半段请求后的最大端到端延迟。",
  "vs FlexLB": "相对相同流量 FlexLB 基线的平均延迟变化，负值更好。",
  "card-time s": "测量窗口内所有 Prefill batch 执行时间在 8 张卡上的总和。",
  "card util": "Prefill 总卡时除以 8 张卡和测量窗口长度。",
  "node CV": "8 个实例请求数的总体标准差除以均值，越低越均衡。",
  "cache requests": "实际至少命中一个完整 1024-token Cache block 的请求数。",
  "failures": "测量运行中的失败和超时请求数。",
  "profile": "固定发送节奏中的到达流量类型。",
  "prediction point": "预测值采集于 Agent 更新前或 Master 调度决策时。",
  "samples": "参与预测误差统计的 batch 样本数。",
  "predicted ms": "模型在执行前给出的平均预测延迟。",
  "predicted total ms": "Master 决策时预测的 Queue 与执行总延迟均值。",
  "actual total ms": "与预测 batch 对齐后的事后 Queue 与执行总延迟均值。",
  "prediction MAPE": "逐 batch 事前总延迟预测相对事后真值的平均绝对百分比误差。",
  "unmatched batches": "无法把事前决策与事后物理 batch 对齐的样本数；非零结果不可用于精度结论。",
  "actual ms": "对应 batch 事后观测到的平均真实延迟。",
  "bias ms": "真实值减预测值；正值表示低估。",
  "MAE ms": "预测与真实值绝对误差的平均值。",
  "MAPE": "逐样本绝对百分比误差的平均值。",
  "reason": "该轮运行不进入性能结论的直接原因。",
  "evidence": "用于复核异常的日志或结果状态。",
};

const LABEL_ZH = {
  experiment: "实验", run: "运行", scheduler: "Scheduler", agents: "Agent 数",
  loop: "循环数", window: "凑批窗口", status: "状态", pool: "资源池",
  "pool layout": "资源池布局", "traffic segment": "流量分段",
  "biz / source": "Biz / 来源", requests: "请求数", share: "占比",
  schedules: "调度批次", mean: "平均值", stddev: "标准差", loops: "循环数",
  parallelism: "并行度", implementation: "实现方式", area: "阶段",
  "measured evidence": "实测证据", "selected node": "所选 Node",
  "agent id": "Agent ID", server: "服务地址", blocks: "Block 数",
  decision: "调度决策", "model revision": "模型版本",
  "target address": "目标地址", "timed out": "是否超时", step: "Step",
  "elapsed us": "耗时 us", "trace elapsed us": "Trace 累计耗时 us",
  "speedup vs serial": "相对串行加速比",
  "loss reduction vs first": "相对首轮 Loss 降幅",
  "actual hit tokens": "实际命中 token", "actual miss tokens": "实际未命中 token",
  "actual cache hit": "实际 Cache 命中率", "pool cache hit": "资源池 Cache 命中率",
  "predicted total ms": "预测总延迟 ms", "actual total ms": "真实总延迟 ms",
  "prediction MAPE": "预测 MAPE", "unmatched batches": "未对齐 Batch",
  reason: "原因", evidence: "证据",
};

function tableColumns(labels) {
  return labels.map((label) => {
    if (typeof label !== "string") {
      return label;
    }
    return {
      label: LABEL_ZH[label] || label,
      description: COLUMN_HELP[label] || `“${label}”列的数据口径。`,
    };
  });
}

const comparisonHeaders = tableColumns([
  "scheduler",
  "agents",
  "lambda",
  "loop",
  "alpha",
  "window",
  "loss avg",
  "loss p99",
  "sched req",
  "sched avg ms",
  "sched p99 ms",
  "queue avg ms",
  "queue p99 ms",
  "e2e avg ms",
  "e2e p99 ms",
  "predicted total ms",
  "actual total ms",
  "prediction MAPE",
  "unmatched batches",
  "qps",
  "exec ms",
  "other ms",
  "batch avg",
  "utilization avg",
  "busy agents",
  "utilization range",
  "utilization stddev",
  "cache hit",
  "requests / agent",
  "drain ms",
]);

function comparisonRow(run) {
  const loopDistribution = run.sampled_loop_count;
  const loopValue = loopDistribution
    ? `${number(loopDistribution.average, 1)} ` +
      `[${number(loopDistribution.min)}-${number(loopDistribution.max)}]`
    : run.max_loop_count == null ? "-" : number(run.max_loop_count);
  const prediction = run.decision_prediction_accuracy || {};
  const predictionTotal = prediction.total || {};
  return [
    run.scheduler,
    number(run.agent_count),
    run.lambda == null ? "-" : lambdaNumber(run.lambda),
    loopValue,
    run.scheduler_config?.alpha == null
      ? "-"
      : number(run.scheduler_config.alpha, 3),
    `${number(run.schedule_wait_count)} / ` +
      `${number(run.schedule_wait_time_ms)} ms`,
    run.loss ? number(run.loss.average, 6) : "-",
    run.loss ? number(run.loss.p99, 6) : "-",
    run.average_requests_per_schedule == null
      ? "-"
      : number(run.average_requests_per_schedule, 2),
    run.schedule_latency_us
      ? number(run.schedule_latency_us.average / 1000, 3)
      : "-",
    run.schedule_latency_us
      ? number(run.schedule_latency_us.p99 / 1000, 3)
      : "-",
    number(run.queue_wait.average_ms, 2),
    number(run.queue_wait.p99_ms, 2),
    number(run.end_to_end_latency.average_ms, 2),
    number(run.end_to_end_latency.p99_ms, 2),
    predictionTotal.predicted_mean_ms == null
      ? "-" : number(predictionTotal.predicted_mean_ms, 2),
    predictionTotal.actual_mean_ms == null
      ? "-" : number(predictionTotal.actual_mean_ms, 2),
    predictionTotal.mape_pct == null
      ? "-" : `${number(predictionTotal.mape_pct, 2)}%`,
    prediction.unmatched_batch_count == null
      ? "-" : number(prediction.unmatched_batch_count),
    number(run.completion_throughput_qps, 2),
    number(run.execution_latency.average_ms, 2),
    number(run.other_latency_ms, 2),
    number(run.average_batch_size, 3),
    `${number(run.average_utilization * 100, 2)}%`,
    number(run.busy_agent_equivalents, 2),
    `${number(run.min_utilization * 100, 2)}%-${number(
      run.max_utilization * 100, 2,
    )}%`,
    `${number(run.utilization_stddev * 100, 2)} pp`,
    `${number(run.cache_hit_rate * 100, 2)}%`,
    `${number(run.agent_request_min)}-${number(run.agent_request_max)}`,
    number(run.logical_drain_time_ms, 2),
  ];
}

function promptDistributionText(distribution, prefix = "prompt tokens") {
  if (!distribution) {
    return "该历史运行没有 Prompt 分布数据";
  }
  return `${prefix}：平均 ${number(distribution.average, 1)}，` +
    `P50 ${number(distribution.p50)}, P90 ${number(distribution.p90)}, ` +
    `P99 ${number(distribution.p99)}，最大 ${number(distribution.max)}`;
}

function appendPromptDistributions(heading, run) {
  if (run.scheduler_config) {
    const config = document.createElement("p");
    const waitCount = run.schedule_wait_count ?? "-";
    const waitMs = run.schedule_wait_time_ms ?? "-";
    config.textContent =
      `Scheduler 配置：${JSON.stringify(run.scheduler_config)}；` +
      `Runtime 生效值：${JSON.stringify(
        run.effective_scheduler_config || run.scheduler_config,
      )}；` +
      `凑批窗口 ${waitMs} ms / ${waitCount} 请求；` +
      `随机种子 ${run.random_seed ?? "-"}`;
    heading.append(config);
  }
  const overall = document.createElement("p");
  overall.textContent = promptDistributionText(
    run.prompt_token_distribution,
  );
  heading.append(overall);
  const bizMetrics = run.biz_metrics || [];
  if (run.traffic_mode === "mixed" && bizMetrics.length === 1) {
    for (const source of bizMetrics[0].traffic_sources || []) {
      const line = document.createElement("p");
      line.textContent = promptDistributionText(
        source.prompt_token_distribution, `来源 ${source.name}`,
      );
      heading.append(line);
    }
    return;
  }
  if (bizMetrics.length <= 1) {
    return;
  }
  for (const biz of bizMetrics) {
    const line = document.createElement("p");
    line.textContent = promptDistributionText(
      biz.prompt_token_distribution, `Biz ${biz.biz_name}`,
    );
    heading.append(line);
  }
}

const GROUP_HELP = {
  "Strategy baseline":
    "在相同单 biz 流量和 Agent 规模下比较 random、max_cache_hit 与 cost，" +
    "作为策略效果基线。",
  "Single-biz lambda sweep":
    "固定单 biz 流量、资源和 CostScheduler 参数，只改变 lambda，观察执行代价" +
    "与排队代价权衡对延迟、吞吐、命中和利用率的影响。",
  "Batch window sweep":
    "固定单 biz 流量和 CostScheduler，联动调整最大凑批请求数与等待窗口，比较" +
    "批大小、调度开销和端到端延迟。",
  "Independent multi-biz lambda sweep":
    "多个不同 Prompt 分布的 biz 分别使用独立 Agent 池，在各自池内调度；只改变" +
    " lambda，用作共享池实验的对照。",
  "Unified mixed-pool lambda sweep":
    "将多个 biz 的长短请求混合送入同一个 Agent 池统一调度；只改变 lambda，检查" +
    "混跑下的延迟、Cache 命中、吞吐和节点利用率。",
  "Independent and unified strategy baseline":
    "在相同总流量和资源规模下，同时比较独立池与共享池中的 random、" +
    "max_cache_hit 和 cost，区分池化收益与策略收益。",
  "Unified pool capacity validation":
    "保持共享混合流量和 CostScheduler 不变，减少 Agent 数量，验证共享池能够节省" +
    "多少容量以及延迟是否仍可接受。",
  "Small-lambda loop cap sweep":
    "固定共享混合流量和小 lambda，只改变最大优化循环数，比较进一步优化 Loss 的" +
    "收益、调度耗时与最终服务效果。",
};

const GROUP_ZH = {
  "Strategy baseline": "策略基线",
  "Single-biz lambda sweep": "单 Biz lambda 扫描",
  "Batch window sweep": "凑批窗口扫描",
  "Independent multi-biz lambda sweep": "多 Biz 独立池 lambda 扫描",
  "Unified mixed-pool lambda sweep": "统一混合池 lambda 扫描",
  "Independent and unified strategy baseline": "独立池与统一池策略基线",
  "Unified pool capacity validation": "统一池容量验证",
  "Small-lambda loop cap sweep": "小 lambda 优化循环上限扫描",
  "Scheduler comparison": "Scheduler 对比",
};

function groupDescription(name) {
  return GROUP_HELP[name] ||
    "同一表内保持流量和资源口径一致，通过行参数比较调度结果。";
}

function renderComparisonGroups(runs) {
  const groups = new Map();
  for (const run of runs) {
    const name = run.group || "Scheduler comparison";
    if (!groups.has(name)) {
      groups.set(name, []);
    }
    groups.get(name).push(run);
  }
  const sections = [];
  for (const [name, groupRuns] of groups) {
    const section = document.createElement("section");
    section.className = "comparison-group";
    const heading = document.createElement("div");
    heading.className = "comparison-group-heading";
    const title = document.createElement("h3");
    title.textContent = GROUP_ZH[name] || name;
    const description = document.createElement("p");
    description.className = "section-description";
    description.textContent = groupDescription(name);
    heading.append(title, description);
    appendPromptDistributions(heading, groupRuns[0]);
    const table = document.createElement("div");
    table.className = "table-scroll report-table";
    replaceTable(
      table,
      comparisonHeaders,
      groupRuns.map(comparisonRow),
      `${name}：每行是一组完整模拟运行，列中同时给出配置、调度成本和服务结果。`,
    );
    section.append(heading, table);
    sections.push(section);
  }
  elements.reportTable.replaceChildren(...sections);
  return groups.size;
}

function renderWorkloadProfiles(runs) {
  const rows = [];
  for (const run of runs) {
    for (const segment of run.traffic_segments || []) {
      const prompt = segment.prompt_token_distribution || {};
      const cache = segment.actual_cache;
      const batch = segment.batch_size_distribution;
      if (!cache) {
        continue;
      }
      rows.push([
        run.group || "Scheduler comparison",
        run.label,
        run.scheduler,
        run.lambda == null ? "-" : lambdaNumber(run.lambda),
        run.traffic_mode,
        segment.name,
        number(segment.request_count),
        segment.request_share == null
          ? "-"
          : `${number(segment.request_share * 100, 2)}%`,
        number(prompt.average, 1),
        number(prompt.stddev, 1),
        number(prompt.p50),
        number(prompt.p90),
        number(prompt.p99),
        number(prompt.max),
        number(cache.cache_hit_tokens),
        number(cache.cache_miss_tokens),
        `${number(cache.cache_hit_rate * 100, 2)}%`,
        batch ? number(batch.average, 2) : "-",
        number(segment.queue_wait.average_ms, 2),
        number(segment.queue_wait.p99_ms, 2),
        number(segment.execution_latency.average_ms, 2),
        number(segment.execution_latency.p99_ms, 2),
        number(segment.end_to_end_latency.average_ms, 2),
        number(segment.end_to_end_latency.p99_ms, 2),
      ]);
    }
  }
  elements.reportWorkloadSection.hidden = !rows.length;
  if (!rows.length) {
    return;
  }
  replaceTable(
    elements.reportWorkload,
    tableColumns([
      "experiment",
      "run",
      "scheduler",
      "lambda",
      "pool",
      "biz / source",
      "requests",
      "share",
      "prompt avg",
      "prompt stddev",
      "prompt p50",
      "prompt p90",
      "prompt p99",
      "prompt max",
      "actual hit tokens",
      "actual miss tokens",
      "actual cache hit",
      "batch avg",
      "queue avg ms",
      "queue p99 ms",
      "execution avg ms",
      "execution p99 ms",
      "e2e avg ms",
      "e2e p99 ms",
    ]),
    rows,
    "每行是一个实验运行中的流量段，流量特征和对应结果放在同一行。",
  );
}

function renderPoolInsights(runs) {
  const independent = runs?.find(
    (run) => run.traffic_mode === "independent",
  );
  const sameSizeShared = runs?.find((run) => (
    run.traffic_mode === "mixed" &&
    run.agent_count === independent?.agent_count
  ));
  const reducedShared = runs
    ?.filter((run) => run.traffic_mode === "mixed")
    .reduce((best, run) => (
      !best || run.agent_count < best.agent_count ? run : best
    ), null);
  if (!independent || !sameSizeShared || !reducedShared) {
    elements.reportPoolInsights.replaceChildren();
    return;
  }
  const shortest = [...independent.traffic_segments].sort((left, right) => (
    left.prompt_token_distribution.average -
    right.prompt_token_distribution.average
  ))[0];
  const sharedShortest = sameSizeShared.traffic_segments.find(
    (segment) => segment.name === shortest.name,
  );
  if (!sharedShortest) {
    elements.reportPoolInsights.replaceChildren();
    return;
  }
  const e2eImprovement = 1 - (
    sameSizeShared.end_to_end_latency.average_ms /
    independent.end_to_end_latency.average_ms
  );
  const cacheGain = sameSizeShared.cache_hit_rate - independent.cache_hit_rate;
  const shortestImprovement = 1 - (
    sharedShortest.end_to_end_latency.average_ms /
    shortest.end_to_end_latency.average_ms
  );
  const shortestMaximum = sharedShortest.end_to_end_latency.max_ms;
  const capacitySaving = 1 - (
    reducedShared.agent_count / independent.agent_count
  );
  elements.reportPoolInsights.replaceChildren(
    insightItem(
      "共享池 E2E 改善",
      sameSizeShared,
      `${number(e2eImprovement * 100, 2)}%`,
    ),
    insightItem(
      "Cache 命中率提升",
      sameSizeShared,
      `${number(cacheGain * 100, 2)} pp`,
    ),
    insightItem(
      `最短请求分段改善（${shortest.name}）`,
      sameSizeShared,
      `${number(shortestImprovement * 100, 2)}%`,
    ),
    insightItem(
      "共享池最短请求分段最大 E2E",
      sameSizeShared,
      `${number(shortestMaximum, 2)} ms`,
    ),
    insightItem(
      "已验证的 Agent 节省比例",
      reducedShared,
      `${number(capacitySaving * 100, 2)}%`,
    ),
  );
}

function renderPoolComparison(runs) {
  const groups = new Map();
  for (const run of runs) {
    const segments = run.traffic_segments || [];
    if (segments.length <= 1) {
      continue;
    }
    const name = run.group || "Scheduler comparison";
    if (!groups.has(name)) {
      groups.set(name, []);
    }
    groups.get(name).push(run);
  }
  const rows = [];
  let insightRuns = null;
  for (const [group, groupRuns] of groups) {
    const modes = new Set(groupRuns.map((run) => run.traffic_mode));
    if (!modes.has("independent") || !modes.has("mixed")) {
      continue;
    }
    insightRuns = groupRuns;
    for (const run of groupRuns) {
      for (const segment of run.traffic_segments) {
        const prompt = segment.prompt_token_distribution || {};
        rows.push([
          group,
          run.label,
          number(run.agent_count),
          `${number(run.cache_hit_rate * 100, 2)}%`,
          segment.name,
          number(segment.request_count),
          number(prompt.average, 1),
          number(prompt.p50),
          number(prompt.p99),
          number(segment.queue_wait.average_ms, 2),
          number(segment.queue_wait.p99_ms, 2),
          number(segment.execution_latency.average_ms, 2),
          number(segment.execution_latency.p99_ms, 2),
          number(segment.end_to_end_latency.average_ms, 2),
          number(segment.end_to_end_latency.p99_ms, 2),
          number(segment.end_to_end_latency.max_ms, 2),
        ]);
      }
    }
  }
  elements.reportPoolSection.hidden = !rows.length;
  if (!rows.length) {
    return;
  }
  renderPoolInsights(insightRuns);
  replaceTable(
    elements.reportPool,
    tableColumns([
      "experiment",
      "pool layout",
      "agents",
      "pool cache hit",
      "traffic segment",
      "requests",
      "prompt mean",
      "prompt p50",
      "prompt p99",
      "queue avg ms",
      "queue p99 ms",
      "execution avg ms",
      "execution p99 ms",
      "e2e avg ms",
      "e2e p99 ms",
      "e2e max ms",
    ]),
    rows,
    "每行是某种池化布局下的一个流量段，用于比较相同流量的延迟变化。",
  );
}

function signedPercent(value) {
  if (value == null) {
    return "-";
  }
  return `${value >= 0 ? "+" : ""}${number(value * 100, 2)}%`;
}

function signedPoints(value) {
  return `${value >= 0 ? "+" : ""}${number(value * 100, 2)} pp`;
}

function renderRegression(regression) {
  const comparisons = regression?.comparisons || [];
  elements.reportRegressionSection.hidden = !comparisons.length;
  if (!comparisons.length) {
    return;
  }
  replaceTable(
    elements.reportRegression,
    tableColumns([
      "experiment",
      "scheduler",
      "lambda",
      "agents",
      "status",
      "e2e avg",
      "e2e p99",
      "qps",
      "cache hit",
      "utilization",
    ]),
    comparisons.map((comparison) => [
      comparison.group,
      comparison.scheduler,
      comparison.lambda == null ? "-" : lambdaNumber(comparison.lambda),
      number(comparison.agent_count),
      comparison.status,
      signedPercent(comparison.e2e_average_change),
      signedPercent(comparison.e2e_p99_change),
      signedPercent(comparison.completion_qps_change),
      signedPoints(comparison.cache_hit_change),
      signedPoints(comparison.utilization_change),
    ]),
    "每行是一组同参数的新旧版本对比；正值表示指标增大。",
  );
}

function decisionRoute(nodes) {
  return `[${(nodes || []).join(", ")}]`;
}

function renderDecisionScene(scene) {
  const configurations = scene?.configurations || [];
  if (!configurations.length) {
    return;
  }
  const requests = scene.requests || [];
  const nodes = scene.nodes || [];
  const cacheHitTotal = requests.reduce(
    (sum, request) => sum + request.cache_hit_blocks, 0,
  );
  const bestRoute = configurations[0].best_assignment.nodes;
  elements.reportRealSceneContext.replaceChildren(
    metricItem("现场来源", scene.is_simulate === false ? "真实 RTP-LLM 运行" : "未标注"),
    metricItem("现场请求", number(requests.length)),
    metricItem("候选分配", number(scene.candidate_count)),
    metricItem("Cache hit blocks", number(cacheHitTotal)),
    metricItem("共同离散最优", decisionRoute(bestRoute)),
  );

  replaceTable(
    elements.reportRealSceneRequests,
    tableColumns([
      "请求", "Request ID", "Prompt tokens", "已等待 ms",
      "剩余 timeout ms", "全节点 Cache hit blocks",
    ]),
    requests.map((request) => [
      `r${request.index}`, request.request_id, number(request.prompt_tokens),
      number(request.wait_ms, 3), number(request.remaining_timeout_ms, 3),
      number(request.cache_hit_blocks),
    ]),
    `真实运行采集时间戳 ${scene.schedule_time_us} us；simulate=${scene.is_simulate === true ? "是" : "否"}；原始现场：${scene.source}。${scene.boundary || ""}`,
  );

  const scheduleDistribution = scene.sampled_schedule_request_distribution || {};
  const scheduleHistogram = Object.entries(scheduleDistribution.histogram || {})
    .sort(([left], [right]) => Number(left) - Number(right))
    .map(([requestCount, scheduleCount]) => (
      `${requestCount} req: ${scheduleCount} schedules`
    )).join("; ");
  replaceTable(
    elements.reportRealSceneScheduleRequests,
    tableColumns([
      "样本调度轮数", "样本请求数", "平均 req/schedule", "最小", "最大",
      "分布",
    ]),
    [[
      number(scheduleDistribution.schedule_count),
      number(scheduleDistribution.request_count),
      number(scheduleDistribution.mean_requests_per_schedule, 3),
      number(scheduleDistribution.min_requests_per_schedule),
      number(scheduleDistribution.max_requests_per_schedule), scheduleHistogram,
    ]],
    `${scheduleDistribution.boundary || ""} 数据源：${scheduleDistribution.source || "-"}`,
  );

  replaceTable(
    elements.reportRealSceneNodes,
    tableColumns([
      "Node", "Agent", "Inflight", "预测 Queue ms", "负载口径", "L820 ms",
      "L220 ms", "L820+220 ms", "Cache blocks (r0 / r1)",
    ]),
    nodes.map((node) => {
      const reservationOnly = node.inflight_requests === 0 && node.queue_ms > 0;
      return [
        `node ${node.node_index}`, node.agent_id, number(node.inflight_requests),
        reservationOnly ? `${number(node.queue_ms, 3)}（重点）` : number(node.queue_ms, 3),
        reservationOnly ? "Master 有效 Queue 含 pending reservation" : "Agent 状态与有效 Queue 一致",
        number(node.long_request_execution_ms, 3),
        number(node.short_request_execution_ms, 3),
        number(node.combined_batch_execution_ms, 3),
        node.cache_hit_blocks.map((value) => number(value)).join(" / "),
      ];
    }),
    "Inflight 是 Agent 最近上报的已接收请求数；有效 Queue 还叠加 Master 在请求传输/确认期间保留的 Batch reservation。Replay 未分别保存 base Queue 与 reservation。",
  );
  elements.reportRealSceneNodes.querySelectorAll("tbody tr").forEach((row, index) => {
    const node = nodes[index];
    row.classList.toggle(
      "attention-row",
      node.inflight_requests === 0 && node.queue_ms > 0,
    );
  });

  replaceTable(
    elements.reportRealSceneDecisions,
    tableColumns([
      "目标", "p / scale / k", "请求权重", "Scheduler 返回", "排名",
      "离散最优", "Cost", "Cost Gap", "预测卡时 ms", "卡时 Gap",
      "投影 route 分布",
    ]),
    configurations.map((config) => {
      const selected = config.selected_assignment;
      const counts = Object.entries(config.projected_route_counts)
        .map(([route, count]) => `[${route}]: ${count}`)
        .join("; ");
      return [
        config.label,
        `${number(config.power, 2)} / ${number(config.scale, 6)} / ${number(config.concavity)}`,
        config.request_weights.map((value) => number(value, 6)).join(" / "),
        decisionRoute(selected.nodes), number(selected.objective_rank),
        decisionRoute(config.best_assignment.nodes), number(selected.objective, 6),
        `+${number(config.cost_gap_percent, 2)}%`,
        number(selected.predicted_card_time_ms, 3),
        `+${number(config.card_time_gap_percent, 2)}%`, counts,
      ];
    }),
    "四种目标的离散最优都是 [2,2]；返回 route 不同来自连续优化和投影路径，而不是离散目标最优点发生变化。",
  );

  replaceTable(
    elements.reportRealSceneCostDefinitions,
    tableColumns([
      "目标", "请求权重 w_i", "非线性权重 h(z)", "返回 route",
      "执行项原值", "执行项贡献", "延迟项原值", "延迟项贡献", "总 Cost",
    ]),
    configurations.map((config) => {
      const selected = config.selected_assignment;
      const weightFormula = config.power === 0
        ? "w_i = 1"
        : `w_i = ${number(config.scale, 6)} / tokens^${number(config.power, 2)}`;
      const latencyFormula = config.concavity > 0
        ? `h(z) = ln(1 + ${number(config.concavity)}z) / ln(${number(config.concavity)})`
        : "h(z) = z";
      return [
        config.label, weightFormula, latencyFormula,
        decisionRoute(selected.nodes), number(selected.execution_raw, 3),
        number(selected.execution_weighted, 6),
        number(selected.latency_raw, 6),
        number(selected.latency_weighted, 6), number(selected.objective, 6),
      ];
    }),
    "统一形式为 J = lambda * ExecRaw / totalTokens + (1-lambda) * LatencyRaw / requestCount；本现场 lambda=0、requestCount=2。",
  );

  const batchRows = [];
  for (const config of configurations) {
    const selected = config.selected_assignment;
    for (const batch of selected.batches) {
      batchRows.push([
        config.label, decisionRoute(selected.nodes), `node ${batch.node_index}`,
        batch.request_indexes.map((index) => `r${index}`).join(" + "),
        number(batch.request_cost_sum, 6), number(batch.execution_ms, 6),
        number(batch.queue_ms, 6), number(batch.assignment_mass, 9),
        number(batch.latency_weight, 9), number(batch.weighted_latency, 6),
        number(batch.weighted_latency / requests.length, 6),
      ]);
    }
  }
  replaceTable(
    elements.reportRealSceneBatchCosts,
    tableColumns([
      "目标", "返回 route", "Batch 节点", "包含请求", "请求线性量 U_n",
      "执行 L_n ms", "Queue Q_n ms", "分配质量 z_n", "h(z_n)",
      "延迟项贡献 (L_n+Q_n)h(z_n)", "总 Cost 贡献 / 2",
    ]),
    batchRows,
    "同节点 route 只有一个 Batch；matched p=0.25 的 [7,5] 被拆成两个 Batch，因此产生两行贡献并失去合 Batch 的卡时收益。",
  );

  const shortLabels = {
    token_normalized: "Token normalized",
    request_level_concave: "Request concave",
    matched_intermediate: "Matched p=0.25",
    request_level_linear: "Request linear",
  };
  replaceTable(
    elements.reportRealSceneCrossCosts,
    tableColumns([
      "Route 来源", "固定 route",
      ...configurations.map((config) => `${shortLabels[config.name]} Cost / Rank`),
      "预测卡时 ms", "预测请求均值 ms",
    ]),
    configurations.map((selectedBy) => {
      const route = selectedBy.selected_assignment.nodes;
      const evaluations = configurations.map((objective) => {
        const assignment = objective.all_assignments.find(
          (candidate) => candidate.nodes.every(
            (node, index) => node === route[index],
          ),
        );
        return `${number(assignment.objective, 6)} / #${number(assignment.objective_rank)}`;
      });
      return [
        selectedBy.label, decisionRoute(route), ...evaluations,
        number(selectedBy.selected_assignment.predicted_card_time_ms, 3),
        number(selectedBy.selected_assignment.predicted_request_total_mean_ms, 3),
      ];
    }),
    "Cost 数值的绝对尺度因 normalization 不同而不可横向比较；Rank 可横向判断同一 route 在各自目标下的相对质量。",
  );

  elements.reportRealSceneObjective.replaceChildren(
    ...configurations.map((config) => {
      const option = document.createElement("option");
      option.value = config.name;
      option.textContent = config.label;
      return option;
    }),
  );

  const renderObjective = () => {
    const config = configurations.find(
      (item) => item.name === elements.reportRealSceneObjective.value,
    ) || configurations[0];
    const bestCost = config.best_assignment.objective;
    replaceTable(
      elements.reportRealSceneCandidates,
      tableColumns([
        "排名", "标记", "Route (r0 / r1)", "Batch 数", "执行项原值",
        "延迟项原值", "总 Cost", "相对最优", "预测卡时 ms",
        "预测 Queue ms", "预测请求均值 ms",
      ]),
      config.all_assignments.map((assignment) => {
        const isBest = assignment.objective_rank === 1;
        const isSelected = assignment.nodes.every(
          (node, index) => node === config.selected_assignment.nodes[index],
        );
        const marker = isBest && isSelected
          ? "离散最优 + Scheduler 返回"
          : isBest ? "离散最优" : isSelected ? "Scheduler 返回" : "";
        return [
          number(assignment.objective_rank), marker,
          decisionRoute(assignment.nodes), number(assignment.batch_count),
          number(assignment.execution_raw, 3),
          number(assignment.latency_raw, 6), number(assignment.objective, 6),
          signedPercent(assignment.objective / bestCost - 1),
          number(assignment.predicted_card_time_ms, 3),
          number(assignment.predicted_request_queue_sum_ms, 3),
          number(assignment.predicted_request_total_mean_ms, 3),
        ];
      }),
      `${config.label}：全部 ${scene.candidate_count} 个组合按离散 Cost 排序。`,
    );

    replaceTable(
      elements.reportRealSceneTrajectory,
      tableColumns([
        "Iteration", "更新前 route", "更新前 Loss", "投影后 route",
        "alpha", "本步 us",
      ]),
      config.steps.map((step) => [
        number(step.iteration), decisionRoute(step.route_before_update),
        number(step.forward_value, 6),
        decisionRoute(step.route_after_projection), number(step.alpha, 4),
        number(step.elapsed_us),
      ]),
      `初始 argmax ${decisionRoute(config.initial_route)}；第一次投影 ${decisionRoute(config.first_projection)}；` +
        `最佳保存 route ${decisionRoute(config.best_route_seen)}；最终 route ${decisionRoute(config.final_route)}；` +
        `50 步总计 ${number(config.optimizer_elapsed_us)} us。`,
    );
  };
  elements.reportRealSceneObjective.onchange = renderObjective;
  renderObjective();

  const conclusion = document.createElement("div");
  const heading = document.createElement("h3");
  heading.textContent = "这个现场说明了什么";
  const list = document.createElement("ul");
  for (const line of [...(scene.conclusions || []), `证据边界：${scene.boundary}`]) {
    const entry = document.createElement("li");
    entry.textContent = line;
    list.append(entry);
  }
  conclusion.append(heading, list);
  elements.reportRealSceneConclusion.replaceChildren(conclusion);
}

function renderVllmValidation(validation) {
  if (!validation) {
    return;
  }
  const audit = validation.audit || {};
  replaceTable(
    elements.reportVllmAudit,
    tableColumns([
      "模型 / 拓扑", "请求", "有效 Block", "Batch 上限", "凑批窗口",
      "Chunked prefill", "参数审计", "结果校验",
    ]),
    [[
      `${audit.model} / ${audit.topology}`,
      `${number(audit.request_count)} / ${number(audit.prompt_tokens)} tokens`,
      `${number(audit.block_size_tokens)} tokens`,
      number(audit.max_batch_size),
      `${number(audit.batch_window_ms)} ms / ${number(audit.batch_window_count)} requests`,
      `${audit.chunked_prefill ? "开启" : "关闭"}；partial=${number(audit.max_num_partial_prefills)}`,
      `traffic ${audit.traffic_spec_sha256}; state ${audit.warm_state_sha256}`,
      audit.result_validation,
    ]],
    "Block 1056 是 VLLM 对 Mamba cache alignment 后的真实有效值；所有配置均从运行日志和 sidecar 反向核验。",
  );

  replaceTable(
    elements.reportVllmRuns,
    tableColumns([
      "策略", "平均 / P50 ms", "P95 / P99 ms", "总耗时 s",
      "Master rounds / req", "Master Batch / 平均大小", "卡时 s / vs FlexLB",
      "Cache hit", "节点卡时 CV", "请求结果",
    ]),
    (validation.real_runs || []).map((run) => [
      run.label,
      `${number(run.http_mean_ms, 2)} / ${number(run.http_p50_ms, 2)}`,
      `${number(run.http_p95_ms, 2)} / ${number(run.http_p99_ms, 2)}`,
      number(run.elapsed_s, 2),
      `${number(run.schedule_rounds)} / ${number(run.requests_per_round, 3)}`,
      `${number(run.execution_batches)} / ${number(run.mean_batch_size, 3)}`,
      `${number(run.card_time_s, 3)} / ${run.card_time_change_percent == null
        ? "baseline" : `${number(run.card_time_change_percent, 2)}%`}`,
      `${number(run.cache_hit_percent, 2)}%`,
      number(run.node_card_time_cv, 4),
      run.correctness,
    ]),
    "这里的 Batch 是 NaviSched Master Batch，不是单次 VLLM GPU forward。Cost 的平均延迟、尾延迟和卡时在本次审计流量中更低，但 P50 仅下降 1.28%，且 Cache hit 低于 FlexLB。",
  );

  const cardTime = validation.card_time_diagnostics || {};
  replaceTable(
    elements.reportVllmCardTime,
    tableColumns([
      "策略", "Master Batch 大小", "单 Batch 耗时 ms", "单请求卡时 ms",
      "Cache-miss tokens", "每 miss token us", "多 chunk Master Batch", "解释",
    ]),
    (cardTime.runs || []).map((run) => [
      run.label, number(run.master_batch_size, 3),
      number(run.master_batch_duration_ms, 2), number(run.card_time_per_request_ms, 2),
      number(run.cache_miss_tokens), number(run.card_time_us_per_miss_token, 2),
      `${number(run.multi_chunk_master_batches)} / ${number(run.master_batches)} ` +
        `(${number(run.multi_chunk_percent, 2)}%)`,
      run.explanation,
    ]),
    cardTime.conclusion,
  );

  const predictionRows = [];
  for (const run of validation.real_runs || []) {
    for (const metric of run.prediction || []) {
      predictionRows.push([
        run.label, metric.metric, number(metric.samples),
        number(metric.predicted_mean_ms, 2), number(metric.actual_mean_ms, 2),
        number(metric.bias_actual_minus_predicted_ms, 2), number(metric.mae_ms, 2),
        metric.mape_percent == null ? "-" : `${number(metric.mape_percent, 2)}%`,
      ]);
    }
  }
  replaceTable(
    elements.reportVllmPrediction,
    tableColumns([
      "策略", "预测位置", "样本", "预测 ms", "真值 ms",
      "真值 - 预测 ms", "MAE ms", "MAPE",
    ]),
    predictionRows,
    "Agent 行是在线更新前的执行预测；Master total 包含排队与执行。正偏差表示调度时低估。",
  );

  replaceTable(
    elements.reportVllmSimulate,
    tableColumns([
      "策略", "窗口", "Master rounds / req", "Master Batch / 平均大小",
      "卡时 s", "Cache hit", "执行真值 ms", "Queue 真值 ms", "Total 真值 ms",
      "执行预测 / 真值 ms", "执行 MAE / MAPE", "边界",
    ]),
    (validation.simulate_runs || []).map((run) => [
      run.label, `${number(run.batch_window_ms)} ms`,
      `${number(run.schedule_rounds)} / ${number(run.requests_per_round, 3)}`,
      `${number(run.execution_batches)} / ${number(run.mean_batch_size, 3)}`,
      run.card_time_s == null ? "未采集" : number(run.card_time_s, 3),
      run.cache_hit_percent == null ? "未采集" : `${number(run.cache_hit_percent, 2)}%`,
      number(run.execution_actual_ms, 2), number(run.queue_actual_ms, 2),
      number(run.total_actual_ms, 2),
      run.execution_predicted_ms == null ? "-" :
        `${number(run.execution_predicted_ms, 2)} / ${number(run.execution_actual_ms, 2)}`,
      run.execution_mae_ms == null ? "-" :
        `${number(run.execution_mae_ms, 2)} / ${number(run.execution_mape_percent, 2)}%`,
      run.note,
    ]),
    "Cost real/sim 的 batch、卡时、Cache 和执行耗时接近；simulate 的 Queue 绝对值没有与真实运行对齐，不能替代真实 E2E 延迟。",
  );

  const list = document.createElement("ul");
  for (const line of validation.conclusions || []) {
    const item = document.createElement("li");
    item.textContent = line;
    list.append(item);
  }
  elements.reportVllmConclusion.replaceChildren(list);
}

function drawKVCacheLatencyChart(fit) {
  const canvas = elements.reportKVFitLatencyChart;
  if (!canvas || !fit) {
    return;
  }
  const context = canvas.getContext("2d");
  const scatter = fit.full_fit?.scatter || [];
  const curve = fit.full_fit?.rolling_curve || [];
  const xValues = scatter.map((point) => Number(point.tokens)).filter(Number.isFinite);
  const yValues = scatter.flatMap((point) => [
    Number(point.actual_ms),
    Number(point.predicted_ms),
  ]).filter(Number.isFinite);
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (!xValues.length || !yValues.length) {
    return;
  }

  const padding = {left: 78, right: 28, top: 34, bottom: 62};
  const plotWidth = canvas.width - padding.left - padding.right;
  const plotHeight = canvas.height - padding.top - padding.bottom;
  const xMax = Math.max(...xValues) * 1.03;
  const yMin = Math.min(0, ...yValues);
  const yMax = Math.max(...yValues) * 1.06;
  const xPixel = (value) => padding.left + value / xMax * plotWidth;
  const yPixel = (value) => padding.top + (yMax - value) / (yMax - yMin) * plotHeight;

  context.font = '13px "IBM Plex Sans", "Noto Sans SC", sans-serif';
  context.lineWidth = 1;
  context.strokeStyle = "#d8dfd9";
  context.fillStyle = "#60706b";
  for (let index = 0; index <= 5; index += 1) {
    const value = yMin + (yMax - yMin) * index / 5;
    const y = yPixel(value);
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(canvas.width - padding.right, y);
    context.stroke();
    context.textAlign = "right";
    context.fillText(`${number(value, 0)} ms`, padding.left - 10, y + 4);
  }
  for (let index = 0; index <= 6; index += 1) {
    const value = xMax * index / 6;
    const x = xPixel(value);
    context.beginPath();
    context.moveTo(x, padding.top);
    context.lineTo(x, canvas.height - padding.bottom);
    context.stroke();
    context.textAlign = "center";
    context.fillText(`${number(value / 1000, 0)}k`, x, canvas.height - padding.bottom + 24);
  }

  for (const point of scatter) {
    const x = xPixel(Number(point.tokens));
    context.fillStyle = "rgba(41, 111, 168, 0.18)";
    context.beginPath();
    context.arc(x, yPixel(Number(point.actual_ms)), 2.1, 0, Math.PI * 2);
    context.fill();
    context.fillStyle = "rgba(201, 87, 61, 0.18)";
    context.beginPath();
    context.arc(x, yPixel(Number(point.predicted_ms)), 2.1, 0, Math.PI * 2);
    context.fill();
  }

  const drawCurve = (field, color) => {
    context.strokeStyle = color;
    context.lineWidth = 3;
    context.beginPath();
    curve.forEach((point, index) => {
      const x = xPixel(Number(point.token_median));
      const y = yPixel(Number(point[field]));
      if (index === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    });
    context.stroke();
  };
  drawCurve("actual_mean_ms", "#296fa8");
  drawCurve("predicted_mean_ms", "#c9573d");

  context.fillStyle = "#152421";
  context.textAlign = "center";
  context.font = '600 14px "IBM Plex Sans", "Noto Sans SC", sans-serif';
  context.fillText("单个调度 Batch 的总 Token 数", padding.left + plotWidth / 2, canvas.height - 12);
  context.save();
  context.translate(18, padding.top + plotHeight / 2);
  context.rotate(-Math.PI / 2);
  context.fillText("KV Cache 初始化耗时", 0, 0);
  context.restore();

  const legend = [
    ["rgba(41, 111, 168, 0.65)", "全部实际 KV init 点 + 滚动均值"],
    ["rgba(201, 87, 61, 0.65)", "全部拟合 KV init 点 + 滚动均值"],
  ];
  context.font = '13px "IBM Plex Sans", "Noto Sans SC", sans-serif';
  context.textAlign = "left";
  legend.forEach(([color, label], index) => {
    const x = padding.left + 18;
    const y = padding.top + 18 + index * 24;
    context.strokeStyle = color;
    context.lineWidth = 4;
    context.beginPath();
    context.moveTo(x, y);
    context.lineTo(x + 26, y);
    context.stroke();
    context.fillStyle = "#152421";
    context.fillText(label, x + 36, y + 4);
  });
}

function drawKVCacheFitChart(fit) {
  const canvas = elements.reportKVFitChart;
  if (!canvas || !fit) {
    return;
  }
  const context = canvas.getContext("2d");
  const scatter = fit.full_fit?.scatter || [];
  const fullCurve = fit.full_fit?.rolling_curve || [];
  const heldoutCurve = fit.heldout?.rolling_curve || [];
  const allX = [
    ...scatter.map((point) => Number(point.tokens)),
    ...fullCurve.map((point) => Number(point.token_median)),
    ...heldoutCurve.map((point) => Number(point.token_median)),
  ].filter(Number.isFinite);
  const allY = [
    ...scatter.map((point) => Number(point.residual_ms)),
    ...fullCurve.map((point) => Number(point.bias_actual_minus_predicted_ms)),
    ...heldoutCurve.map((point) => Number(point.bias_actual_minus_predicted_ms)),
  ].filter(Number.isFinite);
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (!allX.length || !allY.length) {
    return;
  }

  const padding = {left: 78, right: 28, top: 34, bottom: 62};
  const plotWidth = canvas.width - padding.left - padding.right;
  const plotHeight = canvas.height - padding.top - padding.bottom;
  const xMin = 0;
  const xMax = Math.max(...allX) * 1.03;
  const rawYMin = Math.min(...allY, 0);
  const rawYMax = Math.max(...allY, 0);
  const yPadding = Math.max(10, (rawYMax - rawYMin) * 0.08);
  const yMin = rawYMin - yPadding;
  const yMax = rawYMax + yPadding;
  const xPixel = (value) => padding.left + (value - xMin) / (xMax - xMin) * plotWidth;
  const yPixel = (value) => padding.top + (yMax - value) / (yMax - yMin) * plotHeight;

  context.font = '13px "IBM Plex Sans", "Noto Sans SC", sans-serif';
  context.lineWidth = 1;
  context.strokeStyle = "#d8dfd9";
  context.fillStyle = "#60706b";
  for (let index = 0; index <= 5; index += 1) {
    const value = yMin + (yMax - yMin) * index / 5;
    const y = yPixel(value);
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(canvas.width - padding.right, y);
    context.stroke();
    context.textAlign = "right";
    context.fillText(`${number(value, 0)} ms`, padding.left - 10, y + 4);
  }
  for (let index = 0; index <= 6; index += 1) {
    const value = xMax * index / 6;
    const x = xPixel(value);
    context.beginPath();
    context.moveTo(x, padding.top);
    context.lineTo(x, canvas.height - padding.bottom);
    context.stroke();
    context.textAlign = "center";
    context.fillText(`${number(value / 1000, 0)}k`, x, canvas.height - padding.bottom + 24);
  }

  context.strokeStyle = "#75827e";
  context.lineWidth = 1.5;
  context.beginPath();
  context.moveTo(padding.left, yPixel(0));
  context.lineTo(canvas.width - padding.right, yPixel(0));
  context.stroke();

  context.fillStyle = "rgba(41, 111, 168, 0.22)";
  for (const point of scatter) {
    context.beginPath();
    context.arc(xPixel(Number(point.tokens)), yPixel(Number(point.residual_ms)), 2.4, 0, Math.PI * 2);
    context.fill();
  }

  const drawCurve = (points, color) => {
    context.strokeStyle = color;
    context.lineWidth = 3;
    context.beginPath();
    points.forEach((point, index) => {
      const x = xPixel(Number(point.token_median));
      const y = yPixel(Number(point.bias_actual_minus_predicted_ms));
      if (index === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    });
    context.stroke();
  };
  drawCurve(fullCurve, "#c9573d");
  drawCurve(heldoutCurve, "#087f75");

  context.fillStyle = "#152421";
  context.textAlign = "center";
  context.font = '600 14px "IBM Plex Sans", "Noto Sans SC", sans-serif';
  context.fillText("单个调度 Batch 的总 Token 数", padding.left + plotWidth / 2, canvas.height - 12);
  context.save();
  context.translate(18, padding.top + plotHeight / 2);
  context.rotate(-Math.PI / 2);
  context.fillText("实际 KV init - 预测 KV init", 0, 0);
  context.restore();

  const legend = [
    ["rgba(41, 111, 168, 0.55)", "全部真实 Batch 的拟合残差"],
    ["#c9573d", "2000 样本训练回算：100 Batch 滚动 bias"],
    ["#087f75", "后 500 held-out：50 Batch 滚动 bias"],
  ];
  context.font = '13px "IBM Plex Sans", "Noto Sans SC", sans-serif';
  context.textAlign = "left";
  legend.forEach(([color, label], index) => {
    const x = padding.left + 18;
    const y = padding.top + 18 + index * 24;
    context.strokeStyle = color;
    context.lineWidth = 4;
    context.beginPath();
    context.moveTo(x, y);
    context.lineTo(x + 26, y);
    context.stroke();
    context.fillStyle = "#152421";
    context.fillText(label, x + 36, y + 4);
  });
}

function renderKVCacheFit(fit) {
  if (!fit) {
    elements.reportKVFitSummary.replaceChildren();
    elements.reportKVFitBins.replaceChildren();
    elements.reportKVFitRequestBins.replaceChildren();
    elements.reportKVFitParameters.replaceChildren();
    elements.reportKVFitNotes.replaceChildren();
    return;
  }
  const full = fit.full_fit?.metrics || {};
  const heldout = fit.heldout?.metrics || {};
  elements.reportKVFitSummary.replaceChildren(
    metricItem("训练回算 MAE", `${number(full.mae_ms, 2)} ms`),
    metricItem("训练回算 RMSE", `${number(full.rmse_ms, 2)} ms`),
    metricItem("训练回算 R²", number(full.r_squared, 3)),
    metricItem("Held-out MAE", `${number(heldout.mae_ms, 2)} ms`),
    metricItem("Held-out bias", `${number(heldout.bias_actual_minus_predicted_ms, 2)} ms`),
  );
  drawKVCacheLatencyChart(fit);
  drawKVCacheFitChart(fit);

  const binRows = [];
  for (const [scope, result] of [
    ["2000 样本训练回算", fit.full_fit],
    ["后 500 样本 held-out", fit.heldout],
  ]) {
    for (const bin of result?.token_bins || []) {
      const range = bin.token_max == null
        ? `>= ${number(bin.token_min)}`
        : `${number(bin.token_min)}-${number(bin.token_max)}`;
      binRows.push([
        scope, range, number(bin.samples), number(bin.token_mean, 0),
        number(bin.actual_mean_ms, 2), number(bin.predicted_mean_ms, 2),
        number(bin.bias_actual_minus_predicted_ms, 2),
        number(bin.mae_ms, 2), number(bin.rmse_ms, 2),
      ]);
    }
  }
  replaceTable(
    elements.reportKVFitBins,
    tableColumns([
      "评估范围", "Batch Token 区间", "样本", "Token 均值",
      "实际 KV ms", "预测 KV ms", "实际 - 预测 ms", "MAE ms", "RMSE ms",
    ]),
    binRows,
    "分桶按整个调度 Batch 的 hit + miss token 总数统计；正 bias 表示模型低估 KV 初始化耗时。",
  );

  replaceTable(
    elements.reportKVFitRequestBins,
    tableColumns([
      "Batch request 数", "样本", "实际均值 ms", "实际 P50 ms",
      "< 1 ms 占比", ">= 50 ms 占比", ">= 100 ms 占比",
    ]),
    (fit.full_fit?.request_bins || []).map((bin) => [
      bin.label, number(bin.samples), number(bin.actual_mean_ms, 2),
      number(bin.actual_p50_ms, 2), `${number(bin.under_1ms_percent, 1)}%`,
      `${number(bin.at_least_50ms_percent, 1)}%`,
      `${number(bin.at_least_100ms_percent, 1)}%`,
    ]),
    "按 Batch 内 request 数统计 KV init 慢路径概率；小 Batch 也会出现百毫秒级延迟，说明耗时不只由 token 数决定。",
  );

  replaceTable(
    elements.reportKVFitParameters,
    tableColumns(["参数", "2000 样本拟合值", "前 1500 样本拟合值"]),
    (fit.full_fit?.parameters || []).map((value, index) => [
      `P${index}`,
      Number(value).toExponential(6),
      Number(fit.heldout?.parameters?.[index] || 0).toExponential(6),
    ]),
    "参数用于精确复算。该 9 参数公式存在不可辨识性，不能把单个参数的绝对大小解释为独立物理成本。",
  );

  const list = document.createElement("ul");
  for (const note of fit.notes || []) {
    const item = document.createElement("li");
    item.textContent = note;
    list.append(item);
  }
  elements.reportKVFitNotes.replaceChildren(list);
}

function renderRealWorldPerformance(performance) {
  const runs = performance?.valid_runs || [];
  elements.reportRealSection.hidden = !runs.length;
  if (!runs.length) {
    return;
  }
  renderVllmValidation(performance.vllm_validation);
  renderQueueCacheAccuracyAudit(performance.queue_cache_accuracy_audit);
  const environment = performance.environment || {};
  const methodology = performance.methodology || {};
  elements.reportRealMetrics.replaceChildren(
    metricItem("GPU / 实例数", `${number(environment.gpu_count)} / ${number(environment.instance_count)}`),
    metricItem("模型", environment.model || "-"),
    metricItem("统计请求数", number(methodology.measured_requests)),
    metricItem("凑批窗口", `${number(methodology.batch_window_ms)} ms / ${number(methodology.batch_window_count)} 请求`),
    metricItem("Block 大小", `${number(environment.block_size_tokens)} tokens`),
  );
  const contextRows = [
    ["运行环境", `${environment.runtime || "-"}; ${environment.mode || "-"}`],
    ["请求流量", methodology.traffic || "-"],
    ["稳态统计规则", methodology.steady_state_rule || "-"],
    ["构建产物", (performance.artifacts || []).map((item) => (
      `${item.name}: ${item.sha256}`
    )).join("; ") || "-"],
  ];
  elements.reportRealContext.replaceChildren(...contextRows.map(([label, value]) => {
    const item = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = label;
    const detail = document.createElement("span");
    detail.textContent = value;
    item.append(name, detail);
    return item;
  }));

  const bestMean = minimumRun(runs, (run) => run.latency_mean_ms);
  const bestP95 = minimumRun(runs, (run) => run.latency_p95_ms);
  const lowestCardTime = minimumRun(runs, (run) => run.prefill_card_time_s);
  const bestPrediction = minimumRun(
    runs.filter((run) => run.prediction?.master_total),
    (run) => run.prediction.master_total.mape_percent,
  );
  const flexlbRun = runs.find((run) => run.scheduler === "flexlb");
  const flexlbCardUtilization = flexlbRun?.prefill_card_utilization_percent;
  elements.reportRealInsights.replaceChildren(
    insightItem("最低平均延迟", bestMean, `${number(bestMean.latency_mean_ms, 2)} ms`),
    insightItem("最低 P95 延迟", bestP95, `${number(bestP95.latency_p95_ms, 2)} ms`),
    insightItem("最低 Prefill 卡时", lowestCardTime, `${number(lowestCardTime.prefill_card_time_s, 2)} card-s`),
    insightItem(
      "最低 Master 总延迟 MAPE",
      bestPrediction,
      `${number(bestPrediction.prediction.master_total.mape_percent, 2)}%`,
    ),
  );

  replaceTable(
    elements.reportRealRuns,
    tableColumns([
      "运行", "产物", "目标函数", "lambda", "凹度参数",
      "卡时利用率 / vs FlexLB", "平均 ms", "P50",
      "P95 ms", "最大 ms", "相对 FlexLB", "QPS", "Cache 命中请求",
      "Agent 执行 Batch", "平均执行 Batch", "Master 调度轮次",
      "Master req/轮", "Master req/轮分布", "卡时 s", "节点 CV",
      "每节点请求数", "失败 / 超时",
    ]),
    runs.map((run) => [
      run.label,
      run.artifact,
      run.objective,
      run.lambda == null ? "-" : lambdaNumber(run.lambda),
      run.concavity == null ? "-" : number(run.concavity),
      `${number(run.prefill_card_utilization_percent, 2)}% / ${signedPercent(
        flexlbCardUtilization > 0
          ? run.prefill_card_utilization_percent / flexlbCardUtilization - 1
          : null,
      )}`,
      number(run.latency_mean_ms, 2),
      number(run.latency_p50_ms, 2),
      number(run.latency_p95_ms, 2),
      number(run.latency_max_ms, 2),
      run.mean_change_vs_flexlb == null
        ? "baseline"
        : signedPercent(run.mean_change_vs_flexlb),
      number(run.qps_auxiliary, 2),
      number(run.cache_hit_requests),
      number(run.batch_count),
      number(run.mean_batch_size, 2),
      run.master_schedule_rounds == null
        ? "未采集" : number(run.master_schedule_rounds),
      run.master_requests_per_schedule == null
        ? "未采集" : number(run.master_requests_per_schedule, 3),
      run.master_schedule_request_histogram == null
        ? "未采集"
        : Object.entries(run.master_schedule_request_histogram)
          .map(([requestCount, roundCount]) => `${requestCount}:${roundCount}`)
          .join("; "),
      number(run.prefill_card_time_s, 2),
      number(run.node_count_cv, 3),
      `${Math.min(...run.node_request_counts)}-${Math.max(...run.node_request_counts)}`,
      `${number(run.failure_count)} / ${number(run.timeout_count)}`,
    ]),
    "Master 调度轮次和 Agent 执行 Batch 是不同口径；历史运行未采集 Master histogram，后续运行可在 Replay 关闭时直接从 snapshot 读取。",
  );

  const discreteRegression = performance.discrete_repair_regression || {};
  const problemScene = discreteRegression.problem_scene || {};
  const beforeScene = problemScene.before || {};
  const afterScene = problemScene.after || {};
  replaceTable(
    elements.reportRealDiscreteScene,
    tableColumns([
      "版本", "Request / Node", "选中 Node", "物理 Batch",
      "连续 Loss", "离散 Objective", "证据",
    ]),
    [
      [
        "修复前", `${number(problemScene.requests)} / ${number(problemScene.nodes)}`,
        (beforeScene.selected_nodes || []).join(", "), number(beforeScene.batch_count),
        number(beforeScene.continuous_loss, 3), number(beforeScene.discrete_objective, 3),
        problemScene.source || "-",
      ],
      [
        "离散修复后", `${number(problemScene.requests)} / ${number(problemScene.nodes)}`,
        (afterScene.selected_nodes || []).join(", "), number(afterScene.batch_count),
        number(afterScene.continuous_loss, 3), number(afterScene.discrete_objective, 3),
        `runner ${discreteRegression.runner_sha256 || "-"}`,
      ],
    ],
    "连续 Loss 只用于优化轨迹；最终返回决策必须按真实离散 Objective 比较。",
  );
  replaceTable(
    elements.reportRealDiscreteRegression,
    tableColumns([
      "场景", "运行", "状态", "稳态请求", "平均 Batch", "卡时 s",
      "卡时 vs 修复前", "平均 ms", "P95 ms", "预测 MAPE",
      "调度平均 / 最大 us",
    ]),
    (discreteRegression.rows || []).map((run) => [
      run.scenario, run.run, run.status, number(run.request_count),
      number(run.batch_mean, 3), number(run.card_time_s, 3),
      signedPercent(run.card_time_change_vs_old), number(run.latency_mean_ms, 2),
      number(run.latency_p95_ms, 2), `${number(run.prediction_mape_percent, 3)}%`,
      `${number(run.schedule_mean_us, 1)} / ${number(run.schedule_max_us)}`,
    ]),
    `实际离散参数：check interval=${number(discreteRegression.effective_config?.discrete_check_interval)}，` +
      `gap threshold=${number(discreteRegression.effective_config?.discrete_gap_threshold, 2)}，` +
      `repair alpha decay=${number(discreteRegression.effective_config?.discrete_repair_alpha_decay, 2)}。`,
  );
  const discreteConclusion = document.createElement("ul");
  for (const line of discreteRegression.conclusions || []) {
    const item = document.createElement("li");
    item.textContent = line;
    discreteConclusion.append(item);
  }
  elements.reportRealDiscreteConclusion.replaceChildren(discreteConclusion);

  const productionRange = performance.production_range_regression || {};
  replaceTable(
    elements.reportRealProductionRange,
    tableColumns([
      "场景", "策略", "轮次", "状态", "稳态请求", "平均 Batch",
      "卡时 s / vs proxy", "平均 ms", "P95 ms", "排队 / 执行 ms",
      "预测 / 真实 ms", "MAPE", "失败 / 超限", "证据",
    ]),
    (productionRange.rows || []).map((run) => [
      run.scenario, run.label, number(run.repetition), run.status,
      number(run.request_count),
      number(run.batch_mean, 3),
      `${number(run.card_time_s, 3)} / ${run.card_time_change_vs_proxy == null
        ? "baseline" : signedPercent(run.card_time_change_vs_proxy)}`,
      number(run.latency_mean_ms, 2), number(run.latency_p95_ms, 2),
      `${number(run.queue_mean_ms, 2)} / ${number(run.execution_mean_ms, 2)}`,
      `${number(run.predicted_mean_ms, 2)} / ${number(run.actual_mean_ms, 2)}`,
      `${number(run.prediction_mape_percent, 3)}%`,
      `${number(run.failure_count)} / ${number(run.over_token_limit_count)}`,
      `${run.archive_sha256?.slice(0, 12) || "-"} / ${run.driver_sha256?.slice(0, 12) || "-"}`,
    ]),
    productionRange.scope || "同一流量、模型参数和 archive 下的策略对比。",
  );
  const productionConclusion = document.createElement("ul");
  for (const line of productionRange.conclusions || []) {
    const item = document.createElement("li");
    item.textContent = line;
    productionConclusion.append(item);
  }
  elements.reportRealProductionConclusion.replaceChildren(productionConclusion);

  const scaling = performance.scaling_reliability || {};
  const scalingRows = scaling.rows || [];
  replaceTable(
    elements.reportRealScaling,
    tableColumns([
      "GPU", "配置", "轮次", "平均 ms / vs FlexLB", "P95 ms / vs FlexLB",
      "卡时 s / vs FlexLB", "平均 Batch / vs FlexLB", "节点 CV",
      "预测 / 真实 ms", "MAPE", "证据范围",
    ]),
    scalingRows.map((run) => [
      number(run.gpu_count), run.configuration, number(run.run_count),
      `${number(run.latency_mean_ms, 2)} / ${signedPercent(
        run.latency_mean_change_vs_flexlb_percent / 100,
      )}`,
      `${number(run.latency_p95_ms, 2)} / ${signedPercent(
        run.latency_p95_change_vs_flexlb_percent / 100,
      )}`,
      `${number(run.prefill_card_time_ms / 1000, 2)} / ${signedPercent(
        run.prefill_card_time_change_vs_flexlb_percent / 100,
      )}`,
      `${number(run.physical_batch_mean, 3)} / ${signedPercent(
        run.batch_change_vs_flexlb_percent / 100,
      )}`,
      number(run.node_cv, 3),
      run.prediction_mean_ms == null
        ? "-"
        : `${number(run.prediction_mean_ms, 2)} / ${number(run.actual_mean_ms, 2)}`,
      run.prediction_mape_percent == null
        ? "-" : `${number(run.prediction_mape_percent, 2)}%`,
      run.validity,
    ]),
    scaling.scope || "不同卡数对应不同单卡压力，只在同一卡数内计算相对 FlexLB 的变化。",
  );
  const scalingConclusion = document.createElement("ul");
  for (const line of scaling.conclusions || []) {
    const item = document.createElement("li");
    item.textContent = line;
    scalingConclusion.append(item);
  }
  elements.reportRealScalingConclusion.replaceChildren(scalingConclusion);

  const powerScan = performance.normalization_power_runs || {};
  const powerRows = powerScan.rows || [];
  replaceTable(
    elements.reportRealPowerScan,
    tableColumns([
      "Power", "Scale", "轮次", "平均 ms / vs FlexLB", "P50",
      "P95 ms / vs FlexLB", "最大 ms", "卡时 s / vs FlexLB",
      "平均 Batch / vs FlexLB", "节点 CV", "预测 / 真实 ms", "MAPE",
    ]),
    powerRows.map((run) => {
      const aggregate = run.aggregate;
      const change = run.change_vs_flexlb_percent;
      return [
        number(run.power, 2), number(run.scale, 6), number(run.run_count),
        `${number(aggregate.latency_mean_ms, 2)} / ${signedPercent(
          change.latency_mean_ms / 100,
        )}`,
        number(aggregate.latency_p50_ms, 2),
        `${number(aggregate.latency_p95_ms, 2)} / ${signedPercent(
          change.latency_p95_ms / 100,
        )}`,
        number(aggregate.latency_max_ms, 2),
        `${number(aggregate.prefill_card_time_ms / 1000, 2)} / ${signedPercent(
          change.prefill_card_time_ms / 100,
        )}`,
        `${number(aggregate.physical_batch_mean, 3)} / ${signedPercent(
          change.physical_batch_mean / 100,
        )}`,
        number(aggregate.node_cv, 3),
        aggregate.predicted_ms == null
          ? "-"
          : `${number(aggregate.predicted_ms, 2)} / ${number(aggregate.actual_ms, 2)}`,
        aggregate.mape_percent == null
          ? "-" : `${number(aggregate.mape_percent, 2)}%`,
      ];
    }),
    powerScan.scope || "Power 使用平均权重匹配 scale；不同轮次按均值展示。",
  );
  const powerConclusion = document.createElement("ul");
  for (const line of powerScan.conclusions || []) {
    const item = document.createElement("li");
    item.textContent = line;
    powerConclusion.append(item);
  }
  elements.reportRealPowerConclusion.replaceChildren(powerConclusion);

  renderDecisionScene(performance.normalization_decision_scene);

  const normalizationReplay = performance.normalization_replay || {};
  const normalizationRuns = normalizationReplay.runs || [];
  replaceTable(
    elements.reportRealNormalizationReplay,
    tableColumns([
      "模式", "Power", "Scale", "Concavity", "请求权重（820 / 220）",
      "选中节点", "Batch", "调度 us", "预测卡时 ms", "预测请求均值 ms",
      "执行项原值", "执行项贡献", "延迟项原值", "延迟项贡献", "总 Cost",
    ]),
    normalizationRuns.map((run) => [
      run.mode, number(run.power, 2), number(run.scale, 6),
      number(run.concavity), run.request_weights.map((value) => number(value, 6)).join(" / "),
      run.selected_nodes.join(" / "), number(run.batch_count),
      `${number(run.schedule_us_mean, 2)} +/- ${number(run.schedule_us_stddev, 2)}`,
      number(run.card_time_ms, 3), number(run.request_total_mean_ms, 3),
      number(run.execution_raw, 3), number(run.execution_weighted, 3),
      number(run.latency_raw, 3), number(run.latency_weighted, 3),
      number(run.selected_objective, 3),
    ]),
    normalizationReplay.note || "同一真实现场的离线 Replay 对比。",
  );
  const normalizationAggregate = normalizationReplay.aggregate_runs || [];
  replaceTable(
    elements.reportRealNormalizationAggregate,
    tableColumns([
      "模式", "Power", "Scale", "Concavity", "现场 / 请求", "Batch",
      "平均 Batch", "调度 us/现场", "预测队列 ms/请求",
      "平均 Batch 执行 ms", "预测总卡时 ms", "与现场原决策一致",
    ]),
    normalizationAggregate.map((run) => [
      run.mode, number(run.power, 2), number(run.scale, 6),
      number(run.concavity), `${number(run.scene_count)} / ${number(run.request_count)}`,
      number(run.batch_count), number(run.mean_batch_size, 3),
      number(run.schedule_us_mean, 2), number(run.queue_ms_per_request, 3),
      number(run.batch_execution_ms_mean, 3), number(run.predicted_card_time_ms, 3),
      `${number(run.recorded_matches)} / ${number(run.scene_count)}`,
    ]),
    normalizationReplay.aggregate_note || "同轮真实 snapshot 的聚合 Replay 对比。",
  );

  const sweeps = performance.supplemental_real_runs || [];
  replaceTable(
    elements.reportRealSweeps,
    tableColumns([
      "扫描项", "分组", "公式/配置", "窗口 ms", "lambda", "k", "alpha",
      "平均 ms", "P50", "P95 ms", "最大 ms", "QPS", "Cache 命中请求",
      "节点 CV", "可比性说明",
    ]),
    sweeps.map((run) => [
      run.label, run.group, run.variant, number(run.window_ms),
      run.lambda == null ? "-" : lambdaNumber(run.lambda),
      run.concavity == null ? "-" : number(run.concavity),
      run.alpha == null ? "-" : number(run.alpha, 4),
      number(run.latency_mean_ms, 2), number(run.latency_p50_ms, 2),
      number(run.latency_p95_ms, 2), number(run.latency_max_ms, 2),
      number(run.qps_auxiliary, 2), number(run.cache_hit_requests),
      number(run.node_count_cv, 3), run.comparability,
    ]),
    "这些运行补充参数探索过程；只有标明使用相同公式、流量和 alpha 的行才适合作为严格单变量比较。",
  );

  const profileRows = [];
  for (const run of runs) {
    for (const [profile, metrics] of Object.entries(run.arrival_profiles || {})) {
      profileRows.push([
        run.label, profile, number(metrics.request_count),
        number(metrics.latency_mean_ms, 2), number(metrics.latency_p50_ms, 2),
        number(metrics.latency_p95_ms, 2), number(metrics.latency_max_ms, 2),
        number(metrics.qps_auxiliary, 2), number(metrics.cache_hit_requests),
        number(metrics.node_count_cv, 3),
      ]);
    }
  }
  replaceTable(
    elements.reportRealProfiles,
    tableColumns([
      "运行", "流量类型", "请求数", "平均 ms", "P50", "P95 ms",
      "最大 ms", "QPS", "Cache 命中请求", "节点 CV",
    ]),
    profileRows,
    "每行是一种到达 profile 的稳态结果，用于区分稀疏流量和突发流量下的策略差异。",
  );

  const predictionRows = [];
  for (const run of runs) {
    for (const [point, metrics] of Object.entries(run.prediction || {})) {
      predictionRows.push([
        run.label,
        point === "agent_pre_update" ? "Agent 更新前执行延迟" : "Master 预测总延迟",
        number(metrics.sample_count), number(metrics.predicted_mean_ms, 2),
        number(metrics.actual_mean_ms, 2), number(metrics.bias_actual_minus_predicted_ms, 2),
        number(metrics.mae_ms, 2), `${number(metrics.mape_percent, 2)}%`,
      ]);
    }
  }
  replaceTable(
    elements.reportRealPrediction,
    tableColumns([
      "运行", "预测位置", "样本数", "预测 ms", "真实 ms",
      "偏差 ms", "MAE ms", "MAPE",
    ]),
    predictionRows,
    "每行对齐一次运行中的事前预测与事后真值；正 bias 表示调度时低估延迟。",
  );

  const simulate = performance.simulate_runs || [];
  replaceTable(
    elements.reportRealSimulate,
    tableColumns([
      "运行", "目标函数", "窗口 ms", "lambda", "凹度参数", "平均 ms", "P50",
      "P95 ms", "最大 ms", "调度批次", "平均 Batch", "卡时 s",
      "卡时利用率", "节点 CV", "样本数", "MAPE",
    ]),
    simulate.map((run) => [
      run.label, run.objective, number(run.window_ms),
      lambdaNumber(run.lambda), number(run.concavity),
      number(run.latency_mean_ms, 2), number(run.latency_p50_ms, 2),
      number(run.latency_p95_ms, 2), number(run.latency_max_ms, 2),
      number(run.batch_count), number(run.mean_batch_size, 2),
      number(run.prefill_card_time_s, 2),
      `${number(run.prefill_card_utilization_percent, 2)}%`,
      number(run.node_count_cv, 3), number(run.prediction_sample_count),
      `${number(run.execution_prediction_mape_percent, 2)}%`,
    ]),
    "模拟绝对延迟不包含 RTP Python/frontend 固定开销；主要用于核对 batch、卡时和不均衡趋势。",
  );

  renderKVCacheFit(performance.kv_cache_curve_fit);

  const fitting = performance.fitting_experiments || [];
  replaceTable(
    elements.reportRealFitting,
    tableColumns([
      "实验", "评估范围", "样本数", "更新窗口", "learning rate",
      "MAPE", "MAE ms", "偏差 ms", "结论",
    ]),
    fitting.map((run) => [
      run.experiment, run.scope, run.samples, run.window, run.learning_rate,
      run.mape_percent == null ? "-" : `${number(run.mape_percent, 2)}%`,
      run.mae_ms == null ? "-" : number(run.mae_ms, 2),
      run.bias_ms == null ? "-" : number(run.bias_ms, 2),
      run.conclusion,
    ]),
    "MAPE 均按样本在梯度更新前的预测计算；OLS 只表示离线可达精度，不是线上算法。",
  );

  const ledger = performance.experiment_ledger || [];
  replaceTable(
    elements.reportRealLedger,
    tableColumns(["实验想法", "状态", "现有证据", "边界与后续限制"]),
    ledger.map((item) => [item.idea, item.status, item.evidence, item.boundary]),
    "台账同时保留已完成、部分完成和未执行项目，避免把探索性结果写成已经充分验证的结论。",
  );

  const invalidRuns = performance.invalid_runs || [];
  replaceTable(
    elements.reportRealInvalid,
    tableColumns(["运行", "状态", "原因", "证据"]),
    invalidRuns.map((run) => [run.label, run.status, run.reason, run.evidence]),
    "这些运行因运行时异常或输入格式错误被排除，不参与最优配置判断。",
  );

  const conclusion = document.createElement("div");
  const heading = document.createElement("h3");
  heading.textContent = "总结、结论与限制";
  const list = document.createElement("ul");
  for (const line of [
    ...(performance.conclusions || []),
    ...(performance.limitations || []).map((item) => `限制：${item}`),
  ]) {
    const entry = document.createElement("li");
    entry.textContent = line;
    list.append(entry);
  }
  conclusion.append(heading, list);
  elements.reportRealConclusion.replaceChildren(conclusion);
}

function renderQueueCacheAccuracyAudit(audit) {
  elements.reportRealQueueCacheAudit.hidden = !audit;
  if (!audit) {
    return;
  }

  const config = audit.configuration || {};
  const cache = audit.pending_cache || {};
  const queue = audit.queue_boundary || {};
  elements.reportRealQueueCacheMetrics.replaceChildren(
    metricItem("真实请求", number(cache.total_requests)),
    metricItem("Cache exact", `${number(cache.exact_percent, 4)}%`),
    metricItem("Cache under", number(cache.under_count)),
    metricItem("Queue 对齐样本", number(queue.joined_samples)),
    metricItem("稳态 Queue MAE", `${number(audit.queue_stages?.steady_401_plus?.queue_mae_ms, 2)} ms`),
  );

  const contextRows = [
    ["数据范围", `${number(config.agent_count)} Agents / ${number(cache.total_requests)} finished requests / ${audit.source || "-"}`],
    ["Cache 配置", `block=${number(config.block_size_tokens)} tokens; kernel block=${number(config.kernel_block_size_tokens)} tokens; TP=${number(config.tp_size)}; linear_step=${number(config.linear_step)}; hybrid_tail=${config.hybrid_tail_required ? 1 : 0}`],
    ["Pending 语义", cache.reservation_semantics || "-"],
    ["实现与产物", `${audit.implementation || "-"}; SDK SHA256=${audit.sdk_sha256 || "-"}`],
  ];
  elements.reportRealQueueCacheContext.replaceChildren(...contextRows.map(([label, value]) => {
    const item = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = label;
    const detail = document.createElement("span");
    detail.textContent = value;
    item.append(name, detail);
    return item;
  }));

  replaceTable(
    elements.reportRealCacheAccuracy,
    tableColumns(["结果", "请求数", "占比", "Block MAE", "解释"]),
    [
      ["Exact", number(cache.exact_count), `${number(cache.exact_percent, 4)}%`, number(cache.block_mae, 6), "Master 预测 hit blocks 与 RTP 执行时真实值完全一致"],
      ["Under", number(cache.under_count), `${number(cache.under_percent, 4)}%`, "-", "前序排队请求产生的 929 个 under 已全部消除"],
      ["Over", number(cache.over_count), `${number(cache.over_percent, 4)}%`, "-", cache.over_reason || "-"],
    ],
    "Exact/Under/Over 均以同一 request 的 Master 调度时预测与 RTP 执行开始时真值比较。",
  );
  elements.reportRealCacheAccuracy.querySelectorAll("tbody tr").forEach((row, index) => {
    row.classList.toggle("audit-attention-row", index === 1);
    row.classList.toggle("audit-warning-row", index === 2);
  });

  replaceTable(
    elements.reportRealQueueBoundary,
    tableColumns(["校验项", "样本", "结果", "结论"]),
    [
      ["Batch request count", queue.joined_samples, queue.request_count_match, "Agent、RTP 和 latency components 完全一致"],
      ["Agent actual Queue vs ready→canrun", queue.joined_samples, `${number(queue.agent_queue_delta_mean_ms, 3)} ms`, "Queue 真值边界一致"],
      ["canrun→running", queue.joined_samples, `mean ${number(queue.canrun_to_running_mean_ms, 3)} ms / max ${number(queue.canrun_to_running_max_ms, 3)} ms`, "Execution 起点没有数百毫秒固定漏项"],
      ["ready→RTP scheduler enqueue", queue.joined_samples, `mean ${number(queue.ready_to_scheduler_enqueue_mean_ms, 3)} ms`, "只存在毫秒级 Adapter/RTP 交接"],
      ["loading_cache_ms", queue.joined_samples, `${number(queue.loading_cache_zero_count)}/${number(queue.joined_samples)} 为 0`, "本组样本没有 KV Cache loading 延迟"],
    ],
    "31 个样本跨 10 个真实 Agent 完成 request、batch 和时间戳三方 join。",
  );

  const stages = audit.queue_stages || {};
  const stageRows = [
    stages.cold,
    stages.transition,
    stages.steady_100_plus,
    stages.steady_200_plus,
    stages.steady_401_plus,
  ].filter(Boolean);
  replaceTable(
    elements.reportRealQueueStages,
    tableColumns(["阶段", "范围", "实际 Queue 均值 ms", "Queue bias ms", "Queue MAE ms", "Execution MAE ms", "判定"]),
    stageRows.map((stage) => [
      stage.label, stage.range,
      stage.actual_queue_mean_ms == null ? "-" : number(stage.actual_queue_mean_ms, 2),
      stage.queue_bias_ms == null ? "-" : number(stage.queue_bias_ms, 2),
      number(stage.queue_mae_ms, 2),
      stage.execution_mae_ms == null ? "-" : number(stage.execution_mae_ms, 2),
      stage.conclusion,
    ]),
    "fit_step 分层来自三方 join；Batch 序号分层来自全部 4,743 条 BATCH_FINISHED training sample。",
  );
  elements.reportRealQueueStages.querySelectorAll("tbody tr").forEach((row, index) => {
    row.classList.toggle("audit-warning-row", index < 2);
    row.classList.toggle("audit-attention-row", index === stageRows.length - 1);
  });

  const recurrence = audit.recurrence_example || {};
  replaceTable(
    elements.reportRealQueueRecurrence,
    tableColumns(["Agent", "误差来源", "预测 ms", "真实 ms", "实际 - 预测 ms", "传递结果"]),
    [
      [recurrence.agent, "Batch 1 Execution", number(recurrence.batch_1_execution_predicted_ms, 3), number(recurrence.batch_1_execution_actual_ms, 3), number(recurrence.batch_1_execution_error_ms, 3), "成为下一 Batch 的 Queue 增量"],
      [recurrence.agent, "Batch 2 Queue", number(recurrence.batch_2_queue_predicted_ms, 3), number(recurrence.batch_2_queue_actual_ms, 3), number(recurrence.batch_2_queue_error_ms, 3), `与前序 Execution error 相差 ${number(recurrence.error_delta_ms, 3)} ms`],
    ],
    recurrence.conclusion || "-",
  );
  elements.reportRealQueueRecurrence.querySelectorAll("tbody tr").forEach((row) => {
    row.classList.add("audit-warning-row");
  });

  const conclusion = document.createElement("div");
  const heading = document.createElement("h3");
  heading.textContent = "证据结论与下一步";
  const list = document.createElement("ul");
  for (const line of audit.conclusions || []) {
    const item = document.createElement("li");
    item.textContent = line;
    list.append(item);
  }
  conclusion.append(heading, list);
  elements.reportRealQueueCacheConclusion.replaceChildren(conclusion);
}

function mean(values) {
  return values.length
    ? values.reduce((total, value) => total + Number(value), 0) / values.length
    : 0;
}

function shortHost(host) {
  const parts = String(host).split(".");
  return parts[parts.length - 1];
}

function formulaTransferClass(delta) {
  if (delta <= 0) return "formula-transfer-good";
  if (delta <= 1) return "formula-transfer-near";
  if (delta <= 5) return "formula-transfer-warning";
  return "formula-transfer-bad";
}

function makeFormulaMatrix(capture) {
  const section = document.createElement("section");
  section.className = "formula-matrix-panel";
  const heading = document.createElement("div");
  heading.className = "comparison-group-heading";
  const title = document.createElement("h3");
  title.textContent = capture.name;
  const detail = document.createElement("p");
  detail.textContent = `${capture.hosts.length} Agents; 统一初值; 时间顺序 70/30 train/test`;
  heading.append(title, detail);

  const scroll = document.createElement("div");
  scroll.className = "table-scroll formula-matrix-scroll";
  const table = document.createElement("table");
  table.className = "formula-matrix";
  const head = document.createElement("thead");
  const axisRow = document.createElement("tr");
  axisRow.className = "formula-matrix-axis";
  const axisCorner = document.createElement("th");
  axisCorner.textContent = "";
  axisCorner.setAttribute("aria-hidden", "true");
  const sourceAxis = document.createElement("th");
  sourceAxis.colSpan = capture.hosts.length;
  sourceAxis.textContent = "列 = 公式来源节点 A ->";
  axisRow.append(axisCorner, sourceAxis);
  const headRow = document.createElement("tr");
  const corner = document.createElement("th");
  corner.textContent = "行 = 样本节点 B ↓";
  corner.title = "每一行是被评估节点 B 的 held-out 样本";
  headRow.append(corner);
  for (const host of capture.hosts) {
    const cell = document.createElement("th");
    cell.textContent = shortHost(host);
    cell.title = host;
    headRow.append(cell);
  }
  head.append(axisRow, headRow);
  table.append(head);

  const body = document.createElement("tbody");
  for (const target of capture.hosts) {
    const row = document.createElement("tr");
    const targetCell = document.createElement("th");
    targetCell.textContent = shortHost(target);
    targetCell.title = target;
    row.append(targetCell);
    for (const source of capture.hosts) {
      const value = capture.offline_fitted_cross_host[target][source];
      const delta = value.source_minus_reference_mape_pct_points;
      const cell = document.createElement("td");
      cell.className = source === target
        ? "formula-transfer-self"
        : formulaTransferClass(delta);
      const change = document.createElement("strong");
      change.textContent = `${delta >= 0 ? "+" : ""}${number(delta, 2)}%`;
      cell.title = `${source} 公式 -> ${target} 样本: MAPE ${number(value.source_mape_pct, 4)}%; 误差率相对本机变化 ${number(delta, 4)}%`;
      cell.append(change);
      row.append(cell);
    }
    body.append(row);
  }
  table.append(body);
  scroll.append(table);
  const legend = document.createElement("p");
  legend.className = "matrix-legend";
  legend.textContent = "单元格为相对目标节点自身公式的 MAPE 误差率变化；对角线为 0，绿色不劣于自身，红色表示误差率增加超过 5%；悬浮可看原始 MAPE。";
  section.append(heading, scroll, legend);
  return section;
}

function transferablePairs(capture, threshold) {
  const pairs = [];
  capture.hosts.forEach((left, leftIndex) => {
    capture.hosts.slice(leftIndex + 1).forEach((right) => {
      const leftOnRight = capture.offline_fitted_cross_host[right][left]
        .source_minus_reference_mape_pct_points;
      const rightOnLeft = capture.offline_fitted_cross_host[left][right]
        .source_minus_reference_mape_pct_points;
      const worst = Math.max(leftOnRight, rightOnLeft);
      if (worst <= threshold) {
        pairs.push({left, right, leftOnRight, rightOnLeft, worst});
      }
    });
  });
  return pairs;
}

function renderPrefillFormulaAudit(audit) {
  const available = Boolean(audit?.cross_host?.captures?.length);
  elements.formulaEmpty.hidden = available;
  elements.formulaContent.hidden = !available;
  elements.formulaAblationEmpty.hidden = available;
  elements.formulaAblationContent.hidden = !available;
  if (!available) return;

  const crossCaptures = audit.cross_host.captures;
  const offlineCaptures = audit.offline_ablation?.captures || [];
  const onlineCaptures = audit.online_ablation?.captures || [];
  const lastPhase = audit.online_ablation?.long_horizon?.combined_phases?.slice(-1)[0] || {};
  elements.formulaMeta.textContent = `生成时间：${audit.generated_at}; 数据：${audit.scope}`;
  elements.formulaAblationMeta.textContent = `生成时间：${audit.generated_at}; 数据：${audit.scope}`;
  elements.formulaMetrics.replaceChildren(
    metricItem("交叉验证窗口", number(crossCaptures.length)),
    metricItem("跨节点迁移惩罚", crossCaptures.map((capture) => (
      `${number(capture.offline_fitted_summary.mean_transfer_penalty_pct_points, 2)}%`
    )).join(" / ")),
    metricItem("共同覆盖迁移惩罚", crossCaptures.map((capture) => (
      `${number(capture.common_support_summary.mean_transfer_penalty_pct_points, 2)}%`
    )).join(" / ")),
    metricItem("等权全局公式惩罚", crossCaptures.map((capture) => (
      `${number(capture.equal_weight_pooled_summary.mean_penalty_pct_points, 2)}%`
    )).join(" / ")),
  );
  elements.formulaAblationMetrics.replaceChildren(
    ...offlineCaptures.map((capture) => metricItem(
      `${capture.name} MAPE`,
      `${number(capture.host_equal_summary.baseline_9_mape_mean_pct, 2)}% -> ` +
        `${number(capture.host_equal_summary.candidate_10_mape_mean_pct, 2)}%`,
    )),
    metricItem("在线最终 MAPE", `${number(lastPhase.base_mape_pct, 2)}% -> ${number(lastPhase.new_mape_pct, 2)}%`),
  );

  const contextRows = [
    ["老公式", audit.formula_9],
    ["10 参数公式", audit.formula_10],
    ["矩阵口径", "每台节点前 70% 拟合、后 30% 验证；全部节点使用相同初始参数"],
    ["共同覆盖控制", "仅保留落在源节点训练样本近邻覆盖范围内的目标节点样本"],
  ];
  elements.formulaContext.replaceChildren(...contextRows.map(([label, value]) => {
    const item = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = label;
    const detail = document.createElement("span");
    detail.textContent = value;
    item.append(name, detail);
    return item;
  }));

  const conclusion = document.createElement("div");
  const conclusionHeading = document.createElement("h3");
  conclusionHeading.textContent = "结论";
  const conclusionList = document.createElement("ul");
  for (const line of (audit.conclusions || []).slice(0, 6)) {
    const item = document.createElement("li");
    item.textContent = line;
    conclusionList.append(item);
  }
  conclusion.append(conclusionHeading, conclusionList);
  elements.formulaConclusions.replaceChildren(conclusion);

  const ablationConclusion = document.createElement("div");
  const ablationHeading = document.createElement("h3");
  ablationHeading.textContent = "消融结论";
  const ablationList = document.createElement("ul");
  for (const line of (audit.conclusions || []).slice(6)) {
    const item = document.createElement("li");
    item.textContent = line;
    ablationList.append(item);
  }
  ablationConclusion.append(ablationHeading, ablationList);
  elements.formulaAblationConclusions.replaceChildren(ablationConclusion);

  replaceTable(
    elements.formulaCrossSummary,
    tableColumns(["窗口", "Agents", "本机公式 MAPE", "其他节点公式误差增加", "共同覆盖误差增加", "等权 pooled 误差增加"]),
    crossCaptures.map((capture) => {
      const diagonal = capture.hosts.map((host) => (
        capture.offline_fitted_cross_host[host][host].reference_mape_pct
      ));
      return [
        capture.name,
        number(capture.hosts.length),
        `${number(mean(diagonal), 3)}%`,
        `${number(capture.offline_fitted_summary.mean_transfer_penalty_pct_points, 3)}%`,
        `${number(capture.common_support_summary.mean_transfer_penalty_pct_points, 3)}%`,
        `${number(capture.equal_weight_pooled_summary.mean_penalty_pct_points, 3)}%`,
      ];
    }),
    "误差增加为相对目标节点自身公式的 MAPE 误差率绝对变化；正值越大表示跨节点复用越差。",
  );
  const transferRows = [];
  for (const capture of crossCaptures) {
    const strict = transferablePairs(capture, 1);
    const near = transferablePairs(capture, 2).filter((pair) => pair.worst > 1);
    transferRows.push([
      capture.name,
      strict.length
        ? strict.map((pair) => `${shortHost(pair.left)}<->${shortHost(pair.right)} (${number(pair.worst, 2)}%)`).join(", ")
        : "无",
      near.length
        ? near.map((pair) => `${shortHost(pair.left)}<->${shortHost(pair.right)} (${number(pair.worst, 2)}%)`).join(", ")
        : "无",
      capture.name === "20260904_141313"
        ? "有阶段性分族：29/37/44、38/43，以及 31/52；但跨窗口不稳定"
        : "31 单独偏慢；其余节点存在多条可迁移边，但不是所有节点两两可迁移",
    ]);
  }
  replaceTable(
    elements.formulaTransferGroups,
    tableColumns(["窗口", "可直接迁移（双向最坏 <= 1%）", "近似迁移（1-2%）", "分族判断"]),
    transferRows,
    "双向判据同时要求 A 公式用于 B 样本、B 公式用于 A 样本均不过阈值；它比单向最佳公式更严格。",
  );
  elements.formulaCrossMatrices.replaceChildren(
    ...crossCaptures.map(makeFormulaMatrix),
  );

  const crossWindowRows = [];
  for (const window of audit.cross_host.cross_window || []) {
    for (const host of window.hosts || []) {
      const result = window.cross_host[host][host];
      crossWindowRows.push([
        host,
        number(result.source_mape_pct, 3),
        number(result.reference_mape_pct, 3),
        number(result.source_minus_reference_mape_pct_points, 3),
      ]);
    }
  }
  replaceTable(
    elements.formulaCrossWindow,
    tableColumns(["目标 Agent", "同 Agent 上一窗口公式 MAPE%", "当前窗口自身公式 MAPE%", "时间漂移误差变化%"]),
    crossWindowRows,
    (audit.cross_host.cross_window || []).map((window) => (
      `同 Agent 旧公式平均误差增加 ${number(window.summary.same_host_prior_window_mean_penalty_pct_points, 3)}%；` +
      `其他 Agent 旧公式平均误差增加 ${number(window.summary.other_host_prior_window_mean_penalty_pct_points, 3)}%。`
    )).join(" "),
  );

  const offlineRows = [];
  for (const capture of offlineCaptures) {
    for (const [host, result] of Object.entries(capture.by_host || {})) {
      offlineRows.push([
        capture.name, host, number(result.test_samples),
        number(result.baseline_9.mape_pct, 3),
        number(result.candidate_10.mape_pct, 3),
        number(result.mape_delta_pct_points, 3),
        number(result.baseline_9.ape_p95_pct, 3),
        number(result.candidate_10.ape_p95_pct, 3),
        number(result.candidate_w_total_ms_per_1024_tokens, 4),
      ]);
    }
  }
  replaceTable(
    elements.formulaOfflineAblation,
    tableColumns(["窗口", "Agent", "测试样本", "9 参数 MAPE%", "10 参数 MAPE%", "MAPE 变化%", "9 参数 P95%", "10 参数 P95%", "w_total"]),
    offlineRows,
    "负的 MAPE 变化表示 10 参数改善；两段数据共 17 个节点次的 MAPE 全部改善。",
  );

  replaceTable(
    elements.formulaOnlineAblation,
    tableColumns(["窗口", "范围", "9 参数 MAPE%", "10 参数 MAPE%", "MAPE 变化%", "9 参数 P95%", "10 参数 P95%"]),
    onlineCaptures.map((capture) => {
      const result = capture.fresh_start.combined.last_50pct;
      return [
        shortHost(capture.capture), "后 50%", number(result.baseline.mape_pct, 3),
        number(result.candidate.mape_pct, 3), number(result.mape_delta_pct_points, 3),
        number(result.baseline.ape_p95_pct, 3), number(result.candidate.ape_p95_pct, 3),
      ];
    }),
    "从默认参数全新启动，按线上配置进行 Adam 更新；不是离线 curve_fit 回算。",
  );
  replaceTable(
    elements.formulaOnlinePhases,
    tableColumns(["串联十分位", "样本", "9 参数 MAPE%", "10 参数 MAPE%", "MAPE 变化%", "9 参数 P95%", "10 参数 P95%"]),
    (audit.online_ablation?.long_horizon?.combined_phases || []).map((phase) => [
      number(phase.phase), number(phase.samples), number(phase.base_mape_pct, 3),
      number(phase.new_mape_pct, 3), number(phase.mape_delta_pct_points, 3),
      number(phase.base_p95_pct, 3), number(phase.new_p95_pct, 3),
    ]),
    "两个真实窗口按节点串联后重新分成 10 段；用于观察新增参数随在线训练逐步收敛的效果。",
  );
}

function renderComparison(report) {
  renderPrefillFormulaAudit(report?.real_world_performance?.prefill_formula_audit);
  const runs = report?.runs || [];
  if (!runs.length) {
    elements.reportEmpty.hidden = false;
    elements.reportContent.hidden = true;
    return;
  }
  elements.reportEmpty.hidden = true;
  elements.reportContent.hidden = false;
  const experiment = report.experiment || {};
  elements.reportTitle.textContent = report.title === "NaviSched scheduler experiments"
    ? "NaviSched Scheduler 实验"
    : (report.title || "Scheduler 对比");
  elements.reportMeta.textContent =
    `生成时间：${report.generated_at || "-"}；数据来源：simulate 汇总`;
  const agentCounts = experiment.agent_counts || [experiment.agent_count];
  const agentLabel = agentCounts.filter((value) => value != null).join(", ");
  const requestCounts = [...new Set(runs.map((run) => run.request_count))];
  const metrics = [
    ["请求数", requestCounts.map((value) => number(value)).join(", ")],
    ["Agent 数", agentLabel || "-"],
    ["Client 数", number(experiment.client_count)],
    ["时间压缩比", `${number(experiment.time_compression, 1)}x`],
    ["运行数", number(runs.length)],
  ];
  elements.reportMetrics.replaceChildren(
    ...metrics.map(([label, value]) => metricItem(label, value)),
  );

  renderRealWorldPerformance(report.real_world_performance);
  const groupCount = renderComparisonGroups(runs);
  renderWorkloadProfiles(runs);
  renderPoolComparison(runs);
  renderRegression(report.regression);
  elements.reportInsightsHeading.hidden = groupCount > 1;
  elements.reportInsights.hidden = groupCount > 1;
  if (groupCount === 1) {
    const bestLatency = minimumRun(
      runs, (run) => run.end_to_end_latency.average_ms,
    );
    const bestQueue = minimumRun(runs, (run) => run.queue_wait.average_ms);
    const bestCache = maximumRun(runs, (run) => run.cache_hit_rate);
    const bestDrain = minimumRun(runs, (run) => run.logical_drain_time_ms);
    elements.reportInsights.replaceChildren(
      insightItem(
        "最低平均 E2E",
        bestLatency,
        `${number(bestLatency.end_to_end_latency.average_ms, 2)} ms`,
      ),
      insightItem(
        "最低平均 Queue",
        bestQueue,
        `${number(bestQueue.queue_wait.average_ms, 2)} ms`,
      ),
      insightItem(
        "最高 Cache 命中率",
        bestCache,
        `${number(bestCache.cache_hit_rate * 100, 2)}%`,
      ),
      insightItem(
        "最短逻辑排空时间",
        bestDrain,
        `${number(bestDrain.logical_drain_time_ms, 2)} ms`,
      ),
    );
  }
  renderLossComparison(runs);
  renderNearOneComparison(runs);
  renderOptimizationComparison(report.optimization);
  renderScalability(report.scalability);
}

function renderLossComparison(runs) {
  const lossRuns = runs.filter((run) => run.loss);
  elements.reportLossSection.hidden = !lossRuns.length;
  if (!lossRuns.length) {
    return;
  }
  const rows = lossRuns.map((run) => [
    run.group || "Scheduler comparison",
    run.lambda == null ? "-" : lambdaNumber(run.lambda),
    run.scheduler,
    number(run.loss.count),
    number(run.loss.average, 9),
    number(run.loss.stddev, 9),
    number(run.loss.min, 9),
    number(run.loss.p50, 9),
    number(run.loss.p90, 9),
    number(run.loss.p99, 9),
    number(run.loss.max, 9),
  ]);
  replaceTable(
    elements.reportLoss,
    tableColumns([
      "experiment",
      "lambda",
      "scheduler",
      "schedules",
      "mean",
      "stddev",
      "min",
      "p50",
      "p90",
      "p99",
      "max",
    ]),
    rows,
    "每行汇总一个实验运行中所有去重调度轮次的最终可行 Loss。",
  );
}

function renderNearOneComparison(runs) {
  const nearOneRuns = runs.filter((run) => (
    run.lambda != null && run.lambda >= 0.99
  ));
  elements.reportNearOneSection.hidden = !nearOneRuns.length;
  if (!nearOneRuns.length) {
    return;
  }
  const rows = nearOneRuns.map((run) => [
    run.group || "Scheduler comparison",
    lambdaNumber(run.lambda),
    number(run.end_to_end_latency.average_ms, 2),
    number(run.end_to_end_latency.p99_ms, 2),
    number(run.queue_wait.average_ms, 2),
    number(run.queue_wait.p99_ms, 2),
    number(run.completion_throughput_qps, 2),
    `${number(run.cache_hit_rate * 100, 2)}%`,
    `${number(run.average_utilization * 100, 2)}%`,
    `${number(run.min_utilization * 100, 2)}%-${number(
      run.max_utilization * 100, 2,
    )}%`,
    `${number(run.utilization_stddev * 100, 2)} pp`,
    number(run.logical_drain_time_ms, 2),
    run.loss ? number(run.loss.average, 9) : "-",
    run.loss ? number(run.loss.stddev, 9) : "-",
    run.loss ? number(run.loss.p50, 9) : "-",
    run.loss ? number(run.loss.p90, 9) : "-",
    run.loss ? number(run.loss.p99, 9) : "-",
    run.loss ? number(run.loss.max, 9) : "-",
  ]);
  replaceTable(
    elements.reportNearOne,
    tableColumns([
      "experiment",
      "lambda",
      "e2e avg ms",
      "e2e p99 ms",
      "queue avg ms",
      "queue p99 ms",
      "qps",
      "cache hit",
      "utilization avg",
      "utilization range",
      "utilization stddev",
      "drain ms",
      "loss avg",
      "loss stddev",
      "loss p50",
      "loss p90",
      "loss p99",
      "loss max",
    ]),
    rows,
    "每行是一组接近离线优先边界的 lambda 配置及其完整运行结果。",
  );
}

function renderOptimizationComparison(optimization) {
  const checkpoints = optimization?.checkpoints || [];
  elements.reportOptimizationSection.hidden = !checkpoints.length;
  if (!checkpoints.length) {
    return;
  }
  const initial = checkpoints[0];
  const rows = checkpoints.map((checkpoint) => [
    number(checkpoint.loop_count),
    number(checkpoint.projected_loss, 9),
    number(checkpoint.elapsed_us),
    `${number(
      100 * (initial.projected_loss - checkpoint.projected_loss) /
        initial.projected_loss,
      2,
    )}%`,
    number(checkpoint.phases_us.forward_us),
    number(checkpoint.phases_us.backward_us),
    number(checkpoint.phases_us.projection_us),
  ]);
  replaceTable(
    elements.reportOptimization,
    tableColumns([
      "loops",
      "projected loss",
      "trace elapsed us",
      "loss reduction vs first",
      "forward us",
      "backward us",
      "projection us",
    ]),
    rows,
    "每行是同一 Replay 优化到指定循环数时的 Loss 和累计阶段耗时。",
  );
}

function renderScalability(scalability) {
  const rows = scalability?.rows || [];
  const hotspots = scalability?.hotspots || [];
  elements.reportScalabilitySection.hidden = !rows.length;
  elements.reportProfileSection.hidden = !hotspots.length;
  if (rows.length) {
    replaceTable(
      elements.reportScalability,
      tableColumns([
        "requests",
        "agents",
        "loops",
        "parallelism",
        "min ms",
        "median ms",
        "max ms",
        "speedup vs serial",
        {
          label: `within ${number(scalability.latency_budget_ms)} ms`,
          description: "中位调度耗时是否满足本次规模实验的延迟预算。",
        },
      ]),
      rows.map((row) => [
        number(row.request_count),
        number(row.agent_count),
        number(row.loop_count),
        number(row.parallelism || 1),
        number(row.min_ms, 3),
        number(row.median_ms, 3),
        number(row.max_ms, 3),
        row.speedup_vs_serial == null
          ? "-"
          : `${number(row.speedup_vs_serial, 2)}x`,
        row.median_ms <= scalability.latency_budget_ms ? "是" : "否",
      ]),
      "每行是一种请求规模、节点规模、循环数和并行度组合的重复压测结果。",
    );
  }
  if (hotspots.length) {
    replaceTable(
      elements.reportProfile,
      tableColumns(["area", "measured evidence", "implementation"]),
      hotspots.map((hotspot) => [
        hotspot.area,
        hotspot.evidence,
        hotspot.implementation || hotspot.parallel_path,
      ]),
      "每行对应一个运行阶段，并给出实测证据和当前实现方式。",
    );
  }
}

async function loadComparison() {
  try {
    const response = await fetch("/api/comparison", {cache: "no-store"});
    if (!response.ok) {
      throw new Error(`对比报告不可用：HTTP ${response.status}`);
    }
    renderComparison(await response.json());
  } catch (error) {
    elements.reportEmpty.hidden = false;
    elements.reportContent.hidden = true;
    elements.reportEmpty.querySelector("span").textContent = error.message;
  }
}

function activeRun() {
  return state.runs.find((run) => run.id === state.activeRunId) || null;
}

function number(value, digits = 0) {
  const parsed = Number(value || 0);
  return parsed.toLocaleString("en-US", {
    notation: "standard",
    useGrouping: false,
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function milliseconds(value, digits = 0) {
  return `${number(value, digits)} ms`;
}

function textCell(value) {
  const cell = document.createElement("td");
  cell.textContent = value;
  return cell;
}

function sortableValue(value) {
  const valueText = String(value ?? "").trim();
  const numericText = valueText.replaceAll(",", "").replace(
    /(?:%| pp| ms|x)$/,
    "",
  );
  if (/^-?(?:\d+(?:\.\d*)?|\.\d+)$/.test(numericText)) {
    return {numeric: true, value: Number(numericText)};
  }
  return {numeric: false, value: valueText.toLocaleLowerCase()};
}

function compareTableValues(left, right) {
  const leftValue = sortableValue(left);
  const rightValue = sortableValue(right);
  if (leftValue.numeric && rightValue.numeric) {
    return leftValue.value - rightValue.value;
  }
  return String(leftValue.value).localeCompare(String(rightValue.value),
    undefined, {numeric: true});
}

function headerLabel(header) {
  return typeof header === "string" ? header : header.label;
}

function headerDescription(header) {
  const label = headerLabel(header);
  if (typeof header === "string") {
    return COLUMN_HELP[label] || `“${label}”列的数据口径。`;
  }
  return header.description || COLUMN_HELP[label] ||
    `“${label}”列的数据口径。`;
}

function appendTableCaption(table, description) {
  if (!description) {
    return;
  }
  const caption = document.createElement("caption");
  caption.textContent = description;
  table.append(caption);
}

function makeTable(headers, rows, description) {
  const table = document.createElement("table");
  appendTableCaption(table, description);
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  const body = document.createElement("tbody");
  let sortColumn = -1;
  let sortDirection = 1;

  const renderRows = () => {
    const orderedRows = rows.map((row, index) => ({row, index}));
    if (sortColumn >= 0) {
      orderedRows.sort((left, right) => {
        const result = compareTableValues(
          left.row[sortColumn], right.row[sortColumn],
        );
        return result === 0
          ? left.index - right.index
          : result * sortDirection;
      });
    }
    body.replaceChildren(...orderedRows.map(({row, index}) => {
      const tableRow = document.createElement("tr");
      tableRow.dataset.rowIndex = String(index);
      row.forEach((value) => tableRow.append(textCell(value)));
      return tableRow;
    }));
  };

  headers.forEach((header, column) => {
    const cell = document.createElement("th");
    const label = headerLabel(header);
    cell.textContent = label;
    cell.className = "sortable-header";
    cell.tabIndex = 0;
    cell.title = headerDescription(header);
    const sort = () => {
      if (sortColumn !== column) {
        sortColumn = column;
        sortDirection = 1;
      } else if (sortDirection > 0) {
        sortDirection = -1;
      } else {
        sortColumn = -1;
        sortDirection = 1;
      }
      headRow.querySelectorAll("th").forEach((headerCell, index) => {
        headerCell.classList.toggle(
          "sort-ascending",
          sortColumn === column && index === column && sortDirection > 0,
        );
        headerCell.classList.toggle(
          "sort-descending",
          sortColumn === column && index === column && sortDirection < 0,
        );
        headerCell.setAttribute("aria-sort", index !== sortColumn
          ? "none"
          : sortDirection > 0 ? "ascending" : "descending");
      });
      renderRows();
      table.dispatchEvent(new CustomEvent("table-sorted", {bubbles: true}));
    };
    cell.addEventListener("click", sort);
    cell.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        sort();
      }
    });
    headRow.append(cell);
  });
  head.append(headRow);
  table.append(head);
  renderRows();
  table.append(body);
  return table;
}

function replaceTable(container, headers, rows, description) {
  container.replaceChildren(makeTable(headers, rows, description));
}

function metricItem(label, value) {
  const item = document.createElement("div");
  item.className = "metric";
  const labelNode = document.createElement("span");
  labelNode.className = "metric-label";
  labelNode.textContent = label;
  const valueNode = document.createElement("strong");
  valueNode.className = "metric-value";
  valueNode.textContent = value;
  valueNode.title = value;
  item.append(labelNode, valueNode);
  return item;
}

function selectOverviewRequest(requestIndex, nodeIndex) {
  state.overviewRequestIndex = requestIndex;
  state.overviewNodeIndex = Number.isInteger(nodeIndex) ? nodeIndex : null;
  elements.requestTable.querySelectorAll("tbody tr").forEach((row) => {
    const selected = Number(row.dataset.rowIndex) ===
      state.overviewRequestIndex;
    row.classList.toggle("selected-row", selected);
    row.setAttribute("aria-selected", String(selected));
  });
  const nodeRows = elements.nodeTable.querySelectorAll("tbody tr");
  nodeRows.forEach((row) => {
    row.classList.toggle(
      "linked-row",
      Number(row.dataset.rowIndex) === state.overviewNodeIndex,
    );
  });
  const linkedRow = state.overviewNodeIndex == null
    ? null
    : elements.nodeTable.querySelector(
      `tbody tr[data-row-index="${state.overviewNodeIndex}"]`,
    );
  if (linkedRow) {
    elements.nodeTable.scrollTop = Math.max(
      0,
      linkedRow.offsetTop -
        (elements.nodeTable.clientHeight - linkedRow.offsetHeight) / 2,
    );
  }
  elements.cacheTable.querySelectorAll(
    ".linked-cell, .linked-cache-column",
  ).forEach((cell) => {
    cell.classList.remove("linked-cell", "linked-cache-column");
  });
  elements.cacheTable.querySelectorAll("tr.linked-cache-row").forEach((row) => {
    row.classList.remove("linked-cache-row");
  });
  const cacheRow = state.overviewRequestIndex == null
    ? null
    : elements.cacheTable.querySelector(
      `tbody tr[data-row-index="${state.overviewRequestIndex}"]`,
    );
  const linkedCell = state.overviewNodeIndex == null
    ? null
    : cacheRow?.cells[state.overviewNodeIndex + 1];
  if (linkedCell) {
    const columnIndex = state.overviewNodeIndex + 1;
    cacheRow.classList.add("linked-cache-row");
    elements.cacheTable.querySelectorAll("tr").forEach((row) => {
      row.cells[columnIndex]?.classList.add("linked-cache-column");
    });
    linkedCell.classList.add("linked-cell");
    elements.cacheTable.scrollLeft = Math.max(
      0,
      linkedCell.offsetLeft -
        (elements.cacheTable.clientWidth - linkedCell.offsetWidth) / 2,
    );
    elements.cacheTable.scrollTop = Math.max(
      0,
      cacheRow.offsetTop -
        (elements.cacheTable.clientHeight - cacheRow.offsetHeight) / 2,
    );
  }
}

function decorateOverviewRequestRows() {
  elements.requestTable.querySelectorAll("tbody tr").forEach((row) => {
    row.tabIndex = 0;
    row.setAttribute("role", "button");
    row.setAttribute(
      "aria-selected",
      String(Number(row.dataset.rowIndex) === state.overviewRequestIndex),
    );
  });
}

function renderRequests(replay) {
  const decisions = replay.decisions || [];
  const nodes = replay.nodes || [];
  const matches = replay.node_matches || [];
  const schedInfos = replay.scheduler_info?.sched_infos || [];
  const selectedNodeIndices = [];
  const nodeByAddress = new Map();
  nodes.forEach((node, index) => {
    for (const address of [
      node.agent_id,
      node.collector_address,
      node.server_address,
    ]) {
      if (address) {
        nodeByAddress.set(address, {index, node});
      }
    }
  });
  const rows = (replay.requests || []).map((item, index) => {
    const request = item.request || {};
    const decision = decisions[index] || {};
    const target = decision.target || {};
    const targetAddress = target.agent_address || target.address || "";
    const selected = nodeByAddress.get(targetAddress);
    const selectedIndex = selected?.index;
    selectedNodeIndices[index] = selectedIndex;
    const matchedBlockCount = selectedIndex == null
      ? 0
      : Number(matches[index]?.[selectedIndex]?.matched_block_count || 0);
    const blockSize = Number(request.block_size || 0);
    const matchedTokenCount = matchedBlockCount * blockSize;
    const promptTokenCount = Number(request.prompt_token_count || 0);
    const selectedSchedInfo = selectedIndex == null
      ? {}
      : schedInfos[selectedIndex]?.[index] || {};
    return [
      `request[${index}]`,
      selected ? `node[${selected.index}]` : "-",
      selected?.node.agent_id || "-",
      number(matchedBlockCount),
      number(matchedTokenCount),
      number(Math.max(0, promptTokenCount - matchedTokenCount)),
      number(selectedSchedInfo.queue_wait_ms, 3),
      number(selectedSchedInfo.cost, 6),
      number(selectedSchedInfo.weight, 8),
      decision.reason || "-",
      String(decision.batch_id ?? "-"),
      number(decision.expected_batch_size),
      String(request.request_id ?? "-"),
      targetAddress || "-",
      number(promptTokenCount),
      number(blockSize),
      number((request.block_hashes || []).length),
      number(item.wait_time_ms),
      number(item.remaining_timeout_ms),
      String(Boolean(item.timeout_triggered)),
    ];
  });
  replaceTable(
    elements.requestTable,
    tableColumns([
      "request",
      "selected node",
      "agent id",
      "cache hit blocks",
      "cache hit tokens",
      "cache miss tokens",
      "queue ms",
      "cost",
      "weight",
      "decision",
      "batch id",
      "batch size",
      "request id",
      "target address",
      "prompt tokens",
      "block size",
      "block hashes",
      "wait ms",
      "timeout left ms",
      "timed out",
    ]),
    rows,
    "每行是本轮调度中的一个请求，并展示最终节点、命中、代价和决策结果。",
  );
  state.overviewNodeIndices = selectedNodeIndices;
  decorateOverviewRequestRows();
}

function renderNodes(replay) {
  const rows = (replay.nodes || []).map((node, index) => [
    `node[${index}]`,
    node.agent_id || "-",
    node.server_address || "-",
    number(node.total_blocks),
    number(node.inflight_requests),
    number(node.batch_waiting_requests),
    number(node.queued_batch_requests),
    number(node.running_requests),
    number(node.queue_total_predicted_latency_ms, 3),
    String(node.latency_model_revision ?? "-"),
  ]);
  replaceTable(
    elements.nodeTable,
    tableColumns([
      "node",
      "agent id",
      "server",
      "blocks",
      "inflight req",
      "batching req",
      "queued req",
      "running req",
      "queue wait ms",
      "model revision",
    ]),
    rows,
    "每行是本轮可参与调度的一个节点及其调度时刻状态。",
  );
}

function renderCacheMatrix(replay) {
  const nodes = replay.nodes || [];
  const requests = replay.requests || [];
  const matches = replay.node_matches || [];
  const headers = [{
    label: "request / node",
    description: "行表示请求，后续列表示候选节点。",
  }];
  nodes.forEach((node, index) => {
    headers.push({
      label: `node[${index}] ${node.agent_id || ""}`,
      description: `该请求在 node[${index}] 上预计命中的完整 block 数。`,
    });
  });
  const rows = requests.map((item, requestIndex) => {
    const row = [
      `request[${requestIndex}] id=${item.request?.request_id ?? "-"}`,
    ];
    nodes.forEach((unused, nodeIndex) => {
      row.push(number(matches[requestIndex]?.[nodeIndex]?.matched_block_count));
    });
    return row;
  });
  replaceTable(
    elements.cacheTable,
    headers,
    rows,
    "每行是一个请求，每个节点列给出该请求预计命中的完整 Cache block 数。",
  );
}

function schedulerValue(info, metric) {
  if (metric === "queue") {
    return Number(info.queue_wait_ms || 0);
  }
  if (metric === "weight") {
    return Number(info.weight || 0);
  }
  return Number(info.cost || 0);
}

function schedulerRange(schedInfos) {
  const values = [];
  schedInfos.forEach((nodeInfos) => {
    nodeInfos.forEach((info) => values.push(schedulerValue(info, state.metric)));
  });
  return {
    min: values.length ? Math.min(...values) : 0,
    max: values.length ? Math.max(...values) : 0,
  };
}

function heatColor(value, range) {
  const ratio = range.max === range.min
    ? 0.25
    : (value - range.min) / (range.max - range.min);
  const alpha = 0.06 + ratio * 0.24;
  if (state.metric === "queue") {
    return `rgba(196, 123, 22, ${alpha})`;
  }
  if (state.metric === "weight") {
    return `rgba(8, 127, 117, ${alpha})`;
  }
  return `rgba(201, 87, 61, ${alpha})`;
}

function schedulerCell(info, requestIndex, nodeIndex, range) {
  const cell = document.createElement("td");
  cell.className = "sched-cell";
  cell.dataset.requestIndex = String(requestIndex);
  cell.dataset.nodeIndex = String(nodeIndex);
  if (
    state.selectedCell?.requestIndex === requestIndex &&
    state.selectedCell?.nodeIndex === nodeIndex
  ) {
    cell.classList.add("selected");
  }
  cell.style.backgroundColor = heatColor(
    schedulerValue(info, state.metric),
    range,
  );
  const values = [
    ["C", number(info.cost, 3)],
    ["Q", number(info.queue_wait_ms, 3)],
    ["W", number(info.weight, 6)],
  ];
  values.forEach(([label, value]) => {
    const line = document.createElement("span");
    line.className = "cell-line";
    const key = document.createElement("b");
    key.textContent = label;
    const output = document.createElement("span");
    output.textContent = value;
    line.append(key, output);
    cell.append(line);
  });
  cell.addEventListener("click", () => {
    state.selectedCell = {requestIndex, nodeIndex};
    renderSchedulerMatrix(activeRun()?.data);
  });
  return cell;
}

function appendHeader(table, labels) {
  const head = document.createElement("thead");
  const row = document.createElement("tr");
  labels.forEach((header) => {
    const cell = document.createElement("th");
    cell.textContent = headerLabel(header);
    cell.title = headerDescription(header);
    row.append(cell);
  });
  head.append(row);
  table.append(head);
}

function renderSelection(replay, schedInfos) {
  const selected = state.selectedCell;
  if (!selected) {
    elements.selectionDetail.replaceChildren();
    return;
  }
  const request = replay.requests?.[selected.requestIndex]?.request || {};
  const node = replay.nodes?.[selected.nodeIndex] || {};
  const info = schedInfos?.[selected.nodeIndex]?.[selected.requestIndex] || {};
  const hit = replay.node_matches?.[selected.requestIndex]?.[selected.nodeIndex];
  const fields = [
    ["Request", `request[${selected.requestIndex}] / ${request.request_id ?? "-"}`],
    ["Node", `node[${selected.nodeIndex}] / ${node.agent_id || "-"}`],
    ["Cache 命中 Block", number(hit?.matched_block_count)],
    ["Cost", number(info.cost, 6)],
    ["Queue 等待 ms", number(info.queue_wait_ms, 6)],
    ["Weight", number(info.weight, 8)],
  ];
  const items = fields.map(([label, value]) => {
    const item = document.createElement("div");
    item.className = "detail-item";
    const name = document.createElement("span");
    name.textContent = label;
    const output = document.createElement("strong");
    output.textContent = value;
    output.title = value;
    item.append(name, output);
    return item;
  });
  elements.selectionDetail.replaceChildren(...items);
}

function renderSchedulerMatrix(replay) {
  if (!replay) {
    return;
  }
  const schedulerInfo = replay.scheduler_info || {};
  const schedInfos = schedulerInfo.sched_infos;
  if (!Array.isArray(schedInfos)) {
    const error = document.createElement("div");
    error.className = "error-banner";
    error.textContent = "该 Replay 不包含 CostScheduler sched_infos。";
    elements.schedulerMatrix.replaceChildren(error);
    elements.selectionDetail.replaceChildren();
    return;
  }
  const requests = replay.requests || [];
  const nodes = replay.nodes || [];
  const range = schedulerRange(schedInfos);
  const table = document.createElement("table");
  appendTableCaption(
    table,
    "每个单元格依次显示 Cost、Queue 和 Weight；点击单元格可查看请求与节点详情。",
  );
  const body = document.createElement("tbody");

  if (state.orientation === "request") {
    appendHeader(table, [
      {
        label: "request / node",
        description: "每行是请求，每个后续列是候选节点。",
      },
      ...nodes.map((node, index) => ({
        label: `node[${index}] ${node.agent_id || ""}`,
        description: `各请求在 node[${index}] 上的代价、排队和权重。`,
      })),
    ]);
    requests.forEach((item, requestIndex) => {
      const row = document.createElement("tr");
      row.append(textCell(
        `request[${requestIndex}] id=${item.request?.request_id ?? "-"}`,
      ));
      nodes.forEach((unused, nodeIndex) => {
        const info = schedInfos[nodeIndex]?.[requestIndex] || {};
        row.append(schedulerCell(info, requestIndex, nodeIndex, range));
      });
      body.append(row);
    });
  } else {
    appendHeader(table, [
      {
        label: "node / request",
        description: "每行是节点，每个后续列是参与调度的请求。",
      },
      ...requests.map(
        (item, index) => ({
          label: `request[${index}] ${item.request?.request_id ?? ""}`,
          description: `request[${index}] 在各节点上的代价、排队和权重。`,
        }),
      ),
    ]);
    nodes.forEach((node, nodeIndex) => {
      const row = document.createElement("tr");
      row.append(textCell(`node[${nodeIndex}] ${node.agent_id || ""}`));
      requests.forEach((unused, requestIndex) => {
        const info = schedInfos[nodeIndex]?.[requestIndex] || {};
        row.append(schedulerCell(info, requestIndex, nodeIndex, range));
      });
      body.append(row);
    });
  }
  table.append(body);
  elements.schedulerMatrix.replaceChildren(table);
  renderSelection(replay, schedInfos);
}

function optimizationValue(values, requestIndex, nodeIndex, requestCount) {
  return Number(values?.[requestIndex + nodeIndex * requestCount] || 0);
}

function renderOptimizationChart(run) {
  const canvas = elements.optimizationChart;
  const context = canvas.getContext("2d");
  const steps = run.steps || [];
  const before = steps.map((step) => Number(step.forward_value || 0));
  const projected = steps.map((step, index) => Number(
    steps[index + 1]?.forward_value ?? run.final_forward_value ?? 0,
  ));
  const values = [...before, ...projected];
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (!values.length) {
    return;
  }
  const padding = {left: 72, right: 24, top: 20, bottom: 38};
  const width = canvas.width - padding.left - padding.right;
  const height = canvas.height - padding.top - padding.bottom;
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const range = maximum === minimum ? 1 : maximum - minimum;
  const x = (index) => padding.left +
    (steps.length <= 1 ? 0 : index * width / (steps.length - 1));
  const y = (value) => padding.top + (maximum - value) * height / range;

  context.strokeStyle = "#d8dfd9";
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(padding.left, padding.top);
  context.lineTo(padding.left, padding.top + height);
  context.lineTo(padding.left + width, padding.top + height);
  context.stroke();

  const drawLine = (series, color) => {
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.beginPath();
    series.forEach((value, index) => {
      if (index === 0) {
        context.moveTo(x(index), y(value));
      } else {
        context.lineTo(x(index), y(value));
      }
    });
    context.stroke();
  };
  drawLine(before, "#c47b16");
  drawLine(projected, "#087f75");

  context.fillStyle = "#60706b";
  context.font = "12px IBM Plex Mono, monospace";
  context.textAlign = "right";
  context.fillText(number(maximum, 6), padding.left - 8, padding.top + 4);
  context.fillText(
    number(minimum, 6), padding.left - 8, padding.top + height + 4,
  );
  context.textAlign = "left";
  context.fillStyle = "#c47b16";
  context.fillText("更新前", padding.left + 10, 14);
  context.fillStyle = "#087f75";
  context.fillText("投影后", padding.left + 130, 14);
}

function renderOptimizationHistory(run) {
  const steps = run.steps || [];
  const rows = steps.map((step, index) => [
    number(step.iteration),
    number(step.forward_value, 9),
    number(steps[index + 1]?.forward_value ?? run.final_forward_value, 9),
    number(step.elapsed_us),
  ]);
  replaceTable(
    elements.optimizationHistory,
    tableColumns([
      "step",
      "feasible before update",
      "feasible after projection",
      "elapsed us",
    ]),
    rows,
    "每行是一次投影梯度更新；投影后结果是下一行更新前的可行状态。",
  );
  elements.optimizationHistory.querySelectorAll("tbody tr").forEach(
    (row, index) => {
      row.classList.toggle("selected-row", index === state.optimizationStepIndex);
      row.addEventListener("click", () => {
        state.optimizationStepIndex = index;
        renderOptimization(activeRun()?.data);
      });
    },
  );
}

function renderOptimizationTiming(run) {
  const steps = run.steps || [];
  const phases = [
    ["forward", "forward_us"],
    ["backward", "backward_us"],
    ["gradient update", "update_us"],
    ["simplex projection", "projection_us"],
    ["parameter bind", "bind_us"],
  ];
  const totals = phases.map(([label, field]) => [
    label,
    steps.reduce((sum, step) => sum + Number(step[field] || 0), 0),
  ]);
  const measuredTotal = totals.reduce((sum, item) => sum + item[1], 0);
  const rows = totals.map(([label, total]) => [
    label,
    number(total),
    number(steps.length ? total / steps.length : 0, 3),
    `${number(measuredTotal ? total * 100 / measuredTotal : 0, 2)}%`,
  ]);
  replaceTable(
    elements.optimizationTiming,
    tableColumns([
      "phase",
      "total us",
      "average us / step",
      "measured share",
    ]),
    rows,
    "每行汇总一个优化阶段在当前样本全部循环中的耗时。",
  );
}

function optimizationCell(step, requestIndex, nodeIndex, requestCount) {
  const cell = document.createElement("td");
  cell.className = "optimization-cell";
  const values = [
    ["B", optimizationValue(
      step.weights_before_update, requestIndex, nodeIndex, requestCount,
    )],
    ["G", optimizationValue(
      step.gradients, requestIndex, nodeIndex, requestCount,
    )],
    ["U", optimizationValue(
      step.weights_before_projection, requestIndex, nodeIndex, requestCount,
    )],
    ["P", optimizationValue(
      step.weights_after_projection, requestIndex, nodeIndex, requestCount,
    )],
  ];
  values.forEach(([label, value]) => {
    const line = document.createElement("span");
    line.className = "cell-line";
    const key = document.createElement("b");
    key.textContent = label;
    const output = document.createElement("span");
    output.textContent = number(value, 8);
    line.append(key, output);
    cell.append(line);
  });
  return cell;
}

function renderOptimizationMatrix(replay, step) {
  const requests = replay.requests || [];
  const nodes = replay.nodes || [];
  const table = document.createElement("table");
  appendTableCaption(
    table,
    "每个单元格显示更新前权重、梯度、投影前权重和投影后权重。",
  );
  const body = document.createElement("tbody");
  if (state.orientation === "request") {
    appendHeader(table, [
      {
        label: "request / node",
        description: "每行是请求，每个后续列是候选节点。",
      },
      ...nodes.map((node, index) => ({
        label: `node[${index}] ${node.agent_id || ""}`,
        description: `当前优化步骤中 node[${index}] 对各请求的权重状态。`,
      })),
    ]);
    requests.forEach((item, requestIndex) => {
      const row = document.createElement("tr");
      row.append(textCell(
        `request[${requestIndex}] id=${item.request?.request_id ?? "-"}`,
      ));
      nodes.forEach((unused, nodeIndex) => {
        row.append(optimizationCell(
          step, requestIndex, nodeIndex, requests.length,
        ));
      });
      body.append(row);
    });
  } else {
    appendHeader(table, [
      {
        label: "node / request",
        description: "每行是节点，每个后续列是参与调度的请求。",
      },
      ...requests.map(
        (item, index) => ({
          label: `request[${index}] ${item.request?.request_id ?? ""}`,
          description: `当前优化步骤中 request[${index}] 在各节点上的权重状态。`,
        }),
      ),
    ]);
    nodes.forEach((node, nodeIndex) => {
      const row = document.createElement("tr");
      row.append(textCell(`node[${nodeIndex}] ${node.agent_id || ""}`));
      requests.forEach((unused, requestIndex) => {
        row.append(optimizationCell(
          step, requestIndex, nodeIndex, requests.length,
        ));
      });
      body.append(row);
    });
  }
  table.append(body);
  elements.optimizationMatrix.replaceChildren(table);
}

function renderOptimization(replay) {
  const schedulerInfo = replay?.scheduler_info || {};
  const runs = schedulerInfo.optimization_runs;
  if (!Array.isArray(runs) || !runs.length) {
    elements.optimizationEmpty.hidden = false;
    elements.optimizationEmpty.textContent =
      "该 Replay 不包含 CostScheduler 优化过程。";
    elements.optimizationContent.hidden = true;
    return;
  }
  elements.optimizationEmpty.hidden = true;
  elements.optimizationContent.hidden = false;
  if (state.optimizationRunIndex >= runs.length) {
    state.optimizationRunIndex = 0;
  }
  elements.optimizationRun.replaceChildren(
    ...runs.map((unused, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = `起点[${index}]`;
      return option;
    }),
  );
  elements.optimizationRun.value = String(state.optimizationRunIndex);
  const run = runs[state.optimizationRunIndex];
  const steps = run.steps || [];
  if (state.optimizationStepIndex >= steps.length) {
    state.optimizationStepIndex = Math.max(0, steps.length - 1);
  }
  elements.optimizationStep.max = String(Math.max(0, steps.length - 1));
  elements.optimizationStep.value = String(state.optimizationStepIndex);
  elements.optimizationStepOutput.value = String(state.optimizationStepIndex);
  elements.optimizationStepLabel.textContent =
    `Step ${state.optimizationStepIndex}`;
  const step = steps[state.optimizationStepIndex] || {};
  const projectedLoss = Number(
    steps[state.optimizationStepIndex + 1]?.forward_value ??
      run.final_forward_value ?? 0,
  );
  const metrics = [
    ["起点", number(state.optimizationRunIndex)],
    ["Step 数", number(steps.length)],
    ["停止原因", run.stop_reason || "-"],
    ["预算 us", number(schedulerInfo.optimization_time_budget_us)],
    ["初始 Loss", number(steps[0]?.forward_value, 9)],
    ["最终 Loss", number(run.final_forward_value, 9)],
    ["最佳 Loss", number(run.best_forward_value, 9)],
    ["耗时 us", number(run.elapsed_us)],
  ];
  elements.optimizationMetrics.replaceChildren(
    ...metrics.map(([label, value]) => metricItem(label, value)),
  );
  const stepMetrics = [
    ["Alpha", number(step.alpha, 9)],
    ["更新前", number(step.forward_value, 9)],
    ["投影后", number(projectedLoss, 9)],
    ["耗时 us", number(step.elapsed_us)],
    ["Forward us", number(step.forward_us)],
    ["Backward us", number(step.backward_us)],
    ["Update us", number(step.update_us)],
    ["Projection us", number(step.projection_us)],
    ["Bind us", number(step.bind_us)],
  ];
  elements.optimizationStepMetrics.replaceChildren(
    ...stepMetrics.map(([label, value]) => metricItem(label, value)),
  );
  renderOptimizationChart(run);
  renderOptimizationTiming(run);
  renderOptimizationHistory(run);
  renderOptimizationMatrix(replay, step);
}

function numericRange(values, digits = 0) {
  const finiteValues = values.map(Number).filter(Number.isFinite);
  if (!finiteValues.length) {
    return "-";
  }
  return `${number(Math.min(...finiteValues), digits)}-${number(
    Math.max(...finiteValues), digits,
  )}`;
}

function replayBatchSummary(replay) {
  const batches = new Map();
  for (const decision of replay?.decisions || []) {
    const batchId = String(decision.batch_id ?? "-");
    const batch = batches.get(batchId) || {
      actualSize: 0,
      expectedSize: Number(decision.expected_batch_size || 0),
    };
    batch.actualSize += 1;
    batches.set(batchId, batch);
  }
  const values = [...batches.values()];
  const sizes = values.map((batch) => batch.actualSize);
  return {
    count: sizes.length,
    meanSize: sizes.length
      ? sizes.reduce((sum, value) => sum + value, 0) / sizes.length
      : 0,
    sizes,
    expectedSizes: values.map((batch) => batch.expectedSize),
  };
}

function selectedCacheSummary(replay) {
  const nodeIndexes = new Map();
  (replay?.nodes || []).forEach((node, index) => {
    [node.agent_id, node.collector_address, node.server_address]
      .filter(Boolean)
      .forEach((address) => nodeIndexes.set(address, index));
  });
  let blocks = 0;
  let hitTokens = 0;
  let promptTokens = 0;
  (replay?.requests || []).forEach((item, requestIndex) => {
    const request = item.request || {};
    const prompt = Number(request.prompt_token_count || 0);
    promptTokens += prompt;
    const target = replay.decisions?.[requestIndex]?.target || {};
    const nodeIndex = nodeIndexes.get(
      target.agent_address || target.address || "",
    );
    const matchedBlocks = nodeIndex == null
      ? 0
      : Number(replay.node_matches?.[requestIndex]?.[nodeIndex]
        ?.matched_block_count || 0);
    blocks += matchedBlocks;
    hitTokens += Math.min(
      prompt,
      matchedBlocks * Number(request.block_size || 0),
    );
  });
  return {blocks, tokenRate: promptTokens ? hitTokens / promptTokens : 0};
}

function renderReplaySummary(replay) {
  const requests = replay?.requests || [];
  const nodes = replay?.nodes || [];
  const batch = replayBatchSummary(replay);
  const promptTokens = requests.map(
    (item) => Number(item.request?.prompt_token_count || 0),
  );
  const queueValues = nodes.map(
    (node) => Number(node.queue_total_predicted_latency_ms || 0),
  );
  const cache = selectedCacheSummary(replay);
  const revisions = nodes.map((node) => node.latency_model_revision)
    .filter((value) => value != null);
  const queueMean = queueValues.length
    ? queueValues.reduce((sum, value) => sum + value, 0) / queueValues.length
    : 0;
  const metrics = [
    ["请求 / Agent", `${number(requests.length)} / ${number(nodes.length)}`],
    ["物理 Batch", number(batch.count)],
    ["Batch size 平均 / 范围",
      `${number(batch.meanSize, 2)} / ${numericRange(batch.sizes)}`],
    ["Master 期望 Batch size", numericRange(batch.expectedSizes)],
    ["Prompt tokens 总计", number(
      promptTokens.reduce((sum, value) => sum + value, 0),
    )],
    ["Prompt 长度范围", numericRange(promptTokens)],
    ["选中节点 Cache hit",
      `${number(cache.blocks)} blocks / ${number(cache.tokenRate * 100, 2)}% tokens`],
    ["预测 Queue 平均 / 最大",
      `${milliseconds(queueMean, 2)} / ${milliseconds(Math.max(0, ...queueValues), 2)}`],
    ["Inflight / Running",
      `${number(nodes.reduce((sum, node) => sum + Number(node.inflight_requests || 0), 0))} / ` +
      number(nodes.reduce((sum, node) => sum + Number(node.running_requests || 0), 0))],
    ["模型 revision 范围", numericRange(revisions)],
  ];
  elements.replaySummary.replaceChildren(
    ...metrics.map(([label, value]) => metricItem(label, value)),
  );
}

function renderRun() {
  const run = activeRun();
  if (!run || !run.data) {
    elements.empty.hidden = false;
    elements.workspace.hidden = true;
    return;
  }
  elements.empty.hidden = true;
  elements.workspace.hidden = false;
  elements.sourceLine.textContent = run.name;
  state.selectedCell = null;
  state.overviewRequestIndex = null;
  state.overviewNodeIndex = null;
  state.overviewNodeIndices = [];
  state.optimizationRunIndex = 0;
  state.optimizationStepIndex = 0;
  renderReplaySummary(run.data);
  renderRequests(run.data);
  renderNodes(run.data);
  renderCacheMatrix(run.data);
  renderSchedulerMatrix(run.data);
  renderOptimization(run.data);
  elements.rawJson.textContent = JSON.stringify(run.data, null, 2);
}

function schedulerName(run) {
  return run.data?.scheduler_info?.scheduler ||
    run.data?.scheduler_info?.scheduler_name ||
    run.data?.scheduler_name || run.scheduler || "-";
}

function schedulerParameter(run, name) {
  const value = run.data?.scheduler_info?.[name] ?? run[name];
  if (value == null) {
    return "-";
  }
  if (name === "lambda") {
    return lambdaNumber(Number(value));
  }
  const digits = {alpha: 3, alpha_decay: 5, min_alpha: 3}[name] ?? 0;
  return number(value, digits);
}

function actualLoopCount(run) {
  const optimizationRuns = run.data?.scheduler_info?.optimization_runs;
  const value = optimizationRuns?.[0]?.actual_loop_count ??
    run.actual_loop_count;
  return value == null ? "-" : number(value);
}

function optimizationBudgetMs(run) {
  const value = run.data?.scheduler_info
    ?.optimization_time_budget_us ?? run.optimization_time_budget_us;
  return value == null ? "-" : number(value / 1000, 3);
}

function replayCount(replay, run, field, metadataField) {
  const values = replay?.[field];
  if (Array.isArray(values)) {
    return number(values.length);
  }
  const value = run[metadataField];
  return value == null ? "-" : number(value);
}

function promptTokenCount(replay, run) {
  if (Array.isArray(replay?.requests)) {
    return number(replay.requests.reduce(
      (sum, item) => sum + Number(item.request?.prompt_token_count || 0),
      0,
    ));
  }
  return run.prompt_token_count == null
    ? "-"
    : number(run.prompt_token_count);
}

function replayListFeatures(replay, run) {
  if (!replay) {
    return {
      batchCount: number(run.batch_count),
      batchSize: `${number(run.mean_batch_size, 2)} / ` +
        `${number(run.min_batch_size)}-${number(run.max_batch_size)}；期望 ` +
        `${number(run.min_expected_batch_size)}-${number(run.max_expected_batch_size)}`,
      promptRange: `${number(run.min_prompt_tokens)}-${number(run.max_prompt_tokens)}`,
      cacheHit: `${number(run.selected_cache_hit_blocks)} blocks / ` +
        `${number(Number(run.selected_cache_hit_token_rate || 0) * 100, 2)}%`,
      queue: `${milliseconds(run.queue_mean_ms, 2)} / ` +
        milliseconds(run.queue_max_ms, 2),
      load: `${number(run.inflight_request_count)} / ` +
        number(run.running_request_count),
      revision: `${number(run.min_model_revision)}-${number(run.max_model_revision)}`,
    };
  }
  const batch = replayBatchSummary(replay);
  const prompts = (replay.requests || []).map(
    (item) => item.request?.prompt_token_count,
  );
  const cache = selectedCacheSummary(replay);
  const nodes = replay.nodes || [];
  const queues = nodes.map(
    (node) => Number(node.queue_total_predicted_latency_ms || 0),
  );
  const queueMean = queues.length
    ? queues.reduce((sum, value) => sum + value, 0) / queues.length
    : 0;
  return {
    batchCount: number(batch.count),
    batchSize: `${number(batch.meanSize, 2)} / ${numericRange(batch.sizes)}；` +
      `期望 ${numericRange(batch.expectedSizes)}`,
    promptRange: numericRange(prompts),
    cacheHit: `${number(cache.blocks)} blocks / ` +
      `${number(cache.tokenRate * 100, 2)}%`,
    queue: `${milliseconds(queueMean, 2)} / ` +
      milliseconds(Math.max(0, ...queues), 2),
    load: `${number(nodes.reduce((sum, node) => sum + Number(node.inflight_requests || 0), 0))} / ` +
      number(nodes.reduce((sum, node) => sum + Number(node.running_requests || 0), 0)),
    revision: numericRange(nodes.map((node) => node.latency_model_revision)
      .filter((value) => value != null)),
  };
}

function formatBytes(value) {
  if (value == null) {
    return "-";
  }
  if (value < 1024) {
    return `${value} B`;
  }
  if (value < 1024 * 1024) {
    return `${number(value / 1024, 1)} KiB`;
  }
  return `${number(value / (1024 * 1024), 1)} MiB`;
}

const SAMPLE_CONTEXTS = {
  "00_real_rtp_llm_8gpu_runtime_sample.json":
    "RTP-LLM 8 卡真实 Prefill 运行中采集的原始调度现场，不是 simulate，也不是离线生成的反事实 Replay。",
  "01_single_online_lambda_0.jsonl":
    "单 biz 在线低 lambda 场景，用于观察以排队延迟为主时的调度决策。",
  "02_shared_online_lambda_0.jsonl":
    "共享混合池 lambda=0 在线优先场景，用于观察只考虑排队代价时的调度决策。",
  "03_shared_balanced_lambda_0_5.jsonl":
    "共享混合池中等 lambda 场景，用于观察排队与执行代价平衡时的调度决策。",
  "04_shared_regression_lambda_0_999.jsonl":
    "共享混合池近 1 回归场景，用于分析偏重执行效率时的延迟与吞吐退化。",
  "05_independent_offline_lambda_1.jsonl":
    "独立池 lambda=1 离线边界场景，用于观察完全忽略排队代价时的调度行为。",
};

const SAMPLE_LABELS = {
  "00_real_rtp_llm_8gpu_runtime_sample.json":
    "真实运行现场 #1（RTP-LLM 8卡，非 simulate）",
};

function sampleLabel(run) {
  const fileName = run.name.replace(/ #\d+$/, "");
  const realSampleMatch = fileName.match(
    /^00_real_rtp_llm_8gpu_runtime_sample_(\d+)\.json$/,
  );
  if (realSampleMatch) {
    const agentCount = run.data?.nodes?.length ?? run.agent_count;
    const deployment = agentCount == null
      ? "RTP-LLM"
      : `RTP-LLM ${Number(agentCount)} Agent`;
    return `真实运行现场 #${Number(realSampleMatch[1])}` +
      `（${deployment}，非 simulate）`;
  }
  return SAMPLE_LABELS[fileName] || run.name;
}

function sampleKind(run) {
  const fileName = run.name.replace(/ #\d+$/, "");
  return fileName.startsWith("00_real_rtp_llm_8gpu_runtime_sample")
    ? "真实运行"
    : "历史 Replay";
}

const SAMPLE_TIME_FORMATTER = new Intl.DateTimeFormat("zh-CN", {
  timeZone: "Asia/Shanghai",
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hourCycle: "h23",
});

function sampleDescription(run) {
  const scheduleTimeUs = run.data?.schedule_time_us ?? run.schedule_time_us;
  const capturedAt = Number.isFinite(Number(scheduleTimeUs))
    ? `${SAMPLE_TIME_FORMATTER.format(new Date(Number(scheduleTimeUs) / 1000))}` +
      "（Asia/Shanghai）"
    : "Replay 未提供 schedule_time_us";
  const fileName = run.name.replace(/ #\d+$/, "");
  const context = SAMPLE_CONTEXTS[fileName] ||
    (fileName.startsWith("00_real_rtp_llm_8gpu_runtime_sample_")
      ? "RTP-LLM 真实 Prefill 运行中采集的原始调度现场，Agent 数以该 Replay 的 Nodes 为准；不是 simulate，也不是离线生成的反事实 Replay。"
      : null) ||
    "本地导入的调度 Replay，用于还原该轮请求、节点状态和优化过程。";
  const sampleMatch = run.name.match(/ #(\d+)$/);
  const position = sampleMatch
    ? `${fileName} 中的第 ${sampleMatch[1]} 个调度轮次。`
    : `${fileName} 中记录的调度轮次。`;
  return `采集时间：${capturedAt}\n采集场景：${context}\n样本位置：${position}`;
}

function updateRunTable() {
  elements.runTable.replaceChildren();
  if (!state.runs.length) {
    state.activeRunId = "";
    renderRun();
    return;
  }
  if (!state.runs.some((run) => run.id === state.activeRunId)) {
    state.activeRunId = state.runs[0].id;
  }
  const table = document.createElement("table");
  const realRuns = state.runs.filter((run) => sampleKind(run) === "真实运行");
  const realRequestCounts = realRuns
    .map((run) => Number(run.data?.requests?.length ?? run.request_count))
    .filter(Number.isFinite);
  const realSampleSummary = realRequestCounts.length
    ? `当前包含 ${realRuns.length} 个 RTP-LLM 真实运行现场，` +
      `每轮 ${Math.min(...realRequestCounts)}–${Math.max(...realRequestCounts)} 个请求；`
    : "";
  appendTableCaption(
    table,
    `${realSampleSummary}每行是一份可选择的调度 Replay，节点规模以 Agents 列为准；` +
      "选中后页面下方展示该轮完整现场。",
  );
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  tableColumns([
    "类型",
    "sample",
    "biz",
    "scheduler",
    "lambda",
    "alpha",
    "alpha decay",
    "min alpha",
    "max loops",
    "actual loops",
    "budget ms",
    "parallelism",
    "requests",
    "Batch 数",
    "Batch size 平均 / 范围 / 期望",
    "agents",
    "prompt tokens",
    "Prompt 长度范围",
    "选中节点 Cache hit",
    "预测 Queue 平均 / 最大",
    "Inflight / Running",
    "模型 revision",
    "size",
    "source",
  ]).forEach((header) => {
      const cell = document.createElement("th");
      cell.textContent = headerLabel(header);
      cell.title = headerDescription(header);
      headRow.append(cell);
  });
  head.append(headRow);
  table.append(head);
  const body = document.createElement("tbody");
  for (const run of state.runs) {
    const row = document.createElement("tr");
    row.classList.toggle("selected-row", run.id === state.activeRunId);
    row.tabIndex = 0;
    row.setAttribute("role", "button");
    row.setAttribute("aria-pressed", String(run.id === state.activeRunId));
    const replay = run.data;
    const features = replayListFeatures(replay, run);
    const values = [
      sampleKind(run),
      sampleLabel(run),
      replay?.biz_name || run.biz_name || "-",
      schedulerName(run),
      schedulerParameter(run, "lambda"),
      schedulerParameter(run, "alpha"),
      schedulerParameter(run, "alpha_decay"),
      schedulerParameter(run, "min_alpha"),
      schedulerParameter(run, "max_loop_count"),
      actualLoopCount(run),
      optimizationBudgetMs(run),
      schedulerParameter(run, "parallelism"),
      replayCount(replay, run, "requests", "request_count"),
      features.batchCount,
      features.batchSize,
      replayCount(replay, run, "nodes", "agent_count"),
      promptTokenCount(replay, run),
      features.promptRange,
      features.cacheHit,
      features.queue,
      features.load,
      features.revision,
      formatBytes(run.size),
      run.source,
    ];
    values.forEach((value) => row.append(textCell(value)));
    const sampleCell = row.cells[1];
    sampleCell.title = sampleDescription(run);
    const select = () => selectRun(run.id);
    row.addEventListener("click", select);
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        select();
      }
    });
    body.append(row);
  }
  table.append(body);
  elements.runTable.append(table);
}

async function ensureRunLoaded(run) {
  if (run.data || !run.url) {
    return;
  }
  const response = await fetch(run.url, {cache: "no-store"});
  if (!response.ok) {
    throw new Error(`加载 ${run.name} 失败：HTTP ${response.status}`);
  }
  run.data = await response.json();
}

async function selectRun(runId) {
  const run = state.runs.find((item) => item.id === runId);
  if (!run) {
    return;
  }
  try {
    await ensureRunLoaded(run);
    state.activeRunId = run.id;
    updateRunTable();
    renderRun();
  } catch (error) {
    elements.empty.hidden = false;
    elements.empty.innerHTML = "";
    const banner = document.createElement("div");
    banner.className = "error-banner";
    banner.textContent = error.message;
    elements.empty.append(banner);
  }
}

async function refreshDirectoryRuns() {
  try {
    const response = await fetch("/api/replays", {cache: "no-store"});
    if (!response.ok) {
      throw new Error(`Replay 目录不可用：HTTP ${response.status}`);
    }
    const descriptors = await response.json();
    const uploads = state.runs.filter((run) => run.source === "browser");
    const directoryRuns = descriptors.map((item) => ({
      id: `directory:${item.name}`,
      name: item.name,
      source: "directory",
      url: item.url,
      size: item.size,
      biz_name: item.biz_name,
      scheduler: item.scheduler,
      lambda: item.lambda,
      alpha: item.alpha,
      alpha_decay: item.alpha_decay,
      min_alpha: item.min_alpha,
      max_loop_count: item.max_loop_count,
      actual_loop_count: item.actual_loop_count,
      optimization_stop_reason: item.optimization_stop_reason,
      optimization_time_budget_us: item.optimization_time_budget_us,
      parallelism: item.parallelism,
      schedule_time_us: item.schedule_time_us,
      request_count: item.request_count,
      agent_count: item.agent_count,
      prompt_token_count: item.prompt_token_count,
      batch_count: item.batch_count,
      mean_batch_size: item.mean_batch_size,
      min_batch_size: item.min_batch_size,
      max_batch_size: item.max_batch_size,
      min_expected_batch_size: item.min_expected_batch_size,
      max_expected_batch_size: item.max_expected_batch_size,
      min_prompt_tokens: item.min_prompt_tokens,
      max_prompt_tokens: item.max_prompt_tokens,
      selected_cache_hit_blocks: item.selected_cache_hit_blocks,
      selected_cache_hit_token_rate: item.selected_cache_hit_token_rate,
      queue_mean_ms: item.queue_mean_ms,
      queue_max_ms: item.queue_max_ms,
      inflight_request_count: item.inflight_request_count,
      running_request_count: item.running_request_count,
      min_model_revision: item.min_model_revision,
      max_model_revision: item.max_model_revision,
      data: null,
    }));
    state.runs = [...directoryRuns, ...uploads];
    updateRunTable();
    if (state.runs.length) {
      await selectRun(state.activeRunId || state.runs[0].id);
    }
  } catch (error) {
    console.error(error);
    updateRunTable();
  }
}

function parseReplayFile(name, text) {
  try {
    return [{name, data: JSON.parse(text)}];
  } catch (singleError) {
    const runs = [];
    text.split(/\r?\n/).forEach((line, index) => {
      if (line.trim()) {
        runs.push({name: `${name} #${index + 1}`, data: JSON.parse(line)});
      }
    });
    return runs;
  }
}

async function openFiles(fileList) {
  const added = [];
  for (const file of fileList) {
    const text = await file.text();
    parseReplayFile(file.name, text).forEach((entry, index) => {
      added.push({
        id: `browser:${file.name}:${file.lastModified}:${index}`,
        name: entry.name,
        source: "browser",
        size: file.size,
        data: entry.data,
      });
    });
  }
  state.runs = [...added, ...state.runs];
  if (added.length) {
    state.activeRunId = added[0].id;
  }
  updateRunTable();
  renderRun();
}

elements.files.addEventListener("change", () => {
  openFiles(elements.files.files).catch(console.error);
  elements.files.value = "";
});
elements.refresh.addEventListener("click", refreshDirectoryRuns);

function selectOverviewRequestRow(row) {
  const requestIndex = Number(row?.dataset.rowIndex);
  if (!Number.isInteger(requestIndex)) {
    return;
  }
  if (state.overviewRequestIndex === requestIndex) {
    selectOverviewRequest(null, null);
    return;
  }
  selectOverviewRequest(
    requestIndex,
    state.overviewNodeIndices[requestIndex],
  );
}

elements.requestTable.addEventListener("click", (event) => {
  selectOverviewRequestRow(event.target.closest("tbody tr"));
});
elements.requestTable.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    const row = event.target.closest("tbody tr");
    if (row) {
      event.preventDefault();
      selectOverviewRequestRow(row);
    }
  }
});
elements.requestTable.addEventListener("table-sorted", () => {
  decorateOverviewRequestRows();
  if (state.overviewRequestIndex != null) {
    selectOverviewRequest(
      state.overviewRequestIndex,
      state.overviewNodeIndex,
    );
  }
});
elements.nodeTable.addEventListener("table-sorted", () => {
  if (state.overviewRequestIndex != null) {
    selectOverviewRequest(
      state.overviewRequestIndex,
      state.overviewNodeIndex,
    );
  }
});
elements.cacheTable.addEventListener("table-sorted", () => {
  if (state.overviewRequestIndex != null) {
    selectOverviewRequest(
      state.overviewRequestIndex,
      state.overviewNodeIndex,
    );
  }
});

document.querySelectorAll(".view-tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".view-tab").forEach((item) => {
      item.classList.toggle("active", item === button);
    });
    document.querySelectorAll(".view-panel").forEach((panel) => {
      panel.classList.toggle("active", panel.id === `${button.dataset.view}-view`);
    });
  });
});

const pageNames = new Set(
  [...document.querySelectorAll("[data-page]")].map((button) => button.dataset.page),
);

function activatePage(pageName) {
  const selected = pageNames.has(pageName) ? pageName : "comparison";
  document.querySelectorAll("[data-page]").forEach((item) => {
    item.classList.toggle("active", item.dataset.page === selected);
  });
  document.querySelectorAll(".page-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `${selected}-page`);
  });
}

document.querySelectorAll("[data-page]").forEach((button) => {
  button.addEventListener("click", () => {
    window.location.hash = button.dataset.page;
    activatePage(button.dataset.page);
  });
});
window.addEventListener("hashchange", () => {
  activatePage(window.location.hash.slice(1));
});
activatePage(window.location.hash.slice(1));

document.querySelectorAll("[data-orientation]").forEach((button) => {
  button.addEventListener("click", () => {
    state.orientation = button.dataset.orientation;
    document.querySelectorAll("[data-orientation]").forEach((item) => {
      item.classList.toggle(
        "active", item.dataset.orientation === state.orientation,
      );
    });
    renderSchedulerMatrix(activeRun()?.data);
    renderOptimization(activeRun()?.data);
  });
});

document.querySelectorAll("[data-metric]").forEach((button) => {
  button.addEventListener("click", () => {
    state.metric = button.dataset.metric;
    document.querySelectorAll("[data-metric]").forEach((item) => {
      item.classList.toggle("active", item === button);
    });
    renderSchedulerMatrix(activeRun()?.data);
  });
});

elements.optimizationRun.addEventListener("change", () => {
  state.optimizationRunIndex = Number(elements.optimizationRun.value);
  state.optimizationStepIndex = 0;
  renderOptimization(activeRun()?.data);
});

elements.optimizationStep.addEventListener("input", () => {
  state.optimizationStepIndex = Number(elements.optimizationStep.value);
  renderOptimization(activeRun()?.data);
});

loadComparison();
refreshDirectoryRuns();
