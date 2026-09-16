# Priority live 探针：构造与验证

基线 `168ad89170`；远端租约 `priority-live-design-20260909`，111，Java mock，无 GPU。只改 YAML、Python 编排、framework action 与测试。Master 与 mock 引擎代码不变。

## Decode 两实例的构造

`decode_reserved_live_single::single-batch` 和 `decode_reserved_live_window::batch-window` 都使用 `FIXED_WINDOW`（结果 `effective_axes` 明示），收集窗 3000ms。placeholder 先 Fetch 完成并释放资源；然后只向 Decode **上报值**注入 1000 token 压力。victim 在 master 本地排队，经 debug snapshot 与全引擎 lifecycle 双源验证后才发 incoming。保护条件包括 queued、reserved、无 dispatch permit、无 protocol owner、所有引擎均未见 victim；证据缺失或越过边界记构造 ERROR。

YAML 配置 victim 为 priority30、1536/2，incoming 为 priority70、3500/2。D 池仍5块，每块1024，Master默认90%使用率门槛、引擎5%水位均不变。所有新增参数与请求数值由YAML提供。

| 检查 | 值 |
| --- | --- |
| 实际总容量 | 5120 token / 5块 |
| 注入后上报 available | 5120−1000=4120 |
| victim 本地预留后 available | 4120−1536=2584 |
| incoming 的 Master 缺口 | 3500−2584=916，可由1536释放覆盖 |
| 驱逐后预计使用量 | 1000+3502=4502 ≤ 5120×90%=4608 |
| 引擎实际准入 | ceil(3500/1024)+ceil(5×5%)=4+1=5块 |

victim 不持引擎租约，实际池不受上报压力影响。压力操作前后核对真实块池不变，并在异常清理中撤销压力。恢复请求在撤压并同步之后执行。

## 两个已证伪的布局

旧 `.task_specs/priority-eviction-case-fix-20260909/fix-requirement.md` 的“victim 持1块、空闲4块”布局不能复用：持租约立即产生引擎 lifecycle，违反 never_seen，且不再是 MASTER_LOCAL 候选。该文档的 drain 前移有效，继续保留。

本次方向A初试保留victim512，压力1320、上报3800。它能形成Master本地候选与硬KV缺口，但仍返回8431。独立诊断YAML只将压力同步暂停延长到120秒，使用Arthas只读trace/watch取证：所有身份与本地所有权检查通过，提交后的预计使用为1320+3502=4822，大于90%门槛4608，容量检查正确拒绝。固定512时，硬缺口要求上报<4012，90%门槛要求上报≥4014，无交集。因此按需求FR-2方向B调整victim输入，未修改容量或门槛。

## PR6 的指标读取修复

方向B初试已得到 `[8400,200]`、victim never_seen、incoming完整成功；PR6失败来自旧取数代码。`RequestSchedulerReporter` 将 `AUTO_TPM_VICTIM_KV_TOKENS` 注册为 TIMER；`MicrometerFlexMonitor` 将数值按毫秒记录，Prometheus导出 `..._seconds_sum=1.536`。

旧case用substring匹配，混加bucket/count/max/quantile/sum，得到30.977580032。修复只取exact `..._seconds_sum×1000` 得到1536，兼容直接token样本；count只取exact counter。PR10、PR5、PR6的布尔条件和阈值保持不变，427仍失败，8429仍失败。删除裁决中重复写死512的历史参数校验，保留cohort角色顺序检查。

## Prefill single 的可行性裁决

FR-3方向A不成立：`PrefillState.batchPublicationCapacity` 的本地publication容量是 `min(maxWaiting, freeBatches × maxRequestsPerDecision)`；SINGLE每次1请求。在飞额度2被placeholder和victim_a占满后，freeBatches=0，本地publication容量同样为0，victim_b无法稳定排队。仅把4改2不能修好case。

保持该single实例原样，等待用户按FR-3方向B裁决；不自主移除或改变decision。`prefill_queued_live_window` 是保留的通过对照。

## 验证状态

Java mock全量：366项，0失败/错误/跳过。Python框架：91个测试文件、848项通过；指标修复后针对性5项再次通过（含427与8429反例）。现有Maven与parallel_runner入口已获用户明确授权；未编写测试包装脚本。

两个Decode实例在priority家族跑批与confirm复跑中连续两轮PASS：schedule_codes=[8400,200]，count=1，KV=1536，never_seen=true，incoming完整成功。prefill window对照两轮PASS；对应confirm-sb/confirm-bw作业退出码均0。

Priority家族首轮37项：28 PASS、7 FAIL、2 ERROR；各profile实际结果如下，保留原始结果，不用复跑覆盖：

| Profile | PASS | FAIL | ERROR | 作业退出码 |
| --- | ---: | ---: | ---: | ---: |
| single-batch | 4 | 3 | 0 | 1 |
| batch-window | 6 | 1 | 0 | 1 |
| single-nonbatch | 14 | 3 | 2 | 1 |
| window-nonbatch | 4 | 0 | 0 | 0 |

对所有9个未通过实例用基线168ad89170独立副本复核：8个同样非PASS（prefill live single、cancel_tombstoned两profile、single-nonbatch的prefill_queued/same_priority_zero_eviction/observability_integrity/error_code_family/comparator_frozen_weak）。normalize_default50::single-batch首轮PR3前两项顺序交换；基线与候选原样复跑均PASS，记偶发失败，未修改其用例。故不能称全家族通过；本轮未发现经复核确认的新增回归。

原始方向A及方向B指标修复前的FAIL均保留，不把任何失败改为预期失败。功能性结果按实例status、退出码和产物记录，不补造test_valid字段。

外部完整证据目录：`/Users/wangziyi/code/case-refactor-reports/2026-09-09/priority-live-design/`。
