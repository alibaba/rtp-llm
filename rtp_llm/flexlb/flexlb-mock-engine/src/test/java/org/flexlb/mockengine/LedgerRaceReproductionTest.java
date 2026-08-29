package org.flexlb.mockengine;

import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import com.google.protobuf.ByteString;
import org.flexlb.balance.scheduler.LedgerReconciliationHarness;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * m1race 诊断复现用例（test-only，主代码零改动）。
 *
 * <p>背景：rebase 到 v2 intake3 之后的 12 分钟浸泡（loop 5ms / confirm 3，
 * ~1400 req/s）在静默后出现 79/1M 请求的 slot 永久冻结（四类指纹：
 * ENGINE_FENCE_UNBACKED / DECODE_FENCE_PROTECTION_ORPHAN /
 * SLOT_RESERVATION_UNBACKED / SLOT_PROJECTION_UNCONFIRMED，两两交集为空，
 * 双子窗口流特异，静默 342s 不自愈）。静态链路分析把第一嫌疑收敛为
 * “decode 终结投影（WorkerStatus finished 观察）丢失/静默丢弃 → 30s
 * delivered-not-accepted 兜底到期 → fence/残骸固化”。本用例用四个层次
 * 的实验裁决该假设：</p>
 *
 * <ol>
 *   <li>微浸泡压缩复现：deliveredNotAcceptedTimeoutMs 30s→200ms（可调），
 *       完整复刻浸泡拓扑（2P/2D/UnsupportedCancelStub/rid%2 prefill/pump
 *       10ms/shadow 5ms confirm 3），把 12 分钟浸泡压缩成秒级，统计四类
 *       指纹的 rid 集合/奇偶分布/时间分布，并关联 WARN+ERROR 日志。</li>
 *   <li>单请求手动 pump 基线：正常路径对照（无冻结）。</li>
 *   <li>holdBatchAck 探针：delivery 确认（ACK）被人为扣到引擎 finished
 *       观察之后——时序反转窗口的确定性注入。</li>
 *   <li>acceptance 到期先于 finished 观察：完全手动 pump 制造
 *       “delivery 已确认、引擎已完成、无人观察”窗口，让 acceptance
 *       兜底先触发，再恢复观察——fence 装载与迟到 WorkerTerminal 投影
 *       交互的确定性裁决。</li>
 * </ol>
 *
 * <p>默认跳过，显式 {@code -Dflexlb.race=true} 启用（诊断用例，
 * 复现即产物，不作为回归门禁）：</p>
 *
 * <pre>
 * ./mvnw -P'opensource,!internal' -pl flexlb-mock-engine -am test \
 *   -Dtest=LedgerRaceReproductionTest -Dsurefire.failIfNoSpecifiedTests=false \
 *   -Dflexlb.race=true -Dflexlb.race.seconds=90
 * </pre>
 */
class LedgerRaceReproductionTest {

    private static final int RACE_PORT = 63710;
    private static final int BATCH_SIZE = 60;
    private static final int[] PRIORITIES = {70, 50, 30};

    /** 浸泡中观察到的四类冻结指纹规则名。 */
    private static final List<String> FROZEN_RULES = List.of(
            "ENGINE_FENCE_UNBACKED",
            "DECODE_FENCE_PROTECTION_ORPHAN",
            "SLOT_RESERVATION_UNBACKED",
            "SLOT_PROJECTION_UNCONFIRMED");

    private static final Pattern REQUEST_ID_IN_LOG =
            Pattern.compile("request_id[=: ](\\d+)");

    // ================================================================
    // Test 1: 压缩 acceptance 窗的微浸泡 —— 复现主载体
    // ================================================================

    @Test
    void compressedAcceptanceTimeoutMicroSoakReproducesFrozenSlots()
            throws Exception {
        Assumptions.assumeTrue(
                Boolean.getBoolean("flexlb.race"),
                "race reproduction is opt-in: -Dflexlb.race=true");
        long raceSeconds = Long.getLong("flexlb.race.seconds", 90L);
        long acceptanceTimeoutMs =
                Long.getLong("flexlb.race.acceptance.timeout", 200L);
        // harness 默认给每个请求 30s request deadline（startTime+30_000）；
        // race4 起可配置，用于解耦 acceptance 兜底与 request deadline 两个
        // 独立定时器对 EFU 显形时刻的影响。
        long requestDeadlineMs =
                Long.getLong("flexlb.race.deadline.ms", 30_000L);
        long quiesceMs = Long.getLong("flexlb.race.quiesce.ms", 3_000L);
        int samplingRounds =
                Integer.getInteger("flexlb.race.sampling.rounds", 60);
        long samplingIntervalMs =
                Long.getLong("flexlb.race.sampling.interval", 100L);

        AttachedAppender diagnostics = attachDiagnosticAppender(
                "flexlbLogger", "syncLogger");
        try (AutoTpmE2EHarness h = new AutoTpmE2EHarness(
                RACE_PORT, 2, 2, "5", 1.0, false, true)) {
            h.fixedWindowDecision().setMaxCollectionWaitMs(5);
            h.fixedWindowDecision().setMaxRequests(2);
            h.config.queueScheduler().getCapacity()
                    .setMaxWaitingRequestsPerPrefillWorker(1024);
            // 核心杠杆：delivered-not-accepted 兜底 30s → 压缩值。
            h.config.queueScheduler().getLifecycle()
                    .setDeliveredNotAcceptedTimeoutMs(acceptanceTimeoutMs);
            h.config.queueScheduler().getLifecycle()
                    .setMaxDeliveredNotAcceptedRequestsGlobal(100_000);
            h.prefillSelector = ctx -> (int) (ctx.getRequestId() % 2);
            h.startAutoPump(10);

            List<LedgerReconciliationHarness.LedgerDiff> confirmedReal =
                    new CopyOnWriteArrayList<>();
            List<LedgerReconciliationHarness.LedgerDiff> pendingSeen =
                    new CopyOnWriteArrayList<>();
            LedgerReconciliationHarness reconciler =
                    h.scheduler.attachLedgerReconciliation(report -> {
                        confirmedReal.addAll(report.realDiffs());
                        pendingSeen.addAll(report.pendingRealDiffs());
                    }, 3);
            reconciler.startShadowLoop(5);

            // 预热一笔压热链路（不计入浸泡流量）。
            h.scheduler.submit(raceContext(h, 2999, 50, requestDeadlineMs))
                    .get(10, TimeUnit.SECONDS);

            Map<Long, Long> submitMillis = new ConcurrentHashMap<>();
            Map<Long, Long> doneMillis = new ConcurrentHashMap<>();
            long rid = 3000;
            int batches = 0;
            long failures = 0;
            long batchTimeouts = 0;
            long deadline = System.nanoTime()
                    + TimeUnit.SECONDS.toNanos(raceSeconds);
            while (System.nanoTime() < deadline) {
                List<CompletableFuture<Response>> futures =
                        new ArrayList<>(BATCH_SIZE);
                for (int i = 0; i < BATCH_SIZE; i++) {
                    submitMillis.put(rid, System.currentTimeMillis());
                    futures.add(h.scheduler.submit(raceContext(
                            h, rid, PRIORITIES[i % PRIORITIES.length],
                            requestDeadlineMs)));
                    rid++;
                }
                try {
                    AutoTpmE2EHarness.await(
                            () -> futures.stream()
                                    .allMatch(CompletableFuture::isDone),
                            60_000,
                            "race batch must fully terminate");
                } catch (AssertionError batchStall) {
                    batchTimeouts++;
                }
                for (int i = 0; i < futures.size(); i++) {
                    CompletableFuture<Response> future = futures.get(i);
                    if (!future.isDone()) {
                        failures++;
                        continue;
                    }
                    doneMillis.put(rid - futures.size() + i,
                            System.currentTimeMillis());
                    Response response = future.get(1, TimeUnit.SECONDS);
                    if (!response.isSuccess()) {
                        failures++;
                    }
                }
                batches++;
            }

            // 静默收敛：pump 尾部 settle + 事件投影排空；随后密集采样
            // （默认 60 轮 × 100ms）逐轮记录 EFU rid 集合，拿每个 rid 的
            // 精确首现/末见轮次，与 submit + requestDeadline 对齐裁决
            // 触发定时器（滚动出现 = per-rid 定时器；同一绝对时刻集中
            // 显形 = 单一事件）。
            Thread.sleep(quiesceMs);
            long samplingStartMs = System.currentTimeMillis();
            List<LedgerReconciliationHarness.ReconciliationReport> samples =
                    new ArrayList<>(samplingRounds);
            for (int i = 0; i < samplingRounds; i++) {
                samples.add(reconciler.reconcileOnce());
                Thread.sleep(samplingIntervalMs);
            }
            LedgerReconciliationHarness.ReconciliationReport finalReport =
                    samples.get(Math.max(0, samples.size() - 3));
            LedgerReconciliationHarness.ReconciliationReport finalReport2 =
                    samples.get(samples.size() - 2);
            LedgerReconciliationHarness.ReconciliationReport finalReport3 =
                    samples.get(samples.size() - 1);
            dumpRollingTimeline(
                    "micro-soak", samples, samplingStartMs, submitMillis);

            dumpRaceSummary(
                    "micro-soak",
                    confirmedReal,
                    pendingSeen,
                    finalReport,
                    finalReport2,
                    finalReport3,
                    submitMillis,
                    doneMillis,
                    batches,
                    failures,
                    batchTimeouts,
                    raceSeconds,
                    acceptanceTimeoutMs,
                    requestDeadlineMs,
                    diagnostics);
            reconciler.close();
        } finally {
            detachDiagnosticAppender(diagnostics);
        }
    }

    // ================================================================
    // Test 2: 单请求手动 pump 基线 —— 正常路径对照
    // ================================================================

    @Test
    void singleRequestManualPumpBaselineStaysClean() throws Exception {
        Assumptions.assumeTrue(
                Boolean.getBoolean("flexlb.race"),
                "race reproduction is opt-in: -Dflexlb.race=true");
        long acceptanceTimeoutMs =
                Long.getLong("flexlb.race.acceptance.timeout", 200L);
        try (AutoTpmE2EHarness h = new AutoTpmE2EHarness(
                RACE_PORT + 20, 2, 2, "5", 1.0, false, true)) {
            h.fixedWindowDecision().setMaxCollectionWaitMs(5);
            h.fixedWindowDecision().setMaxRequests(2);
            h.config.queueScheduler().getLifecycle()
                    .setDeliveredNotAcceptedTimeoutMs(acceptanceTimeoutMs);
            h.config.queueScheduler().getLifecycle()
                    .setMaxDeliveredNotAcceptedRequestsGlobal(1_000);

            long rid = 6100;
            CompletableFuture<Response> future =
                    h.scheduler.submit(h.context(rid, 50));
            int pumps = 0;
            while (!future.isDone() && pumps < 3_000) {
                h.pumpOnce();
                pumps++;
                Thread.sleep(2);
            }
            for (int i = 0; i < 10; i++) {
                h.pumpOnce();
                Thread.sleep(2);
            }
            Thread.sleep(acceptanceTimeoutMs + 500);

            LedgerReconciliationHarness reconciler =
                    h.scheduler.attachLedgerReconciliation(report -> { }, 1);
            LedgerReconciliationHarness.ReconciliationReport report =
                    reconciler.reconcileOnce();
            Thread.sleep(150);
            LedgerReconciliationHarness.ReconciliationReport report2 =
                    reconciler.reconcileOnce();

            boolean success = future.isDone()
                    && future.get(1, TimeUnit.SECONDS).isSuccess();
            System.out.printf(
                    "[race-baseline] rid=%d pumps=%d futureDone=%b success=%b"
                            + " real=%d pending=%d transient=%d slots=%d%n",
                    rid, pumps, future.isDone(), success,
                    report.realDiffs().size(),
                    report.pendingRealDiffs().size(),
                    report.transientDiffs().size(),
                    report.slotCount());
            dumpRuleCounts("race-baseline", report, report2);

            assertTrue(future.isDone(),
                    "baseline: future must complete under manual pump");
            assertTrue(success, "baseline: request must succeed");
            assertTrue(report.pendingRealDiffs().isEmpty()
                            && report.realDiffs().isEmpty(),
                    "baseline single request must stay clean: "
                            + report.pendingRealDiffs() + report.realDiffs());
            reconciler.close();
        }
    }

    // ================================================================
    // Test 3: holdBatchAck 探针 —— delivery 确认晚于引擎 finished 观察
    // ================================================================

    @Test
    void holdBatchAckDelaysDeliveryPastEngineFinish() throws Exception {
        Assumptions.assumeTrue(
                Boolean.getBoolean("flexlb.race"),
                "race reproduction is opt-in: -Dflexlb.race=true");
        long acceptanceTimeoutMs =
                Long.getLong("flexlb.race.acceptance.timeout", 200L);
        AttachedAppender diagnostics = attachDiagnosticAppender(
                "flexlbLogger", "syncLogger");
        try (AutoTpmE2EHarness h = new AutoTpmE2EHarness(
                RACE_PORT + 40, 2, 2, "5", 1.0, false, true)) {
            h.fixedWindowDecision().setMaxCollectionWaitMs(5);
            h.fixedWindowDecision().setMaxRequests(2);
            h.config.queueScheduler().getLifecycle()
                    .setDeliveredNotAcceptedTimeoutMs(acceptanceTimeoutMs);
            h.config.queueScheduler().getLifecycle()
                    .setMaxDeliveredNotAcceptedRequestsGlobal(1_000);

            long rid = 6600;
            CompletableFuture<Response> future;
            try (AutoCloseable gate = h.holdBatchAck(rid)) {
                future = h.scheduler.submit(h.context(rid, 50));
                // ACK 被扣：delivery 停在 DISPATCHING，但 mock 引擎照常执行，
                // finished 会被 pump 观察到 —— WorkerTerminal 投影将先于
                // delivery 确认到达 slot。
                long t0 = System.currentTimeMillis();
                while (System.currentTimeMillis() - t0
                        < acceptanceTimeoutMs + 300) {
                    h.pumpOnce();
                    Thread.sleep(2);
                }
            }
            // 释放 ACK → dispatcher 收到 → delivery 确认（或 Stale）。
            for (int i = 0; i < 20; i++) {
                h.pumpOnce();
                Thread.sleep(2);
            }
            Thread.sleep(acceptanceTimeoutMs + 500);

            LedgerReconciliationHarness reconciler =
                    h.scheduler.attachLedgerReconciliation(report -> { }, 1);
            LedgerReconciliationHarness.ReconciliationReport report =
                    reconciler.reconcileOnce();
            Thread.sleep(150);
            LedgerReconciliationHarness.ReconciliationReport report2 =
                    reconciler.reconcileOnce();

            boolean success = future.isDone()
                    && future.get(1, TimeUnit.SECONDS).isSuccess();
            System.out.printf(
                    "[race-holdack] rid=%d futureDone=%b success=%b"
                            + " real=%d pending=%d transient=%d slots=%d%n",
                    rid, future.isDone(), success,
                    report.realDiffs().size(),
                    report.pendingRealDiffs().size(),
                    report.transientDiffs().size(),
                    report.slotCount());
            dumpRuleCounts("race-holdack", report, report2);
            dumpDiagnosticLogs("race-holdack", diagnostics, Map.of());

            assertTrue(future.isDone(),
                    "hold-ack probe: future must complete");
            reconciler.close();
        } finally {
            detachDiagnosticAppender(diagnostics);
        }
    }

    // ================================================================
    // Test 4: acceptance 到期先于 finished 观察 —— fence 兜底与迟到
    //         WorkerTerminal 投影交互的确定性裁决
    // ================================================================

    @Test
    void acceptanceExpiryBeforeFinishedObservationIsObserved()
            throws Exception {
        Assumptions.assumeTrue(
                Boolean.getBoolean("flexlb.race"),
                "race reproduction is opt-in: -Dflexlb.race=true");
        long acceptanceTimeoutMs =
                Long.getLong("flexlb.race.acceptance.timeout", 200L);
        AttachedAppender diagnostics = attachDiagnosticAppender(
                "flexlbLogger", "syncLogger");
        try (AutoTpmE2EHarness h = new AutoTpmE2EHarness(
                RACE_PORT + 60, 2, 2, "5", 1.0, false, true)) {
            h.fixedWindowDecision().setMaxCollectionWaitMs(5);
            h.fixedWindowDecision().setMaxRequests(2);
            h.config.queueScheduler().getLifecycle()
                    .setDeliveredNotAcceptedTimeoutMs(acceptanceTimeoutMs);
            h.config.queueScheduler().getLifecycle()
                    .setMaxDeliveredNotAcceptedRequestsGlobal(1_000);
            // 无 autoPump：finished 观察完全由手动 pump 控制。

            long rid = 7100;
            CompletableFuture<Response> future =
                    h.scheduler.submit(h.context(rid, 50));
            // 阶段 1：不 pump。引擎自走（集批→dispatch→ACK→delivery 确认→
            // acceptance deadline arm→引擎 finished 入 completions 队列），
            // 但无人观察 running/finished。
            Thread.sleep(Math.max(150, acceptanceTimeoutMs / 2));
            // 阶段 2：跨过 acceptance 到期点（fence 兜底应已触发）。
            Thread.sleep(acceptanceTimeoutMs + 300);
            // 阶段 3：恢复观察 —— doCalibrate 消费 finished，WorkerTerminal
            // fact 投影到达带 fence 的 slot。
            for (int i = 0; i < 30; i++) {
                h.pumpOnce();
                Thread.sleep(2);
            }
            Thread.sleep(400);

            LedgerReconciliationHarness reconciler =
                    h.scheduler.attachLedgerReconciliation(report -> { }, 1);
            LedgerReconciliationHarness.ReconciliationReport report =
                    reconciler.reconcileOnce();
            Thread.sleep(150);
            LedgerReconciliationHarness.ReconciliationReport report2 =
                    reconciler.reconcileOnce();

            boolean success = future.isDone()
                    && future.get(1, TimeUnit.SECONDS).isSuccess();
            System.out.printf(
                    "[race-expiry] rid=%d futureDone=%b success=%b"
                            + " real=%d pending=%d transient=%d slots=%d%n",
                    rid, future.isDone(), success,
                    report.realDiffs().size(),
                    report.pendingRealDiffs().size(),
                    report.transientDiffs().size(),
                    report.slotCount());
            dumpRuleCounts("race-expiry", report, report2);
            dumpDiagnosticLogs("race-expiry", diagnostics, Map.of());

            assertTrue(future.isDone(),
                    "expiry race: future must complete after pump resumes");
            reconciler.close();
        } finally {
            detachDiagnosticAppender(diagnostics);
        }
    }

    // ================================================================
    // diagnostics helpers
    // ================================================================

    private record AttachedAppender(
            List<Logger> loggers,
            ListAppender<ILoggingEvent> appender) {
    }

    /** Attach one shared in-memory appender to the given named loggers. */
    private static AttachedAppender attachDiagnosticAppender(
            String... loggerNames) {
        List<Logger> classicLoggers = new ArrayList<>();
        ListAppender<ILoggingEvent> appender = new ListAppender<>();
        appender.start();
        for (String name : loggerNames) {
            org.slf4j.Logger slf4j = LoggerFactory.getLogger(name);
            if (slf4j instanceof Logger classic) {
                classic.addAppender(appender);
                classicLoggers.add(classic);
            }
        }
        return new AttachedAppender(classicLoggers, appender);
    }

    private static void detachDiagnosticAppender(AttachedAppender attached) {
        for (Logger classic : attached.loggers()) {
            classic.detachAppender(attached.appender());
        }
        attached.appender().stop();
    }

    /** Mirror of {@code AutoTpmE2EHarness#context} with a configurable deadline. */
    private static BalanceContext raceContext(
            AutoTpmE2EHarness h, long requestId, int priority,
            long requestDeadlineMs) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(128);
        request.setMaxNewTokens(8);
        request.setNumBeams(1);
        request.setModel("test-model");
        request.setPriority(priority);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setConfig(h.config);
        ctx.setGenerateInputPb(ByteString.copyFrom(
                AutoTpmE2EHarness.generateInputBytes(requestId, 128, 8)));
        ctx.setSchedulingMetadata(SchedulingMetadata.explicit(
                priority, ctx.getStartTime() + requestDeadlineMs));
        return ctx;
    }

    /**
     * Rolling EFU timeline: per-round EFU rid sets, first/last seen round per
     * rid, and the per-round EFU counts, to decide whether EFU emergence
     * tracks each rid's submit+requestDeadline or one absolute event.
     */
    private static void dumpRollingTimeline(
            String tag,
            List<LedgerReconciliationHarness.ReconciliationReport> samples,
            long samplingStartMs,
            Map<Long, Long> submitMillis) {
        Map<Long, int[]> ridWindows = new TreeMap<>();
        StringBuilder rolling = new StringBuilder();
        int round = 0;
        for (LedgerReconciliationHarness.ReconciliationReport sample
                : samples) {
            Set<Long> efuHere = new HashSet<>();
            Map<String, Integer> frozenHere = new TreeMap<>();
            for (LedgerReconciliationHarness.LedgerDiff diff
                    : sample.pendingRealDiffs()) {
                if (FROZEN_RULES.contains(diff.rule().name())) {
                    frozenHere.merge(diff.rule().name(), 1, Integer::sum);
                }
                if (diff.rule().name().equals("ENGINE_FENCE_UNBACKED")) {
                    efuHere.add(diff.requestId());
                }
            }
            for (LedgerReconciliationHarness.LedgerDiff diff
                    : sample.realDiffs()) {
                if (FROZEN_RULES.contains(diff.rule().name())) {
                    frozenHere.merge(diff.rule().name(), 1, Integer::sum);
                }
                if (diff.rule().name().equals("ENGINE_FENCE_UNBACKED")) {
                    efuHere.add(diff.requestId());
                }
            }
            if (!efuHere.isEmpty()) {
                System.out.printf(
                        "[race-%s] TIMELINE round=%d tMs=%d efu=%d"
                                + " minRid=%d maxRid=%d frozenCounts=%s%n",
                        tag, round,
                        System.currentTimeMillis() - samplingStartMs,
                        efuHere.size(),
                        efuHere.stream().min(Long::compare).orElse(-1L),
                        efuHere.stream().max(Long::compare).orElse(-1L),
                        frozenHere);
            } else if (!frozenHere.isEmpty()) {
                System.out.printf(
                        "[race-%s] TIMELINE round=%d tMs=%d efu=0"
                                + " frozenCounts=%s%n",
                        tag, round,
                        System.currentTimeMillis() - samplingStartMs,
                        frozenHere);
            }
            for (Long rid : efuHere) {
                int[] window = ridWindows.computeIfAbsent(
                        rid, k -> new int[] {Integer.MAX_VALUE, -1});
                window[0] = Math.min(window[0], round);
                window[1] = Math.max(window[1], round);
            }
            rolling.append(efuHere.size()).append(',');
            round++;
        }
        System.out.printf("[race-%s] ROLLING_EFU_PER_ROUND %s%n",
                tag, rolling);
        System.out.printf("[race-%s] RIDWIN total=%d%n",
                tag, ridWindows.size());
        int printed = 0;
        for (Map.Entry<Long, int[]> entry : ridWindows.entrySet()) {
            if (printed++ >= 12) {
                System.out.printf("[race-%s]   ... %d more rid windows%n",
                        tag, ridWindows.size() - 12);
                break;
            }
            Long submit = submitMillis.get(entry.getKey());
            System.out.printf(
                    "[race-%s]   RIDWIN rid=%d firstRound=%d lastRound=%d"
                            + " submitMs=%s%n",
                    tag, entry.getKey(), entry.getValue()[0],
                    entry.getValue()[1], submit);
        }
    }

    private static void dumpRaceSummary(
            String tag,
            List<LedgerReconciliationHarness.LedgerDiff> confirmedReal,
            List<LedgerReconciliationHarness.LedgerDiff> pendingSeen,
            LedgerReconciliationHarness.ReconciliationReport finalReport,
            LedgerReconciliationHarness.ReconciliationReport finalReport2,
            LedgerReconciliationHarness.ReconciliationReport finalReport3,
            Map<Long, Long> submitMillis,
            Map<Long, Long> doneMillis,
            int batches,
            long failures,
            long batchTimeouts,
            long raceSeconds,
            long acceptanceTimeoutMs,
            long requestDeadlineMs,
            AttachedAppender diagnostics) {
        Map<String, TreeMap<Long, String>> pass1 = new TreeMap<>();
        Map<String, TreeMap<Long, String>> pass2 = new TreeMap<>();
        Map<String, TreeMap<Long, String>> pass3 = new TreeMap<>();
        collect(pass1, finalReport.pendingRealDiffs());
        collect(pass1, finalReport.realDiffs());
        collect(pass2, finalReport2.pendingRealDiffs());
        collect(pass2, finalReport2.realDiffs());
        collect(pass3, finalReport3.pendingRealDiffs());
        collect(pass3, finalReport3.realDiffs());

        System.out.printf(
                "[race-%s] acceptance=%dms deadline=%dms duration=%ds"
                        + " batches=%d failures=%d batchTimeouts=%d"
                        + " runtimeConfirmedReal=%d runtimePending=%d"
                        + " quiescedReal=%d quiescedPending=%d"
                        + " quiescedTransient=%d slots=%d decode=%d%n",
                tag, acceptanceTimeoutMs, requestDeadlineMs, raceSeconds,
                batches, failures,
                batchTimeouts, confirmedReal.size(), pendingSeen.size(),
                finalReport.realDiffs().size(),
                finalReport.pendingRealDiffs().size(),
                finalReport.transientDiffs().size(),
                finalReport.slotCount(),
                finalReport.decodeEndpointCount());

        // 冻结永久性：三轮 REAL 规则 rid 集合的稳定性。
        boolean reproduced = false;
        Map<Long, String> frozenRids = new TreeMap<>();
        for (Map.Entry<String, TreeMap<Long, String>> ruleEntry
                : pass2.entrySet()) {
            String rule = ruleEntry.getKey();
            TreeMap<Long, String> rids2 = ruleEntry.getValue();
            boolean frozenFingerprint = FROZEN_RULES.contains(rule);
            if (frozenFingerprint && !rids2.isEmpty()) {
                reproduced = true;
            }
            for (Long frozenRid : rids2.keySet()) {
                frozenRids.put(frozenRid, rule);
            }
        }
        for (String rule : FROZEN_RULES) {
            Set<Long> s1 = pass1.getOrDefault(rule, new TreeMap<>()).keySet();
            Set<Long> s2 = pass2.getOrDefault(rule, new TreeMap<>()).keySet();
            Set<Long> s3 = pass3.getOrDefault(rule, new TreeMap<>()).keySet();
            int stable = 0;
            for (Long rid : s2) {
                if (s1.contains(rid) && s3.contains(rid)) {
                    stable++;
                }
            }
            System.out.printf(
                    "[race-%s] CONVERGENCE %s pass1=%d pass2=%d pass3=%d"
                            + " stableAcrossAllThree=%d%n",
                    tag, rule, s1.size(), s2.size(), s3.size(), stable);
        }

        // 冻结 rid 分布：奇偶 / 直方图 / 提交→完成窗口。
        for (String rule : FROZEN_RULES) {
            TreeMap<Long, String> rids = pass2.get(rule);
            if (rids == null || rids.isEmpty()) {
                continue;
            }
            long odd = rids.keySet().stream()
                    .filter(r -> r % 2 == 1).count();
            long minRid = rids.firstKey();
            long maxRid = rids.lastKey();
            long minSubmit = Long.MAX_VALUE;
            long maxSubmit = Long.MIN_VALUE;
            long minDone = Long.MAX_VALUE;
            long maxDone = Long.MIN_VALUE;
            for (Long frozenRid : rids.keySet()) {
                Long submit = submitMillis.get(frozenRid);
                Long done = doneMillis.get(frozenRid);
                if (submit != null) {
                    minSubmit = Math.min(minSubmit, submit);
                    maxSubmit = Math.max(maxSubmit, submit);
                }
                if (done != null) {
                    minDone = Math.min(minDone, done);
                    maxDone = Math.max(maxDone, done);
                }
            }
            System.out.printf(
                    "[race-%s] RULE %s count=%d odd=%d even=%d"
                            + " ridRange=[%d..%d] submitWindowMs=[%s..%s]"
                            + " doneWindowMs=[%s..%s] gapsInWindow=%d%n",
                    tag, rule, rids.size(), odd, rids.size() - odd,
                    minRid, maxRid,
                    minSubmit == Long.MAX_VALUE ? "n/a" : minSubmit,
                    maxSubmit == Long.MIN_VALUE ? "n/a" : maxSubmit,
                    minDone == Long.MAX_VALUE ? "n/a" : minDone,
                    maxDone == Long.MIN_VALUE ? "n/a" : maxDone,
                    (maxRid - minRid + 1) / 2 - rids.size());
            // rid 直方图：每 500 个 rid 一桶（区分奇偶流）。
            Map<Long, int[]> buckets = new TreeMap<>();
            for (Long frozenRid : rids.keySet()) {
                long bucket = (frozenRid - minRid) / 500;
                int[] counts = buckets.computeIfAbsent(bucket,
                        k -> new int[2]);
                counts[frozenRid % 2 == 0 ? 0 : 1]++;
            }
            for (Map.Entry<Long, int[]> bucket : buckets.entrySet()) {
                System.out.printf(
                        "[race-%s]   HIST rid~%d even=%d odd=%d%n",
                        tag, minRid + bucket.getKey() * 500,
                        bucket.getValue()[0], bucket.getValue()[1]);
            }
            int printed = 0;
            for (Map.Entry<Long, String> diff : rids.entrySet()) {
                if (printed++ >= 10) {
                    System.out.printf(
                            "[race-%s]   ... %d more%n",
                            tag, rids.size() - 10);
                    break;
                }
                System.out.printf(
                        "[race-%s]   rid=%d detail=%s%n",
                        tag, diff.getKey(), diff.getValue());
            }
        }

        // transient 规则分布（撕裂候选的健康面）。
        Map<String, Integer> transientRules = new TreeMap<>();
        for (LedgerReconciliationHarness.LedgerDiff diff
                : finalReport.transientDiffs()) {
            transientRules.merge(diff.rule().name(), 1, Integer::sum);
        }
        System.out.printf("[race-%s] TRANSIENT_RULES %s%n",
                tag, transientRules);

        dumpDiagnosticLogs(tag, diagnostics, frozenRids);
        System.out.printf("[race-%s] REPRODUCED=%b%n", tag, reproduced);
    }

    private static void collect(
            Map<String, TreeMap<Long, String>> byRule,
            List<LedgerReconciliationHarness.LedgerDiff> diffs) {
        for (LedgerReconciliationHarness.LedgerDiff diff : diffs) {
            byRule.computeIfAbsent(
                            diff.rule().name(), k -> new TreeMap<>())
                    .putIfAbsent(diff.requestId(), diff.detail());
        }
    }

    private static void dumpRuleCounts(
            String tag,
            LedgerReconciliationHarness.ReconciliationReport report,
            LedgerReconciliationHarness.ReconciliationReport report2) {
        Map<String, TreeMap<Long, String>> byRule = new LinkedHashMap<>();
        collect(byRule, report.pendingRealDiffs());
        collect(byRule, report.realDiffs());
        collect(byRule, report2.pendingRealDiffs());
        collect(byRule, report2.realDiffs());
        for (Map.Entry<String, TreeMap<Long, String>> entry
                : byRule.entrySet()) {
            System.out.printf(
                    "[race-rule/%s] %s count=%d rids=%s%n",
                    tag, entry.getKey(), entry.getValue().size(),
                    entry.getValue().keySet());
        }
    }

    /** Print WARN/ERROR log lines; flag the ones whose request_id is frozen. */
    private static void dumpDiagnosticLogs(
            String tag,
            AttachedAppender diagnostics,
            Map<Long, String> frozenRids) {
        List<ILoggingEvent> events = diagnostics.appender().list;
        int warnError = 0;
        List<String> frozenRelated = new ArrayList<>();
        for (ILoggingEvent event : events) {
            if (!event.getLevel()
                    .isGreaterOrEqual(ch.qos.logback.classic.Level.WARN)) {
                continue;
            }
            warnError++;
            String message = event.getFormattedMessage();
            if (frozenRids.isEmpty()) {
                continue;
            }
            Matcher matcher = REQUEST_ID_IN_LOG.matcher(message);
            if (matcher.find()) {
                long logRid = Long.parseLong(matcher.group(1));
                String rule = frozenRids.get(logRid);
                if (rule != null) {
                    frozenRelated.add(
                            "rid=" + logRid + " (" + rule + ") "
                                    + event.getLevel() + " "
                                    + event.getLoggerName() + ": " + message);
                }
            }
        }
        System.out.printf(
                "[race-%s] warnErrorLogs=%d frozenRelatedLogs=%d%n",
                tag, warnError, frozenRelated.size());
        for (int i = 0; i < Math.min(frozenRelated.size(), 30); i++) {
            System.out.printf("[race-%s] frozenLog %s%n",
                    tag, frozenRelated.get(i));
        }
    }
}
