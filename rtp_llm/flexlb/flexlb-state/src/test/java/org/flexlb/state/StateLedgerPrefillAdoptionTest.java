package org.flexlb.state;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.flexlb.state.spi.EnginePhase;
import org.flexlb.state.spi.StateRole;
import org.junit.jupiter.api.Test;

/**
 * P 侧引擎收养三段语义（master 重启首报窗口的收养路径回归——真机
 * master 重启演练发现的 per-EP 计数簿漂移修复验收）：
 *
 * <ol>
 *   <li><b>首报三相位入账口径</b>：未知 running 首报按引擎观察相位收养
 *       （RECEIVED→P_RECEIVED / KV_ALLOCATED→P_WAITING_LOADED /
 *       RUNNING→P_RUNNING），全局账与端点簿的相位人口、engineOwned、
 *       活跃总数全部按收养相位精确入账。</li>
 *   <li><b>收养 → 观察 → 终局归零</b>：收养条目经后续引擎观察推进相位、
 *       finished 终局移除——全局账与端点簿全字段随出账对称归零，无悬挂。</li>
 *   <li><b>迟到 running 不复活已终局请求</b>：本地终局事实（墓碑）优先于
 *       引擎收养——迟到 running 被墓碑吸收，不产生新条目、不产生任何计数
 *       （复活条目无后续出账路径，收养入账将成为永久悬挂漂移）。</li>
 * </ol>
 */
class StateLedgerPrefillAdoptionTest {

    /** 观察时刻基准：真实墙钟（墓碑过期判定用真实时钟，假时间戳会瞬间“过期”）。 */
    private static final long NOW = System.currentTimeMillis();

    /**
     * 首报三相位入账：一次 observeAdopting 携带三个未知 requestId（各处
     * 不同引擎观察相位）——收养入账按收养相位落账，全局账与端点簿
     * 逐字段一致，且收养条目语义（batchId=-1、binding=观察端点世代、
     * 引擎事实 KV 随条目）正确。
     */
    @Test
    void firstReportThreePhasesBookedOnGlobalAndEndpointBooks() {
        StateLedger ledger = new StateLedger();
        TestEndpoints.Endpoint pEp = TestEndpoints.ep(1L, StateRole.PREFILL, 5_000L);

        ledger.observeAdopting(TestEndpoints.observation(pEp, 1L, NOW, List.of(
                TestEndpoints.running(301L, StateRole.PREFILL, EnginePhase.RECEIVED, -1L, 0L, 1L),
                TestEndpoints.running(302L, StateRole.PREFILL, EnginePhase.KV_ALLOCATED, -1L, 512L, 2L),
                TestEndpoints.running(303L, StateRole.PREFILL, EnginePhase.RUNNING, -1L, 1_024L, 3L)),
                List.of()));

        // 端点簿：收养即引擎事实——三相位全部 engineOwned、活跃合计 3；
        // 收养条目无批次预测历史（等待估算计 0）
        PrefillEndpointCounters book = ledger.prefill().endpointCounters(1);
        assertEquals(3, book.activeTotal());
        assertEquals(3, book.engineOwnedCount());
        assertEquals(0L, book.estimatedWaitMs());
        assertEquals(1L, book.phaseCounts().get(5), "P_RECEIVED 收养相位人口");
        assertEquals(1L, book.phaseCounts().get(7), "P_WAITING_LOADED 收养相位人口");
        assertEquals(1L, book.phaseCounts().get(8), "P_RUNNING 收养相位人口");

        // 全局账同口径（refresh 后精确）
        ledger.prefill().refreshSnapshot();
        PrefillCounterSnapshot global = ledger.prefill().snapshot();
        assertEquals(3L, global.inflight());
        assertEquals(3L, global.engineOwned());
        assertEquals(1L, global.phaseCounts().get(5));
        assertEquals(1L, global.phaseCounts().get(7));
        assertEquals(1L, global.phaseCounts().get(8));

        // 收养条目语义：散请求、绑定观察端点世代、引擎事实 KV 随条目
        PrefillRequestStateView v = ledger.prefill().get(303L).orElseThrow();
        assertTrue(v.engineOwned());
        assertEquals(-1L, v.batchId());
        assertEquals(new GenerationTriple(1, 5_000L, -1L), v.binding());
        assertEquals(1_024L, v.kvTokensReported());

        assertTrue(ledger.auditAndDrift().clean(), () -> ledger.auditAndDrift().toString());
    }

    /**
     * 收养 → 观察 → 终局全链路归零：收养（KV 装载位）→ 正常观察推进到
     * 执行位（桶迁移账：收养相位出账、执行相位入账）→ 引擎 finished
     * 终局——全局账与端点簿全字段随移除对称出账归零，墓碑恰一条。
     */
    @Test
    void adoptionThenObserveThenSettleRetiresAllBooks() {
        StateLedger ledger = new StateLedger();
        TestEndpoints.Endpoint pEp = TestEndpoints.ep(1L, StateRole.PREFILL, 5_000L);

        ledger.observeAdopting(TestEndpoints.runningOnly(pEp, 1L, NOW,
                TestEndpoints.running(401L, StateRole.PREFILL, EnginePhase.KV_ALLOCATED, -1L, 512L, 1L)));
        PrefillEndpointCounters adopted = ledger.prefill().endpointCounters(1);
        assertEquals(1, adopted.activeTotal());
        assertEquals(1, adopted.engineOwnedCount());
        assertEquals(1L, adopted.phaseCounts().get(7), "收养相位人口入账");

        // 后续正常观察：推进到执行位（入账以收养相位为迁移 from——任意
        // 交错下配平），引擎事实 KV 随观察刷新
        ledger.observe(TestEndpoints.runningOnly(pEp, 2L, NOW + 100L,
                TestEndpoints.running(401L, StateRole.PREFILL, EnginePhase.RUNNING, -1L, 1_024L, 2L)));
        PrefillEndpointCounters advanced = ledger.prefill().endpointCounters(1);
        assertEquals(1, advanced.activeTotal());
        assertEquals(0L, advanced.phaseCounts().get(7), "桶迁移账：收养相位人口随推进出账");
        assertEquals(1L, advanced.phaseCounts().get(8));
        assertEquals(1_024L, ledger.prefill().get(401L).orElseThrow().kvTokensReported());

        // 引擎 finished 终局：端点簿与全局账全字段归零
        ledger.observe(TestEndpoints.finishedOnly(pEp, 3L, NOW + 200L,
                TestEndpoints.finished(401L, StateRole.PREFILL, 0, NOW + 200L, 3L)));
        PrefillEndpointCounters retired = ledger.prefill().endpointCounters(1);
        assertEquals(0, retired.activeTotal(), "活跃随终局出账归零");
        assertEquals(0, retired.engineOwnedCount());
        assertEquals(0L, retired.estimatedWaitMs());
        for (int p = 0; p < retired.phaseCounts().size(); p++) {
            assertEquals(0L, retired.phaseCounts().get(p), "phase[" + p + "] 归零");
        }
        ledger.prefill().refreshSnapshot();
        assertEquals(0L, ledger.prefill().snapshot().inflight());
        assertEquals(0L, ledger.prefill().snapshot().engineOwned());
        assertEquals(1L, ledger.snapshot().prefillTombstones());
        assertTrue(ledger.prefill().get(401L).isEmpty());
        assertTrue(ledger.auditAndDrift().clean(), () -> ledger.auditAndDrift().toString());
    }

    /**
     * 迟到 running 不复活已终局请求（墓碑优先）：收养条目终局移除后，
     * 迟到的收养观察（observeAdopting 重放 running）必须被墓碑吸收——
     * 不新建条目、不产生任何计数；后续迟到 finished 同样吸收，
     * 墓碑不增殖。若复活成立，收养入账（活跃/首见/收养相位三账各 +1）
     * 将无出账路径而永久悬挂。
     */
    @Test
    void lateRunningAfterTerminalIsAbsorbedByTombstoneNotResurrected() {
        StateLedger ledger = new StateLedger();
        TestEndpoints.Endpoint pEp = TestEndpoints.ep(1L, StateRole.PREFILL, 5_000L);

        ledger.observeAdopting(TestEndpoints.runningOnly(pEp, 1L, NOW,
                TestEndpoints.running(501L, StateRole.PREFILL, EnginePhase.RUNNING, -1L, 256L, 1L)));
        ledger.observe(TestEndpoints.finishedOnly(pEp, 2L, NOW + 100L,
                TestEndpoints.finished(501L, StateRole.PREFILL, 0, NOW + 100L, 2L)));
        assertEquals(0, ledger.prefill().endpointCounters(1).activeTotal());

        // 迟到 running（收养观察重放，低相位高版本）：墓碑优先——不复活
        ledger.observeAdopting(TestEndpoints.runningOnly(pEp, 3L, NOW + 200L,
                TestEndpoints.running(501L, StateRole.PREFILL, EnginePhase.RECEIVED, -1L, 999L, 3L)));
        assertTrue(ledger.prefill().get(501L).isEmpty(), "迟到 running 不得复活已终局请求");
        PrefillEndpointCounters absorbed = ledger.prefill().endpointCounters(1);
        assertEquals(0, absorbed.activeTotal(), "复活若发生，收养入账将永久悬挂（活跃恒高 1）");
        assertEquals(0, absorbed.engineOwnedCount());
        assertEquals(0L, absorbed.phaseCounts().get(5));
        assertEquals(0L, absorbed.phaseCounts().get(8));

        // 再次迟到 running（高相位）与迟到 finished：均被墓碑吸收（无新条目、无计数）
        ledger.observeAdopting(TestEndpoints.runningOnly(pEp, 4L, NOW + 300L,
                TestEndpoints.running(501L, StateRole.PREFILL, EnginePhase.RUNNING, -1L, 999L, 4L)));
        ledger.observe(TestEndpoints.finishedOnly(pEp, 5L, NOW + 400L,
                TestEndpoints.finished(501L, StateRole.PREFILL, 0, NOW + 400L, 5L)));

        LedgerSnapshot s = ledger.snapshot();
        assertEquals(3L, s.lateEventsAbsorbed(), "三次迟到事件全部被墓碑吸收");
        assertEquals(1L, s.prefillTombstones(), "墓碑不随迟到事件增殖");
        assertTrue(ledger.prefill().get(501L).isEmpty());
        assertTrue(ledger.auditAndDrift().clean(), () -> ledger.auditAndDrift().toString());
    }
}
