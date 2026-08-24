package org.flexlb.sync.shadow;

import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.state.StateLedger;
import org.flexlb.state.spi.EngineObservation;
import org.flexlb.state.spi.EnginePhase;
import org.flexlb.state.spi.StateRole;
import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * G1 影子翻译器单测：WorkerStatusResponse → EngineObservation 字段映射、
 * 上报完整性标记（runningDetailCount）、P/D 侧过滤与端点世代缓存。
 */
class WorkerStatusObservationTranslatorTest {

    private final StateLedger ledger = new StateLedger();
    private final WorkerStatusObservationTranslator translator = new WorkerStatusObservationTranslator(ledger);

    @Test
    void shouldMapRunningAndFinishedFields_whenTranslatingDecodeReport() {
        TaskInfo running = new TaskInfo();
        running.setRequestId(101L);
        running.setPhase(TaskPhase.RUNNING);
        running.setBatchId(7L);
        running.setKvTokens(2048L);

        TaskInfo finished = new TaskInfo();
        finished.setRequestId(100L);
        finished.setErrorCode(0L);
        finished.setEndTimeMs(123456789L);

        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setStatusVersion(42L);
        response.setRole(RoleType.DECODE);
        response.setRunningDetailCount(1L);
        response.setRunningTaskInfo(mapOf("100-replaced", running));
        response.setFinishedTaskInfo(mapOf("100", finished));

        EngineObservation observation = translator.translate(response, RoleType.DECODE, "10.0.0.1:9000");

        assertNotNull(observation);
        // 报级字段：round=报级 statusVersion（跨报严格单调，相位裁决矩阵的版本屏障输入）
        assertEquals(42L, observation.round());
        assertEquals(1, observation.detailCount());
        assertTrue(observation.isComplete());
        // 端点身份：endpointId 为 ipPort 稳定哈希，side 按 roleType
        assertEquals(WorkerStatusObservationTranslator.endpointIdOf("10.0.0.1:9000"),
                observation.endpointRef().endpointId());
        assertEquals(StateRole.DECODE, observation.endpointRef().role());

        assertEquals(1, observation.running().size());
        EngineObservation.RunningObservation r = observation.running().get(0);
        assertEquals(101L, r.requestId());
        assertEquals(StateRole.DECODE, r.side());
        assertEquals(EnginePhase.RUNNING, r.enginePhase());
        assertEquals(7L, r.batchId());
        assertEquals(2048L, r.kvTokens());
        assertEquals(42L, r.version());

        assertEquals(1, observation.finished().size());
        EngineObservation.FinishedObservation f = observation.finished().get(0);
        assertEquals(100L, f.requestId());
        assertEquals(StateRole.DECODE, f.side());
        assertEquals(0, f.errorCode());
        assertEquals(123456789L, f.endTimeMs());
        assertEquals(42L, f.version());
    }

    @Test
    void shouldMarkIncomplete_whenOldEngineReportsZeroDetailCountWithRunningTasks() {
        // 旧引擎未填 runningDetailCount（0）而 running 非空 → 按上报完整性判不完整
        TaskInfo running = new TaskInfo();
        running.setRequestId(1L);
        running.setPhase(TaskPhase.RECEIVED);

        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setStatusVersion(1L);
        response.setRole(RoleType.PREFILL);
        response.setRunningDetailCount(0L);
        response.setRunningTaskInfo(mapOf("1", running));

        EngineObservation observation = translator.translate(response, RoleType.PREFILL, "10.0.0.2:9000");

        assertNotNull(observation);
        assertEquals(0, observation.detailCount());
        assertEquals(1, observation.running().size());
        assertEquals(false, observation.isComplete(), "runningDetailCount=0 且 running 非空应判不完整（旧引擎）");
    }

    @Test
    void shouldReturnNull_whenRoleIsNotPrefillOrDecode() {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setStatusVersion(1L);
        response.setRole(RoleType.PDFUSION);

        assertNull(translator.translate(response, RoleType.PDFUSION, "10.0.0.3:9000"),
                "PDFUSION/VIT 融合模式影子挂载（G1）不覆盖，返回 null");
    }

    @Test
    void shouldFallbackToPending_whenPhaseIsNull() {
        // 相位缺失保守倒推：旧引擎未报相位 → 保守 PENDING，不丢弃明细
        TaskInfo running = new TaskInfo();
        running.setRequestId(2L);
        running.setPhase(null);

        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setStatusVersion(1L);
        response.setRole(RoleType.DECODE);
        response.setRunningDetailCount(1L);
        response.setRunningTaskInfo(mapOf("2", running));

        EngineObservation observation = translator.translate(response, RoleType.DECODE, "10.0.0.4:9000");

        assertNotNull(observation);
        assertEquals(EnginePhase.PENDING, observation.running().get(0).enginePhase());
    }

    @Test
    void shouldTolerateNullMapsAndNullStatusVersion() {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.DECODE);
        // statusVersion/running/finished 均未设置（null/默认 0）

        EngineObservation observation = translator.translate(response, RoleType.DECODE, "10.0.0.5:9000");

        assertNotNull(observation);
        assertEquals(0L, observation.round());
        assertTrue(observation.running().isEmpty());
        assertTrue(observation.finished().isEmpty());
        assertTrue(observation.isComplete(), "空报（0=0）完整");
    }

    @Test
    void shouldRegisterEndpointOnceAndReuseGeneration_forSameIpPort() {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setStatusVersion(1L);
        response.setRole(RoleType.DECODE);

        EngineObservation first = translator.translate(response, RoleType.DECODE, "10.0.0.6:9000");
        EngineObservation second = translator.translate(response, RoleType.DECODE, "10.0.0.6:9000");

        // 同 role:ipPort：端点缓存命中，世代稳定（首次 newGeneration 注册后复用）
        assertEquals(first.endpointRef().endpointId(), second.endpointRef().endpointId());
        assertEquals(first.endpointRef().generation(), second.endpointRef().generation());

        // 不同 role 同 ipPort：独立端点（P/D 两侧分开注册）
        EngineObservation prefillSide = translator.translate(response, RoleType.PREFILL, "10.0.0.6:9000");
        assertEquals(StateRole.PREFILL, prefillSide.endpointRef().role());
        assertEquals(first.endpointRef().endpointId(), prefillSide.endpointRef().endpointId(),
                "endpointId 只由 ipPort 决定（跨 role 稳定）");
    }

    @Test
    void shouldLazilyRegisterEndpointAndReuseSameBinding_whenBindingRequested() {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setStatusVersion(1L);
        response.setRole(RoleType.DECODE);

        // 惰性注册：未见过的端点直接注册并返回绑定（不依赖事件泵先到）
        WorkerStatusObservationTranslator.GenerationTripleLike binding =
                translator.bindingOf(RoleType.DECODE, "10.0.0.7:9000");
        assertNotNull(binding);

        // 后续 translate 复用同一端点世代（缓存命中）
        EngineObservation observation = translator.translate(response, RoleType.DECODE, "10.0.0.7:9000");
        assertNotNull(observation);
        assertEquals(binding.endpointId(), observation.endpointRef().endpointId());
        assertEquals(binding.generation(), observation.endpointRef().generation());

        // 再次 bindingOf 仍稳定
        WorkerStatusObservationTranslator.GenerationTripleLike again =
                translator.bindingOf(RoleType.DECODE, "10.0.0.7:9000");
        assertEquals(binding.endpointId(), again.endpointId());
        assertEquals(binding.generation(), again.generation());
    }

    @Test
    void shouldFeedLedgerViaObservation_whenLedgerConsumesTranslatedReport() {
        // 端到端冒烟：翻译结果可被 StateLedger.observe 消费（契约兼容性）。
        // 注意开账前置：正常 observe 不收养未开条目，先本地 reserve 开 D 账。
        WorkerStatusObservationTranslator.GenerationTripleLike binding =
                translator.bindingOf(RoleType.DECODE, "10.0.0.8:9000");
        assertNotNull(binding);
        ledger.decode().reserve(9L, 100L, 200L,
                new org.flexlb.state.GenerationTriple((int) binding.endpointId(), binding.generation(), -1L));

        TaskInfo running = new TaskInfo();
        running.setRequestId(9L);
        running.setPhase(TaskPhase.RUNNING);
        running.setKvTokens(512L);

        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setStatusVersion(5L);
        response.setRole(RoleType.DECODE);
        response.setRunningDetailCount(1L);
        response.setRunningTaskInfo(mapOf("9", running));

        EngineObservation observation = translator.translate(response, RoleType.DECODE, "10.0.0.8:9000");
        ledger.observe(observation);

        // 已开账条目收到 running 观察 → 相位/版本推进（引擎上报观察账更新）
        assertTrue(ledger.decode().get(9L).isPresent(), "D 侧已开账条目应保留并推进");
        assertSame(StateRole.DECODE, observation.running().get(0).side());
    }

    private static Map<String, TaskInfo> mapOf(String key, TaskInfo info) {
        Map<String, TaskInfo> map = new LinkedHashMap<>();
        map.put(key, info);
        return map;
    }
}
