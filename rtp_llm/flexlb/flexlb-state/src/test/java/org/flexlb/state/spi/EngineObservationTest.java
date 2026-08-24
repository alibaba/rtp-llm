package org.flexlb.state.spi;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.Test;

/**
 * EngineObservation 值对象契约：防御拷贝 / 不可变性 / 上报完整性 / TaskPhase 静态映射。
 */
class EngineObservationTest {

    /** 测试用端点身份（record 实现 SPI 接口）。 */
    private record TestEndpointRef(long endpointId, StateRole role, long generation)
            implements StateEndpointRef {
    }

    private static EngineObservation.RunningObservation running(long requestId, long version) {
        return new EngineObservation.RunningObservation(
                requestId, StateRole.PREFILL, EnginePhase.RUNNING, 7, 1024, version);
    }

    private static EngineObservation.FinishedObservation finished(long requestId, long version) {
        return new EngineObservation.FinishedObservation(
                requestId, StateRole.PREFILL, 0, 12345L, version);
    }

    /** 防御拷贝：构造后修改源 List 不影响 record 内部状态。 */
    @Test
    void defensiveCopyOnConstruction() {
        List<EngineObservation.RunningObservation> running = new ArrayList<>();
        running.add(running(1, 5));
        List<EngineObservation.FinishedObservation> finished = new ArrayList<>();

        EngineObservation observation = new EngineObservation(
                new TestEndpointRef(3, StateRole.PREFILL, 2), 1, 1000L, 1, running, finished);

        running.add(running(2, 6));
        finished.add(finished(3, 7));

        assertEquals(1, observation.running().size(), "构造后修改源 running 不得影响 record");
        assertEquals(0, observation.finished().size(), "构造后修改源 finished 不得影响 record");
    }

    /** 不可变性：内部 List 不可修改。 */
    @Test
    void internalListsAreImmutable() {
        EngineObservation observation = new EngineObservation(
                new TestEndpointRef(3, StateRole.DECODE, 2), 1, 1000L, 1,
                List.of(running(1, 5)), List.of(finished(2, 5)));

        assertThrows(UnsupportedOperationException.class,
                () -> observation.running().add(running(9, 9)));
        assertThrows(UnsupportedOperationException.class,
                () -> observation.finished().add(finished(9, 9)));
    }

    /** 构造校验：endpointRef / running / finished 不可为 null，detailCount 非负。 */
    @Test
    void constructorRejectsIllegalArguments() {
        TestEndpointRef ref = new TestEndpointRef(1, StateRole.PREFILL, 0);
        assertThrows(NullPointerException.class,
                () -> new EngineObservation(null, 1, 1, 0, List.of(), List.of()));
        assertThrows(NullPointerException.class,
                () -> new EngineObservation(ref, 1, 1, 0, null, List.of()));
        assertThrows(NullPointerException.class,
                () -> new EngineObservation(ref, 1, 1, 0, List.of(), null));
        assertThrows(IllegalArgumentException.class,
                () -> new EngineObservation(ref, 1, 1, -1, List.of(), List.of()));
    }

    /** 上报完整性：detailCount == running.size() 时完整（截断上报不完整）。 */
    @Test
    void completenessFlag() {
        TestEndpointRef ref = new TestEndpointRef(1, StateRole.PREFILL, 0);
        List<EngineObservation.RunningObservation> one = List.of(running(1, 5));

        assertTrue(new EngineObservation(ref, 1, 1, 1, one, List.of()).isComplete(),
                "detailCount == running.size() → 完整");
        assertFalse(new EngineObservation(ref, 1, 1, 3, one, List.of()).isComplete(),
                "detailCount > running.size() → 截断上报，不完整");
    }

    /** TaskPhase（引擎报文值域）→ EnginePhase 静态映射：一一对应，null 拒绝。 */
    @Test
    void taskPhaseMapping() {
        assertSame(EnginePhase.PENDING, EnginePhase.fromTaskPhase(TaskPhase.PENDING));
        assertSame(EnginePhase.RECEIVED, EnginePhase.fromTaskPhase(TaskPhase.RECEIVED));
        assertSame(EnginePhase.KV_ALLOCATED, EnginePhase.fromTaskPhase(TaskPhase.KV_ALLOCATED));
        assertSame(EnginePhase.RUNNING, EnginePhase.fromTaskPhase(TaskPhase.RUNNING));
        assertNull(TaskPhase.fromValue("nonexistent"), "前置确认：未知字符串反序列化为 null");
        assertThrows(NullPointerException.class, () -> EnginePhase.fromTaskPhase(null),
                "无显式相位不得静默吞掉");
    }
}
