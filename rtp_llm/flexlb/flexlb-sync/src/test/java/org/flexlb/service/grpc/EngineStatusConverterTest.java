package org.flexlb.service.grpc;

import com.google.protobuf.CodedOutputStream;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.KvCacheGroupMode;
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EngineStatusConverterTest {

    private final WorkerStatus owner = WorkerStatus.createDiscovered(
            RoleType.PREFILL, "default", "127.0.0.1", 8001, 18002, "test");

    @Test
    void preservesAllActiveTaskPhasesAndKeepsFinishedTasksSeparate() {
        var status = statusBuilder()
                .setRunningQueryLen(2)
                .setWaitingQueryLen(4)
                .addRunningTaskInfo(EngineRpcService.TaskInfoPB.newBuilder()
                        .setRequestId("running").setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING))
                .addRunningTaskInfo(EngineRpcService.TaskInfoPB.newBuilder()
                        .setRequestId("received").setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED))
                .addRunningTaskInfo(EngineRpcService.TaskInfoPB.newBuilder()
                        .setRequestId("allocated").setPhase(EngineRpcService.TaskPhase.TASK_PHASE_KV_ALLOCATED))
                .addRunningTaskInfo(EngineRpcService.TaskInfoPB.newBuilder()
                        .setRequestId("pending").setPhase(EngineRpcService.TaskPhase.TASK_PHASE_PENDING)
                        .setIsWaiting(true))
                .addRunningTaskInfo(EngineRpcService.TaskInfoPB.newBuilder()
                        .setRequestId("legacy-waiting").setIsWaiting(true))
                .addRunningTaskInfo(EngineRpcService.TaskInfoPB.newBuilder()
                        .setRequestId("legacy-running"))
                .addFinishedTaskList(EngineRpcService.TaskInfoPB.newBuilder()
                        .setRequestId("finished").setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING))
                .build();

        var observation = EngineStatusConverter.convertToStatusObservation(owner, status);
        var activeTasks = observation.runningTasks();

        assertEquals(Set.of("running", "received", "allocated", "pending", "legacy-waiting", "legacy-running"),
                activeTasks.keySet());
        assertEquals(TaskPhase.RUNNING, activeTasks.get("running").phase());
        assertEquals(TaskPhase.RECEIVED, activeTasks.get("received").phase());
        assertEquals(TaskPhase.KV_ALLOCATED, activeTasks.get("allocated").phase());
        assertEquals(TaskPhase.PENDING, activeTasks.get("pending").phase());
        assertEquals(TaskPhase.PENDING, activeTasks.get("legacy-waiting").phase());
        assertEquals(TaskPhase.RUNNING, activeTasks.get("legacy-running").phase());
        assertEquals(Set.of("finished"), observation.finishedTasks().keySet());
        assertEquals(2, observation.runningQueryLen());
        assertEquals(4, observation.waitingQueryLen());
    }

    @Test
    void distinctWireStringIdsSurviveRunningAndFinishedTaskConversion() throws Exception {
        var status = statusBuilder().setAlive(true).setStatusVersion(1);
        for (String id : List.of("req-a-p-001", "req-b-p-002", "00123")) {
            var bytes = new ByteArrayOutputStream();
            var wire = CodedOutputStream.newInstance(bytes);
            wire.writeString(1, id);
            wire.writeInt64(3, 3584);
            wire.writeInt64(4, 3912);
            wire.writeBool(16, true);
            wire.flush();
            var task = EngineRpcService.TaskInfoPB.parseFrom(bytes.toByteArray());
            status.addRunningTaskInfo(task).addFinishedTaskList(task);
        }
        var nativeBytes = new ByteArrayOutputStream();
        var nativeWire = CodedOutputStream.newInstance(nativeBytes);
        nativeWire.writeInt64(1, 123);
        nativeWire.writeInt64(3, 3584);
        nativeWire.writeBool(16, true);
        nativeWire.flush();
        var nativeTask = EngineRpcService.TaskInfoPB.parseFrom(nativeBytes.toByteArray());
        status.addRunningTaskInfo(nativeTask).addFinishedTaskList(nativeTask);

        var observation = EngineStatusConverter.convertToStatusObservation(owner, status.build());

        var expected = Set.of("req-a-p-001", "req-b-p-002", "00123", "123");
        assertEquals(expected, observation.runningTasks().keySet());
        assertEquals(expected, observation.finishedTasks().keySet());
        for (var task : observation.finishedTasks().values()) {
            assertEquals(3584, task.prefixLength());
            assertTrue(task.telemetry().prefixLengthValid());
        }
    }

    @Test
    void preservesStepMetricsWireNumbersAndDecodeZeros() throws Exception {
        var bytes = new ByteArrayOutputStream();
        var wire = CodedOutputStream.newInstance(bytes);
        wire.writeInt64(2, 42);
        wire.writeInt64(3, 1700000000000L);
        wire.writeInt64(4, 64);
        wire.writeInt64(5, 0);
        wire.writeInt64(6, 0);
        wire.writeInt64(7, 32000);
        wire.writeDouble(8, 0.002);
        wire.flush();
        var statusBytes = new ByteArrayOutputStream();
        var statusWire = CodedOutputStream.newInstance(statusBytes);
        statusWire.writeByteArray(27, bytes.toByteArray());
        statusWire.flush();
        var status = EngineRpcService.WorkerStatusPB.parseFrom(statusBytes.toByteArray())
                .toBuilder().setRole("RoleType.DECODE").build();

        var observation = EngineStatusConverter.convertToStatusObservation(owner, status);
        var step = observation.engine().lastStepMetrics();

        assertEquals(42, step.stepId());
        assertEquals(1700000000000L, step.completedTimeMs());
        assertEquals(64, step.totalScheduledTokens());
        assertEquals(0, step.prefillRequestCount());
        assertEquals(0, step.prefillTokens());
        assertEquals(32000, step.tokenBudget());
        assertEquals(0.002, step.budgetFillRatio());
        assertNull(EngineStatusConverter.convertToStatusObservation(owner, statusBuilder().build())
                .engine().lastStepMetrics());
    }

    @Test
    void preservesStringTaskIdsAndBatchId() {
        var runningTask = EngineRpcService.TaskInfoPB.newBuilder().setRequestId("123")
                .setBatchId(42).setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING).build();
        var finishedTask = EngineRpcService.TaskInfoPB.newBuilder().setRequestId("456").build();
        var status = statusBuilder().addRunningTaskInfo(runningTask).addFinishedTaskList(finishedTask).build();

        var observation = EngineStatusConverter.convertToStatusObservation(owner, status);

        assertEquals("123", observation.runningTasks().get("123").requestId());
        assertEquals(42, observation.runningTasks().get("123").batchId());
        assertEquals("456", observation.finishedTasks().get("456").requestId());
    }

    @Test
    void convertsKvCacheGroupMode() {
        var status = statusBuilder()
                .setKvCacheGroupMode(EngineRpcService.KvCacheGroupModePB.KV_CACHE_GROUP_MODE_WITH_MAMBA)
                .build();

        var observation = EngineStatusConverter.convertToStatusObservation(owner, status);

        assertEquals(KvCacheGroupMode.WITH_MAMBA, observation.engine().kvCacheGroupMode());
    }

    @Test
    void preservesPrefixLengthValidityFromWorkerStatus() {
        var runningTask = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId("1")
                .setPrefixLength(128)
                .setPrefixLengthValid(true)
                .build();
        var status = statusBuilder().addRunningTaskInfo(runningTask).build();

        var observation = EngineStatusConverter.convertToStatusObservation(owner, status);
        var task = observation.runningTasks().get("1");

        assertEquals(128, task.prefixLength());
        assertTrue(task.telemetry().prefixLengthValid());
    }

    @Test
    void preservesPrefillTimingAndCacheBreakdownFromWorkerStatus() {
        var finishedTask = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId("1")
                .setInputQueueEnqueueTimeMs(1000)
                .setInputQueueDrainTimeMs(1100)
                .setRemoteKvWaitMs(200)
                .setFirstTokenTimeMs(1500)
                .setHbmLocalMatchTokens(512)
                .setRemoteKvAddedMatchTokens(256)
                .setFirstPrefillStepId(7)
                .setLastPrefillStepId(9)
                .setPrefillStepCount(3)
                .setPrefillNonfinalChunkTokensMin(128)
                .setPrefillNonfinalChunkTokensMax(256)
                .build();
        var status = statusBuilder().addFinishedTaskList(finishedTask).build();

        var observation = EngineStatusConverter.convertToStatusObservation(owner, status);
        var telemetry = observation.finishedTasks().get("1").telemetry();

        assertEquals(1000, telemetry.inputQueueEnqueueTimeMs());
        assertEquals(1100, telemetry.inputQueueDrainTimeMs());
        assertEquals(200, telemetry.remoteKvWaitMs());
        assertEquals(1500, telemetry.firstTokenTimeMs());
        assertEquals(512, telemetry.hbmLocalMatchTokens());
        assertEquals(256, telemetry.remoteKvAddedMatchTokens());
        assertEquals(7, telemetry.firstPrefillStepId());
        assertEquals(9, telemetry.lastPrefillStepId());
        assertEquals(3, telemetry.prefillStepCount());
        assertEquals(128, telemetry.prefillNonfinalChunkTokensMin());
        assertEquals(256, telemetry.prefillNonfinalChunkTokensMax());
    }

    @Test
    void preservesCacheMatchMetadataFromWorkerStatus() {
        var status = statusBuilder()
                .setBlockHashLookaheadTokens(1)
                .setCacheMatchRollbackBlocks(1)
                .build();

        var observation = EngineStatusConverter.convertToStatusObservation(owner, status);

        assertEquals(1, observation.engine().blockHashLookaheadTokens());
        assertEquals(1, observation.engine().cacheMatchRollbackBlocks());
    }

    @Test
    void preservesCanonicalWorkerResourceFields() {
        var status = statusBuilder()
                .setDpSize(2)
                .setDpRank(1)
                .setAvailableKvCache(2_000_000)
                .setTotalKvCache(2_100_000)
                .setMaxSeqLen(131_072)
                .setMaxBatchTokensSize(262_144)
                .setBlockSize(1152)
                .setBlockHashLookaheadTokens(1)
                .setKvCacheGroupMode(EngineRpcService.KvCacheGroupModePB.KV_CACHE_GROUP_MODE_WITH_MAMBA)
                .setCacheMatchRollbackBlocks(1)
                .build();

        var observation = EngineStatusConverter.convertToStatusObservation(owner, status);
        var engine = observation.engine();

        assertEquals(2, engine.dpSize());
        assertEquals(1, engine.dpRank());
        assertEquals(2_000_000, engine.availableKvCacheTokens());
        assertEquals(2_100_000, engine.totalKvCacheTokens());
        assertEquals(131_072, engine.maxSeqLen());
        assertEquals(262_144, engine.maxBatchTokensSize());
        assertEquals(1152, engine.blockSize());
        assertEquals(1, engine.blockHashLookaheadTokens());
        assertEquals(KvCacheGroupMode.WITH_MAMBA, engine.kvCacheGroupMode());
        assertEquals(1, engine.cacheMatchRollbackBlocks());
    }

    private static EngineRpcService.WorkerStatusPB.Builder statusBuilder() {
        return EngineRpcService.WorkerStatusPB.newBuilder().setRole("RoleType.PREFILL");
    }
}
