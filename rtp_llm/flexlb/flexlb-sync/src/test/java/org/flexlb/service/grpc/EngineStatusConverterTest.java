package org.flexlb.service.grpc;

import com.google.protobuf.CodedOutputStream;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.KvCacheGroupMode;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EngineStatusConverterTest {

    @Test
    void convertsOldIntegerAndNewStringTaskIdsWithoutChangingOtherFields() throws Exception {
        var bytes = new ByteArrayOutputStream();
        var wire = CodedOutputStream.newInstance(bytes);
        wire.writeInt64(1, 123);
        wire.flush();
        var oldTask = EngineRpcService.TaskInfoPB.parseFrom(bytes.toByteArray()).toBuilder()
                .setBatchId(42).setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING).build();
        var newTask = EngineRpcService.TaskInfoPB.newBuilder().setRequestId("req-abc-001").build();
        var response = EngineStatusConverter.convertToWorkerStatusResponse(EngineRpcService.WorkerStatusPB.newBuilder()
                .addRunningTaskInfo(oldTask).addFinishedTaskList(newTask).build());
        assertEquals("123", response.getRunningTaskInfo().get("123").getRequestId());
        assertEquals(42, response.getRunningTaskInfo().get("123").getBatchId());
        assertEquals("req-abc-001", response.getFinishedTaskInfo().get("req-abc-001").getRequestId());
    }

    @Test
    void convertsKvCacheGroupMode() {
        EngineRpcService.WorkerStatusPB workerStatus = EngineRpcService.WorkerStatusPB.newBuilder()
                .setKvCacheGroupMode(
                        EngineRpcService.KvCacheGroupModePB.KV_CACHE_GROUP_MODE_WITH_MAMBA)
                .build();

        WorkerStatusResponse response =
                EngineStatusConverter.convertToWorkerStatusResponse(workerStatus);

        assertEquals(KvCacheGroupMode.WITH_MAMBA, response.getKvCacheGroupMode());
    }

    @Test
    void leavesCacheStatusEmptyWhenWorkerDoesNotReportBlockSize() {
        EngineRpcService.WorkerStatusPB workerStatus =
                EngineRpcService.WorkerStatusPB.newBuilder().build();

        WorkerStatusResponse response =
                EngineStatusConverter.convertToWorkerStatusResponse(workerStatus);

        assertNull(response.getCacheStatus());
    }

    @Test
    void preservesRequestIdFromWorkerStatus() {
        long requestId = 123L;
        EngineRpcService.TaskInfoPB finishedTask = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(String.valueOf(requestId))
                .build();
        EngineRpcService.WorkerStatusPB workerStatus = EngineRpcService.WorkerStatusPB.newBuilder()
                .addFinishedTaskList(finishedTask)
                .build();

        WorkerStatusResponse response =
                EngineStatusConverter.convertToWorkerStatusResponse(workerStatus);

        assertEquals(String.valueOf(requestId),
                response.getFinishedTaskInfo().get(String.valueOf(requestId)).getRequestId());
    }

    @Test
    void preservesPrefixLengthValidityFromWorkerStatus() {
        EngineRpcService.TaskInfoPB runningTask = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(String.valueOf(1L))
                .setPrefixLength(128)
                .setPrefixLengthValid(true)
                .build();
        EngineRpcService.WorkerStatusPB workerStatus = EngineRpcService.WorkerStatusPB.newBuilder()
                .addRunningTaskInfo(runningTask)
                .build();

        WorkerStatusResponse response =
                EngineStatusConverter.convertToWorkerStatusResponse(workerStatus);

        assertEquals(128, response.getRunningTaskInfo().get("1").getPrefixLength());
        assertTrue(response.getRunningTaskInfo().get("1").isPrefixLengthValid());
    }

    @Test
    void preservesPrefillTimingAndCacheBreakdownFromWorkerStatus() {
        EngineRpcService.TaskInfoPB finishedTask = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(String.valueOf(1L))
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
        EngineRpcService.WorkerStatusPB workerStatus = EngineRpcService.WorkerStatusPB.newBuilder()
                .addFinishedTaskList(finishedTask)
                .build();

        WorkerStatusResponse response =
                EngineStatusConverter.convertToWorkerStatusResponse(workerStatus);

        var task = response.getFinishedTaskInfo().get("1");
        assertEquals(1000, task.getInputQueueEnqueueTimeMs());
        assertEquals(1100, task.getInputQueueDrainTimeMs());
        assertEquals(200, task.getRemoteKvWaitMs());
        assertEquals(1500, task.getFirstTokenTimeMs());
        assertEquals(512, task.getHbmLocalMatchTokens());
        assertEquals(256, task.getRemoteKvAddedMatchTokens());
        assertEquals(7, task.getFirstPrefillStepId());
        assertEquals(9, task.getLastPrefillStepId());
        assertEquals(3, task.getPrefillStepCount());
        assertEquals(128, task.getPrefillNonfinalChunkTokensMin());
        assertEquals(256, task.getPrefillNonfinalChunkTokensMax());
    }

    @Test
    void preservesPostForwardPrefillProgressWithPresence() {
        EngineRpcService.TaskInfoPB runningTask = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(String.valueOf(1L))
                .setCompletedPrefillTokens(0)
                .setRemainingPrefillTokens(48_000)
                .setLastCompletedPrefillStepId(0)
                .build();
        EngineRpcService.WorkerStatusPB workerStatus = EngineRpcService.WorkerStatusPB.newBuilder()
                .addRunningTaskInfo(runningTask)
                .build();

        var task = EngineStatusConverter.convertToWorkerStatusResponse(workerStatus)
                .getRunningTaskInfo().get("1");

        assertEquals(0, task.getCompletedPrefillTokens());
        assertEquals(48_000, task.getRemainingPrefillTokens());
        assertEquals(0, task.getLastCompletedPrefillStepId());
    }

    @Test
    void keepsMissingRemainingPrefillTokensAsNegativeOne() {
        EngineRpcService.TaskInfoPB runningTask = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(String.valueOf(2L))
                .build();
        EngineRpcService.WorkerStatusPB workerStatus = EngineRpcService.WorkerStatusPB.newBuilder()
                .addRunningTaskInfo(runningTask)
                .build();

        var task = EngineStatusConverter.convertToWorkerStatusResponse(workerStatus)
                .getRunningTaskInfo().get("2");

        assertEquals(-1, task.getRemainingPrefillTokens());
    }

    @Test
    void preservesExplicitZeroRemainingPrefillTokens() {
        EngineRpcService.TaskInfoPB runningTask = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(String.valueOf(3L))
                .setRemainingPrefillTokens(0)
                .build();
        EngineRpcService.WorkerStatusPB workerStatus = EngineRpcService.WorkerStatusPB.newBuilder()
                .addRunningTaskInfo(runningTask)
                .build();

        var task = EngineStatusConverter.convertToWorkerStatusResponse(workerStatus)
                .getRunningTaskInfo().get("3");

        assertEquals(0, task.getRemainingPrefillTokens());
    }

    @Test
    void preservesCacheMatchMetadataFromWorkerStatus() {
        EngineRpcService.WorkerStatusPB workerStatus = EngineRpcService.WorkerStatusPB.newBuilder()
                .setBlockHashLookaheadTokens(1)
                .setCacheMatchRollbackBlocks(1)
                .build();

        WorkerStatusResponse response =
                EngineStatusConverter.convertToWorkerStatusResponse(workerStatus);

        assertEquals(1, response.getBlockHashLookaheadTokens());
        assertEquals(1, response.getCacheMatchRollbackBlocks());
    }

    @Test
    void preservesCanonicalWorkerResourceFields() {
        EngineRpcService.WorkerStatusPB workerStatus = EngineRpcService.WorkerStatusPB.newBuilder()
                .setDpSize(2)
                .setDpRank(1)
                .setAvailableKvCache(2_000_000)
                .setTotalKvCache(2_100_000)
                .setMaxSeqLen(131_072)
                .setMaxBatchTokensSize(262_144)
                .setBlockSize(1152)
                .setBlockHashLookaheadTokens(1)
                .setKvCacheGroupMode(
                        EngineRpcService.KvCacheGroupModePB.KV_CACHE_GROUP_MODE_WITH_MAMBA)
                .setCacheMatchRollbackBlocks(1)
                .build();

        WorkerStatusResponse response =
                EngineStatusConverter.convertToWorkerStatusResponse(workerStatus);

        assertEquals(1, response.getDpRank());
        assertEquals(2_000_000, response.getAvailableKvCacheTokens());
        assertEquals(2_100_000, response.getTotalKvCacheTokens());
        assertEquals(131_072, response.getMaxSeqLen());
        assertEquals(262_144, response.getMaxBatchTokensSize());
        assertEquals(1152, response.getCacheStatus().getBlockSize());
        assertEquals(1, response.getBlockHashLookaheadTokens());
        assertEquals(KvCacheGroupMode.WITH_MAMBA, response.getKvCacheGroupMode());
        assertEquals(1, response.getCacheMatchRollbackBlocks());
    }
}
