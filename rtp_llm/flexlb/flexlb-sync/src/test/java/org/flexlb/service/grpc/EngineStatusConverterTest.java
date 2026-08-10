package org.flexlb.service.grpc;

import org.flexlb.domain.worker.WorkerStatusResponse;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.KvCacheGroupMode;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EngineStatusConverterTest {

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
    void preservesStringRequestIdFromWorkerStatus() {
        String requestId = "c68b72ff-982d-944f-9834-bc0e8bf2f43f";
        EngineRpcService.TaskInfoPB finishedTask = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(requestId)
                .build();
        EngineRpcService.WorkerStatusPB workerStatus = EngineRpcService.WorkerStatusPB.newBuilder()
                .addFinishedTaskList(finishedTask)
                .build();

        WorkerStatusResponse response =
                EngineStatusConverter.convertToWorkerStatusResponse(workerStatus);

        assertEquals(requestId, response.getFinishedTaskInfo().get(requestId).getRequestId());
    }

    @Test
    void preservesPrefixLengthValidityFromWorkerStatus() {
        EngineRpcService.TaskInfoPB runningTask = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId("request-1")
                .setPrefixLength(128)
                .setPrefixLengthValid(true)
                .build();
        EngineRpcService.WorkerStatusPB workerStatus = EngineRpcService.WorkerStatusPB.newBuilder()
                .addRunningTaskInfo(runningTask)
                .build();

        WorkerStatusResponse response =
                EngineStatusConverter.convertToWorkerStatusResponse(workerStatus);

        assertEquals(128, response.getRunningTaskInfo().get("request-1").getPrefixLength());
        assertTrue(response.getRunningTaskInfo().get("request-1").isPrefixLengthValid());
    }

    @Test
    void preservesPrefillTimingAndCacheBreakdownFromWorkerStatus() {
        EngineRpcService.TaskInfoPB finishedTask = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId("request-1")
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

        var task = response.getFinishedTaskInfo().get("request-1");
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
                .setRequestId("request-1")
                .setCompletedPrefillTokens(0)
                .setRemainingPrefillTokens(48_000)
                .setLastCompletedPrefillStepId(0)
                .build();
        EngineRpcService.WorkerStatusPB workerStatus = EngineRpcService.WorkerStatusPB.newBuilder()
                .addRunningTaskInfo(runningTask)
                .build();

        var task = EngineStatusConverter.convertToWorkerStatusResponse(workerStatus)
                .getRunningTaskInfo().get("request-1");

        assertEquals(0, task.getCompletedPrefillTokens());
        assertEquals(48_000, task.getRemainingPrefillTokens());
        assertEquals(0, task.getLastCompletedPrefillStepId());
    }

    @Test
    void keepsMissingRemainingPrefillTokensAsNegativeOne() {
        EngineRpcService.TaskInfoPB runningTask = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId("missing-progress")
                .build();
        EngineRpcService.WorkerStatusPB workerStatus = EngineRpcService.WorkerStatusPB.newBuilder()
                .addRunningTaskInfo(runningTask)
                .build();

        var task = EngineStatusConverter.convertToWorkerStatusResponse(workerStatus)
                .getRunningTaskInfo().get("missing-progress");

        assertEquals(-1, task.getRemainingPrefillTokens());
    }

    @Test
    void preservesExplicitZeroRemainingPrefillTokens() {
        EngineRpcService.TaskInfoPB runningTask = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId("completed-prefill")
                .setRemainingPrefillTokens(0)
                .build();
        EngineRpcService.WorkerStatusPB workerStatus = EngineRpcService.WorkerStatusPB.newBuilder()
                .addRunningTaskInfo(runningTask)
                .build();

        var task = EngineStatusConverter.convertToWorkerStatusResponse(workerStatus)
                .getRunningTaskInfo().get("completed-prefill");

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
}
