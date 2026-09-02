package org.flexlb.service.grpc;

import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.KvCacheGroupMode;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
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
    void preservesRequestIdFromWorkerStatus() {
        long requestId = 123L;
        EngineRpcService.TaskInfoPB finishedTask = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(requestId)
                .build();
        EngineRpcService.WorkerStatusPB workerStatus = EngineRpcService.WorkerStatusPB.newBuilder()
                .addFinishedTaskList(finishedTask)
                .build();

        WorkerStatusResponse response =
                EngineStatusConverter.convertToWorkerStatusResponse(workerStatus);

        assertEquals(requestId,
                response.getFinishedTaskInfo().get(String.valueOf(requestId)).getRequestId());
    }

    @Test
    void preservesPrefixLengthValidityFromWorkerStatus() {
        EngineRpcService.TaskInfoPB runningTask = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(1L)
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
                .setRequestId(1L)
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
                .setRequestId(1L)
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
                .setRequestId(2L)
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
                .setRequestId(3L)
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
    void remapsKvcmCompatibilityWorkerResourceFields() {
        EngineRpcService.WorkerStatusPB workerStatus = EngineRpcService.WorkerStatusPB.newBuilder()
                // KVCM layout field 16: available_kv_cache.
                .setDpRank(2_076_905)
                // KVCM layout field 17: total_kv_cache.
                .setAvailableKvCache(2_076_905)
                // KVCM layout field 18: block_size.
                .setTotalKvCache(1152)
                .setDpSize(1)
                // KVCM layout fields 21-23: lookahead, group mode, rollback.
                .setMaxSeqLen(1)
                .setMaxBatchTokensSize(2)
                .setBlockSize(1)
                .build();

        WorkerStatusResponse response =
                EngineStatusConverter.convertToWorkerStatusResponse(workerStatus);

        assertEquals(0, response.getDpRank());
        assertEquals(2_076_905, response.getAvailableKvCacheTokens());
        assertEquals(2_076_905, response.getTotalKvCacheTokens());
        assertEquals(1152, response.getCacheStatus().getBlockSize());
        assertEquals(1, response.getBlockHashLookaheadTokens());
        assertEquals(1, response.getCacheMatchRollbackBlocks());
        assertEquals(KvCacheGroupMode.WITH_MAMBA, response.getKvCacheGroupMode());
        assertEquals(0, response.getMaxSeqLen());
        assertEquals(0, response.getMaxBatchTokensSize());
    }

    @Test
    void preservesStandardWorkerResourceFields() {
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
