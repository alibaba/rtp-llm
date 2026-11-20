package org.flexlb.worker.status;

import org.flexlb.config.service.RoleType;
import org.flexlb.worker.rpc.EngineGrpcClient;
import org.flexlb.worker.rpc.WorkerStatusRpcClient;
import org.flexlb.worker.rpc.protocol.EngineRpcService;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.mockito.ArgumentCaptor;

import static org.flexlb.worker.status.WorkerStatusTestSupport.createDiscovered;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;

/**
 * Checks cache-version and Worker-state conversion at the RPC boundary.
 */
class WorkerStatusRpcClientTest {

    @org.junit.jupiter.api.Test
    void detailedCountsRaiseIncompleteSummariesWithoutLoweringReportedOccupancy() {
        var owner = createDiscovered(RoleType.PREFILL, null, "127.0.0.1", 8080, 8081, null);
        var builder =
                EngineRpcService.WorkerStatusPB.newBuilder()
                        .setRunningQueryLen(3)
                        .setWaitingQueryLen(0)
                        .addRunningTaskInfo(
                                EngineRpcService.TaskInfoPB.newBuilder()
                                        .setRequestId(1)
                                        .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING))
                        .addRunningTaskInfo(
                                EngineRpcService.TaskInfoPB.newBuilder()
                                        .setRequestId(2)
                                        .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED))
                        .addRunningTaskInfo(
                                EngineRpcService.TaskInfoPB.newBuilder()
                                        .setRequestId(3)
                                        .setPhase(
                                                EngineRpcService.TaskPhase
                                                        .TASK_PHASE_KV_ALLOCATED));
        var observation = WorkerStatusRpcClient.convertToStatusObservation(owner, builder.build());
        assertEquals(3, observation.engine().runningQueryLen());
        assertEquals(2, observation.engine().waitingQueryLen());
    }

    @org.junit.jupiter.api.Test
    void cacheVersionPreservesItsFullInt64Range() {
        EngineGrpcClient client = mock(EngineGrpcClient.class);
        long version = (long) Integer.MAX_VALUE + 10L;
        new WorkerStatusRpcClient(client)
                .getCacheStatusAsync("127.0.0.1", 8081, version, 100L, RoleType.PREFILL);
        var request = ArgumentCaptor.forClass(EngineRpcService.CacheVersionPB.class);
        verify(client).getCacheStatusAsync(eq("127.0.0.1"), eq(8081), request.capture(), eq(100L));
        assertEquals(version, request.getValue().getLatestCacheVersion());
    }

    @ParameterizedTest
    @CsvSource({"DECODE,false", "PREFILL,true", "PDFUSION,true", "VIT,false"})
    void cacheRequestOnlyFetchesDetailedKeysWhenNeeded(RoleType role, boolean needsCacheKeys) {
        EngineGrpcClient client = mock(EngineGrpcClient.class);
        WorkerStatusRpcClient service = new WorkerStatusRpcClient(client);

        service.getCacheStatusAsync("127.0.0.1", 8081, 7L, 100L, role);

        ArgumentCaptor<EngineRpcService.CacheVersionPB> request =
                ArgumentCaptor.forClass(EngineRpcService.CacheVersionPB.class);
        if (role == RoleType.VIT) {
            verify(client)
                    .getMultimodalCacheStatusAsync(
                            eq("127.0.0.1"), eq(8081), request.capture(), eq(100L));
        } else {
            verify(client)
                    .getCacheStatusAsync(eq("127.0.0.1"), eq(8081), request.capture(), eq(100L));
        }
        verifyNoMoreInteractions(client);
        assertEquals(needsCacheKeys, request.getValue().getNeedCacheKeys());
        assertEquals(7, request.getValue().getLatestCacheVersion());
    }
}
