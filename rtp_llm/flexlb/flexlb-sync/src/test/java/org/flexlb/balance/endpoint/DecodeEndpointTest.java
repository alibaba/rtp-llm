package org.flexlb.balance.endpoint;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DecodeEndpointTest {

    private WorkerStatus status;
    private DecodeEndpoint endpoint;

    @BeforeEach
    void setUp() {
        status = new WorkerStatus();
        status.setIp("10.0.0.1");
        status.setPort(8080);
        status.setGrpcPort(8081);
        endpoint = new DecodeEndpoint(status);
    }

    @Test
    void reserve_updatesSnapshotAndInflight() {
        updateStatus(null, null, 10000);
        endpoint.reserve(100L, 500);
        assertEquals(1, endpoint.getInflightCount());
        assertEquals(9500, endpoint.realKvAvailable());
    }

    @Test
    void release_decrementsInflight() {
        endpoint.reserve(100L, 500);
        endpoint.reserve(101L, 300);
        endpoint.release(100L);

        assertEquals(1, endpoint.getInflightCount());
    }

    @Test
    void release_unknownRequestId_noEffect() {
        endpoint.reserve(100L, 500);
        endpoint.release(999L);
        assertEquals(1, endpoint.getInflightCount());
    }

    @Test
    void release_neverGoesNegative() {
        endpoint.reserve(100L, 100);
        endpoint.release(100L);
        endpoint.release(100L);
        assertEquals(0, endpoint.getInflightCount());
        assertEquals(0, endpoint.realKvAvailable());
    }

    @Test
    void duplicateLiveRequestDoesNotReplaceExactReservationOwner() {
        updateStatus(null, null, 10_000);

        DecodeEndpoint.Lease owner = endpoint.reserve(100L, 500);

        assertThrows(IllegalStateException.class,
                () -> endpoint.reserve(100L, 300));
        assertSame(owner, endpoint.leaseFor(100L));
        assertEquals(1, endpoint.getInflightCount());
        assertEquals(9_500, endpoint.realKvAvailable());
    }

    @Test
    void staleLeaseCannotReleaseReplacementWithSameRequestId() {
        updateStatus(null, null, 10_000);
        DecodeEndpoint.Lease stale = endpoint.reserve(100L, 500);
        stale.release();
        DecodeEndpoint.Lease replacement = endpoint.reserve(100L, 300);

        stale.release();

        assertSame(replacement, endpoint.leaseFor(100L));
        assertEquals(1, endpoint.getInflightCount());
        assertEquals(9_700, endpoint.realKvAvailable());
        replacement.release();
        assertEquals(0, endpoint.getInflightCount());
        assertEquals(10_000, endpoint.realKvAvailable());
    }

    @Test
    void leaseLookupRequiresExactRouteOwner() {
        BalanceContext firstRoute = new BalanceContext();
        BalanceContext replacementRoute = new BalanceContext();

        DecodeEndpoint.Lease lease = endpoint.reserve(100L, 500, firstRoute);

        assertSame(lease, endpoint.leaseFor(100L, firstRoute));
        assertNull(endpoint.leaseFor(100L, replacementRoute));
    }

    @Test
    void calibrate_kvAllocatedReleasesFromInflight() {
        reserveBound(100L, 500, 10L);

        TaskInfo running = task(100L);
        running.setBatchId(10L);
        running.setPhase(TaskPhase.KV_ALLOCATED);
        updateStatus(Map.of("100", running), null, 10000);

        assertEquals(0, endpoint.getInflightCount());
        assertEquals(10000, endpoint.realKvAvailable());
    }

    @Test
    void calibrate_finishedFailureReleasesFromInflight() {
        reserveBound(100L, 500, 11L);

        TaskInfo failed = task(100L);
        failed.setBatchId(11L);
        failed.setErrorCode(1);
        failed.setErrorMessage("timeout");
        updateStatus(null, Map.of("100", failed), 10000);

        assertEquals(0, endpoint.getInflightCount());
    }

    @Test
    void calibrate_finishedSuccessReleasesIfStillPresent() {
        reserveBound(100L, 500, 12L);

        TaskInfo success = task(100L);
        success.setBatchId(12L);
        success.setErrorCode(0);
        updateStatus(null, Map.of("100", success), 10000);

        assertEquals(0, endpoint.getInflightCount());
    }

    @Test
    void staleWorkerBatchCannotReleaseReplacementReservation() {
        DecodeEndpoint.Lease stale = reserveBound(100L, 500, 20L);
        stale.release();
        DecodeEndpoint.Lease replacement = reserveBound(100L, 300, 21L);

        TaskInfo staleFinished = task(100L);
        staleFinished.setBatchId(20L);
        updateStatus(null, Map.of("100", staleFinished), 10_000);

        assertSame(replacement, endpoint.leaseFor(100L));
        assertEquals(1, endpoint.getInflightCount());
    }

    @Test
    void workerStatusWithoutPositiveBatchIdCannotReleaseReservation() {
        DecodeEndpoint.Lease reservation = reserveBound(100L, 500, 22L);

        TaskInfo legacyFinished = task(100L);
        legacyFinished.setBatchId(0L);
        updateStatus(null, Map.of("100", legacyFinished), 10_000);

        assertSame(reservation, endpoint.leaseFor(100L));
        assertEquals(1, endpoint.getInflightCount());
    }

    @Test
    void bindingBatchRefreshesReservationTtl() throws Exception {
        DecodeEndpoint.Lease lease = endpoint.reserve(100L, 500);
        java.lang.reflect.Field ttlBase = lease.getClass().getDeclaredField("ttlBaseAtMs");
        ttlBase.setAccessible(true);
        ttlBase.setLong(lease, System.currentTimeMillis() - 100_000L);

        assertTrue(lease.bindBatch(23L));

        assertEquals(0, endpoint.evictExpiredRequests(60_000L));
        assertSame(lease, endpoint.leaseFor(100L));
    }

    @Test
    void calibrate_updatesReportedKvAvailable() {
        endpoint.reserve(100L, 500);
        updateStatus(null, null, 10000);

        assertEquals(9500, endpoint.realKvAvailable());
    }

    @Test
    void availableKvTokens_accountsForReservations() {
        updateStatus(null, null, 10000);

        endpoint.reserve(100L, 3000);
        endpoint.reserve(101L, 2000);

        assertEquals(5000, endpoint.realKvAvailable());
    }

    @Test
    void ipPort_format() {
        assertEquals("10.0.0.1:8080", endpoint.ipPort());
    }

    private void updateStatus(Map<String, TaskInfo> running, Map<String, TaskInfo> finished,
                              long availableKvCacheTokens) {
        status.getAvailableKvCacheTokens().set(availableKvCacheTokens);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(running);
        response.setFinishedTaskInfo(finished);
        endpoint.onWorkerStatusUpdate(status, response);
    }

    private TaskInfo task(long requestId) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        return task;
    }

    private DecodeEndpoint.Lease reserveBound(long requestId, long kvTokens,
                                               long batchId) {
        DecodeEndpoint.Lease lease = endpoint.reserve(requestId, kvTokens);
        assertTrue(lease.bindBatch(batchId));
        return lease;
    }
}
