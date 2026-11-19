package org.flexlb.service.lease;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.PrefillChunk;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.runner.RunnerTestSupport;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class PrefillLeaseManagerTest {
    @Test
    void revocationSerializesWithChunkReservationAndDoesNotCancelReservedRequest() throws Exception {
        WorkerStatus worker = RunnerTestSupport.discovered(RoleType.PREFILL, "g", "127.0.0.1", 8000, 8001, "site");
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillLeaseManager manager = manager(worker, endpoint);
        CountDownLatch attempted = new CountDownLatch(1);
        try (var executor = Executors.newSingleThreadExecutor()) {
            java.util.concurrent.Future<?> revoke;
            PrefillChunk reserved;
            worker.lock.lock();
            try {
                worker.prefillChunkCapacity()
                        .update(new PrefillChunk("e", 1, 1024, 1, System.currentTimeMillis() + 60000), 1024);
                revoke = executor.submit(() -> {
                    attempted.countDown();
                    manager.change("g", worker.getLogicalIpPort(), false);
                });
                assertTrue(attempted.await(2, TimeUnit.SECONDS));
                reserved = worker.prefillChunkCapacity().reserve(1024, System.currentTimeMillis(), Long.MAX_VALUE);
                assertNotNull(reserved);
            } finally {
                worker.lock.unlock();
            }
            revoke.get(2, TimeUnit.SECONDS);
            worker.lock.lock();
            try {
                assertNull(worker.prefillChunkCapacity().reserve(1, System.currentTimeMillis(), Long.MAX_VALUE));
                worker.prefillChunkCapacity().refundUnpublished(reserved, 1024);
                assertNull(worker.prefillChunkCapacity().definition());
                assertEquals(1L, reserved.chunkId());
            } finally {
                worker.lock.unlock();
            }
        }
    }

    @Test
    void acknowledgedRevocationStillDrainsMasterCommittedRequests() {
        WorkerStatus worker = RunnerTestSupport.discovered(RoleType.PREFILL, "g", "127.0.0.1", 8000, 8001, "site");
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillLeaseManager manager = manager(worker, endpoint);
        manager.change("g", worker.getLogicalIpPort(), false);
        worker.lock.lock();
        try {
            worker.prefillChunkLease().confirm(worker.prefillChunkLease().command().version(), false);
        } finally {
            worker.lock.unlock();
        }
        when(endpoint.getCommittedRequestCount()).thenReturn(1);
        assertEquals("DRAINING", manager.status("g").getFirst().state());
        when(endpoint.getCommittedRequestCount()).thenReturn(0);
        assertEquals("STANDBY", manager.status("g").getFirst().state());
    }

    private PrefillLeaseManager manager(WorkerStatus worker, PrefillEndpoint endpoint) {
        ConfigService configs = mock(ConfigService.class);
        FlexlbConfig config = new FlexlbConfig();
        config.getRouter().getRoles().getPrefill().getChunkLease().setEnabled(true);
        config.getRouter().getRoles().getPrefill().getChunkLease().setGroups(List.of("g"));
        when(configs.loadBalanceConfig()).thenReturn(config);
        WorkerDirectory directory = mock(WorkerDirectory.class);
        when(directory.statusSnapshot(RoleType.PREFILL)).thenReturn(Map.of(worker.getLogicalIpPort(), worker));
        when(directory.getWorkerStatuses(RoleType.PREFILL, "g")).thenReturn(List.of(worker));
        when(directory.isCurrentStatus(RoleType.PREFILL, worker.getLogicalIpPort(), worker)).thenReturn(true);
        when(directory.exactEndpoint(any(), any(), any())).thenReturn(endpoint);
        return new PrefillLeaseManager(configs, directory);
    }
}
