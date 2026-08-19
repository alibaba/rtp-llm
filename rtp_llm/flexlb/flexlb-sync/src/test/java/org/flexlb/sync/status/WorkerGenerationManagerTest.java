package org.flexlb.sync.status;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

class WorkerGenerationManagerTest {

    @Test
    void version_rotation_should_fence_old_endpoint_before_publishing_replacement()
            throws Exception {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus old = status(ipPort, 100);
        ConcurrentMap<String, WorkerStatus> statuses = new ConcurrentHashMap<>();
        statuses.put(ipPort, old);

        EndpointRegistry registry = mock(EndpointRegistry.class);
        CacheAwareService cache = mock(CacheAwareService.class);
        WorkerGenerationManager manager = new WorkerGenerationManager(
                registry, cache, new WorkerGenerationFence());
        CountDownLatch retirementEntered = new CountDownLatch(1);
        CountDownLatch allowRetirement = new CountDownLatch(1);
        doAnswer(invocation -> {
            retirementEntered.countDown();
            if (!allowRetirement.await(5, TimeUnit.SECONDS)) {
                throw new IllegalStateException("test retirement release timed out");
            }
            return true;
        }).when(registry).beginEndpointRetirement(
                eq(RoleType.PREFILL), eq(ipPort), eq(old));

        CompletableFuture<Boolean> rotation = CompletableFuture.supplyAsync(() ->
                manager.rotateOnVersionRollback(
                        statuses, RoleType.PREFILL, ipPort, old, 0));

        assertTrue(retirementEntered.await(5, TimeUnit.SECONDS));
        assertSame(old, statuses.get(ipPort));
        allowRetirement.countDown();

        assertTrue(rotation.get(5, TimeUnit.SECONDS));
        assertNotSame(old, statuses.get(ipPort));
        verify(registry).remove(RoleType.PREFILL, ipPort, old);
        verify(cache).clearEngineCache(ipPort);
    }

    private static WorkerStatus status(String ipPort, long version) {
        int separator = ipPort.lastIndexOf(':');
        WorkerStatus status = new WorkerStatus();
        status.setRole(RoleType.PREFILL);
        status.setIp(ipPort.substring(0, separator));
        status.setPort(Integer.parseInt(ipPort.substring(separator + 1)));
        status.setAlive(true);
        status.getStatusVersion().set(version);
        return status;
    }
}
