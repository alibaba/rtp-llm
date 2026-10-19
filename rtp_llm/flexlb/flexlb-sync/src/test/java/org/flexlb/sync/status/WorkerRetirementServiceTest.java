package org.flexlb.sync.status;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class WorkerRetirementServiceTest {
    @Test void slowCleanupDetachesImmediatelyAndFencesAddressReuse() throws Exception {
        var endpoints = mock(EndpointRegistry.class);
        var caches = mock(CacheAwareService.class);
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var returned = new CountDownLatch(1);
        String address="127.0.0.1:8000";
        doAnswer(call -> { entered.countDown(); assertTrue(release.await(5, TimeUnit.SECONDS)); return null; })
                .when(caches).retireEngineCache(address);
        var service = new WorkerRetirementService(endpoints, caches);
        var worker = new WorkerStatus();
        worker.setAlive(true);
        var current = new ConcurrentHashMap<String,WorkerStatus>();
        current.put(address, worker);
        try {
            Thread caller = new Thread(() -> { service.retire(current, RoleType.PREFILL, address, worker); returned.countDown(); });
            caller.start();
            assertTrue(returned.await(2, TimeUnit.SECONDS), "health caller must not wait for cache cleanup");
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            assertFalse(current.containsKey(address));
            assertTrue(worker.getPollState().isRetired());
            assertTrue(service.isRetiring(RoleType.PREFILL, address));
            var healthy = new WorkerStatus();
            healthy.getPollState().rpcSucceeded(100);
            assertFalse(healthy.getPollState().isExpired(101,1,10000));
            release.countDown();
            caller.join(1000);
            long until=System.nanoTime()+TimeUnit.SECONDS.toNanos(2);
            while (service.isRetiring(RoleType.PREFILL,address) && System.nanoTime()<until) Thread.sleep(5);
            assertFalse(service.isRetiring(RoleType.PREFILL,address));
            verify(caches, times(1)).retireEngineCache(address);
        } finally { release.countDown(); service.close(); }
    }

    @Test void lateRetirementCannotDetachANewerGeneration() {
        var endpoints=mock(EndpointRegistry.class);
        var caches=mock(CacheAwareService.class);
        var service=new WorkerRetirementService(endpoints,caches);
        var current=new ConcurrentHashMap<String,WorkerStatus>();
        var old=new WorkerStatus();var replacement=new WorkerStatus();
        current.put("worker:80",replacement);
        try {
            service.retire(current,RoleType.PREFILL,"worker:80",old);
            assertSame(replacement,current.get("worker:80"));
            assertFalse(old.getPollState().isRetired());
            assertFalse(service.isRetiring(RoleType.PREFILL,"worker:80"));
            verifyNoInteractions(caches,endpoints);
        } finally { service.close(); }
    }
}
