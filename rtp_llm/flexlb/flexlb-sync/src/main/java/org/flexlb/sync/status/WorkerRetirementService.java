package org.flexlb.sync.status;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/** Detach immediately; endpoint drain and address-scoped cache retirement never occupy polling threads. */
@Component
public class WorkerRetirementService {
    private static final Logger LOG = LoggerFactory.getLogger("syncLogger");
    private final EndpointRegistry endpoints;
    private final CacheAwareService caches;
    private static final class Pending {
        final RoleType role;
        final String address;
        final WorkerStatus worker;
        final AtomicBoolean submitted = new AtomicBoolean();
        volatile boolean detached;
        volatile long retryAfterNanos;
        Pending(RoleType role, String address, WorkerStatus worker) {
            this.role=role; this.address=address; this.worker=worker;
        }
    }
    private final Map<String, Pending> retiring = new ConcurrentHashMap<>();
    private final ExecutorService executor = new ThreadPoolExecutor(16, 16, 60, TimeUnit.SECONDS,
            new LinkedBlockingQueue<>(1024), new NamedThreadFactory("worker-retirement"),
            new ThreadPoolExecutor.AbortPolicy());
    private final ScheduledExecutorService retries = Executors.newSingleThreadScheduledExecutor(
            new NamedThreadFactory("worker-retirement-retry"));

    public WorkerRetirementService(EndpointRegistry endpoints, CacheAwareService caches) {
        this.endpoints = endpoints;
        this.caches = caches;
        retries.scheduleWithFixedDelay(() -> retiring.forEach((key, pending) -> submit(key,pending)),
                100, 100, TimeUnit.MILLISECONDS);
    }

    public boolean isRetiring(RoleType role, String address) { return retiring.containsKey(role + ":" + address); }

    public void retire(Map<String, WorkerStatus> current, RoleType role, String address, WorkerStatus worker) {
        String key = role + ":" + address;
        Pending pending = new Pending(role,address,worker);
        // Reserve this address before detaching it so rediscovery cannot publish a new owner during cleanup.
        if (retiring.putIfAbsent(key, pending) != null) return;
        if (current.get(address) != worker || !worker.getPollState().retire()) {
            retiring.remove(key, pending);
            return;
        }
        worker.setAlive(false);
        current.remove(address, worker);
        pending.detached=true;
        submit(key,pending);
    }

    private void submit(String key, Pending pending) {
        if (!pending.detached || System.nanoTime()<pending.retryAfterNanos || !pending.submitted.compareAndSet(false,true)) return;
        try { executor.execute(() -> {
            WorkerStatus worker=pending.worker;
            RoleType role=pending.role;
            String address=pending.address;
            try {
                worker.lock.lock();
                try { endpoints.remove(role, address, worker); }
                finally { worker.lock.unlock(); }
                worker.cacheMutationLock.lock();
                try { caches.retireEngineCache(address); }
                finally { worker.cacheMutationLock.unlock(); }
                LOG.info("Completed retired worker cleanup: {}, role: {}", address, role);
                retiring.remove(key, pending);
            } catch (Throwable error) {
                // Keep the tombstone: a new generation must never inherit partially retired cache state.
                LOG.error("Worker retirement cleanup failed; address remains fenced: {}, role: {}", address, role, error);
                pending.retryAfterNanos=System.nanoTime()+TimeUnit.SECONDS.toNanos(5);
                pending.submitted.set(false);
            }
        }); } catch (RejectedExecutionException rejected) { pending.submitted.set(false); }
    }

    @PreDestroy
    public void close() { retries.shutdownNow(); executor.shutdown(); }
}
