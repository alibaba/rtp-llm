package org.flexlb.sync.schedule;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.AppenderBase;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.scheduling.TaskScheduler;
import org.springframework.scheduling.annotation.ScheduledAnnotationBeanPostProcessor;
import org.springframework.scheduling.config.FixedRateTask;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class ExpirationCleanerTest {

    @Test
    void onlyConsecutiveRpcFailuresTriggerRetirement() {
        EndpointRegistry registry = mock(EndpointRegistry.class);
        CacheAwareService cache = mock(CacheAwareService.class);
        WorkerStatus status = status("127.0.0.1", 8080);
        WorkerDirectory directory = new WorkerDirectory(registry, Runnable::run);
        directory.currentOrDiscover(RoleType.PREFILL, status.getIpPort(), () -> status);
        ExpirationCleaner cleaner = new ExpirationCleaner(cache, directory);
        cleaner.retireUnavailableWorkers();
        assertTrue(status.isActiveGeneration());
        for (int failures = 1; failures <= 3; failures++) {
            WorkerStatus.PollLease lease = status.tryBeginStatusPoll(failures);
            status.recordStatusRpcResult(lease, false, false, new IllegalStateException("RPC failed"));
            lease.close();
            cleaner.retireUnavailableWorkers();
            assertEquals(failures < 3, status.isActiveGeneration());
        }
        verify(cache).removeEngineBlockCache(status, status.trace());
    }

    @Test
    @Timeout(10)
    void logsCountsWithoutWaitingForWorkerLock() throws Exception {
        var directory = new WorkerDirectory(mock(EndpointRegistry.class));
        WorkerStatus worker = status("127.0.0.1", 8080);
        directory.currentOrDiscover(RoleType.PREFILL, worker.getIpPort(), () -> worker);
        var cleaner = new ExpirationCleaner(mock(CacheAwareService.class), directory);
        var emitted = new CountDownLatch(1);
        var line = new AtomicReference<String>();
        var appender = new AppenderBase<ILoggingEvent>() {
            @Override
            protected void append(ILoggingEvent event) {
                if (event.getMessage().startsWith("event=worker_counts ")) {
                    line.set(event.getFormattedMessage());
                    emitted.countDown();
                }
            }
        };
        var logger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("syncLogger");
        Level previousLevel = logger.getLevel();
        logger.setLevel(Level.INFO);
        appender.start();
        logger.addAppender(appender);
        try (ExecutorService executor = Executors.newSingleThreadExecutor()) {
            Future<?> cleaning;
            worker.lock.lock();
            try {
                cleaning = executor.submit(cleaner::retireUnavailableWorkers);
                assertTrue(emitted.await(2, TimeUnit.SECONDS));
                cleaning.get(5, TimeUnit.SECONDS);
                assertTrue(line.get().contains("prefill_discovered=1 prefill_published=0"));
                assertTrue(line.get().contains("decode_discovered=0 decode_published=0"));
            } finally {
                worker.lock.unlock();
            }
        } finally {
            logger.detachAppender(appender);
            appender.stop();
            logger.setLevel(previousLevel);
        }
    }

    @Test
    void springSchedulesWorkerCleanupAtTheConfiguredInterval() {
        FlexlbConfig config = ConfigService.parse("""
                {"requestLifecycle":{"request":{"timeoutMs":60000}},
                 "workerRegistry":{"health":{"cleanupIntervalMs":1250}}}
                """);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        try (AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext()) {
            context.registerBean("configService", ConfigService.class, () -> configService);
            context.registerBean(CacheAwareService.class, () -> mock(CacheAwareService.class));
            context.registerBean(WorkerDirectory.class, () -> mock(WorkerDirectory.class));
            context.registerBean(TaskScheduler.class, () -> mock(TaskScheduler.class));
            context.registerBean(ScheduledAnnotationBeanPostProcessor.class);
            context.registerBean(ExpirationCleaner.class);
            context.refresh();
            var tasks = context.getBean(ScheduledAnnotationBeanPostProcessor.class).getScheduledTasks();
            assertEquals(1, tasks.size());
            FixedRateTask task = assertInstanceOf(FixedRateTask.class, tasks.iterator().next().getTask());
            assertEquals(1250L, task.getInterval());
        }
    }

    @Test
    void detachesEveryUnavailableWorkerBeforeAwaitingAnyRetirement()
            throws Exception {
        EndpointRegistry registry = mock(EndpointRegistry.class);
        CacheAwareService cache = mock(CacheAwareService.class);
        WorkerStatus first = status("127.0.0.1", 8080);
        WorkerStatus second = status("127.0.0.2", 8080);
        for (WorkerStatus status : new WorkerStatus[]{first, second}) {
            WorkerStatus.PollLease lease = status.tryBeginStatusPoll(1L);
            status.recordStatusRpcResult(lease, true, false, null);
            lease.close();
        }
        WorkerDirectory directory = new WorkerDirectory(registry);
        directory.currentOrDiscover(
                RoleType.PREFILL, first.getIpPort(), () -> first);
        directory.currentOrDiscover(
                RoleType.PREFILL, second.getIpPort(), () -> second);

        EndpointRegistry.DetachedGeneration firstDetached =
                mock(EndpointRegistry.DetachedGeneration.class);
        EndpointRegistry.DetachedGeneration secondDetached =
                mock(EndpointRegistry.DetachedGeneration.class);
        WorkerEndpoint firstEndpoint = mock(WorkerEndpoint.class);
        WorkerEndpoint secondEndpoint = mock(WorkerEndpoint.class);
        configureDetach(
                registry, first, firstEndpoint, firstDetached);
        configureDetach(
                registry, second, secondEndpoint, secondDetached);

        CountDownLatch firstAwaitEntered = new CountDownLatch(1);
        CountDownLatch releaseFirstAwait = new CountDownLatch(1);
        org.mockito.Mockito.doAnswer(invocation -> {
            firstAwaitEntered.countDown();
            assertTrue(releaseFirstAwait.await(5, TimeUnit.SECONDS));
            return null;
        }).when(firstDetached).retireAndAwait();

        ExpirationCleaner cleaner = new ExpirationCleaner(cache, directory);
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<?> cleaning = executor.submit(
                    cleaner::retireUnavailableWorkers);

            assertTrue(firstAwaitEntered.await(2, TimeUnit.SECONDS));
            assertFalse(first.isActiveGeneration());
            assertFalse(second.isActiveGeneration(),
                    "the second routing gate must close before the first drain waits");
            releaseFirstAwait.countDown();
            cleaning.get(5, TimeUnit.SECONDS);
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
            while (!directory.statusSnapshot(RoleType.PREFILL).isEmpty() && System.nanoTime() < deadline) {
                java.util.concurrent.locks.LockSupport.parkNanos(TimeUnit.MILLISECONDS.toNanos(1));
            }
            assertTrue(directory.statusSnapshot(RoleType.PREFILL).isEmpty());
            verify(cache).removeEngineBlockCache(
                    first, first.trace());
            verify(cache).removeEngineBlockCache(
                    second, second.trace());
        } finally {
            releaseFirstAwait.countDown();
            executor.shutdownNow();
        }
    }

    private static void configureDetach(
            EndpointRegistry registry,
            WorkerStatus status,
            WorkerEndpoint endpoint,
            EndpointRegistry.DetachedGeneration detached) {
        when(registry.get(
                RoleType.PREFILL, status.getIpPort(), status))
                .thenReturn(endpoint);
        when(detached.ownsEndpoint(endpoint)).thenReturn(true);
        when(registry.detachAndBeginRetirement(
                RoleType.PREFILL, status.getIpPort(), status))
                .thenAnswer(invocation -> {
                    status.beginRetirementAfterEndpointGateClosed();
                    return detached;
                });
    }

    private static WorkerStatus status(String ip, int port) {
        return WorkerStatus.createDiscovered(
                RoleType.PREFILL, null, ip, port, port + 1, "test-site");
    }
}
