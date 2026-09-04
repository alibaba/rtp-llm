package org.flexlb.dispatcher;

import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.http.HttpStatus;
import org.springframework.web.reactive.function.client.ClientResponse;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Sinks;
import reactor.test.StepVerifier;

import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;

import static org.flexlb.dispatcher.DispatcherTestSupport.feHealthChecker;
import static org.flexlb.dispatcher.DispatcherTestSupport.fePool;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class FeHealthCheckerTest {

    private static final String PROBE_PATH = "/frontend_health";

    private MockWebServer feA;
    private MockWebServer feB;

    @BeforeEach
    void setUp() throws Exception {
        feA = new MockWebServer();
        feA.start();
        feB = new MockWebServer();
        feB.start();
    }

    @AfterEach
    void tearDown() throws Exception {
        feA.shutdown();
        feB.shutdown();
    }

    private String url(MockWebServer s) {
        return "http://" + s.getHostName() + ":" + s.getPort();
    }

    @Test
    void healthyAfterSingleSuccessfulProbe() {
        feA.enqueue(new MockResponse().setResponseCode(200).setBody("ok"));
        FeHealthChecker checker = feHealthChecker(
                () -> List.of(url(feA)), WebClient.create(), PROBE_PATH);

        StepVerifier.create(checker.probeOnce()).verifyComplete();

        assertTrue(checker.isAlive(url(feA)));
    }

    @Test
    void unknownUrlAssumedAliveOptimistically() {
        FeHealthChecker checker = feHealthChecker(
                () -> List.of(), WebClient.create(), PROBE_PATH);
        assertTrue(checker.isAlive("http://never-probed"),
                "URL with no probe history must default to alive — never block traffic on missing data");
    }

    @Test
    void singleProbeFailureStillAliveForFlapTolerance() {
        feA.enqueue(new MockResponse().setResponseCode(500));
        FeHealthChecker checker = feHealthChecker(
                () -> List.of(url(feA)), WebClient.create(), PROBE_PATH);

        StepVerifier.create(checker.probeOnce()).verifyComplete();

        assertTrue(checker.isAlive(url(feA)),
                "1 failure is not enough — single-probe flap tolerance prevents transient noise from removing a healthy FE");
    }

    @Test
    void twoConsecutiveFailuresMarkDead() {
        feA.enqueue(new MockResponse().setResponseCode(500));
        feA.enqueue(new MockResponse().setResponseCode(500));
        FeHealthChecker checker = feHealthChecker(
                () -> List.of(url(feA)), WebClient.create(), PROBE_PATH);

        StepVerifier.create(checker.probeOnce()).verifyComplete();
        StepVerifier.create(checker.probeOnce()).verifyComplete();

        assertFalse(checker.isAlive(url(feA)));
    }

    @Test
    void singleSuccessAfterFailuresResetsCounter() {
        feA.enqueue(new MockResponse().setResponseCode(500));
        feA.enqueue(new MockResponse().setResponseCode(500));
        feA.enqueue(new MockResponse().setResponseCode(200).setBody("ok"));
        FeHealthChecker checker = feHealthChecker(
                () -> List.of(url(feA)), WebClient.create(), PROBE_PATH);

        StepVerifier.create(checker.probeOnce()).verifyComplete();
        StepVerifier.create(checker.probeOnce()).verifyComplete();
        assertFalse(checker.isAlive(url(feA)));

        StepVerifier.create(checker.probeOnce()).verifyComplete();
        assertTrue(checker.isAlive(url(feA)),
                "one successful probe wipes the consec counter — recovery must be immediate, not lagged");
    }

    @Test
    void probesAllUrlsInPool() {
        feA.enqueue(new MockResponse().setResponseCode(200).setBody("ok"));
        feB.enqueue(new MockResponse().setResponseCode(500));
        feB.enqueue(new MockResponse().setResponseCode(500));
        FeHealthChecker checker = feHealthChecker(
                () -> List.of(url(feA), url(feB)), WebClient.create(), PROBE_PATH);

        StepVerifier.create(checker.probeOnce()).verifyComplete();
        StepVerifier.create(checker.probeOnce()).verifyComplete();

        assertTrue(checker.isAlive(url(feA)));
        assertFalse(checker.isAlive(url(feB)));
        assertEquals(2, feA.getRequestCount(), "every round must probe FE A exactly once");
        assertEquals(2, feB.getRequestCount(), "every round must probe FE B exactly once");
    }

    @Test
    void fePoolFiltersDeadHosts() {
        feA.enqueue(new MockResponse().setResponseCode(500));
        feA.enqueue(new MockResponse().setResponseCode(500));
        feB.enqueue(new MockResponse().setResponseCode(200).setBody("ok"));
        feB.enqueue(new MockResponse().setResponseCode(200).setBody("ok"));
        FeHealthChecker checker = feHealthChecker(
                () -> List.of(url(feA), url(feB)), WebClient.create(), PROBE_PATH);
        FePool pool = fePool(() -> List.of(url(feA), url(feB)), checker::isAlive);

        StepVerifier.create(checker.probeOnce()).verifyComplete();
        StepVerifier.create(checker.probeOnce()).verifyComplete();

        // 10 picks: all should be feB since feA is dead
        for (int i = 0; i < 10; i++) {
            String picked = pool.next();
            assertTrue(picked.equals(url(feB)),
                    "FePool.next() must skip dead hosts; picked " + picked);
        }
    }

    @Test
    void fePoolFallsBackToRoundRobinWhenAllDead() {
        feA.enqueue(new MockResponse().setResponseCode(500));
        feA.enqueue(new MockResponse().setResponseCode(500));
        feB.enqueue(new MockResponse().setResponseCode(500));
        feB.enqueue(new MockResponse().setResponseCode(500));
        FeHealthChecker checker = feHealthChecker(
                () -> List.of(url(feA), url(feB)), WebClient.create(), PROBE_PATH);
        FePool pool = fePool(() -> List.of(url(feA), url(feB)), checker::isAlive);

        StepVerifier.create(checker.probeOnce()).verifyComplete();
        StepVerifier.create(checker.probeOnce()).verifyComplete();

        assertFalse(checker.isAlive(url(feA)));
        assertFalse(checker.isAlive(url(feB)));

        // Even though all dead, pool must still return something — refusing service
        // when probe data is stale is worse than gambling on a possibly-recovered host.
        String picked = pool.next();
        assertTrue(picked.equals(url(feA)) || picked.equals(url(feB)));
    }

    @Test
    void departedHostCounterIsEvictedOnNextProbeRound() {
        feA.enqueue(new MockResponse().setResponseCode(500));
        feA.enqueue(new MockResponse().setResponseCode(500));
        feB.enqueue(new MockResponse().setResponseCode(200).setBody("ok"));
        feB.enqueue(new MockResponse().setResponseCode(200).setBody("ok"));
        AtomicReference<List<String>> urls = new AtomicReference<>(List.of(url(feA), url(feB)));
        FeHealthChecker checker = feHealthChecker(urls::get, WebClient.create(), PROBE_PATH);

        StepVerifier.create(checker.probeOnce()).verifyComplete();
        StepVerifier.create(checker.probeOnce()).verifyComplete();
        assertFalse(checker.isAlive(url(feA)));

        // feA leaves the pool; the next probe round drops its counter so a re-added feA
        // starts from the optimistic default instead of inheriting stale dead state.
        urls.set(List.of(url(feB)));
        feB.enqueue(new MockResponse().setResponseCode(200).setBody("ok"));
        StepVerifier.create(checker.probeOnce()).verifyComplete();

        assertEquals(0, checker.consecFails(url(feA)));
        assertTrue(checker.isAlive(url(feA)));
    }

    @Test
    void emptyPoolClearsFailuresBeforeSameUrlIsReintroduced() {
        feA.enqueue(new MockResponse().setResponseCode(500));
        feA.enqueue(new MockResponse().setResponseCode(500));
        AtomicReference<List<String>> urls = new AtomicReference<>(List.of(url(feA)));
        FeHealthChecker checker = feHealthChecker(urls::get, WebClient.create(), PROBE_PATH);

        StepVerifier.create(checker.probeOnce()).verifyComplete();
        StepVerifier.create(checker.probeOnce()).verifyComplete();
        assertFalse(checker.isAlive(url(feA)));

        urls.set(List.of());
        StepVerifier.create(checker.probeOnce()).verifyComplete();
        urls.set(List.of(url(feA)));

        assertTrue(checker.isAlive(url(feA)),
                "a URL re-added after an empty snapshot must not inherit stale dead state");
        assertEquals(0, checker.consecFails(url(feA)));
    }

    @Test
    void scheduledLoopIsSingleFlightIdempotentAndStops() {
        ScheduledExecutorService scheduler = mock(ScheduledExecutorService.class);
        AtomicInteger schedulerCreations = new AtomicInteger();
        AtomicInteger exchanges = new AtomicInteger();
        Sinks.One<ClientResponse> firstResponse = Sinks.one();
        WebClient webClient = WebClient.builder().exchangeFunction(request ->
                exchanges.incrementAndGet() == 1
                        ? firstResponse.asMono()
                        : Mono.just(ClientResponse.create(HttpStatus.OK).build()))
                .build();
        FeHealthChecker checker = scheduledChecker(
                () -> List.of("http://fe"), webClient,
                () -> {
                    schedulerCreations.incrementAndGet();
                    return scheduler;
                });

        try {
            checker.start();
            checker.start();

            ArgumentCaptor<Runnable> task = ArgumentCaptor.forClass(Runnable.class);
            verify(scheduler, times(1)).scheduleAtFixedRate(
                    task.capture(), eq(0L), anyLong(), eq(TimeUnit.MILLISECONDS));
            assertEquals(1, schedulerCreations.get(), "repeated start must reuse one scheduler");

            task.getValue().run();
            assertEquals(1, exchanges.get());
            task.getValue().run();
            assertEquals(1, exchanges.get(), "a slow round must suppress the next tick");

            Assertions.assertEquals(
                    Sinks.EmitResult.OK,
                    firstResponse.tryEmitValue(ClientResponse.create(HttpStatus.OK).build()));
            task.getValue().run();
            assertEquals(2, exchanges.get(), "completion must reopen the single-flight gate");
        } finally {
            checker.stop();
            checker.stop();
        }
        verify(scheduler, times(1)).shutdownNow();
    }

    @Test
    void scheduledLoopRecoversAfterSynchronousSupplierFailure() {
        ScheduledExecutorService scheduler = mock(ScheduledExecutorService.class);
        AtomicInteger supplierCalls = new AtomicInteger();
        Supplier<List<String>> urls = () -> {
            if (supplierCalls.getAndIncrement() == 0) {
                throw new IllegalStateException("discovery snapshot unavailable");
            }
            return List.of();
        };
        FeHealthChecker checker = scheduledChecker(
                urls, WebClient.builder().exchangeFunction(request -> Mono.error(
                        new AssertionError("empty pool must not issue HTTP"))).build(),
                () -> scheduler);

        try {
            checker.start();
            ArgumentCaptor<Runnable> task = ArgumentCaptor.forClass(Runnable.class);
            verify(scheduler).scheduleAtFixedRate(
                    task.capture(), eq(0L), anyLong(), eq(TimeUnit.MILLISECONDS));

            task.getValue().run();
            task.getValue().run();
            assertEquals(2, supplierCalls.get(),
                    "a synchronous round failure must reopen the gate for the next tick");
        } finally {
            checker.stop();
        }
    }

    @Test
    void customProbePathHitsConfiguredEndpoint() throws Exception {
        feA.enqueue(new MockResponse().setResponseCode(200).setBody("ok"));
        FeHealthChecker checker = feHealthChecker(
                () -> List.of(url(feA)), WebClient.create(), "/health");

        StepVerifier.create(checker.probeOnce()).verifyComplete();

        assertTrue(checker.isAlive(url(feA)));
        // First (and only) recorded request must hit /health, proving the probe path
        // is wired through and not silently falling back to the old default.
        String hit = feA.takeRequest().getPath();
        assertTrue(hit != null && hit.startsWith("/health"),
                "probe must hit configured /health path, got: " + hit);
    }

    private static FeHealthChecker scheduledChecker(
            Supplier<List<String>> urls,
            WebClient webClient,
            Supplier<ScheduledExecutorService> schedulerFactory) {
        DispatcherFePoolRefresher refresher = mock(DispatcherFePoolRefresher.class);
        when(refresher.source()).thenReturn(urls);
        DispatchConfig cfg = new DispatchConfig();
        cfg.setProbePath(PROBE_PATH);
        return new FeHealthChecker(
                refresher, webClient, cfg, DispatcherTestSupport.noopMetrics(), schedulerFactory);
    }
}
