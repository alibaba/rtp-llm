package org.flexlb.dispatcher;

import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.test.StepVerifier;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FeHealthCheckerTest {

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
        FeHealthChecker checker = new FeHealthChecker(() -> List.of(url(feA)), WebClient.create());

        StepVerifier.create(checker.probeOnce()).verifyComplete();

        assertTrue(checker.isAlive(url(feA)));
    }

    @Test
    void unknownUrlAssumedAliveOptimistically() {
        FeHealthChecker checker = new FeHealthChecker(() -> List.of(), WebClient.create());
        assertTrue(checker.isAlive("http://never-probed"),
                "URL with no probe history must default to alive — never block traffic on missing data");
    }

    @Test
    void singleProbeFailureStillAliveForFlapTolerance() {
        feA.enqueue(new MockResponse().setResponseCode(500));
        FeHealthChecker checker = new FeHealthChecker(() -> List.of(url(feA)), WebClient.create());

        StepVerifier.create(checker.probeOnce()).verifyComplete();

        assertTrue(checker.isAlive(url(feA)),
                "1 failure is not enough — single-probe flap tolerance prevents transient noise from removing a healthy FE");
    }

    @Test
    void twoConsecutiveFailuresMarkDead() {
        feA.enqueue(new MockResponse().setResponseCode(500));
        feA.enqueue(new MockResponse().setResponseCode(500));
        FeHealthChecker checker = new FeHealthChecker(() -> List.of(url(feA)), WebClient.create());

        StepVerifier.create(checker.probeOnce()).verifyComplete();
        StepVerifier.create(checker.probeOnce()).verifyComplete();

        assertFalse(checker.isAlive(url(feA)));
    }

    @Test
    void singleSuccessAfterFailuresResetsCounter() {
        feA.enqueue(new MockResponse().setResponseCode(500));
        feA.enqueue(new MockResponse().setResponseCode(500));
        feA.enqueue(new MockResponse().setResponseCode(200).setBody("ok"));
        FeHealthChecker checker = new FeHealthChecker(() -> List.of(url(feA)), WebClient.create());

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
        FeHealthChecker checker = new FeHealthChecker(
                () -> List.of(url(feA), url(feB)), WebClient.create());

        StepVerifier.create(checker.probeOnce()).verifyComplete();
        StepVerifier.create(checker.probeOnce()).verifyComplete();

        assertTrue(checker.isAlive(url(feA)));
        assertFalse(checker.isAlive(url(feB)));
    }

    @Test
    void fePoolFiltersDeadHosts() {
        feA.enqueue(new MockResponse().setResponseCode(500));
        feA.enqueue(new MockResponse().setResponseCode(500));
        feB.enqueue(new MockResponse().setResponseCode(200).setBody("ok"));
        feB.enqueue(new MockResponse().setResponseCode(200).setBody("ok"));
        FeHealthChecker checker = new FeHealthChecker(
                () -> List.of(url(feA), url(feB)), WebClient.create());
        FePool pool = new FePool(() -> List.of(url(feA), url(feB)), checker::isAlive);

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
        FeHealthChecker checker = new FeHealthChecker(
                () -> List.of(url(feA), url(feB)), WebClient.create());
        FePool pool = new FePool(() -> List.of(url(feA), url(feB)), checker::isAlive);

        StepVerifier.create(checker.probeOnce()).verifyComplete();
        StepVerifier.create(checker.probeOnce()).verifyComplete();

        assertFalse(checker.isAlive(url(feA)));
        assertFalse(checker.isAlive(url(feB)));

        // Even though all dead, pool must still return something — refusing service
        // when probe data is stale is worse than gambling on a possibly-recovered host.
        String picked = pool.next();
        assertTrue(picked.equals(url(feA)) || picked.equals(url(feB)));
    }
}
