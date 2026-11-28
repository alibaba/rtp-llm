package org.flexlb.mock.frontend;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import org.flexlb.mock.protocol.MultimodalProtocol;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.flexlb.mock.frontend.FrontendProtocol.*;

/**
 * Verifies paced playback, concurrency, and shutdown for one or more Frontends.
 */
class PlaybackTest {
    @TempDir Path directory;

    Path topology(BackendRpcTransportTest.Harness h) throws Exception {
        var path = directory.resolve(UUID.randomUUID() + ".json");
        new ObjectMapper().writeValue(path.toFile(), List.of(h.endpoint())); return path;
    }

    @Test void ttftHistogramDistinguishesSlowCallsAboveThirtySeconds() {
        var metrics = new FrontendMetrics();
        metrics.finished(new BackendRpcTransport.Result(1, java.time.Instant.now(), TimeUnit.SECONDS.toNanos(45),
            TimeUnit.SECONDS.toNanos(46), 2, BackendRpcTransport.Outcome.success, "OK", 0, "", TimeUnit.SECONDS.toNanos(1)));
        var scrape = metrics.scrape();
        assertTrue(scrape.contains("py_rtp_success_qps_metric_total{role=\"frontend\",} 1.0"));
        assertTrue(scrape.contains("mock_frontend_ttft_seconds_bucket{role=\"frontend\",le=\"30.0\",} 0.0"));
        assertTrue(scrape.contains("mock_frontend_ttft_seconds_bucket{role=\"frontend\",le=\"60.0\",} 1.0"));
    }

    @Test void naturalEndDrainsQueueAndConsumesRealNativeResponses() throws Exception {
        try (var h = new BackendRpcTransportTest.Harness(); var load = worker(frontendReplayPlan(master(h), 3, 10, 2, 4), new FrontendMetrics(), h)) {
            load.begin(System.currentTimeMillis() + 30);
            var result = load.await(5000);
            assertTrue(result.done()); assertFalse(result.stopped()); assertTrue(result.errors().isEmpty(), result.toString());
            assertEquals(3, result.issued()); assertEquals(3, result.finished()); assertEquals(0, result.unissued());
            assertEquals(3, h.schedules.get());
            assertFalse(h.generated.get().getGenerateConfig().getIgnoreEos());
            assertEquals(3, h.generated.get().getGenerateConfig().getMaxNewTokens());
        }
    }

    FrontendReplayPlan frontendReplayPlan(String master, int count, double qps, int slots, int queue) throws Exception {
        var start = InputFixture.plan(directory, List.of(master), count, qps, slots, queue, 3000, 1000);
        try (var input = new IndexedTrafficReader(Path.of(start.requests_file()))) {
            return new FrontendReplayPlan(1, start.masters(), start.requests_file(), input.count, 0, 1, 1, input.identity,
                qps, null, slots, queue, 3000, 1000);
        }
    }

    static String master(BackendRpcTransportTest.Harness h) { return "127.0.0.1:" + h.master.getPort(); }

    @Test void fullQueueFailsInsteadOfBlockingDroppingOrExceedingConcurrency() throws Exception {
        try (var h = new BackendRpcTransportTest.Harness(); var load = worker(frontendReplayPlan(master(h), 30, 500, 1, 1), new FrontendMetrics(), h)) {
            h.tokenDelay = 300;
            load.begin(System.currentTimeMillis() + 30);
            var result = load.await(5000);
            assertTrue(result.errors().getFirst().contains("queue full"));
            assertTrue(result.issued() <= 1); assertEquals(result.issued(), result.finished());
            assertEquals(30 - result.issued(), result.unissued());
        }
    }

    @Test void stopBeforeBeginDoesNotSendAndIsIdempotent() throws Exception {
        try (var h = new BackendRpcTransportTest.Harness(); var load = worker(frontendReplayPlan(master(h), 3, 10, 1, 2), new FrontendMetrics(), h)) {
            load.halt(); load.halt();
            var result = load.await(5000);
            assertEquals(0, result.issued()); assertEquals(3, result.unissued()); assertTrue(result.errors().isEmpty());
            assertThrows(IllegalStateException.class, () -> load.begin(System.currentTimeMillis() + 30));
        }
    }

    @Test void stateRemainsReadableWhileDispatchOwnsItsMonitor() throws Exception {
        try (var h = new BackendRpcTransportTest.Harness();
             var load = worker(frontendReplayPlan(master(h), 3, 10, 1, 2), new FrontendMetrics(), h)) {
            synchronized (load) {
                var state = new FutureTask<>(load::state);
                Thread.ofVirtual().start(state);
                assertFalse(state.get(1, TimeUnit.SECONDS).done());
            }
            load.halt();
            var result = load.await(5000);
            assertEquals(result, new ObjectMapper().readValue(directory.resolve("result.json").toFile(), State.class));
        }
    }

    @Test void stopClosesAdmissionAndDrainTimeoutCancelsInflight() throws Exception {
        try (var h = new BackendRpcTransportTest.Harness()) {
            h.tokenDelay = 5000;
            var plan = frontendReplayPlan(master(h), 10, 20, 1, 10);
            plan = new FrontendReplayPlan(1, plan.masters(), plan.requests_file(), plan.count(), 0, 1, 1, plan.sha256(), plan.qps(), plan.replay_speed(), 1, 10, 10000, 80);
            try (var load = worker(plan, new FrontendMetrics(), h)) {
                load.begin(System.currentTimeMillis() + 30);
                assertTrue(h.scheduleEntered.await(2, TimeUnit.SECONDS));
                load.halt(); long issued = load.state().issued();
                var result = load.await(5000);
                assertEquals(issued, result.issued()); assertEquals(issued, result.finished());
                assertEquals(0, result.inflight()); assertTrue(result.errors().getFirst().contains("drain timed out"));
            }
        }
    }

    @Test void businessErrorsAreRecordedWithoutFailingPlayback() throws Exception {
        try (var h = new BackendRpcTransportTest.Harness()) {
            h.route = r -> r.toBuilder().setSuccess(false).setCode(429).clearLifecycle().build();
            var metrics = new FrontendMetrics();
            try (var load = worker(frontendReplayPlan(master(h), 3, 10, 1, 4), metrics, h)) {
                load.begin(System.currentTimeMillis() + 30);
                var result = load.await(5000);
                assertEquals(3, result.finished()); assertTrue(result.errors().isEmpty());
                assertTrue(metrics.scrape().contains("py_rtp_success_qps_metric_total{role=\"frontend\",} 0.0"));
                assertTrue(metrics.scrape().contains("mock_frontend_requests_finished_total{role=\"frontend\",outcome=\"error\",} 3.0"));
            }
        }
    }

    @Test void threeFrontendsSplitGlobalCountAndUseTcpEvenForFrontendZero() throws Exception {
        Path topology = directory.resolve("topology.json");
        try (var h = new BackendRpcTransportTest.Harness();
             var zero = new FrontendNode("frontend-0", topology, directory.resolve("zero"));
             var one = new FrontendNode("frontend-1", topology, directory.resolve("one"));
             var two = new FrontendNode("frontend-2", topology, directory.resolve("two"))) {
            new ObjectMapper().writeValue(topology.toFile(), List.of(h.endpoint(),
                Map.of("role", "frontend", "control", zero.url()), Map.of("role", "frontend", "control", one.url()), Map.of("role", "frontend", "control", two.url())));
            var start = InputFixture.plan(directory, null, 7, 10, 4, 4, 3000, 1000);
            assertThrows(IllegalStateException.class, () -> one.startReplay(start));
            zero.startReplay(start);
            assertThrows(IllegalStateException.class, () -> zero.startReplay(start));
            eventually(() -> zero.state(new Sequence(1)).done() && one.state(new Sequence(1)).done() && two.state(new Sequence(1)).done());
            var result = zero.stopReplay(new Empty());
            assertTrue(result.errors().isEmpty(), result.toString()); assertEquals(3, result.frontends().size());
            assertEquals(7, result.frontends().values().stream().mapToLong(State::issued).sum());
            assertEquals(7, h.schedules.get()); assertEquals(result, zero.stopReplay(new Empty()));
            zero.startReplay(start);
            assertThrows(IllegalStateException.class, () -> one.halt(new Sequence(1)));
            zero.stopReplay(new Empty());
            assertTrue(Files.exists(directory.resolve("one/playback/1/result.json")));
            assertTrue(Files.exists(directory.resolve("zero/coordinator/2/result.json")));
        }
    }

    static void eventually(java.util.function.BooleanSupplier condition) throws Exception {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(8);
        while (!condition.getAsBoolean() && System.nanoTime() < deadline) Thread.sleep(10);
        assertTrue(condition.getAsBoolean(), "Condition did not become true");
    }

    @Test void transientWorkerStateTimeoutDoesNotStopPlayback() throws Exception {
        var peer = com.sun.net.httpserver.HttpServer.create(new java.net.InetSocketAddress("127.0.0.1", 0), 0);
        var timeouts = new java.util.concurrent.atomic.AtomicInteger();
        var halted = new java.util.concurrent.atomic.AtomicBoolean();
        var json = new ObjectMapper();
        peer.createContext("/invoke", exchange -> {
            try (exchange) {
                String operation = json.readTree(exchange.getRequestBody()).get("interface").asText();
                if (operation.equals("worker/halt")) halted.set(true);
                if (operation.equals("worker/state") && timeouts.getAndIncrement() == 0) {
                    exchange.sendResponseHeaders(504, -1);
                    return;
                }
                Object result = operation.equals("worker/state")
                    ? new State(true, false, 1, 1, 1, 0, 0, 0, Map.of("success", 1L), List.of())
                    : new Ack(true);
                byte[] body = json.writeValueAsBytes(Map.of("result", result));
                exchange.sendResponseHeaders(200, body.length);
                exchange.getResponseBody().write(body);
            }
        });
        peer.start();
        Path topology = directory.resolve("topology.json");
        try (var h = new BackendRpcTransportTest.Harness();
             var zero = new FrontendNode("frontend-0", topology, directory.resolve("zero"))) {
            json.writeValue(topology.toFile(), List.of(h.endpoint(),
                Map.of("role", "frontend", "control", zero.url()),
                Map.of("role", "frontend", "control", "http://127.0.0.1:" + peer.getAddress().getPort())));
            zero.startReplay(InputFixture.plan(directory, null, 2, 10, 2, 4, 3000, 1000));
            var report = directory.resolve("zero/coordinator/1/result.json");
            eventually(() -> Files.exists(report));
            var summary = zero.stopReplay(new Empty());
            assertTrue(summary.errors().isEmpty(), summary.toString());
            assertFalse(halted.get());
            assertEquals(2, timeouts.get());
            assertEquals(1, h.schedules.get());
        } finally { peer.stop(0); }
    }

    @Test void replayCoordinatesWorkersAndEarlyStopInterruptsLongArrivalWaits() throws Exception {
        for (boolean early : List.of(false, true)) {
            Path topology = directory.resolve("replay-" + early + ".json");
            Path source = InputFixture.write(directory.resolve("replay-" + early + ".mckidx"),
                java.util.stream.IntStream.range(0, 6).mapToObj(i -> i * (early ? 30000L : 100L)).toList());
            try (var h = new BackendRpcTransportTest.Harness();
                 var zero = new FrontendNode("frontend-0", topology, directory.resolve("zero-" + early));
                 var one = new FrontendNode("frontend-1", topology, directory.resolve("one-" + early))) {
                new ObjectMapper().writeValue(topology.toFile(), List.of(h.endpoint(),
                    Map.of("role", "frontend", "control", zero.url()), Map.of("role", "frontend", "control", one.url())));
                zero.startReplay(new Start(List.of(master(h)), source.toString(), null, 5.0, 4, 8, 3000, 1000, true, null, null, null));
                assertTrue(h.scheduleEntered.await(3, TimeUnit.SECONDS));
                if (!early) eventually(() -> zero.state(new Sequence(1)).done() && one.state(new Sequence(1)).done());
                var result = zero.stopReplay(new Empty());
                assertTrue(result.errors().isEmpty(), result.toString());
                assertEquals(early ? 1 : 6, result.frontends().values().stream().mapToLong(State::issued).sum());
                assertTrue(h.generated.get().getGenerateConfig().getReuseCache());
                assertEquals(early ? 5 : 0, result.frontends().values().stream().mapToLong(State::unissued).sum());
                assertTrue(result.frontends().values().stream().allMatch(State::done));
            }
        }
    }

    @Test void oneWorkerOverflowStopsWholeGroupAndCannotBecomeSuccessfulSummary() throws Exception {
        Path topology = directory.resolve("topology.json");
        try (var h = new BackendRpcTransportTest.Harness();
             var zero = new FrontendNode("frontend-0", topology, directory.resolve("zero"));
             var one = new FrontendNode("frontend-1", topology, directory.resolve("one"))) {
            h.tokenDelay = 500;
            new ObjectMapper().writeValue(topology.toFile(), List.of(h.endpoint(),
                Map.of("role", "frontend", "control", zero.url()), Map.of("role", "frontend", "control", one.url())));
            zero.startReplay(InputFixture.plan(directory, List.of(master(h)), 100, 500, 2, 1, 3000, 1000));
            eventually(() -> zero.state(new Sequence(1)).done() && one.state(new Sequence(1)).done());
            var result = zero.stopReplay(new Empty());
            assertFalse(result.errors().isEmpty()); assertTrue(result.frontends().values().stream().allMatch(State::done));
            assertTrue(result.frontends().values().stream().mapToLong(State::issued).sum() < 100);
        }
    }

    @Test void unreachablePeerStopsTheRemainingFrontendAndKeepsMissingResultsVisible() throws Exception {
        Path topology = directory.resolve("topology.json");
        try (var h = new BackendRpcTransportTest.Harness();
             var zero = new FrontendNode("frontend-0", topology, directory.resolve("zero"));
             var one = new FrontendNode("frontend-1", topology, directory.resolve("one"))) {
            new ObjectMapper().writeValue(topology.toFile(), List.of(h.endpoint(),
                Map.of("role", "frontend", "control", zero.url()), Map.of("role", "frontend", "control", one.url())));
            zero.startReplay(InputFixture.plan(directory, List.of(master(h)), 20, 2, 2, 4, 3000, 1000));
            one.close();
            eventually(() -> zero.state(new Sequence(1)).done());
            var result = zero.stopReplay(new Empty());
            assertFalse(result.errors().isEmpty());
            assertTrue(zero.state(new Sequence(1)).stopped());
            assertTrue(zero.state(new Sequence(1)).issued() < 10);
        }
    }

    @Test void malformedFileFailsBeforeAnyWorkerStarts() throws Exception {
        Path topology = directory.resolve("topology.json"), input = directory.resolve("bad.mckidx");
        Files.writeString(input, "{\"token_ids\":[1],\"max_new_tokens\":2}\nnot-json\n");
        try (var h = new BackendRpcTransportTest.Harness(); var zero = new FrontendNode("frontend-0", topology, directory.resolve("zero"))) {
            new ObjectMapper().writeValue(topology.toFile(), List.of(h.endpoint(), Map.of("role", "frontend", "control", zero.url())));
            assertThrows(java.io.IOException.class, () -> zero.startReplay(new Start(null, input.toString(), 10.0, null, 1, 4, 3000, 1000)));
            assertFalse(zero.stopReplay(new Empty()).errors().isEmpty());
            assertEquals(0, h.schedules.get());
            assertFalse(Files.exists(directory.resolve("zero/playback/1/plan.json")));
        }
    }

    @Test void corruptedSharedInputStopsTheWholeGroup() throws Exception {
        Path topology = directory.resolve("topology.json"), input = directory.resolve("input.mckidx");
        InputFixture.write(input, Collections.nCopies(20, 0L));
        try (var h = new BackendRpcTransportTest.Harness();
             var zero = new FrontendNode("frontend-0", topology, directory.resolve("zero"));
             var one = new FrontendNode("frontend-1", topology, directory.resolve("one"))) {
            new ObjectMapper().writeValue(topology.toFile(), List.of(h.endpoint(),
                Map.of("role", "frontend", "control", zero.url()), Map.of("role", "frontend", "control", one.url())));
            byte[] data = Files.readAllBytes(input);
            int payload = Math.toIntExact(java.nio.ByteBuffer.wrap(data).getLong(56));
            data[payload] ^= 1;
            Files.write(input, data);
            zero.startReplay(new Start(null, input.toString(), 2.0, null, 2, 4, 3000, 1000));
            eventually(() -> zero.state(new Sequence(1)).done() && one.state(new Sequence(1)).done());
            var result = zero.stopReplay(new Empty());
            assertFalse(result.errors().isEmpty());
            assertTrue(result.frontends().values().stream().allMatch(State::stopped));
            assertTrue(result.frontends().values().stream().mapToLong(State::issued).sum() < 20);
        }
    }

    @Test void impossiblePlansAreRejectedBeforeThreadsStart() {
        assertThrows(IllegalArgumentException.class, () -> new Start(List.of("m:1"), "/input.mckidx", Double.NaN, null, 1, 1, 100, 100));
        assertThrows(IllegalArgumentException.class, () -> new Start(List.of("m:1"), "/input.mckidx", 0.0, null, 1, 1, 100, 100));
        assertThrows(IllegalArgumentException.class, () -> new Start(null, null, 1.0, null, 1, 1, 100, 100));
        assertThrows(IllegalArgumentException.class, () -> new Start(null, "/input.mckidx", 1.0, null, 1, 1, 100, 100, false, null, null, 1));
    }

    @Test void duplicateEveryIssuesAConcurrentSecondCallReusingTheRequestId() throws Exception {
        Path topology = directory.resolve("topology.json");
        Path input = InputFixture.write(directory.resolve("dup.mckidx"), java.util.Collections.nCopies(4, 0L));
        try (var h = new BackendRpcTransportTest.Harness();
             var zero = new FrontendNode("frontend-0", topology, directory.resolve("zero"))) {
            new ObjectMapper().writeValue(topology.toFile(), List.of(h.endpoint(),
                Map.of("role", "frontend", "control", zero.url())));
            zero.startReplay(new Start(null, input.toString(), 50.0, null, 8, 16, 3000, 3000, false, null, null, 2));
            eventually(() -> zero.state(new Sequence(1)).done());
            var result = zero.stopReplay(new Empty());
            assertTrue(result.errors().isEmpty(), result.toString());
            var state = result.frontends().values().iterator().next();
            assertTrue(state.done());
            assertEquals(6, state.issued());
            assertEquals(6, state.finished());
            assertEquals(6L, state.outcomes().values().stream().mapToLong(Long::longValue).sum());
            assertEquals(6L, (long) state.outcomes().get("success"));
            // Positions 0 and 2 are duplicated: two extra schedule calls reuse their request ids.
            assertEquals(6, h.schedules.get());
            var frequency = new java.util.HashMap<Long, Integer>();
            for (var schedule : h.scheduleInputs) frequency.merge(schedule.getRequestId(), 1, Integer::sum);
            assertEquals(4, frequency.size());
            assertEquals(2, frequency.values().stream().filter(count -> count == 2).count());
        }
    }

    private ReplayWorker worker(FrontendReplayPlan plan, FrontendMetrics metrics, BackendRpcTransportTest.Harness harness)
            throws Exception {
        return new ReplayWorker(plan, metrics, topology(harness), directory.resolve(UUID.randomUUID() + ".jsonl"),
            "frontend-0", MultimodalProtocol.IMAGE_TOKEN, 0, 0L, 1);
    }
}
